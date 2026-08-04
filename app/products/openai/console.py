"""Basic-account console.x.ai Responses transport and OpenAI adapters."""

from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
import random
import re
import string
import uuid
from typing import Any, AsyncGenerator, AsyncIterable
from urllib.parse import urlsplit

import orjson

from app.control.account.enums import FeedbackKind
from app.control.account.commands import AccountPatch
from app.control.account.console_usage import (
    EXT_KEY as CONSOLE_USAGE_EXT_KEY,
    console_mode_id_for_model,
    console_usage_key_for_model,
    increment_console_usage,
)
from app.dataplane.proxy.adapters.headers import build_sso_cookie
from app.dataplane.proxy.adapters.profile import (
    browser_from_user_agent,
    extract_cookie_value,
)
from app.dataplane.proxy.adapters.session import ResettableSession, build_session_kwargs
from app.platform.config.snapshot import get_config
from app.platform.errors import RateLimitError, UpstreamError
from app.platform.logging.logger import logger
from app.platform.runtime.clock import now_ms, now_s
from app.platform.tokens import estimate_prompt_tokens, estimate_tokens
from app.products._account_selection import selection_max_retries

from ._format import (
    build_usage,
    format_sse,
    make_chat_response,
    make_resp_id,
    make_response_id,
    make_stream_chunk,
    make_thinking_chunk,
)
from .dpop import DpopMintError, mint_dpop_credentials


_BASIC_POOL_ID = 0
_SYNTHETIC_REASONING_TEXT = "已深度思考。"
_DEFAULT_CONSOLE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0"
)
_DEFAULT_CONSOLE_BROWSER = "edge"

_CONSOLE_MODEL_MAP: dict[str, str] = {
    "grok-4.20-fast-console": "grok-4.20-0309-non-reasoning",
    "grok-4.20-0309-non-reasoning-console": "grok-4.20-0309-non-reasoning",
    "grok-4.20-0309-console": "grok-4.20-0309",
    "grok-4.20-0309-reasoning-console": "grok-4.20-0309-reasoning",
    "grok-4.20-reasoning-console": "grok-4.20-0309-reasoning",
    "grok-4.20-expert-console": "grok-4.20-0309-reasoning",
    "grok-4.3-console": "grok-4.3",
    "grok-4.3-beta-console": "grok-4.3",
    "grok-4.3-low-console": "grok-4.3",
    "grok-4.3-medium-console": "grok-4.3",
    "grok-4.3-high-console": "grok-4.3",
    "grok-4.5-console": "grok-4.5",
    "grok-4.20-multi-agent-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-heavy-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-0309-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-heavy-low-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-heavy-medium-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-heavy-high-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-heavy-xhigh-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-low-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-medium-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-high-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-xhigh-console": "grok-4.20-multi-agent-0309",
    "grok-build-console": "grok-build-0.1",
}

_CONSOLE_FIXED_EFFORT: dict[str, str] = {
    "grok-4.3-low-console": "low",
    "grok-4.3-medium-console": "medium",
    "grok-4.3-high-console": "high",
    "grok-4.20-heavy-low-console": "low",
    "grok-4.20-heavy-medium-console": "medium",
    "grok-4.20-heavy-high-console": "high",
    "grok-4.20-heavy-xhigh-console": "xhigh",
    "grok-4.20-multi-agent-low-console": "low",
    "grok-4.20-multi-agent-medium-console": "medium",
    "grok-4.20-multi-agent-high-console": "high",
    "grok-4.20-multi-agent-xhigh-console": "xhigh",
}

_CONSOLE_EFFORT_MAP: dict[str, str] = {
    "none": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

_CONSOLE_MODELS_WITH_REASONING_FIELD = frozenset({
    "grok-4.3",
    "grok-4.5",
    "grok-4.20-multi-agent-0309",
})

_CONSOLE_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "grok-4.20-multi-agent-0309": 2_000_000,
}
_CONSOLE_RETRYABLE_STATUSES = frozenset({401, 403, 429, 503})


class NoBasicConsoleAccount(Exception):
    """Raised internally when the basic pool cannot serve a console request."""


def is_console_basic_model(model: str) -> bool:
    return model in _CONSOLE_MODEL_MAP


def console_upstream_model(model: str) -> str:
    return _CONSOLE_MODEL_MAP.get(model, model)


def _console_reasoning_include(include: list[str] | None, emit_think: bool) -> list[str] | None:
    if not emit_think:
        return include
    values = list(include or [])
    if "reasoning.encrypted_content" not in values:
        values.append("reasoning.encrypted_content")
    return values


def _make_console_thinking_chunk(response_id: str, model: str, content: str) -> dict:
    chunk = make_thinking_chunk(response_id, model, content)
    try:
        delta = chunk["choices"][0]["delta"]
    except Exception:
        return chunk
    # Keep reasoning_content for existing clients, and add common aliases for
    # clients that only open their thinking UI on reasoning/thinking fields.
    delta.setdefault("reasoning", content)
    delta.setdefault("thinking", content)
    return chunk


def _make_console_chat_response(
    model: str,
    content: str,
    *,
    prompt_content: Any | None = None,
    response_id: str | None = None,
    usage: dict | None = None,
    reasoning_content: str | None = None,
) -> dict:
    resp = make_chat_response(
        model,
        content,
        prompt_content=prompt_content,
        response_id=response_id,
        usage=usage,
        reasoning_content=reasoning_content,
    )
    if reasoning_content:
        msg = resp["choices"][0]["message"]
        msg.setdefault("reasoning", reasoning_content)
        msg.setdefault("thinking", reasoning_content)
    return resp


def _synthetic_response_reasoning_item(reasoning_id: str, *, status: str = "completed") -> dict:
    return {
        "id": reasoning_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": _SYNTHETIC_REASONING_TEXT}],
        "status": status,
    }


def _synthetic_response_reasoning_events(reasoning_id: str) -> list[str]:
    return [
        format_sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": reasoning_id, "type": "reasoning", "summary": [], "status": "in_progress"},
        }),
        format_sse("response.reasoning_summary_part.added", {
            "type": "response.reasoning_summary_part.added",
            "item_id": reasoning_id,
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        }),
        format_sse("response.reasoning_summary_text.delta", {
            "type": "response.reasoning_summary_text.delta",
            "item_id": reasoning_id,
            "output_index": 0,
            "summary_index": 0,
            "delta": _SYNTHETIC_REASONING_TEXT,
        }),
        format_sse("response.reasoning_summary_text.done", {
            "type": "response.reasoning_summary_text.done",
            "item_id": reasoning_id,
            "output_index": 0,
            "summary_index": 0,
            "text": _SYNTHETIC_REASONING_TEXT,
        }),
        format_sse("response.reasoning_summary_part.done", {
            "type": "response.reasoning_summary_part.done",
            "item_id": reasoning_id,
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": _SYNTHETIC_REASONING_TEXT},
        }),
        format_sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": _synthetic_response_reasoning_item(reasoning_id),
        }),
    ]


def _default_console_tools() -> list[dict[str, Any]]:
    if not get_config().get_bool("console.enable_search_tools", True):
        return []
    return [
        {"type": "web_search", "enable_image_understanding": True},
        {"type": "x_search", "enable_video_understanding": True},
    ]


def _console_reasoning_effort(model: str, reasoning_effort: str | None) -> str:
    return _CONSOLE_FIXED_EFFORT.get(model) or _CONSOLE_EFFORT_MAP.get(
        reasoning_effort or "medium",
        "medium",
    )


def _console_url() -> str:
    return get_config().get_str("console.responses_url", "https://console.x.ai/v1/responses")


def _console_dpop_token_url() -> str:
    parsed = urlsplit(_console_url())
    return f"{parsed.scheme}://{parsed.netloc}/v1/dpop/token"


def _console_cluster() -> str:
    return get_config().get_str("console.cluster", "https://us-east-1.api.x.ai")


def _console_stream_heartbeat_interval() -> float:
    interval = get_config().get_float(
        "console.stream_heartbeat_interval",
        get_config().get_float("chat.stream_heartbeat_interval", 15.0),
    )
    return max(0.0, interval)


def _console_referer() -> str:
    team_id = get_config().get_str("console.team_id", "").strip()
    if team_id:
        return f"https://console.x.ai/team/{team_id}/chat-playground"
    return "https://console.x.ai/"


def _lease_user_agent(lease: Any | None) -> str:
    return str(getattr(lease, "user_agent", "") or "").strip()


def _console_user_agent(lease: Any | None = None) -> str:
    # FlareSolverr returns the browser identity that was used to obtain the
    # clearance cookie.  Reuse it for Console requests so the Cookie and
    # User-Agent remain one browser session.  The configured Console value is
    # only a fallback for manual/disabled clearance modes.
    lease_user_agent = _lease_user_agent(lease)
    if lease_user_agent:
        return lease_user_agent
    return get_config().get_str(
        "console.user_agent",
        _DEFAULT_CONSOLE_USER_AGENT,
    ).strip() or _DEFAULT_CONSOLE_USER_AGENT


def _console_browser_override(lease: Any | None = None) -> str | None:
    lease_user_agent = _lease_user_agent(lease)
    if lease_user_agent:
        browser = browser_from_user_agent(lease_user_agent)
        if browser:
            return browser
    browser = get_config().get_str("console.browser", "").strip()
    return browser or _DEFAULT_CONSOLE_BROWSER


def _console_sso_token(token: str) -> str:
    raw = str(token or "").strip()
    tok = extract_cookie_value(raw, "sso") if "sso=" in raw else raw
    if tok.startswith("sso="):
        tok = tok[4:]
    tok = "".join(str(tok).split())
    return tok


def _console_sso_cookie(token: str, lease=None) -> str:
    return build_sso_cookie(_console_sso_token(token), lease=lease)


def _console_statsig_id() -> str:
    if random.choice((True, False)):
        rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        msg = f"x1:TypeError: Cannot read properties of null (reading 'children['{rand}']')"
    else:
        rand = "".join(random.choices(string.ascii_lowercase, k=10))
        msg = f"x1:TypeError: Cannot read properties of undefined (reading '{rand}')"
    return base64.b64encode(msg.encode()).decode()


def _console_client_hints(user_agent: str) -> dict[str, str]:
    ua = user_agent or ""
    edge = re.search(r"Edg/(\d+)", ua)
    chrome = re.search(r"(?:Chrome|Chromium)/(\d+)", ua)
    version = (edge or chrome).group(1) if (edge or chrome) else "120"
    brand = "Microsoft Edge" if edge else "Google Chrome"
    platform = "Windows"
    if "Mac OS X" in ua or "Macintosh" in ua:
        platform = "macOS"
    elif "Android" in ua:
        platform = "Android"
    elif "iPhone" in ua or "iPad" in ua:
        platform = "iOS"
    elif "Linux" in ua:
        platform = "Linux"
    return {
        "Sec-Ch-Ua": f'"Chromium";v="{version}", "{brand}";v="{version}", "Not/A)Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?1" if ("Mobile" in ua or platform in {"Android", "iOS"}) else "?0",
        "Sec-Ch-Ua-Platform": f'"{platform}"',
    }


def _build_console_payload(
    *,
    model: str,
    input_val: str | list[Any],
    instructions: str | None = None,
    stream: bool,
    temperature: float,
    top_p: float,
    max_output_tokens: int | None = None,
    include: list[str] | None = None,
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    reasoning_effort: str | None = None,
) -> bytes:
    upstream_model = console_upstream_model(model)
    payload: dict[str, Any] = {
        "model": upstream_model,
        "input": input_val,
        "temperature": temperature,
        "top_p": top_p,
        "store": False,
        "stream": stream,
    }
    if instructions:
        payload["instructions"] = instructions
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    elif upstream_model in _CONSOLE_MAX_OUTPUT_TOKENS:
        payload["max_output_tokens"] = _CONSOLE_MAX_OUTPUT_TOKENS[upstream_model]
    if include is not None:
        payload["include"] = include
    if upstream_model in _CONSOLE_MODELS_WITH_REASONING_FIELD:
        payload["reasoning"] = {
            "effort": _console_reasoning_effort(model, reasoning_effort),
        }
    effective_tools = tools if tools is not None else _default_console_tools()
    if effective_tools:
        payload["tools"] = effective_tools
        payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
    elif tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return orjson.dumps(payload)


def _build_console_headers(token: str, lease) -> dict[str, str]:
    user_agent = _console_user_agent(lease)
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Authorization": "Bearer anonymous",
        "Content-Type": "application/json",
        "Cookie": _console_sso_cookie(token, lease=lease),
        "Origin": "https://console.x.ai",
        "Priority": "u=1, i",
        "Referer": _console_referer(),
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": user_agent,
        "X-Cluster": _console_cluster(),
        "x-statsig-id": _console_statsig_id(),
        "x-xai-request-id": str(uuid.uuid4()),
    }
    headers.update(_console_client_hints(user_agent))
    return headers


def _response_headers(response: Any) -> dict[str, str]:
    try:
        return {
            str(key).lower(): str(value)
            for key, value in response.headers.items()
        }
    except Exception:
        return {}


def _is_cloudflare_forbidden(response: Any, body: str = "") -> bool:
    """Return whether a 403 has recognizable Cloudflare challenge markers."""
    headers = _response_headers(response)
    if "cf-ray" in headers or "cf-mitigated" in headers:
        return True
    if any(key.startswith("cf-") for key in headers):
        return True
    server = headers.get("server", "").lower()
    if "cloudflare" in server:
        return True
    body_lower = (body or "").lower()
    return any(
        marker in body_lower
        for marker in ("cloudflare", "challenge-platform", "cf-chl-", "just a moment")
    )


def _cookie_names(cookie_header: str) -> tuple[str, ...]:
    names: list[str] = []
    for item in (cookie_header or "").split(";"):
        name, separator, _ = item.strip().partition("=")
        if separator and name and name not in names:
            names.append(name)
    return tuple(names)


def _redact_proxy_url(proxy_url: str | None) -> str:
    raw = str(proxy_url or "").strip()
    if not raw:
        return "direct"
    try:
        parsed = urlsplit(raw)
        host = parsed.hostname or "configured"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme or 'proxy'}://{host}{port}"
    except ValueError:
        return "configured"


def _log_console_forbidden_diagnostics(
    response: Any,
    *,
    body: str,
    model: str,
    lease: Any,
    session_kwargs: dict[str, Any],
) -> None:
    headers = _response_headers(response)
    cf_header_names = tuple(sorted(key for key in headers if key.startswith("cf-")))
    cookies = _cookie_names(str(getattr(lease, "cf_cookies", "") or ""))
    user_agent = _lease_user_agent(lease)
    major = ""
    match = re.search(r"(?:Chrome|Chromium|Edg|Firefox)/([0-9]+)", user_agent)
    if match:
        major = match.group(1)
    logger.warning(
        "console 403 diagnostics: model={} server={} content_type={} "
        "cf_ray_present={} cf_headers={} clearance_host={} cookie_names={} "
        "ua_major={} impersonate={} proxy={}",
        model,
        headers.get("server", ""),
        headers.get("content-type", ""),
        bool(headers.get("cf-ray")),
        cf_header_names,
        getattr(lease, "clearance_host", ""),
        cookies,
        major,
        session_kwargs.get("impersonate", ""),
        _redact_proxy_url(getattr(lease, "proxy_url", "")),
    )


def _proxy_feedback_kind(status: int | None, *, response: Any = None, body: str = ""):
    from app.control.proxy.models import ProxyFeedbackKind

    if status == 429:
        return ProxyFeedbackKind.RATE_LIMITED
    if status == 403:
        # A Console API 403 is not necessarily a Cloudflare challenge.  An
        # empty 403 from the upstream is commonly an account/model denial; do
        # not throw away a valid clearance bundle in that case.
        return (
            ProxyFeedbackKind.CHALLENGE
            if response is not None and _is_cloudflare_forbidden(response, body)
            else ProxyFeedbackKind.FORBIDDEN
        )
    if status == 401:
        return ProxyFeedbackKind.UNAUTHORIZED
    if status and status >= 500:
        return ProxyFeedbackKind.UPSTREAM_5XX
    return ProxyFeedbackKind.TRANSPORT_ERROR


def _console_dpop_mint_message(status: int) -> str:
    if status == 403:
        return (
            "Console DPoP token mint returned 403; the SSO/Cloudflare session "
            "was rejected before the Console request could be signed. Refresh "
            "the SSO and FlareSolverr clearance from the same browser session."
        )
    return f"Console DPoP token mint returned {status}"


async def _build_console_dpop_headers(
    session: Any,
    *,
    headers: dict[str, str],
    proxy: Any,
    lease: Any,
    timeout_s: float,
    request_url: str,
) -> dict[str, str]:
    from app.control.proxy.models import ProxyFeedback

    try:
        credentials = await mint_dpop_credentials(
            session,
            url=_console_dpop_token_url(),
            headers=headers,
            timeout_s=timeout_s,
        )
    except DpopMintError as exc:
        await proxy.feedback(
            lease,
            ProxyFeedback(
                kind=_proxy_feedback_kind(exc.status, body=exc.body),
                status_code=exc.status,
            ),
        )
        raise UpstreamError(
            _console_dpop_mint_message(exc.status),
            status=exc.status,
            body=exc.body,
        ) from exc

    authenticated_headers = dict(headers)
    authenticated_headers.update(
        credentials.headers(method="POST", url=request_url)
    )
    return authenticated_headers


def _feedback_kind_for_status(status: int) -> FeedbackKind:
    if status == 429:
        return FeedbackKind.RATE_LIMITED
    if status == 401:
        return FeedbackKind.UNAUTHORIZED
    if status == 403:
        return FeedbackKind.FORBIDDEN
    return FeedbackKind.SERVER_ERROR


def _console_status_message(
    status: int,
    *,
    model: str | None = None,
    upstream_model: str | None = None,
    body: str = "",
) -> str:
    if status == 404 and model == "grok-4.5-console":
        return (
            "Console model 'grok-4.5-console' upstream is currently unavailable "
            "(404 from grok-4.5)."
        )
    if status == 403:
        body_lower = (body or "").lower()
        if "unauthorized:dpop-required" in body_lower or (
            "dpop" in body_lower
            and "proof required" in body_lower
            and "not verified" in body_lower
        ):
            return (
                "Console upstream requires a verified DPoP proof "
                "(unauthorized:dpop-required). The current SSO/FlareSolverr "
                "flow supplies cookies and Cloudflare clearance, but not the "
                "Console browser's DPoP key and proof; changing the User-Agent, "
                "cookies, or proxy alone cannot satisfy this check."
            )
        return (
            "Console upstream returned 403; console.x.ai rejected the selected "
            "account or browser session. Check the SSO token, FlareSolverr "
            "clearance, Console browser identity, and egress IP consistency."
        )
    if model and upstream_model:
        return f"Console upstream returned {status} for {model} ({upstream_model})"
    return f"Console upstream returned {status}"


async def _post_console_json(
    token: str,
    payload: bytes,
    *,
    timeout_s: float,
    model: str,
) -> dict:
    from app.control.proxy.models import ProxyFeedback, ProxyFeedbackKind
    from app.dataplane.proxy import get_proxy_runtime

    proxy = await get_proxy_runtime()
    lease = await proxy.acquire(clearance_origin="console.x.ai")
    headers = _build_console_headers(token, lease)
    session_kwargs = build_session_kwargs(
        lease=lease,
        browser_override=_console_browser_override(lease),
    )

    try:
        async with ResettableSession(**session_kwargs) as session:
            headers = await _build_console_dpop_headers(
                session,
                headers=headers,
                proxy=proxy,
                lease=lease,
                timeout_s=timeout_s,
                request_url=_console_url(),
            )
            response = await session.post(
                _console_url(),
                headers=headers,
                data=payload,
                timeout=timeout_s,
            )
            body_bytes = response.content
            if response.status_code != 200:
                body = body_bytes.decode("utf-8", "replace")[:400]
                if response.status_code == 403:
                    logger.warning(
                        "console upstream 403: model={} upstream_model={} body={}",
                        model,
                        console_upstream_model(model),
                        body,
                    )
                    _log_console_forbidden_diagnostics(
                        response,
                        body=body,
                        model=model,
                        lease=lease,
                        session_kwargs=session_kwargs,
                    )
                await proxy.feedback(
                    lease,
                    ProxyFeedback(
                        kind=_proxy_feedback_kind(
                            response.status_code,
                            response=response,
                            body=body,
                        ),
                        status_code=response.status_code,
                    ),
                )
                raise UpstreamError(
                    _console_status_message(
                        response.status_code,
                        model=model,
                        upstream_model=console_upstream_model(model),
                        body=body,
                    ),
                    status=response.status_code,
                    body=body,
                )
            await proxy.feedback(
                lease,
                ProxyFeedback(kind=ProxyFeedbackKind.SUCCESS, status_code=200),
            )
            return orjson.loads(body_bytes) if body_bytes.strip() else {}
    except UpstreamError:
        raise
    except Exception as exc:
        await proxy.feedback(
            lease,
            ProxyFeedback(kind=ProxyFeedbackKind.TRANSPORT_ERROR, status_code=None),
        )
        body = str(exc).replace("\n", "\\n")[:400]
        raise UpstreamError(f"Console transport failed: {exc}", status=502, body=body) from exc


async def _post_console_stream(
    token: str,
    payload: bytes,
    *,
    timeout_s: float,
    model: str,
) -> AsyncGenerator[str, None]:
    from app.control.proxy.models import ProxyFeedback, ProxyFeedbackKind
    from app.dataplane.proxy import get_proxy_runtime

    proxy = await get_proxy_runtime()
    lease = await proxy.acquire(clearance_origin="console.x.ai")
    headers = _build_console_headers(token, lease)
    session_kwargs = build_session_kwargs(
        lease=lease,
        browser_override=_console_browser_override(lease),
    )
    session = ResettableSession(**session_kwargs)

    try:
        headers = await _build_console_dpop_headers(
            session,
            headers=headers,
            proxy=proxy,
            lease=lease,
            timeout_s=timeout_s,
            request_url=_console_url(),
        )
        response = await session.post(
            _console_url(),
            headers=headers,
            data=payload,
            timeout=timeout_s,
            stream=True,
        )
        if response.status_code != 200:
            body = response.content.decode("utf-8", "replace")[:400]
            if response.status_code == 403:
                logger.warning(
                    "console upstream 403: model={} upstream_model={} body={}",
                    model,
                    console_upstream_model(model),
                    body,
                )
                _log_console_forbidden_diagnostics(
                    response,
                    body=body,
                    model=model,
                    lease=lease,
                    session_kwargs=session_kwargs,
                )
            await proxy.feedback(
                lease,
                ProxyFeedback(
                    kind=_proxy_feedback_kind(
                        response.status_code,
                        response=response,
                        body=body,
                    ),
                    status_code=response.status_code,
                ),
            )
            await session.close()
            raise UpstreamError(
                _console_status_message(
                    response.status_code,
                    model=model,
                    upstream_model=console_upstream_model(model),
                    body=body,
                ),
                status=response.status_code,
                body=body,
            )
        await proxy.feedback(
            lease,
            ProxyFeedback(kind=ProxyFeedbackKind.SUCCESS, status_code=200),
        )
    except UpstreamError:
        raise
    except Exception as exc:
        await proxy.feedback(
            lease,
            ProxyFeedback(kind=ProxyFeedbackKind.TRANSPORT_ERROR, status_code=None),
        )
        try:
            await session.close()
        except Exception:
            pass
        body = str(exc).replace("\n", "\\n")[:400]
        raise UpstreamError(f"Console transport failed: {exc}", status=502, body=body) from exc

    async def _lines() -> AsyncGenerator[str, None]:
        try:
            async for line in response.aiter_lines():
                yield _coerce_sse_line(line)
        finally:
            try:
                await session.close()
            except Exception:
                pass

    return _lines()


async def _reserve_basic_console_account(directory, model: str, *, exclude_tokens: list[str] | None):
    return await directory.reserve(
        pool_candidates=(_BASIC_POOL_ID,),
        mode_id=console_mode_id_for_model(model),
        exclude_tokens=exclude_tokens,
        now_s_override=now_s(),
    )


async def _record_console_feedback(directory, token: str, model: str, kind: FeedbackKind) -> None:
    repo = getattr(directory, "_repo", None)
    if repo is None:
        return
    try:
        records = await repo.get_accounts([token])
        if not records:
            return
        record = records[0]
        key = console_usage_key_for_model(model)
        usage = increment_console_usage(
            record.ext,
            key,
            success=kind == FeedbackKind.SUCCESS,
        )
        await repo.patch_accounts([
            AccountPatch(
                token=token,
                ext_merge={CONSOLE_USAGE_EXT_KEY: usage},
                usage_use_delta=1 if kind == FeedbackKind.SUCCESS else None,
                usage_fail_delta=1 if kind != FeedbackKind.SUCCESS else None,
                last_use_at=now_ms() if kind == FeedbackKind.SUCCESS else None,
                last_fail_at=now_ms() if kind != FeedbackKind.SUCCESS else None,
            )
        ])
    except Exception as exc:
        logger.warning(
            "console usage persistence failed: token={}... model={} error={}",
            token[:8],
            model,
            exc,
        )


def _inject_synthetic_response_reasoning(obj: dict) -> None:
    output = obj.get("output")
    if not isinstance(output, list):
        return
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        if not extract_response_reasoning({"output": [item]}):
            item["summary"] = [{"type": "summary_text", "text": _SYNTHETIC_REASONING_TEXT}]
        return
    output.insert(0, _synthetic_response_reasoning_item(make_resp_id("rs")))


def _normalize_response_model(obj: dict, model: str, *, emit_think: bool = False) -> dict:
    if isinstance(obj, dict):
        obj["model"] = model
        if emit_think:
            _inject_synthetic_response_reasoning(obj)
        response = obj.get("response")
        if isinstance(response, dict):
            response["model"] = model
            if emit_think:
                _inject_synthetic_response_reasoning(response)
    return obj


def _rewrite_response_sse_line(line: str, model: str) -> str:
    if not line.startswith("data:"):
        return line
    data = line[5:].strip()
    if not data or data == "[DONE]" or not data.startswith("{"):
        return line
    try:
        obj = orjson.loads(data)
    except Exception:
        return line
    if isinstance(obj, dict):
        _normalize_response_model(obj, model)
        return f"data: {orjson.dumps(obj).decode()}"
    return line


def _coerce_sse_line(line: Any) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", "replace")
    return str(line)


async def _iter_with_console_heartbeat(
    source: AsyncIterable[Any],
) -> AsyncGenerator[Any, None]:
    interval = _console_stream_heartbeat_interval()
    if interval <= 0:
        async for item in source:
            yield item
        return

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def _produce() -> None:
        try:
            async for item in source:
                await queue.put(("item", item))
        except BaseException as exc:
            await queue.put(("error", exc))
        else:
            await queue.put(("done", None))

    task = asyncio.create_task(_produce())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield ": ping"
                continue
            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            else:
                break
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _is_sse_comment(line: str) -> bool:
    return line.lstrip().startswith(":")


def extract_response_text(obj: dict) -> str:
    texts: list[str] = []
    direct = obj.get("output_text")
    if direct:
        texts.append(str(direct))
    wrapped = obj.get("response")
    if isinstance(wrapped, dict):
        nested = extract_response_text(wrapped)
        if nested:
            texts.append(nested)
    for item in obj.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part["text"]))
    return "".join(texts)


def extract_response_reasoning(obj: dict) -> str:
    texts: list[str] = []
    wrapped = obj.get("response")
    if isinstance(wrapped, dict):
        nested = extract_response_reasoning(wrapped)
        if nested:
            texts.append(nested)
    for item in obj.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for part in item.get("summary", []) or []:
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part["text"]))
    return "".join(texts)


def _response_has_reasoning(obj: dict) -> bool:
    wrapped = obj.get("response")
    if isinstance(wrapped, dict) and _response_has_reasoning(wrapped):
        return True
    for item in obj.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            return True
    return False


def _event_has_reasoning_item(obj: dict) -> bool:
    item = obj.get("item")
    return isinstance(item, dict) and item.get("type") == "reasoning"


def _append_content_texts(content: Any, texts: list[str]) -> None:
    if isinstance(content, dict):
        text = content.get("text")
        if text:
            texts.append(str(text))
        return
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if text:
            texts.append(str(text))


def _extract_event_text(obj: dict) -> tuple[str, bool]:
    """Return text carried by a console Responses stream event.

    The console endpoint is not perfectly stable about whether text arrives as
    output_text.delta events or only in done/completed payloads. The boolean is
    True when the returned text is a full accumulated value rather than a delta.
    """
    event_type = obj.get("type")
    if event_type == "response.output_text.delta":
        return str(obj.get("delta") or ""), False
    if event_type == "response.output_text.done":
        return str(obj.get("text") or ""), True
    if event_type == "response.content_part.done":
        part = obj.get("part")
        if isinstance(part, dict):
            return str(part.get("text") or ""), True
        return "", True
    if event_type == "response.output_item.done":
        item = obj.get("item")
        if isinstance(item, dict):
            texts: list[str] = []
            _append_content_texts(item.get("content"), texts)
            return "".join(texts), True
        return "", True
    if event_type == "response.completed":
        response = obj.get("response")
        if isinstance(response, dict):
            return extract_response_text(response), True
    return "", False


def _extract_event_reasoning(obj: dict) -> tuple[str, bool]:
    event_type = obj.get("type")
    if event_type == "response.reasoning_summary_text.delta":
        return str(obj.get("delta") or ""), False
    if event_type == "response.reasoning_summary_text.done":
        return str(obj.get("text") or ""), True
    if event_type == "response.reasoning_summary_part.done":
        part = obj.get("part")
        if isinstance(part, dict):
            return str(part.get("text") or ""), True
    if event_type == "response.output_item.done":
        item = obj.get("item")
        if isinstance(item, dict) and item.get("type") == "reasoning":
            texts: list[str] = []
            _append_content_texts(item.get("summary"), texts)
            return "".join(texts), True
    if event_type == "response.completed":
        response = obj.get("response")
        if isinstance(response, dict):
            return extract_response_reasoning(response), True
    return "", False


def _delta_from_full(full_text: str, emitted_text: str) -> str:
    if not full_text:
        return ""
    if emitted_text and full_text.startswith(emitted_text):
        return full_text[len(emitted_text):]
    if full_text == emitted_text:
        return ""
    return full_text


async def maybe_create_response(
    *,
    model: str,
    input_val: str | list[Any],
    instructions: str | None,
    stream: bool,
    emit_think: bool,
    temperature: float,
    top_p: float,
    max_output_tokens: int | None = None,
    include: list[str] | None = None,
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    reasoning_effort: str | None = None,
    synthesize_response_reasoning: bool = True,
) -> dict | AsyncGenerator[str, None] | None:
    if not is_console_basic_model(model):
        return None

    from app.dataplane.account import _directory as _acct_dir

    if _acct_dir is None:
        raise RateLimitError("Account directory not initialised")
    directory = _acct_dir

    payload = _build_console_payload(
        model=model,
        input_val=input_val,
        instructions=instructions,
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        include=_console_reasoning_include(include, emit_think),
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
    )
    timeout_s = get_config().get_float("chat.timeout", 120.0)
    max_retries = selection_max_retries()

    excluded: list[str] = []
    last_exc: UpstreamError | None = None
    for attempt in range(max_retries + 1):
        selected_mode_id = console_mode_id_for_model(model)
        acct = await _reserve_basic_console_account(directory, model, exclude_tokens=excluded or None)
        if acct is None:
            if attempt == 0:
                return None
            break

        success = False
        fail_exc: UpstreamError | None = None
        try:
            if stream:
                upstream = await _post_console_stream(
                    acct.token,
                    payload,
                    timeout_s=timeout_s,
                    model=model,
                )

                async def _run_stream(lease=acct, upstream_lines=upstream):
                    nonlocal success, fail_exc
                    try:
                        synthetic_response_reasoning_sent = False
                        async for raw_line in _iter_with_console_heartbeat(upstream_lines):
                            line = _coerce_sse_line(raw_line)
                            if not line.strip():
                                continue
                            if _is_sse_comment(line):
                                yield line.rstrip("\n") + "\n\n"
                                continue
                            yield _rewrite_response_sse_line(line, model) + "\n\n"
                            if (
                                emit_think
                                and synthesize_response_reasoning
                                and not synthetic_response_reasoning_sent
                                and line.startswith("data:")
                                and line[5:].strip().startswith("{")
                            ):
                                for event in _synthetic_response_reasoning_events(make_resp_id("rs")):
                                    yield event
                                synthetic_response_reasoning_sent = True
                        success = True
                    except UpstreamError as exc:
                        fail_exc = exc
                        raise
                    finally:
                        await directory.release(lease)
                        kind = (
                            FeedbackKind.SUCCESS
                            if success
                            else _feedback_kind_for_status(fail_exc.status)
                            if fail_exc
                            else FeedbackKind.SERVER_ERROR
                        )
                        await directory.feedback(
                            lease.token,
                            kind,
                            selected_mode_id,
                            now_s_val=now_s(),
                        )
                        await _record_console_feedback(directory, lease.token, model, kind)

                return _run_stream()

            obj = await _post_console_json(
                acct.token,
                payload,
                timeout_s=timeout_s,
                model=model,
            )
            success = True
            return _normalize_response_model(
                obj,
                model,
                emit_think=emit_think and synthesize_response_reasoning,
            )
        except UpstreamError as exc:
            fail_exc = exc
            last_exc = exc
            if exc.status not in _CONSOLE_RETRYABLE_STATUSES or attempt >= max_retries:
                raise
            logger.warning(
                "console basic response retry scheduled: attempt={}/{} status={} token={}...",
                attempt + 1,
                max_retries,
                exc.status,
                acct.token[:8],
            )
        finally:
            if not stream:
                await directory.release(acct)
                kind = (
                    FeedbackKind.SUCCESS
                    if success
                    else _feedback_kind_for_status(fail_exc.status)
                    if fail_exc
                    else FeedbackKind.SERVER_ERROR
                )
                await directory.feedback(
                    acct.token,
                    kind,
                    selected_mode_id,
                    now_s_val=now_s(),
                )
                await _record_console_feedback(directory, acct.token, model, kind)
        excluded.append(acct.token)

    if last_exc is not None:
        raise last_exc
    raise NoBasicConsoleAccount()


def chat_messages_to_response_input(messages: list[dict]) -> list[dict]:
    items: list[dict] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        if role == "developer":
            role = "system"
        content = msg.get("content") or ""
        parts: list[dict] = []
        if isinstance(content, str):
            if content:
                ptype = "output_text" if role == "assistant" else "input_text"
                parts.append({"type": ptype, "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text") or ""
                    if text:
                        target = "output_text" if role == "assistant" else "input_text"
                        parts.append({"type": target, "text": text})
                elif ptype == "image_url":
                    image_url = part.get("image_url") or {}
                    url = image_url.get("url") if isinstance(image_url, dict) else ""
                    if url:
                        parts.append({"type": "input_image", "image_url": url})
        if parts:
            items.append({"role": role, "content": parts})
    return items


async def maybe_create_chat_completion(
    *,
    model: str,
    messages: list[dict],
    stream: bool,
    emit_think: bool,
    temperature: float,
    top_p: float,
    max_tokens: int | None = None,
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    reasoning_effort: str | None = None,
) -> dict | AsyncGenerator[str, None] | None:
    if not is_console_basic_model(model):
        return None

    response_input = chat_messages_to_response_input(messages)
    if not response_input:
        return None

    result = await maybe_create_response(
        model=model,
        input_val=response_input,
        instructions=None,
        stream=stream,
        emit_think=emit_think,
        synthesize_response_reasoning=False,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        include=["reasoning.encrypted_content"],
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
    )
    if result is None:
        return None

    if not stream:
        assert isinstance(result, dict)
        text = extract_response_text(result)
        reasoning = extract_response_reasoning(result) if emit_think else ""
        if emit_think and not reasoning and _response_has_reasoning(result):
            reasoning = _SYNTHETIC_REASONING_TEXT
        usage = result.get("usage")
        chat_usage = None
        if isinstance(usage, dict):
            pt = int(usage.get("input_tokens") or estimate_prompt_tokens(messages))
            ct = int(usage.get("output_tokens") or estimate_tokens(text))
            rt = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
            chat_usage = build_usage(pt, ct, reasoning_tokens=rt)
        return _make_console_chat_response(
            model,
            text,
            prompt_content=messages,
            reasoning_content=reasoning or None,
            usage=chat_usage,
        )

    assert not isinstance(result, dict)
    response_id = make_response_id()

    async def _run_chat_stream() -> AsyncGenerator[str, None]:
        done = False
        emitted_text = ""
        emitted_reasoning = ""
        synthetic_reasoning_sent = False
        if emit_think:
            emitted_reasoning += _SYNTHETIC_REASONING_TEXT
            yield f"data: {orjson.dumps(_make_console_thinking_chunk(response_id, model, _SYNTHETIC_REASONING_TEXT)).decode()}\n\n"
            synthetic_reasoning_sent = True
        async for raw in _iter_with_console_heartbeat(result):
            raw_text = _coerce_sse_line(raw)
            if _is_sse_comment(raw_text):
                yield raw_text.rstrip("\n") + "\n\n"
                continue
            for line in raw_text.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    if not done:
                        yield f"data: {orjson.dumps(make_stream_chunk(response_id, model, '', is_final=True)).decode()}\n\n"
                        yield "data: [DONE]\n\n"
                        done = True
                    continue
                try:
                    obj = orjson.loads(data)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                event_type = obj.get("type")
                if (
                    emit_think
                    and not synthetic_reasoning_sent
                    and _event_has_reasoning_item(obj)
                ):
                    emitted_reasoning += _SYNTHETIC_REASONING_TEXT
                    yield f"data: {orjson.dumps(_make_console_thinking_chunk(response_id, model, _SYNTHETIC_REASONING_TEXT)).decode()}\n\n"
                    synthetic_reasoning_sent = True
                text, text_is_full = _extract_event_text(obj)
                if text:
                    delta = _delta_from_full(text, emitted_text) if text_is_full else text
                    if delta:
                        emitted_text += delta
                        yield f"data: {orjson.dumps(make_stream_chunk(response_id, model, delta)).decode()}\n\n"
                reasoning, reasoning_is_full = _extract_event_reasoning(obj)
                if reasoning and emit_think:
                    delta = (
                        _delta_from_full(reasoning, emitted_reasoning)
                        if reasoning_is_full
                        else reasoning
                    )
                    if delta:
                        emitted_reasoning += delta
                        yield f"data: {orjson.dumps(_make_console_thinking_chunk(response_id, model, delta)).decode()}\n\n"
                if event_type == "response.completed" and not done:
                    yield f"data: {orjson.dumps(make_stream_chunk(response_id, model, '', is_final=True)).decode()}\n\n"
                    yield "data: [DONE]\n\n"
                    done = True
        if not done:
            yield f"data: {orjson.dumps(make_stream_chunk(response_id, model, '', is_final=True)).decode()}\n\n"
            yield "data: [DONE]\n\n"

    return _run_chat_stream()


__all__ = [
    "NoBasicConsoleAccount",
    "console_upstream_model",
    "extract_response_text",
    "is_console_basic_model",
    "maybe_create_chat_completion",
    "maybe_create_response",
]
