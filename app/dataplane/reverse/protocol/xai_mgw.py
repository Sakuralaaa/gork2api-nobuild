"""Grok Web mgw WebSocket chat transport.

grok.com 已把文本对话的正文下发迁到 ``/ws/mgw`` 事件流；旧的
``/rest/app-chat/conversations/new`` SSE 端点现在只下发工具调用帧
（tool_usage_card / raw_function_result），拿不到任何正文 token。

本模块实现 mgw 客户端，并把 mgw 事件转换成与旧 SSE 完全相同的
``{"result":{"response":{...}}}`` 形状，使 ``xai_chat.StreamAdapter``
无需改动即可复用（思考、工具卡片、搜索信源、正文、结束信号）。
"""

import os
import asyncio
import time
import uuid
from typing import Any, AsyncGenerator
from urllib.parse import quote, urlparse

import aiohttp
import orjson

from app.platform.config.snapshot import get_config
from app.platform.errors import UpstreamError
from app.platform.logging.logger import logger
from app.control.proxy.models import ProxyLease
from app.dataplane.proxy.adapters.headers import build_sso_cookie, build_ws_headers
from app.dataplane.reverse.runtime.endpoint_table import BASE
from app.dataplane.reverse.transport.http import get_json
from app.dataplane.reverse.transport.websocket import WebSocketClient

MGW_PATH = "/ws/mgw/"
HEARTBEAT_SECONDS = 25.0
MAX_FRAME_BYTES = 16 << 20
UID_TIMEOUT_SECONDS = 15.0
_UID_CACHE_LIMIT = 512

# token -> user_id（UUID）。mgw 端点必须带 uid，账号记录里没有该字段，
# 因此按 SSO token 缓存最近解析结果，避免每次请求都走一次 Session 接口。
_UID_CACHE: dict[str, str] = {}

_ASSISTANT_CHANNEL = "CHANNEL_ASSISTANT_RESPONSE"
_REASONING_MARKERS = ("ANALYSIS", "REASONING")

_TRACE_MGW = os.getenv("GROK_TRACE_MGW", "1").strip().lower() in {"1", "true", "yes", "on"}


def mgw_enabled() -> bool:
    """文本对话是否走 mgw WebSocket；配置 chat.transport=http 可回退旧链路。"""
    return get_config().get_str("chat.transport", "mgw").strip().lower() != "http"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _first_str(value: Any, *keys: str) -> str:
    if not isinstance(value, dict):
        return ""
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def _extract_user_id(data: Any) -> str:
    """按 Go 侧 sessionidentity.Parse 的顺序提取稳定 user id。"""
    if not isinstance(data, dict):
        return ""
    status = _first_str(data, "status").lower()
    if status in ("unauthenticated", "blocked"):
        return ""
    session = data.get("session")
    user = data.get("user")
    for source in (session, user, data):
        value = _first_str(source, "userId", "user_id", "id", "sub")
        if value:
            return value
    return ""


async def resolve_user_id(token: str, *, lease: ProxyLease | None = None) -> str:
    """解析 SSO 账号的 user id（UUID），失败即向上报错而不是静默降级。"""
    cached = _UID_CACHE.get(token)
    if cached:
        return cached
    data = await get_json(
        f"{BASE}/api/auth/session",
        token,
        lease=lease,
        timeout_s=UID_TIMEOUT_SECONDS,
        referer=BASE + "/",
    )
    user_id = _extract_user_id(data)
    if not user_id:
        raise UpstreamError(
            "Grok Session 未返回 user_id，无法建立 mgw 连接",
            status=502,
            body=orjson.dumps(data)[:400].decode("utf-8", "replace"),
        )
    if len(_UID_CACHE) >= _UID_CACHE_LIMIT:
        _UID_CACHE.clear()
    _UID_CACHE[token] = user_id
    return user_id


# ---------------------------------------------------------------------------
# Endpoint / headers
# ---------------------------------------------------------------------------


def mgw_url(user_id: str, base_url: str = BASE) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}{MGW_PATH}?uid={quote(user_id, safe='')}"


def build_mgw_headers(
    token: str, user_id: str, lease: ProxyLease | None = None
) -> dict[str, str]:
    cookie = build_sso_cookie(token, lease=lease)
    if user_id:
        cookie = f"{cookie}; x-userid={user_id}"
    return build_ws_headers(token, origin=BASE, extra={"Cookie": cookie}, lease=lease)


# ---------------------------------------------------------------------------
# Event -> legacy SSE frames
# ---------------------------------------------------------------------------


def _frame(response: dict[str, Any]) -> str:
    return "data: " + orjson.dumps({"result": {"response": response}}).decode()


def _channel_is_reasoning(channel: Any) -> bool:
    text = str(channel or "").upper()
    return any(marker in text for marker in _REASONING_MARKERS)


def _nested_query(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    value = _first_str(tool, "query", "q", "image_description")
    if value:
        return value
    args = tool.get("args")
    if isinstance(args, dict):
        return _first_str(args, "query", "q", "image_description")
    return ""


def _tool_usage_card_frame(card: dict[str, Any]) -> str:
    """把 mgw 的 tool_usage_card 转成 StreamAdapter 认识的 toolUsageCard。"""
    converted: dict[str, Any] = {}
    for key, value in card.items():
        if isinstance(value, dict) and "args" not in value:
            query = _nested_query(value)
            if query:
                converted[key] = {**value, "args": {"query": query}}
                continue
        converted[key] = value
    return _frame(
        {
            "messageTag": "tool_usage_card",
            "isThinking": True,
            "token": "",
            "toolUsageCard": converted,
            "toolUsageCardId": _first_str(card, "tool_usage_card_id", "id"),
        }
    )


def _web_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"url": item["url"], "title": item.get("title") or item["url"]}
        for item in results
        if isinstance(item, dict) and isinstance(item.get("url"), str) and item["url"]
    ]


def _tool_result_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    web = result.get("web_search")
    if isinstance(web, dict):
        for page in web.get("webpages") or []:
            if isinstance(page, dict):
                items.append(page)
    x_post = result.get("x_post")
    if isinstance(x_post, dict):
        for post in x_post.get("posts") or []:
            if not isinstance(post, dict):
                continue
            handle = _first_str(post, "userhandle", "username")
            post_id = _first_str(post, "post_id", "postId")
            if not handle or not post_id:
                continue
            items.append(
                {
                    "url": f"https://x.com/{handle}/status/{post_id}",
                    "title": _first_str(post, "name", "userhandle", "username") or post_id,
                }
            )
    return _web_sources(items)


def _chunk_frames(chunk: Any) -> list[tuple[str, bool]]:
    """把 mgw chunk 转成 (SSE 行, 是否为正文) 列表。"""
    if not isinstance(chunk, dict):
        return []
    frames: list[tuple[str, bool]] = []
    text = chunk.get("text")
    if isinstance(text, dict):
        value = text.get("text")
        channel = text.get("channel")
        if isinstance(value, str) and value:
            if _channel_is_reasoning(channel):
                frames.append((_frame({"token": value, "isThinking": True}), False))
            elif not channel or str(channel).upper() == _ASSISTANT_CHANNEL:
                body = _frame(
                    {"token": value, "isThinking": False, "messageTag": "final"}
                )
                frames.append((body, True))
    card = chunk.get("tool_usage_card")
    if isinstance(card, dict):
        frames.append((_tool_usage_card_frame(card), False))
    result = chunk.get("tool_result")
    if isinstance(result, dict):
        sources = _tool_result_sources(result)
        if sources:
            frames.append((_frame({"webSearchResults": {"results": sources}}), False))
    return frames


# ---------------------------------------------------------------------------
# Outbound events
# ---------------------------------------------------------------------------


def _session_object(model: str) -> dict[str, Any]:
    x_grok: dict[str, Any] = {
        "protocol_capabilities": ["conversation_attached", "custom_methods_v1"],
        "use_chunk": True,
        "enable_side_by_side": True,
        "force_side_by_side": False,
        "enable_image_generation": True,
        "image_generation_count": 2,
        "disable_text_follow_ups": False,
        "disable_artifact": True,
        "force_concise": False,
        "keep_context": False,
        "is_temporary": True,
        "disable_memory": True,
    }
    if get_config().get_bool("chat.mgw_disable_search", False):
        x_grok["disable_search"] = True
    return {"model": model, "x_grok": x_grok}


def _session_create_event(model: str) -> dict[str, Any]:
    return {
        "event": {
            "type": "session.create",
            "event_id": "evt_init_" + str(uuid.uuid4()),
            "session": _session_object(model),
        }
    }


def _item_event(session_id: str, message: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "event": {
            "type": "conversation.item.create",
            "event_id": f"evt_msg_{int(time.time() * 1000)}",
            "item": {
                "type": "message",
                "role": "user",
                "x_grok": {
                    "client_message_id": str(uuid.uuid4()),
                    "input_chunks": [{"text": {"text": message}}],
                },
            },
        },
    }


def _apply_search_limit(message: str) -> str:
    """按 chat.mgw_search_limit 在提示词里约束联网搜索轮次。"""
    limit = get_config().get_int("chat.mgw_search_limit", 3)
    if limit <= 0:
        return message
    return (
        message
        + f"\n\n【重要】最多只允许进行 {limit} 次联网搜索。达到 {limit} 次后必须"
        + "立即停止搜索，直接基于已获得的信息给出最终答案；不要再搜索、"
        + "不要打开更多页面。"
    )


def _response_event(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "event": {
            "type": "response.create",
            "event_id": f"evt_resp_{int(time.time() * 1000)}",
        },
    }


def _error_message(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        text = _first_str(error, "message", "error", "detail")
        if text:
            return text
    return "Grok mgw 返回未知错误"


def _trace_mgw_event(event_type: str, event: dict[str, Any]) -> str:
    """脱敏记录单个 mgw 事件结构，仅用于定位正文/思考通道。"""
    parts = [f"type={event_type or '<empty>'}"]
    chunk = event.get("chunk")
    if isinstance(chunk, dict):
        parts.append("chunk=" + ",".join(sorted(chunk.keys()))[:200])
        text = chunk.get("text")
        if isinstance(text, dict):
            parts.append(f"channel={text.get('channel')!r}")
            value = text.get("text")
            size = str(len(value)) if isinstance(value, str) else type(value).__name__
            parts.append("textlen=" + size)
        for key in ("tool_usage_card", "tool_result", "render_citation"):
            if isinstance(chunk.get(key), dict):
                parts.append(key)
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            parts.append("metadata=" + ",".join(sorted(metadata.keys()))[:200])
            for key in ("is_final", "status", "search_round", "tool_call_id", "reason", "type"):
                if key in metadata:
                    parts.append(f"md.{key}={str(metadata[key])[:60]!r}")
    for key in ("delta", "text", "status", "result"):
        if key in event:
            value = event[key]
            size = str(len(value)) if isinstance(value, str) else type(value).__name__
            parts.append(f"{key}={size}")
    response = event.get("response")
    if isinstance(response, dict):
        parts.append("response=" + ",".join(sorted(response.keys()))[:160])
    if event.get("error"):
        parts.append("ERROR")
    return " ".join(parts)


async def _heartbeat(
    ws: aiohttp.ClientWebSocketResponse, lock: asyncio.Lock
) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        try:
            async with lock:
                await ws.send_json(
                    {
                        "event": {
                            "type": "ping",
                            "event_id": f"evt_hb_{int(time.time() * 1000)}",
                        }
                    }
                )
        except Exception:
            return


# ---------------------------------------------------------------------------
# Public stream
# ---------------------------------------------------------------------------


async def mgw_chat_stream(
    *,
    token: str,
    model: str,
    message: str,
    lease: ProxyLease | None = None,
    timeout_s: float = 120.0,
) -> AsyncGenerator[str, None]:
    """产出行协议与旧 HTTP SSE 一致的文本流。

    产出 ``data: {"result":{"response":...}}`` 行，调用方用
    ``classify_line`` + ``StreamAdapter`` 解析即可。
    """
    user_id = await resolve_user_id(token, lease=lease)
    url = mgw_url(user_id)
    headers = build_mgw_headers(token, user_id, lease)

    # 整个流的总时长上限，避免长回答被连接层提前掐断。
    budget = max(float(timeout_s), 300.0)
    client = WebSocketClient()
    try:
        connection = await client.connect(
            url,
            headers=headers,
            timeout=budget,
            lease=lease,
            ws_kwargs={"max_msg_size": MAX_FRAME_BYTES},
        )
    except Exception as exc:
        raise UpstreamError(f"Grok mgw 连接失败: {exc}", status=502) from exc

    ws = connection.ws
    send_lock = asyncio.Lock()
    heartbeat: asyncio.Task[None] | None = None
    try:
        initial = _session_create_event(model)
        async with send_lock:
            await ws.send_json(initial)
        if _TRACE_MGW:
            logger.info(
                "mgw session.create sent | model={} x_grok={}",
                model,
                orjson.dumps(_session_object(model)["x_grok"]).decode(),
            )
        heartbeat = asyncio.create_task(_heartbeat(ws, send_lock))

        created = False
        attached = False
        turn_sent = False
        session_id = ""
        text_seen = False

        async for raw in ws:
            if raw.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                break
            if raw.type != aiohttp.WSMsgType.TEXT:
                continue
            payload = raw.data
            if isinstance(payload, str) and len(payload) > MAX_FRAME_BYTES:
                raise UpstreamError("Grok mgw 响应帧超过安全上限", status=502)
            try:
                envelope = orjson.loads(payload)
            except (orjson.JSONDecodeError, ValueError, TypeError):
                continue
            if not isinstance(envelope, dict):
                continue
            event = envelope.get("event")
            if not isinstance(event, dict):
                event = {}
            event_type = str(event.get("type") or "")
            frame_session = envelope.get("session_id")
            frame_session = frame_session if isinstance(frame_session, str) else ""
            if _TRACE_MGW:
                logger.info("mgw event | {}", _trace_mgw_event(event_type, event))

            if event_type == "session.created":
                created = True
                session_id = session_id or frame_session
            elif event_type == "conversation.attached":
                attached = True
                conversation = event.get("conversation")
                conv_id = ""
                if isinstance(conversation, dict):
                    conv_id = _first_str(conversation, "id")
                session_id = session_id or frame_session or conv_id
            elif event_type == "response.chunk":
                for line, is_text in _chunk_frames(event.get("chunk")):
                    if is_text:
                        text_seen = True
                    yield line
            elif event_type == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    text_seen = True
                    yield _frame(
                        {"token": delta, "isThinking": False, "messageTag": "final"}
                    )
            elif event_type == "response.output_text.done":
                text = event.get("text")
                if isinstance(text, str) and text and not text_seen:
                    text_seen = True
                    yield _frame(
                        {"token": text, "isThinking": False, "messageTag": "final"}
                    )
            elif event_type == "response.search.result":
                result = event.get("result")
                sources = _tool_result_sources({"web_search": {"webpages": [result]}})
                if sources:
                    yield _frame({"webSearchResults": {"results": sources}})
            elif event_type == "response.done":
                response = event.get("response")
                status = ""
                if isinstance(response, dict):
                    status = _first_str(response, "status")
                if status and status != "completed":
                    raise UpstreamError(
                        f"Grok mgw response 状态为 {status}", status=502
                    )
                logger.info(
                    "mgw response.done: status={} text_seen={}",
                    status or "completed",
                    text_seen,
                )
                yield _frame({"isSoftStop": True})
                return
            elif event_type == "session.ended":
                raise UpstreamError("Grok mgw session 在响应完成前结束", status=502)
            elif event_type == "error":
                raise UpstreamError(_error_message(event), status=502)

            if created and attached and not turn_sent:
                turn_sent = True
                active_session = session_id or frame_session
                async with send_lock:
                    await ws.send_json(_item_event(active_session, _apply_search_limit(message)))
                    await ws.send_json(_response_event(active_session))
                logger.info(
                    "mgw turn sent: session={} model={}", active_session, model
                )
        logger.warning(
            "mgw stream closed without response.done: session={} text_seen={}",
            session_id,
            text_seen,
        )
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        try:
            await connection.close()
        except Exception:
            pass


__all__ = [
    "mgw_enabled",
    "mgw_chat_stream",
    "resolve_user_id",
    "mgw_url",
    "build_mgw_headers",
]
