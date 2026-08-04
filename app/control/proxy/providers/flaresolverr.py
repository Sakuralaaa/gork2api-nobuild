"""FlareSolverr-backed managed clearance provider."""

import asyncio
import json
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlsplit, urlunsplit

from app.platform.logging.logger import logger
from app.platform.config.snapshot import get_config
from ..models import ClearanceBundle, ClearanceMode


def _extract_all_cookies(cookies: list[dict]) -> str:
    return "; ".join(f"{c.get('name')}={c.get('value')}" for c in cookies)


def _redact_proxy_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "<direct>"
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return "<configured>"
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except ValueError:
        return "<configured>"


def _request_proxy(proxy_url: str) -> dict[str, str] | None:
    """Build a request.get proxy only when it has no credentials.

    FlareSolverr's request.get API does not support authenticated proxies in
    the request body. Authenticated global proxies must be configured through
    PROXY_URL, PROXY_USERNAME and PROXY_PASSWORD in the FlareSolverr service.
    """
    raw = str(proxy_url or "").strip()
    if not raw:
        return None
    try:
        parsed = urlsplit(raw)
        has_credentials = parsed.username is not None or parsed.password is not None
    except ValueError:
        has_credentials = True
    if has_credentials:
        # FlareSolverr's request.get schema deliberately rejects credentials in
        # the request body.  The service-level PROXY_* variables are the
        # supported path, so this is an expected delegation rather than a
        # failed proxy request.  Keep it at debug level; a warning here made a
        # correctly configured FlareSolverr instance look broken on every
        # clearance refresh.
        logger.debug(
            "flaresolverr request.get delegated authenticated proxy to service env: proxy={}",
            _redact_proxy_url(raw),
        )
        return None
    return {"url": raw}


class FlareSolverrClearanceProvider:
    """Refresh CF clearance bundles via a FlareSolverr instance."""

    async def refresh_bundle(
        self,
        *,
        affinity_key: str,
        proxy_url:    str,
        target_url:   str = "https://grok.com",
    ) -> ClearanceBundle | None:
        cfg = get_config()
        mode = ClearanceMode.parse(cfg.get_str("proxy.clearance.mode", "none"))
        if mode != ClearanceMode.FLARESOLVERR:
            return None
        fs_url      = cfg.get_str("proxy.clearance.flaresolverr_url", "")
        timeout_sec = cfg.get_int("proxy.clearance.timeout_sec", 60)
        if not fs_url:
            return None

        result = await self._solve(
            fs_url      = fs_url,
            proxy_url   = proxy_url,
            timeout_sec = timeout_sec,
            target_url  = target_url,
        )
        if not result:
            safe_proxy = _redact_proxy_url(proxy_url or affinity_key)
            logger.warning(
                "flaresolverr clearance refresh failed: affinity={} proxy={} target={}",
                safe_proxy, safe_proxy, target_url,
            )
            return None
        host = result.get("clearance_host", "grok.com")

        return ClearanceBundle(
            bundle_id    = f"flaresolverr:{affinity_key}@{host}",
            cf_cookies   = result.get("cookies", ""),
            user_agent   = result.get("user_agent", ""),
            affinity_key = affinity_key,
            clearance_host = host,
        )

    async def _solve(
        self,
        *,
        fs_url:      str,
        proxy_url:   str,
        timeout_sec: int,
        target_url:  str,
    ) -> dict[str, str] | None:
        target = target_url.strip() or "https://grok.com"
        payload: dict = {
            "cmd":        "request.get",
            "url":        target,
            "maxTimeout": timeout_sec * 1000,
        }
        if proxy := _request_proxy(proxy_url):
            payload["proxy"] = proxy

        body    = json.dumps(payload).encode()
        request = urllib_request.Request(
            f"{fs_url.rstrip('/')}/v1",
            data    = body,
            method  = "POST",
            headers = {"Content-Type": "application/json"},
        )

        try:
            def _post() -> dict:
                with urllib_request.urlopen(request, timeout=timeout_sec + 30) as resp:
                    return json.loads(resp.read().decode())

            result = await asyncio.to_thread(_post)
            if result.get("status") != "ok":
                logger.warning(
                    "flaresolverr returned non-ok status: status={} message={}",
                    result.get("status"), result.get("message", ""),
                )
                return None

            solution = result.get("solution") or {}
            if not isinstance(solution, dict):
                logger.warning("flaresolverr returned an invalid solution")
                return None

            cookies = solution.get("cookies") or []
            ua = str(solution.get("userAgent") or "").strip()
            try:
                solution_status = int(solution.get("status") or 0)
            except (TypeError, ValueError):
                solution_status = 0

            # FlareSolverr legitimately returns no cookies when the target is
            # reachable without a Cloudflare challenge.  Keep the successful
            # browser identity in that case so callers still use the same
            # User-Agent and browser fingerprint instead of falling back to a
            # different configured session.
            if not cookies and not 200 <= solution_status < 400:
                logger.warning(
                    "flaresolverr returned no cookies for a non-success solution: "
                    "target={} solution_status={}",
                    target,
                    solution_status,
                )
                return None
            if not cookies:
                logger.info(
                    "flaresolverr challenge not detected: target={} user_agent_present={}",
                    target,
                    bool(ua),
                )

            host = (urlparse(target).hostname or "").lower()
            filtered = [
                cookie for cookie in cookies
                if not host or not cookie.get("domain") or host.endswith(str(cookie.get("domain", "")).lstrip(".").lower())
            ]
            chosen = filtered or cookies
            return {
                "cookies":    _extract_all_cookies(chosen),
                "user_agent": ua,
                "clearance_host": host or "grok.com",
            }

        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", "replace")[:300]
            logger.warning("flaresolverr http request failed: status={} body={}", exc.code, body_text)
        except URLError as exc:
            logger.warning("flaresolverr connection failed: reason={}", exc.reason)
        except Exception as exc:
            logger.warning("flaresolverr request failed: error={}", exc)

        return None


__all__ = ["FlareSolverrClearanceProvider"]
