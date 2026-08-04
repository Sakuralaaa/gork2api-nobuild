"""DPoP helpers used by the console.x.ai web API."""

from __future__ import annotations

import base64
import hashlib
import math
import secrets
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import orjson
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _json_b64(value: dict[str, Any]) -> str:
    return _b64url(orjson.dumps(value))


def _public_jwk(private_key: ec.EllipticCurvePrivateKey) -> dict[str, str]:
    public_numbers = private_key.public_key().public_numbers()
    return {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64url(public_numbers.x.to_bytes(32, "big")),
        "y": _b64url(public_numbers.y.to_bytes(32, "big")),
    }


def normalize_dpop_htu(url: str) -> str:
    """Match the Console frontend's origin + pathname URL normalization."""
    parsed = urlsplit(str(url))
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"DPoP URL must be absolute: {url!r}")

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    host = f"[{hostname}]" if ":" in hostname else hostname
    port = parsed.port
    if port is not None and not (
        (scheme == "https" and port == 443)
        or (scheme == "http" and port == 80)
    ):
        host = f"{host}:{port}"
    return f"{scheme}://{host}{parsed.path or '/'}"


def _server_clock_offset(response: Any, started_at: float) -> int:
    headers = getattr(response, "headers", {}) or {}
    date_header = next(
        (
            str(value)
            for key, value in headers.items()
            if str(key).lower() == "date"
        ),
        "",
    )
    if not date_header:
        return 0
    try:
        server_time = parsedate_to_datetime(date_header).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0
    return round(server_time - started_at)


class DpopMintError(Exception):
    """The Console refused to mint a DPoP access token."""

    def __init__(self, status: int, body: str = "") -> None:
        self.status = int(status or 502)
        self.body = body[:400]
        super().__init__(f"Console DPoP token mint returned {self.status}")


@dataclass(frozen=True)
class DpopCredentials:
    private_key: ec.EllipticCurvePrivateKey
    public_jwk: dict[str, str]
    access_token: str
    clock_offset_s: int = 0

    def headers(self, *, method: str, url: str) -> dict[str, str]:
        """Build the two headers required by a Console API request."""
        htu = normalize_dpop_htu(url)
        payload = {
            "jti": _b64url(secrets.token_bytes(16)),
            "htm": str(method).upper(),
            "htu": htu,
            "iat": int(time.time()) + self.clock_offset_s,
            "ath": _b64url(hashlib.sha256(self.access_token.encode()).digest()),
        }
        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": self.public_jwk,
        }
        signing_input = f"{_json_b64(header)}.{_json_b64(payload)}".encode("ascii")
        der_signature = self.private_key.sign(
            signing_input,
            ec.ECDSA(hashes.SHA256()),
        )
        r, s = decode_dss_signature(der_signature)
        raw_signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        proof = f"{signing_input.decode('ascii')}.{_b64url(raw_signature)}"
        return {
            "Authorization": f"DPoP {self.access_token}",
            "DPoP": proof,
        }


async def mint_dpop_credentials(
    session: Any,
    *,
    url: str,
    headers: dict[str, str],
    timeout_s: float,
) -> DpopCredentials:
    """Mint a DPoP token using the current Console cookie/session lease."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_jwk = _public_jwk(private_key)
    mint_headers = dict(headers)
    mint_headers.pop("Authorization", None)
    mint_headers.pop("DPoP", None)
    mint_headers["Content-Type"] = "application/json"
    started_at = time.time()
    response = await session.post(
        url,
        headers=mint_headers,
        data=orjson.dumps({"jwk": public_jwk}),
        timeout=timeout_s,
    )
    if not 200 <= response.status_code < 300:
        body = response.content.decode("utf-8", "replace")[:400]
        raise DpopMintError(response.status_code, body)

    try:
        result = orjson.loads(response.content)
    except Exception as exc:
        raise DpopMintError(502, "invalid JSON response") from exc
    access_token = result.get("access_token") if isinstance(result, dict) else None
    expires_in = result.get("expires_in") if isinstance(result, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise DpopMintError(502, "missing access_token")
    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, (int, float))
        or not math.isfinite(expires_in)
    ):
        raise DpopMintError(502, "missing expires_in")

    return DpopCredentials(
        private_key=private_key,
        public_jwk=public_jwk,
        access_token=access_token,
        clock_offset_s=_server_clock_offset(response, started_at),
    )


__all__ = [
    "DpopCredentials",
    "DpopMintError",
    "mint_dpop_credentials",
    "normalize_dpop_htu",
]
