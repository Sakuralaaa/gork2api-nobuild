import base64
import json
import unittest

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from app.products.openai.dpop import (
    DpopCredentials,
    normalize_dpop_htu,
    _public_jwk,
)


def _decode_segment(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class ConsoleDpopTests(unittest.TestCase):
    def test_htu_matches_console_frontend_normalization(self):
        self.assertEqual(
            normalize_dpop_htu(
                "https://console.x.ai/v1/responses?model=grok#response"
            ),
            "https://console.x.ai/v1/responses",
        )

    def test_proof_uses_bound_access_token_and_raw_es256_signature(self):
        private_key = ec.generate_private_key(ec.SECP256R1())
        access_token = "dpop-access-token"
        credentials = DpopCredentials(
            private_key=private_key,
            public_jwk=_public_jwk(private_key),
            access_token=access_token,
            clock_offset_s=0,
        )

        headers = credentials.headers(
            method="POST",
            url="https://console.x.ai/v1/responses?ignored=query",
        )
        proof_parts = headers["DPoP"].split(".")

        self.assertEqual(len(proof_parts), 3)
        self.assertEqual(headers["Authorization"], f"DPoP {access_token}")
        proof_header = json.loads(_decode_segment(proof_parts[0]))
        proof_payload = json.loads(_decode_segment(proof_parts[1]))
        signature = _decode_segment(proof_parts[2])

        self.assertEqual(proof_header["typ"], "dpop+jwt")
        self.assertEqual(proof_header["alg"], "ES256")
        self.assertEqual(proof_header["jwk"], credentials.public_jwk)
        self.assertEqual(proof_payload["htm"], "POST")
        self.assertEqual(proof_payload["htu"], "https://console.x.ai/v1/responses")
        self.assertEqual(len(proof_payload["jti"]), 22)
        self.assertEqual(len(signature), 64)

        r = int.from_bytes(signature[:32], "big")
        s = int.from_bytes(signature[32:], "big")
        private_key.public_key().verify(
            encode_dss_signature(r, s),
            f"{proof_parts[0]}.{proof_parts[1]}".encode("ascii"),
            ec.ECDSA(hashes.SHA256()),
        )


if __name__ == "__main__":
    unittest.main()
