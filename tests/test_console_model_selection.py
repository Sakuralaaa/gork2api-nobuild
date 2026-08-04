import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import orjson

from app.control.model import registry
from app.control.account.models import AccountRecord
from app.products.openai.console import (
    _CONSOLE_FIXED_EFFORT,
    _build_console_headers,
    _is_cloudflare_forbidden,
    _console_status_message,
    _proxy_feedback_kind,
    console_upstream_model,
    is_console_basic_model,
)
from app.control.proxy import _clearance_host, _normalize_clearance_origin
from app.control.proxy.models import ProxyFeedbackKind
from app.products._account_selection import mode_candidates

openai_router_module = import_module("app.products.openai.router")


CANONICAL_CONSOLE_MODELS = {
    "grok-4.20-fast-console",
    "grok-4.20-expert-console",
    "grok-4.3-low-console",
    "grok-4.3-medium-console",
    "grok-4.3-high-console",
    "grok-4.20-heavy-low-console",
    "grok-4.20-heavy-medium-console",
    "grok-4.20-heavy-high-console",
    "grok-4.20-heavy-xhigh-console",
    "grok-4.5-console",
    "grok-build-console",
}

LEGACY_CONSOLE_ALIASES = {
    "grok-4.20-0309-non-reasoning-console",
    "grok-4.20-0309-console",
    "grok-4.20-0309-reasoning-console",
    "grok-4.20-reasoning-console",
    "grok-4.3-console",
    "grok-4.3-beta-console",
    "grok-4.20-multi-agent-console",
    "grok-4.20-heavy-console",
    "grok-4.20-multi-agent-0309-console",
    "grok-4.20-multi-agent-low-console",
    "grok-4.20-multi-agent-medium-console",
    "grok-4.20-multi-agent-high-console",
    "grok-4.20-multi-agent-xhigh-console",
}


class _FakeRepo:
    async def runtime_snapshot(self):
        return SimpleNamespace(items=[AccountRecord(token="sso-token", pool="basic")])


def _fake_request():
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(repository=_FakeRepo()))
    )


class ConsoleModelSelectionTests(unittest.TestCase):
    def test_clearance_origin_is_normalized_to_absolute_https_url(self):
        self.assertEqual(
            _normalize_clearance_origin("console.x.ai"),
            "https://console.x.ai",
        )
        self.assertEqual(_clearance_host("console.x.ai"), "console.x.ai")

    def test_console_models_use_existing_virtual_quota_buckets(self):
        cases = {
            "grok-4.20-fast-console": 2,
            "grok-4.20-0309-reasoning-console": 2,
            "grok-4.20-multi-agent-console": 3,
            "grok-4.3-console": 4,
            "grok-4.5-console": 4,
        }
        for model, expected_mode in cases.items():
            with self.subTest(model=model):
                self.assertEqual(mode_candidates(registry.resolve(model)), (expected_mode,))

    def test_canonical_console_models_have_complete_mapping(self):
        for model in CANONICAL_CONSOLE_MODELS:
            with self.subTest(model=model):
                self.assertIsNotNone(registry.resolve(model))
                self.assertTrue(is_console_basic_model(model))
                self.assertNotEqual(console_upstream_model(model), model)
                self.assertEqual(len(mode_candidates(registry.resolve(model))), 1)

    def test_legacy_console_aliases_remain_callable_but_hidden(self):
        for model in LEGACY_CONSOLE_ALIASES:
            with self.subTest(model=model):
                self.assertIsNotNone(registry.resolve(model))
                self.assertTrue(is_console_basic_model(model))
                self.assertIn(model, openai_router_module._MODEL_LIST_HIDDEN_ALIASES)

    def test_fixed_effort_is_driven_by_model_name(self):
        cases = {
            "grok-4.3-low-console": "low",
            "grok-4.3-medium-console": "medium",
            "grok-4.3-high-console": "high",
            "grok-4.20-heavy-low-console": "low",
            "grok-4.20-heavy-medium-console": "medium",
            "grok-4.20-heavy-high-console": "high",
            "grok-4.20-heavy-xhigh-console": "xhigh",
        }
        for model, effort in cases.items():
            with self.subTest(model=model):
                self.assertEqual(_CONSOLE_FIXED_EFFORT[model], effort)

    def test_grok_45_console_404_message_is_explicit(self):
        message = _console_status_message(
            404,
            model="grok-4.5-console",
            upstream_model="grok-4.5",
        )
        self.assertIn("upstream is currently unavailable", message)
        self.assertIn("grok-4.5", message)

    def test_console_headers_reuse_clearance_lease_user_agent(self):
        lease = SimpleNamespace(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            cf_cookies="cf_clearance=clearance-value",
        )

        headers = _build_console_headers("sso-token", lease)

        self.assertEqual(headers["User-Agent"], lease.user_agent)

    def test_console_403_message_does_not_reference_nonexistent_cookie_setting(self):
        message = _console_status_message(403)

        self.assertNotIn("console.cf_cookies", message)
        self.assertIn("SSO token", message)
        self.assertIn("egress IP consistency", message)

    def test_console_empty_403_is_account_forbidden_not_clearance_challenge(self):
        response = SimpleNamespace(headers={"server": "granian"})

        self.assertFalse(_is_cloudflare_forbidden(response, ""))
        self.assertEqual(
            _proxy_feedback_kind(403, response=response, body=""),
            ProxyFeedbackKind.FORBIDDEN,
        )

    def test_console_cloudflare_403_invalidates_clearance(self):
        response = SimpleNamespace(headers={"server": "cloudflare", "cf-ray": "ray"})

        self.assertTrue(_is_cloudflare_forbidden(response, ""))
        self.assertEqual(
            _proxy_feedback_kind(403, response=response, body=""),
            ProxyFeedbackKind.CHALLENGE,
        )


class ConsoleModelListTests(unittest.IsolatedAsyncioTestCase):
    async def test_models_endpoint_only_lists_canonical_console_names(self):
        response = await openai_router_module.list_models(_fake_request())
        body = orjson.loads(response.body)
        model_ids = {item["id"] for item in body["data"]}
        visible_console_models = {
            model for model in model_ids if model.endswith("-console")
        }

        self.assertEqual(visible_console_models, CANONICAL_CONSOLE_MODELS)
        self.assertTrue(LEGACY_CONSOLE_ALIASES.isdisjoint(model_ids))


class ConsoleConfigPageTests(unittest.TestCase):
    def test_config_page_has_console_fields_and_no_placeholder_text(self):
        html = Path("app/statics/admin/config.html").read_text(encoding="utf-8")
        self.assertNotIn("????", html)
        for text in (
            "section: 'console'",
            "responses_url",
            "cluster",
            "team_id",
            "enable_search_tools",
            "user_agent",
            "browser",
        ):
            with self.subTest(text=text):
                self.assertIn(text, html)


if __name__ == "__main__":
    unittest.main()
