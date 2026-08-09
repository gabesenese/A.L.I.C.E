"""Guards against the class of bug where a plugin constructor stalls application startup.

An interactive OAuth flow inside GmailPlugin.__init__ previously blocked
`import app.main` indefinitely, which in turn blocked the entire test suite at
conftest import time.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from ai.plugins.email_plugin import GmailPlugin
from ai.plugins.registry import construct_plugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPORT_BUDGET_SECONDS = 90


def test_importing_app_main_completes_within_budget():
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=IMPORT_BUDGET_SECONDS,
    )
    elapsed = time.perf_counter() - started
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert elapsed < IMPORT_BUDGET_SECONDS


def test_gmail_plugin_construction_does_not_authenticate():
    plugin = GmailPlugin()
    assert plugin._service is None
    assert plugin._auth_attempted is False
    assert plugin.creds is None


def test_gmail_plugin_never_starts_interactive_flow_without_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("ALICE_ALLOW_INTERACTIVE_AUTH", raising=False)
    monkeypatch.setenv("ALICE_PROJECT_ROOT", str(tmp_path))
    cred_dir = tmp_path / "config" / "cred"
    cred_dir.mkdir(parents=True)
    (cred_dir / "gmail_credentials.json").write_text("{}", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("interactive OAuth flow started without ALICE_ALLOW_INTERACTIVE_AUTH")

    monkeypatch.setattr("ai.plugins.email_plugin.InstalledAppFlow", type("Flow", (), {"from_client_secrets_file": staticmethod(fail_if_called)}))

    plugin = GmailPlugin()
    assert plugin.service is None


def test_calendar_plugin_never_starts_interactive_flow_without_opt_in(monkeypatch, tmp_path):
    calendar_plugin = pytest.importorskip("ai.plugins.calendar_plugin")
    if not getattr(calendar_plugin, "GOOGLE_AVAILABLE", False):
        pytest.skip("Google Calendar dependencies not installed")

    monkeypatch.delenv("ALICE_ALLOW_INTERACTIVE_AUTH", raising=False)
    cred_dir = tmp_path / "config" / "cred"
    cred_dir.mkdir(parents=True)
    (cred_dir / "calendar_credentials.json").write_text("{}", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("interactive OAuth flow started without ALICE_ALLOW_INTERACTIVE_AUTH")

    monkeypatch.setattr(
        calendar_plugin,
        "InstalledAppFlow",
        type("Flow", (), {"from_client_secrets_file": staticmethod(fail_if_called)}),
    )

    plugin = calendar_plugin.CalendarPlugin()
    plugin.credentials_file = str(cred_dir / "calendar_credentials.json")
    plugin.token_file = str(cred_dir / "calendar_token.pickle")
    assert plugin._authenticate() is None


def test_construct_plugin_gives_up_on_a_blocking_constructor():
    class BlockingPlugin:
        def __init__(self):
            time.sleep(30)

    import ai.plugins.registry as registry_module

    original = registry_module.CONSTRUCTION_TIMEOUT_SECONDS
    registry_module.CONSTRUCTION_TIMEOUT_SECONDS = 1
    try:
        started = time.perf_counter()
        result = construct_plugin(BlockingPlugin, "BlockingPlugin")
        elapsed = time.perf_counter() - started
    finally:
        registry_module.CONSTRUCTION_TIMEOUT_SECONDS = original

    assert result is None
    assert elapsed < 10


def test_construct_plugin_survives_a_raising_constructor():
    class ExplodingPlugin:
        def __init__(self):
            raise RuntimeError("no credentials")

    assert construct_plugin(ExplodingPlugin, "ExplodingPlugin") is None
