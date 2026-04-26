from types import SimpleNamespace

from app.main import ALICE


def test_resolve_runtime_user_name_defaults_when_missing_attr():
    alice = ALICE.__new__(ALICE)

    resolved = alice._resolve_runtime_user_name()

    assert resolved == "User"
    assert alice.user_name == "User"


def test_resolve_runtime_user_name_uses_context_preference_name():
    alice = ALICE.__new__(ALICE)
    alice.context = SimpleNamespace(user_prefs=SimpleNamespace(name="Gabriel"))

    resolved = alice._resolve_runtime_user_name()

    assert resolved == "Gabriel"
    assert alice.user_name == "Gabriel"


def test_resolve_runtime_user_name_prefers_existing_attribute():
    alice = ALICE.__new__(ALICE)
    alice.user_name = "Casey"
    alice.context = SimpleNamespace(user_prefs=SimpleNamespace(name="Other"))

    resolved = alice._resolve_runtime_user_name()

    assert resolved == "Casey"
