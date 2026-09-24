from types import SimpleNamespace

from app.main import ALICE


def _alice(tmp_path, monkeypatch, *, history=None, privacy=False):
    monkeypatch.setattr(ALICE, "CONVERSATION_STATE_PATH", str(tmp_path / "conversation_state.json"))
    alice = ALICE.__new__(ALICE)
    alice.conversation_summary = []
    alice.conversation_topics = []
    alice.referenced_items = {}
    alice.conversation_state_tracker = None
    alice.executive_controller = None
    alice.context = None
    alice.advanced_context = None
    alice.privacy_mode = privacy
    alice.llm = SimpleNamespace(conversation_history=list(history or []), config=SimpleNamespace(max_history=30))
    return alice


def test_the_conversation_is_still_there_after_a_restart(tmp_path, monkeypatch):
    turns = [
        {"role": "user", "content": "I'm moving the scheduler to asyncio."},
        {"role": "assistant", "content": "Then the blocking calls go first."},
    ]
    _alice(tmp_path, monkeypatch, history=turns)._save_conversation_state()

    restarted = _alice(tmp_path, monkeypatch)
    restarted._load_conversation_state()

    assert restarted.llm.conversation_history == turns


def test_privacy_mode_keeps_the_transcript_off_disk(tmp_path, monkeypatch):
    turns = [{"role": "user", "content": "secret"}, {"role": "assistant", "content": "ok"}]
    _alice(tmp_path, monkeypatch, history=turns, privacy=True)._save_conversation_state()

    restarted = _alice(tmp_path, monkeypatch)
    restarted._load_conversation_state()

    assert restarted.llm.conversation_history == []
