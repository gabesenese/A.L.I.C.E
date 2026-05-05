from pathlib import Path

from app.runtime_modes import RuntimeModeConfig


def test_minimal_runtime_mode_disables_optional_groups():
    cfg = RuntimeModeConfig.for_mode("minimal")
    assert cfg.enable_voice is False
    assert cfg.enable_training is False
    assert cfg.enable_lab_tools is False
    assert cfg.enable_background_learning is False
    assert cfg.enable_proactive_loops is False
    assert cfg.enable_cognitive_orchestrator is False
    assert cfg.enable_autonomous_agent is False
    assert cfg.enable_analytics is False


def test_factory_wrapper_is_thin():
    path = Path("ai/runtime/alice_contract_factory.py")
    line_count = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    assert line_count < 150

