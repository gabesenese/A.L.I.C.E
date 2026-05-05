from app.runtime_modes import RuntimeModeConfig
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from tests.integration.test_contract_pipeline import _FakeAlice


def test_runtime_mode_config_minimal_disables_optional_systems():
    cfg = RuntimeModeConfig.for_mode("minimal")
    assert cfg.enable_voice is False
    assert cfg.enable_training is False
    assert cfg.enable_lab_tools is False
    assert cfg.enable_background_learning is False
    assert cfg.enable_proactive_loops is False
    assert cfg.enable_cognitive_orchestrator is False
    assert cfg.enable_autonomous_agent is False
    assert cfg.enable_analytics is False
    assert cfg.enable_advanced_tiers is False
    assert cfg.enable_contract_pipeline is True
    assert cfg.enable_local_actions is True


def test_runtime_mode_config_agentic_enables_agentic_only():
    cfg = RuntimeModeConfig.for_mode("agentic")
    assert cfg.enable_proactive_loops is True
    assert cfg.enable_cognitive_orchestrator is True
    assert cfg.enable_autonomous_agent is True
    assert cfg.enable_voice is False
    assert cfg.enable_training is False
    assert cfg.enable_lab_tools is False


def test_factory_wrapper_compatibility():
    boundaries = build_runtime_boundaries(_FakeAlice())
    assert boundaries is not None
    assert boundaries.routing is not None
    assert boundaries.tools is not None

