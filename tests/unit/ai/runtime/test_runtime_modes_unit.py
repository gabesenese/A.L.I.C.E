from ai.runtime.runtime_modes import get_runtime_mode


def test_default_runtime_mode_is_minimal():
    mode = get_runtime_mode()
    assert mode.name == "minimal"
    assert "voice" not in mode.enabled_groups
    assert "lab" not in mode.enabled_groups
    assert "training" not in mode.enabled_groups

