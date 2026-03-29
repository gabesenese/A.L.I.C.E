from ai.runtime.user_state_model import UserStateModel


def test_user_state_model_updates_current_and_prior_task():
    model = UserStateModel()

    state1 = model.update_turn(
        user_id="u1",
        intent="notes:search",
        route="tool",
        unresolved_references=["it"],
        last_tool_used="NotesPlugin",
        last_result_produced="Found notes",
        world_state_snapshot={"ok": True},
    )
    assert state1.current_task == "notes:search"
    assert state1.prior_task == ""

    state2 = model.update_turn(
        user_id="u1",
        intent="weather:current",
        route="tool",
        unresolved_references=[],
        last_tool_used="WeatherPlugin",
        last_result_produced="It is sunny",
        world_state_snapshot={"ok": True},
    )
    assert state2.current_task == "weather:current"
    assert state2.prior_task == "notes:search"
    assert state2.last_tool_used == "WeatherPlugin"
