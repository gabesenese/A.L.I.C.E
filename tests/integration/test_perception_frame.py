from ai.runtime.perception_frame import build_perception_frame


def test_long_day_work_on_alice_perception():
    frame = build_perception_frame(
        "good, it was a long day, but its just monday so we are trying to stay positive, I want to work on alice for a little bit"
    )
    assert frame.social_context == "good; it was a long day; its just monday so we are trying to stay positive"
    assert frame.actual_request.lower() == "i want to work on alice for a little bit"
    assert frame.topic == "Alice"
    assert frame.user_energy_signal == "low"
    assert frame.user_mood_signal == "positive"
    assert frame.is_continuation is True
    assert frame.is_action_request is True


def test_weather_context_open_notes_perception():
    frame = build_perception_frame(
        "going pretty good, weather is great today, looking forward to it, I was wondering if i have any open notes?"
    )
    assert frame.social_context == "going pretty good; weather is great today; looking forward to it"
    assert frame.actual_request.lower() == "i have any open notes"
    assert frame.topic == "notes"
    assert frame.is_action_request is True


def test_memory_delete_marks_rights_request():
    frame = build_perception_frame("move on from this convo, and delete the memories from your data")
    assert frame.is_memory_rights_request is True
    assert "delete" in frame.actual_request.lower()


def test_hi_alice_marks_greeting():
    frame = build_perception_frame("hi alice")
    assert frame.is_greeting is True
    assert frame.actual_request == ""
