from ai.core.knowledge_engine import KnowledgeEngine, Relationship


def _new_engine(tmp_path):
    return KnowledgeEngine(storage_path=str(tmp_path / "knowledge"))


def test_rejects_uncertain_llm_knowledge(tmp_path):
    engine = _new_engine(tmp_path)

    result = engine.learn_from_interaction(
        user_input="Explain quantum tunneling",
        alice_response="I'm not sure, maybe it is about particles doing strange things.",
        intent="question:science",
        entities={},
        context={"route": "LLM"},
    )

    assert result["stored"] is False
    assert "contains_uncertainty" in result["validation"]["reasons"]
    assert len(engine.learned_responses["question:science"]) == 0


def test_rejects_generic_llm_knowledge(tmp_path):
    engine = _new_engine(tmp_path)

    result = engine.learn_from_interaction(
        user_input="How does encryption work?",
        alice_response="In general, there are many factors and it depends on the context.",
        intent="question:security",
        entities={},
        context={"route": "LLM"},
    )

    assert result["stored"] is False
    assert "too_generic" in result["validation"]["reasons"]
    assert len(engine.learned_responses["question:security"]) == 0


def test_rejects_contradictory_llm_knowledge(tmp_path):
    engine = _new_engine(tmp_path)

    known = Relationship("Water", "is", "H2O")
    known.confidence = 0.9
    engine.relationships.append(known)

    result = engine.learn_from_interaction(
        user_input="What is water?",
        alice_response="Water is fire.",
        intent="question:science",
        entities={},
        context={"route": "LLM"},
    )

    assert result["stored"] is False
    assert "contradictory" in result["validation"]["reasons"]


def test_accepts_relevant_specific_llm_knowledge(tmp_path):
    engine = _new_engine(tmp_path)

    result = engine.learn_from_interaction(
        user_input="What is photosynthesis?",
        alice_response="Photosynthesis is the process plants use to convert light into chemical energy.",
        intent="question:biology",
        entities={"topic": "photosynthesis"},
        context={"route": "LLM"},
    )

    assert result["stored"] is True
    assert len(engine.learned_responses["question:biology"]) == 1


def test_non_llm_route_skips_strict_validation(tmp_path):
    engine = _new_engine(tmp_path)

    result = engine.learn_from_interaction(
        user_input="What's in my calendar?",
        alice_response="I found two meetings today.",
        intent="calendar:query",
        entities={},
        context={"route": "TOOL"},
    )

    assert result["stored"] is True
    assert result["validation"].get("skipped") is True
