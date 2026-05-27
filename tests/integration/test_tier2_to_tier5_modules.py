from ai.core.adaptive_response_style import AdaptiveResponseStyle
from ai.core.causal_inference_engine import CausalInferenceEngine
from ai.core.cross_session_pattern_detector import CrossSessionPatternDetector
from ai.core.decision_constraint_solver import DecisionConstraintSolver
from ai.core.episodic_memory_engine import EpisodicMemoryEngine
from ai.core.hypothetical_scenario_generator import HypotheticalScenarioGenerator
from ai.core.memory_consolidator import MemoryConsolidator
from ai.core.multi_step_reasoning_engine import MultiStepReasoningEngine
from ai.core.proactive_interruption_manager import ProactiveInterruptionManager
from ai.core.semantic_memory_index import SemanticMemoryIndex


def test_multi_step_reasoning_builds_plan_with_secondary_intents():
    engine = MultiStepReasoningEngine()
    plan = engine.plan_turn(
        user_input="create a note and remind me at 9",
        primary_intent="notes:create",
        primary_confidence=0.88,
        secondary_intents=[{"intent": "reminder:create", "confidence": 0.81, "text": "remind me at 9"}],
    )
    payload = plan.as_dict()
    assert payload["complexity"] == "multi_step"
    assert payload["step_count"] == 2


def test_episodic_memory_and_semantic_index_support_recall():
    episodic = EpisodicMemoryEngine(max_episodes=10)
    semantic = SemanticMemoryIndex()
    episodic.add_episode(
        user_input="debug import error in parser",
        intent="technical:debug",
        response="Check dependency versions first",
        entities={"topic": "parser"},
    )
    semantic.add("doc-1", "technical debug parser import error dependency versions")

    hits = episodic.recall_similar("parser import", limit=2)
    assert hits
    sem_hits = semantic.search("parser import", limit=2)
    assert sem_hits


def test_proactive_interruption_manager_throttles_suggestions():
    manager = ProactiveInterruptionManager(cooldown_seconds=3600, max_suggestions=1)
    first = manager.select(["suggestion one", "suggestion two"], now_ts=1000.0)
    second = manager.select(["suggestion three"], now_ts=1200.0)
    assert first == ["suggestion one"]
    assert second == []


def test_advanced_reasoning_modules_return_structured_output():
    causal = CausalInferenceEngine()
    hyp = HypotheticalScenarioGenerator()
    solver = DecisionConstraintSolver()

    analysis = causal.infer("why did this import error timeout happen")
    assert analysis["likely_causes"]

    scenarios = hyp.generate("what if deployment fails")
    assert len(scenarios) >= 2

    ranked = solver.solve(
        [
            {"name": "a", "quality": 0.6, "speed": 0.9, "risk": 0.5},
            {"name": "b", "quality": 0.8, "speed": 0.7, "risk": 0.2},
        ],
        soft_weights={"quality": 0.6, "speed": 0.3, "risk": -0.2},
    )
    assert ranked and ranked[0]["name"] in {"a", "b"}


def test_memory_consolidation_and_cross_session_patterns():
    episodic = EpisodicMemoryEngine(max_episodes=20)
    detector = CrossSessionPatternDetector()
    consolidator = MemoryConsolidator()

    for _ in range(4):
        episodic.add_episode(
            user_input="check weather",
            intent="weather:current",
            response="It is sunny",
            entities={},
        )
        detector.observe("weather:current")

    consolidated = consolidator.consolidate(episodic.recall_recent(limit=10))
    summary = detector.summary(top_n=3)

    assert consolidated["episode_count"] >= 1
    assert "weather:current" in summary


def test_adaptive_response_style_enforces_word_limit_and_format():
    styler = AdaptiveResponseStyle()
    response = "This is sentence one. This is sentence two. This is sentence three."
    out = styler.apply_constraints(
        response,
        {"format": "bullet_points", "max_words": 10},
    )
    assert out
    assert len(out.split()) <= 10
