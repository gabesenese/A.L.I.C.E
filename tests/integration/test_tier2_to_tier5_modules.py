from ai.core.adaptive_response_style import AdaptiveResponseStyle
from ai.core.cross_session_pattern_detector import CrossSessionPatternDetector
from ai.core.episodic_memory_engine import EpisodicMemoryEngine
from ai.core.memory_consolidator import MemoryConsolidator
from ai.core.semantic_memory_index import SemanticMemoryIndex


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
