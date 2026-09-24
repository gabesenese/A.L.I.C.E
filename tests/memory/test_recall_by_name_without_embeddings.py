"""A name he mentioned is recalled even when semantic search finds nothing.

Without sentence-transformers, embeddings fall back to TF-IDF fitted once on the
first text seen, so a word that arrived later carries no weight. Semantic recall
of "who is Sarah?" then came back empty, as if he had never mentioned her.
"""

import pytest

from ai.memory.memory_system import MemorySystem


@pytest.fixture
def memory(tmp_path, monkeypatch):
    memory = MemorySystem(data_dir=str(tmp_path))
    memory.store_memory(content="My sister Sarah is visiting next week.", memory_type="episodic")
    memory.store_memory(content="The parser rewrite is blocked on the tokenizer.", memory_type="episodic")
    # Semantic recall finding nothing, as it does without an embedding model.
    monkeypatch.setattr(memory, "recall_memory_weighted", lambda **_kwargs: [])
    monkeypatch.setattr(memory, "recall_memory", lambda **_kwargs: [])
    return memory


def test_a_name_is_recalled_when_semantic_search_comes_back_empty(memory):
    hits = memory.search("who is Sarah?")
    assert len(hits) == 1
    assert "Sarah" in hits[0]["content"]
    assert hits[0]["match"] == "literal"


def test_a_question_of_common_words_recalls_nothing(memory):
    assert memory.search("what do you know?") == []
