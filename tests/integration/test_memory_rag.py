"""
Integration Tests for Memory RAG System
Tests retrieval-augmented generation capabilities
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.memory.memory_store import InMemoryMemoryStore, MemoryEntry
from ai.memory.embedding_manager import EmbeddingManager
from ai.memory.rag_engine import RAGEngine
from datetime import datetime


class TestMemoryRAG:
    """Integration tests for RAG (Retrieval-Augmented Generation)"""

    @pytest.fixture
    def store(self):
        """Create fresh memory store"""
        return InMemoryMemoryStore()

    @pytest.fixture
    def embedding(self):
        """Create embedding manager"""
        return EmbeddingManager()

    @pytest.fixture
    def rag(self, store, embedding):
        """Create RAG engine"""
        return RAGEngine(store, embedding)

    @pytest.fixture
    def populated_store(self, store, embedding):
        """Store with test memories"""
        memories = [
            MemoryEntry(
                id="mem_1",
                content="Python is a programming language",
                memory_type="semantic",
                timestamp=datetime.now().isoformat(),
                context={},
                importance=0.8,
                embedding=embedding.create_embedding("Python is a programming language").tolist(),
                tags=["programming", "python"]
            ),
            MemoryEntry(
                id="mem_2",
                content="Alice is an AI assistant",
                memory_type="semantic",
                timestamp=datetime.now().isoformat(),
                context={},
                importance=0.9,
                embedding=embedding.create_embedding("Alice is an AI assistant").tolist(),
                tags=["alice", "ai"]
            ),
            MemoryEntry(
                id="mem_3",
                content="Machine learning uses neural networks",
                memory_type="semantic",
                timestamp=datetime.now().isoformat(),
                context={},
                importance=0.7,
                embedding=embedding.create_embedding("Machine learning uses neural networks").tolist(),
                tags=["ml", "ai"]
            ),
        ]

        for mem in memories:
            store.add(mem)

        return store

    def test_semantic_search_finds_relevant_memories(self, rag, populated_store):
        """RAG should find semantically similar memories"""
        results = rag.query(
            query_text="What is Python?",
            top_k=5,
            min_similarity=0.3
        )

        assert len(results) > 0
        # Should find the Python memory
        python_memory = next((r for r in results if "Python" in r['content']), None)
        assert python_memory is not None
        assert python_memory['similarity'] > 0.3

    def test_rag_filters_by_similarity_threshold(self, rag, populated_store):
        """RAG should respect similarity thresholds"""
        # High threshold should return fewer results
        high_threshold_results = rag.query(
            query_text="programming",
            top_k=10,
            min_similarity=0.8
        )

        # Low threshold should return more results
        low_threshold_results = rag.query(
            query_text="programming",
            top_k=10,
            min_similarity=0.3
        )

        assert len(low_threshold_results) >= len(high_threshold_results)

    def test_rag_respects_top_k_limit(self, rag, populated_store):
        """RAG should respect top_k parameter"""
        results = rag.query(
            query_text="AI and programming",
            top_k=1,
            min_similarity=0.1
        )

        assert len(results) <= 1

    def test_llm_context_generation(self, rag, populated_store):
        """RAG should generate proper LLM context"""
        context = rag.get_llm_context(
            user_input="Tell me about Python",
            max_tokens=1000,
            max_memories=3
        )

        assert isinstance(context, str)
        if context:  # May be empty if no relevant memories
            assert "Relevant memories:" in context or len(context) == 0

    def test_document_search_filters_by_type(self, store, embedding, rag):
        """Document search should only return document types"""
        # Add episodic and document memories
        episodic_mem = MemoryEntry(
            id="episodic_1",
            content="User asked about weather",
            memory_type="episodic",
            timestamp=datetime.now().isoformat(),
            context={},
            importance=0.5,
            embedding=embedding.create_embedding("User asked about weather").tolist(),
            tags=["conversation"]
        )

        document_mem = MemoryEntry(
            id="doc_1",
            content="Python documentation explains functions",
            memory_type="document",
            timestamp=datetime.now().isoformat(),
            context={},
            importance=0.8,
            embedding=embedding.create_embedding("Python documentation explains functions").tolist(),
            tags=["docs"],
            source_file="python_docs.pdf",
            chunk_index=0
        )

        store.add(episodic_mem)
        store.add(document_mem)

        # Search documents only
        doc_results = rag.search_documents(
            query="Python functions",
            top_k=10,
            min_similarity=0.3
        )

        # Should only return document type
        for result in doc_results:
            memory = store.get_by_id(result['id'])
            assert memory.memory_type == "document"

    def test_empty_query_handling(self, rag, populated_store):
        """RAG should handle empty queries gracefully"""
        results = rag.query(
            query_text="",
            top_k=5,
            min_similarity=0.5
        )

        # Should return empty or handle gracefully
        assert isinstance(results, list)

    def test_similarity_scoring(self, rag, populated_store):
        """Similarity scores should be reasonable"""
        results = rag.query(
            query_text="Python programming language",
            top_k=5,
            min_similarity=0.0
        )

        for result in results:
            # Similarity should be between 0 and 1
            assert 0.0 <= result['similarity'] <= 1.0

        # Results should be sorted by similarity (highest first)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]['similarity'] >= results[i + 1]['similarity']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
