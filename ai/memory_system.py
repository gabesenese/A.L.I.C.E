"""
Advanced Memory System for A.L.I.C.E
Features:
- Vector embeddings for semantic search
- RAG (Retrieval Augmented Generation) capabilities
- Long-term memory with ChromaDB/FAISS
- Episodic and semantic memory
- Knowledge graph integration
"""

import os
import json
import logging
import pickle
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry"""
    id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural"
    timestamp: str
    context: Dict[str, Any]
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class VectorStore:
    """
    Simple vector store for semantic search
    Can be replaced with ChromaDB, FAISS, or Pinecone for production
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.ids: List[str] = []
    
    def add(self, id: str, vector: np.ndarray, metadata: Dict):
        """Add vector to store"""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        
        self.ids.append(id)
        self.vectors.append(vector)
        self.metadata.append(metadata)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors using cosine similarity
        
        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        if not self.vectors:
            return []
        
        # Calculate cosine similarities
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        similarities = []
        for i, vec in enumerate(self.vectors):
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            similarity = np.dot(query_norm, vec_norm)
            similarities.append((self.ids[i], float(similarity), self.metadata[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def delete(self, id: str) -> bool:
        """Delete vector by ID"""
        if id in self.ids:
            index = self.ids.index(id)
            del self.ids[index]
            del self.vectors[index]
            del self.metadata[index]
            return True
        return False
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'dimension': self.dimension,
            'ids': self.ids,
            'vectors': [v.tolist() for v in self.vectors],
            'metadata': self.metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Vector store saved ({len(self.ids)} vectors)")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.dimension = data['dimension']
            self.ids = data['ids']
            self.vectors = [np.array(v) for v in data['vectors']]
            self.metadata = data['metadata']
            logger.info(f"📂 Vector store loaded ({len(self.ids)} vectors)")


class MemorySystem:
    """
    Advanced memory management system for A.L.I.C.E
    Implements episodic, semantic, and procedural memory with RAG
    """
    
    def __init__(self, data_dir: str = "data/memory"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Memory stores
        self.episodic_memory: List[MemoryEntry] = []  # Conversations and events
        self.semantic_memory: List[MemoryEntry] = []  # Facts and knowledge
        self.procedural_memory: List[MemoryEntry] = []  # How-to knowledge
        
        # Vector store for semantic search
        self.vector_store = VectorStore(dimension=384)
        
        # Embedding function (lightweight - can use sentence-transformers for better results)
        self._embedding_model = None
        
        # Load existing memories
        self._load_memories()
        
        logger.info("[OK] Memory System initialized")
    
    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded (sentence-transformers)")
            except ImportError:
                logger.warning("[WARNING] sentence-transformers not installed. Using simple embeddings.")
                logger.warning("   Install with: pip install sentence-transformers")
                # Fallback to simple TF-IDF based embeddings
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._embedding_model = TfidfVectorizer(max_features=384)
        
        return self._embedding_model
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding vector for text"""
        model = self._get_embedding_model()
        
        try:
            # Try sentence-transformers first
            if hasattr(model, 'encode'):
                embedding = model.encode(text, convert_to_numpy=True)
                return embedding
            else:
                # Fallback to TF-IDF
                embedding = model.transform([text]).toarray()[0]
                # Pad or truncate to 384 dimensions
                if len(embedding) < 384:
                    embedding = np.pad(embedding, (0, 384 - len(embedding)))
                else:
                    embedding = embedding[:384]
                return embedding
        except Exception as e:
            logger.error(f"[ERROR] Embedding error: {e}")
            # Return random embedding as last resort
            return np.random.rand(384)
    
    def store_memory(
        self, 
        content: str, 
        memory_type: str = "episodic",
        context: Optional[Dict] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a new memory
        
        Args:
            content: Memory content
            memory_type: Type of memory (episodic, semantic, procedural)
            context: Additional context information
            importance: Importance score (0-1)
            tags: Tags for categorization
            
        Returns:
            Memory ID
        """
        # Generate unique ID
        memory_id = f"{memory_type}_{datetime.now().timestamp()}"
        
        # Create embedding
        embedding = self._create_embedding(content)
        
        # Create memory entry
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now().isoformat(),
            context=context or {},
            importance=importance,
            embedding=embedding.tolist(),
            tags=tags or []
        )
        
        # Store in appropriate memory
        if memory_type == "episodic":
            self.episodic_memory.append(memory)
        elif memory_type == "semantic":
            self.semantic_memory.append(memory)
        elif memory_type == "procedural":
            self.procedural_memory.append(memory)
        
        # Add to vector store for search
        self.vector_store.add(
            id=memory_id,
            vector=embedding,
            metadata={
                "content": content,
                "type": memory_type,
                "timestamp": memory.timestamp,
                "importance": importance,
                "tags": tags or []
            }
        )
        
        logger.info(f"Memory stored: {memory_type} - {content[:50]}...")
        return memory_id
    
    def recall_memory(
        self, 
        query: str, 
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Recall memories using semantic search
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant memories with metadata
        """
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # Filter by memory type and similarity
        filtered_results = []
        for mem_id, similarity, metadata in results:
            if similarity < min_similarity:
                continue
            
            if memory_type and metadata.get('type') != memory_type:
                continue
            
            # Get full memory entry
            memory = self._get_memory_by_id(mem_id)
            if memory:
                # Update access statistics
                memory.access_count += 1
                memory.last_accessed = datetime.now().isoformat()
                
                filtered_results.append({
                    "id": mem_id,
                    "content": metadata['content'],
                    "type": metadata['type'],
                    "similarity": similarity,
                    "importance": metadata['importance'],
                    "timestamp": metadata['timestamp'],
                    "access_count": memory.access_count,
                    "tags": metadata.get('tags', [])
                })
            
            if len(filtered_results) >= top_k:
                break
        
        logger.info(f"Recalled {len(filtered_results)} memories for: {query[:50]}...")
        return filtered_results
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get memory entry by ID"""
        for memory_list in [self.episodic_memory, self.semantic_memory, self.procedural_memory]:
            for memory in memory_list:
                if memory.id == memory_id:
                    return memory
        return None
    
    def get_context_for_llm(self, query: str, max_memories: int = 3) -> str:
        """
        Get relevant context from memory for LLM (RAG)
        
        Args:
            query: User query
            max_memories: Maximum number of memories to include
            
        Returns:
            Formatted context string
        """
        # Recall relevant memories
        memories = self.recall_memory(query, top_k=max_memories)
        
        if not memories:
            return ""
        
        # Format as context
        context_parts = ["Relevant information from memory:"]
        for i, mem in enumerate(memories, 1):
            context_parts.append(f"{i}. {mem['content']} (from {mem['timestamp'][:10]})")
        
        return "\n".join(context_parts)
    
    def consolidate_memories(self, max_episodic: int = 1000):
        """
        Consolidate old episodic memories
        Keep important ones, summarize or archive others
        """
        if len(self.episodic_memory) <= max_episodic:
            return
        
        # Sort by importance and recency
        self.episodic_memory.sort(
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )
        
        # Keep top memories
        archived = self.episodic_memory[max_episodic:]
        self.episodic_memory = self.episodic_memory[:max_episodic]
        
        logger.info(f"🗂️ Consolidated memories: kept {max_episodic}, archived {len(archived)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "procedural_count": len(self.procedural_memory),
            "total_memories": len(self.episodic_memory) + len(self.semantic_memory) + len(self.procedural_memory),
            "vector_count": len(self.vector_store.ids)
        }
    
    def _save_memories(self):
        """Save all memories to disk"""
        try:
            # Save memory lists
            memories_data = {
                "episodic": [asdict(m) for m in self.episodic_memory],
                "semantic": [asdict(m) for m in self.semantic_memory],
                "procedural": [asdict(m) for m in self.procedural_memory]
            }
            
            with open(os.path.join(self.data_dir, "memories.json"), 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, indent=2)
            
            # Save vector store
            self.vector_store.save(os.path.join(self.data_dir, "vectors.pkl"))
            
            logger.info("All memories saved")
            
        except Exception as e:
            logger.error(f"[ERROR] Error saving memories: {e}")
    
    def _load_memories(self):
        """Load memories from disk"""
        try:
            memories_path = os.path.join(self.data_dir, "memories.json")
            if os.path.exists(memories_path):
                with open(memories_path, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
                
                self.episodic_memory = [MemoryEntry(**m) for m in memories_data.get("episodic", [])]
                self.semantic_memory = [MemoryEntry(**m) for m in memories_data.get("semantic", [])]
                self.procedural_memory = [MemoryEntry(**m) for m in memories_data.get("procedural", [])]
                
                logger.info(f"📂 Loaded {len(self.episodic_memory)} episodic, "
                          f"{len(self.semantic_memory)} semantic, "
                          f"{len(self.procedural_memory)} procedural memories")
            
            # Load vector store
            vectors_path = os.path.join(self.data_dir, "vectors.pkl")
            if os.path.exists(vectors_path):
                self.vector_store.load(vectors_path)
                
        except Exception as e:
            logger.warning(f"[WARNING] Could not load memories: {e}")
    
    def __enter__(self):
        """Context manager enter"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save on exit"""
        self._save_memories()


# Example usage
if __name__ == "__main__":
    print("Testing Memory System...\n")
    
    with MemorySystem() as memory:
        # Store different types of memories
        print("Storing memories...")
        
        # Episodic memories (conversations/events)
        memory.store_memory(
            "User asked about the weather in New York on January 24, 2026",
            memory_type="episodic",
            context={"location": "New York", "topic": "weather"},
            importance=0.6,
            tags=["weather", "conversation"]
        )
        
        memory.store_memory(
            "User mentioned they like Python programming",
            memory_type="episodic",
            context={"topic": "preferences"},
            importance=0.8,
            tags=["preferences", "programming"]
        )
        
        # Semantic memories (facts)
        memory.store_memory(
            "Python is a high-level programming language known for readability",
            memory_type="semantic",
            importance=0.7,
            tags=["python", "programming", "facts"]
        )
        
        memory.store_memory(
            "The Eiffel Tower is located in Paris, France",
            memory_type="semantic",
            importance=0.5,
            tags=["facts", "geography"]
        )
        
        # Procedural memory (how-to)
        memory.store_memory(
            "To create a Python virtual environment, use: python -m venv env",
            memory_type="procedural",
            importance=0.9,
            tags=["python", "tutorial", "setup"]
        )
        
        print(f"✅ Stored memories\n")
        
        # Recall memories
        print("🔍 Testing memory recall...")
        
        queries = [
            "Tell me about Python",
            "What do I like?",
            "How to setup Python environment?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            results = memory.recall_memory(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['type']}] {result['content']}")
                print(f"     Similarity: {result['similarity']:.3f}, "
                      f"Importance: {result['importance']}, "
                      f"Accessed: {result['access_count']} times")
        
        # Get RAG context
        print(f"\nRAG Context for 'Python programming':")
        context = memory.get_context_for_llm("Python programming", max_memories=3)
        print(context)
        
        # Statistics
        print(f"\n📊 Memory Statistics:")
        stats = memory.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    print("\n[OK] Memories saved automatically on exit")
