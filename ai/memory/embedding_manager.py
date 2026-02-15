"""
Embedding Manager for A.L.I.C.E Memory System
Manages vector embeddings for semantic similarity
"""

import logging
import importlib
import numpy as np
from typing import Optional
import os

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages vector embeddings for semantic similarity"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', dimension: int = 384) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: Optional[any] = None

    def _get_model(self):
        """Lazy load embedding model"""
        if self._model is None:
            try:
                sentence_transformers = importlib.import_module("sentence_transformers")
                SentenceTransformer = sentence_transformers.SentenceTransformer
                # Set longer timeout for model download
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
                self._model = SentenceTransformer(self.model_name, device='cpu')
                logger.info(f"Embedding model loaded: {self.model_name}")
            except ImportError:
                logger.warning("[WARNING] sentence-transformers not installed. Using simple embeddings.")
                logger.warning("   Install with: pip install sentence-transformers")
                # Fallback to simple TF-IDF based embeddings
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    self._model = TfidfVectorizer(max_features=self.dimension)
                except ImportError:
                    logger.error("sklearn not available. Embeddings disabled.")
                    self._model = None
            except Exception as e:
                logger.warning(f"[WARNING] Failed to load model: {e}")
                logger.warning("   Falling back to simple TF-IDF embeddings.")
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    self._model = TfidfVectorizer(max_features=self.dimension)
                except ImportError:
                    logger.error("sklearn not available. Embeddings disabled.")
                    self._model = None

        return self._model

    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding vector for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            return np.zeros(self.dimension)

        model = self._get_model()

        if model is None:
            logger.warning("No embedding model available, using zero vector")
            return np.zeros(self.dimension)

        try:
            # Try sentence-transformers first
            if hasattr(model, 'encode'):
                embedding = model.encode(text, convert_to_numpy=True)
                return embedding
            else:
                # Fallback to TF-IDF (needs fit first)
                if not hasattr(model, 'vocabulary_'):
                    # Fit on the input text
                    model.fit([text])

                embedding = model.transform([text]).toarray()[0]

                # Pad or truncate to target dimension
                if len(embedding) < self.dimension:
                    embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
                else:
                    embedding = embedding[:self.dimension]

                return embedding
        except Exception as e:
            logger.error(f"[ERROR] Embedding creation failed: {e}")
            # Return zero vector as fallback
            return np.zeros(self.dimension)

    def batch_create_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """
        Create embeddings for multiple texts efficiently

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = self._get_model()

        if model is None:
            return [np.zeros(self.dimension) for _ in texts]

        try:
            if hasattr(model, 'encode'):
                # Batch encoding with sentence-transformers
                embeddings = model.encode(texts, convert_to_numpy=True)
                return [emb for emb in embeddings]
            else:
                # Batch with TF-IDF
                if not hasattr(model, 'vocabulary_'):
                    model.fit(texts)

                embeddings = model.transform(texts).toarray()

                # Pad or truncate each embedding
                result = []
                for emb in embeddings:
                    if len(emb) < self.dimension:
                        emb = np.pad(emb, (0, self.dimension - len(emb)))
                    else:
                        emb = emb[:self.dimension]
                    result.append(emb)

                return result
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [np.zeros(self.dimension) for _ in texts]

    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            # Fall back to manual calculation
            try:
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)

                if norm1 == 0 or norm2 == 0:
                    return 0.0

                return dot_product / (norm1 * norm2)
            except Exception as e2:
                logger.error(f"Manual similarity calculation also failed: {e2}")
                return 0.0


# Singleton instance
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    """Get or create the EmbeddingManager singleton"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
