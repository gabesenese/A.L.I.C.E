from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings


class VectorMemory:
    def __init__(self, collection_name: str = "alice_memory") -> None:
        settings = get_settings()
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def store(
        self,
        trace_id: str,
        text: str,
        metadata: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "documents": [text],
            "metadatas": [metadata],
            "ids": [trace_id],
        }
        if embedding:
            payload["embeddings"] = [embedding]
        self.collection.add(**payload)

    async def query(self, text: str, n_results: int = 5) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        return [
            {
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index],
            }
            for index in range(len(ids))
        ]
