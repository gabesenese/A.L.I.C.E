"""Shared model interface and compact Ollama adapter."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import requests


class Model(Protocol):
    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...


@dataclass(slots=True)
class ModelOutput:
    response: str
    confidence: float
    reasoning_used: bool
    model: str
    execution_time_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "confidence": max(0.0, min(1.0, float(self.confidence))),
            "reasoning_used": bool(self.reasoning_used),
            "model": self.model,
            "execution_time_ms": round(float(self.execution_time_ms), 2),
        }


@dataclass(slots=True)
class OllamaRoleModel:
    model_name: str
    system_prompt: str
    reasoning_used: bool
    temperature: float = 0.3
    timeout_seconds: int = 45
    base_url: str = "http://localhost:11434"

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = time.perf_counter()
        text = str(prompt or "").strip()
        if not text:
            return ModelOutput("", 0.0, self.reasoning_used, self.model_name, 0.0).as_dict()

        if os.getenv("ALICE_MULTI_LLM_MOCK", "0") == "1":
            elapsed = (time.perf_counter() - start) * 1000.0
            return ModelOutput(
                response=f"[mock:{self.model_name}] {text[:120]}",
                confidence=0.75,
                reasoning_used=self.reasoning_used,
                model=self.model_name,
                execution_time_ms=elapsed,
            ).as_dict()

        ctx = context or {}
        context_blob = "\n".join(f"{k}: {v}" for k, v in ctx.items() if v is not None)
        user_prompt = text if not context_blob else f"Context:\n{context_blob}\n\nTask:\n{text}"

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": float(self.temperature),
                    "num_ctx": 4096,
                },
            },
            timeout=max(5, int(self.timeout_seconds)),
        )
        resp.raise_for_status()
        answer = (resp.json().get("message") or {}).get("content", "").strip()

        elapsed = (time.perf_counter() - start) * 1000.0
        conf = 0.9 if answer else 0.2
        return ModelOutput(answer, conf, self.reasoning_used, self.model_name, elapsed).as_dict()
