"""Multi-LLM router with classification, complexity scoring, logging, and fallback."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from brain.task_classifier import classify_task
from complexity import score_prompt
from models import CodingModel, FastModel, ReasoningModel
from models.base import Model


class ModelRouter:
    """Select and execute model roles based on task type + complexity."""

    def __init__(self, usage_log: str = "logs/model_usage.log", confidence_floor: float = 0.6) -> None:
        self.models: Dict[str, Model] = {
            "fast": FastModel(),
            "reasoning": ReasoningModel(),
            "coding": CodingModel(),
        }
        self.fallback_key = "reasoning"
        self.confidence_floor = max(0.0, min(1.0, float(confidence_floor)))
        self.usage_log = Path(usage_log)
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        self.usage_log.touch(exist_ok=True)
        self.last_route: Dict[str, Any] = {}

    def describe_models(self) -> Dict[str, str]:
        return {
            role: str(getattr(model, "model_name", role))
            for role, model in self.models.items()
        }

    def route(self, request: str) -> str:
        task = classify_task(request)
        complexity = score_prompt(request)

        if task.task_type == "coding":
            return "coding"
        if task.task_type in {"planning", "reasoning"} or task.multi_step or complexity > 7:
            return "reasoning"
        return "fast"

    def generate(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = dict(context or {})
        selected = self.route(request)
        task = classify_task(request)
        complexity = score_prompt(request)

        try:
            out = self.models[selected].generate(request, context=context)
            if float(out.get("confidence", 0.0)) < self.confidence_floor and selected != "reasoning":
                out = self.models["reasoning"].generate(request, context=context)
                selected = "reasoning"
            model_name = str(out.get("model") or getattr(self.models[selected], "model_name", selected))
            self.last_route = {
                "role": selected,
                "model": model_name,
                "task_type": task.task_type,
                "complexity": int(complexity),
                "success": True,
            }
            self._log_usage(
                selected,
                model_name,
                task.task_type,
                complexity,
                True,
                float(out.get("execution_time_ms", 0.0)),
            )
            return out
        except Exception:
            fallback_key = self.fallback_key
            try:
                out = self.models[fallback_key].generate(request, context=context)
                model_name = str(out.get("model") or getattr(self.models[fallback_key], "model_name", fallback_key))
                self.last_route = {
                    "role": fallback_key,
                    "model": model_name,
                    "task_type": task.task_type,
                    "complexity": int(complexity),
                    "success": True,
                }
                self._log_usage(
                    fallback_key,
                    model_name,
                    task.task_type,
                    complexity,
                    True,
                    float(out.get("execution_time_ms", 0.0)),
                )
                return out
            except Exception as exc:
                self.last_route = {
                    "role": fallback_key,
                    "model": str(getattr(self.models.get(fallback_key), "model_name", fallback_key)),
                    "task_type": task.task_type,
                    "complexity": int(complexity),
                    "success": False,
                }
                self._log_usage(
                    fallback_key,
                    str(getattr(self.models.get(fallback_key), "model_name", fallback_key)),
                    task.task_type,
                    complexity,
                    False,
                    0.0,
                )
                return {
                    "response": f"Model routing failed: {exc}",
                    "confidence": 0.0,
                    "reasoning_used": True,
                    "model": fallback_key,
                    "execution_time_ms": 0.0,
                }

    def _log_usage(
        self,
        model_role: str,
        model_name: str,
        task_type: str,
        complexity: int,
        success: bool,
        latency_ms: float,
    ) -> None:
        row = {
            "ts": time.time(),
            "task_type": task_type,
            "complexity": int(complexity),
            "model_role": model_role,
            "model": model_name,
            "latency_ms": round(float(latency_ms), 2),
            "success": bool(success),
        }
        with self.usage_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
