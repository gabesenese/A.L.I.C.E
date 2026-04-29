"""Multi-LLM router with classification, complexity scoring, logging, and fallback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from brain.task_classifier import classify_task
from complexity import score_prompt
from models import CodingModel, FastModel, ReasoningModel
from models.base import Model


class ModelRouter:
    """Select and execute model roles based on task type + complexity."""

    def __init__(
        self, usage_log: str = "logs/model_usage.log", confidence_floor: float = 0.6
    ) -> None:
        self.models: Dict[str, Model] = {
            "fast": FastModel(),
            "reasoning": ReasoningModel(),
            "coding": CodingModel(),
        }
        self.fallback_key = "reasoning"
        self.confidence_floor = max(0.0, min(1.0, float(confidence_floor)))
        self.require_all_roles = os.getenv(
            "ALICE_MULTI_LLM_REQUIRE_ALL", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self.usage_log = Path(usage_log)
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        self.usage_log.touch(exist_ok=True)
        self.last_route: Dict[str, Any] = {}
        self._recent_roles: list[str] = []
        self._role_health: Dict[str, bool] = {
            "fast": True,
            "reasoning": True,
            "coding": True,
        }
        self._health_error: str = ""
        self.refresh_role_health()

    def describe_models(self) -> Dict[str, str]:
        return {
            role: str(getattr(model, "model_name", role))
            for role, model in self.models.items()
        }

    def refresh_role_health(self) -> Dict[str, bool]:
        """Detect whether each routed model appears in local Ollama tags."""
        if os.getenv("ALICE_MULTI_LLM_MOCK", "0") == "1":
            self._role_health = {role: True for role in self.models.keys()}
            self._health_error = ""
            return dict(self._role_health)

        available_models: set[str] = set()
        self._health_error = ""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            resp.raise_for_status()
            payload = resp.json() if isinstance(resp.json(), dict) else {}
            for item in list(payload.get("models") or []):
                name = str((item or {}).get("name") or "").strip()
                if name:
                    available_models.add(name)
        except Exception as exc:
            self._health_error = str(exc)

        role_health: Dict[str, bool] = {}
        for role, model_name in self.describe_models().items():
            role_health[role] = self._is_model_available(model_name, available_models)
        self._role_health = role_health
        return dict(self._role_health)

    def _is_model_available(self, model_name: str, available_models: set[str]) -> bool:
        if not model_name:
            return False
        if model_name in available_models:
            return True
        base = model_name.split(":", 1)[0].strip().lower()
        for name in available_models:
            low = str(name).strip().lower()
            if low == model_name.lower():
                return True
            if low.startswith(base + ":"):
                return True
        return False

    def all_roles_ready(self) -> bool:
        return all(bool(v) for v in (self._role_health or {}).values())

    def runtime_status(self) -> Dict[str, Any]:
        counts = {
            role: self._recent_roles.count(role)
            for role in ("fast", "reasoning", "coding")
        }
        total = max(1, sum(counts.values()))
        shares = {k: round(v / total, 3) for k, v in counts.items()}
        return {
            "require_all_roles": bool(self.require_all_roles),
            "all_roles_ready": bool(self.all_roles_ready()),
            "health_error": str(self._health_error or ""),
            "role_health": dict(self._role_health or {}),
            "recent_role_counts": counts,
            "recent_role_shares": shares,
            "models": self.describe_models(),
        }

    def route(self, request: str, context: Optional[Dict[str, Any]] = None) -> str:
        task = classify_task(request)
        complexity = score_prompt(request)
        ctx = dict(context or {})
        intent_hint = str(ctx.get("intent") or "").strip().lower()

        if any(
            token in intent_hint
            for token in (
                "code",
                "coding",
                "refactor",
                "debug",
                "bug",
                "pytest",
                "unit_test",
            )
        ):
            return "coding"

        if task.task_type == "coding":
            return "coding"
        if (
            task.task_type in {"planning", "reasoning"}
            or task.multi_step
            or complexity > 7
        ):
            selected = "reasoning"
        elif task.task_type == "simple" and complexity <= 3:
            selected = "fast"
        else:
            selected = "reasoning"

        return self._rebalance_role(
            selected=selected,
            task_type=task.task_type,
            complexity=complexity,
        )

    def _rebalance_role(self, *, selected: str, task_type: str, complexity: int) -> str:
        """Prevent persistent overuse of one role for low-complexity traffic."""
        if selected != "reasoning":
            return selected
        if task_type != "simple" or int(complexity) > 4:
            return selected

        history = [
            r for r in self._recent_roles if r in {"fast", "reasoning", "coding"}
        ]
        if len(history) < 8:
            return selected

        reasoning_share = history.count("reasoning") / max(1, len(history))
        fast_share = history.count("fast") / max(1, len(history))
        if reasoning_share > 0.65 and fast_share < 0.30:
            return "fast"
        return selected

    def _record_recent_role(self, role: str) -> None:
        if not role:
            return
        self._recent_roles.append(str(role))
        self._recent_roles = self._recent_roles[-60:]

    def generate(
        self, request: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        context = dict(context or {})
        self.refresh_role_health()

        if self.require_all_roles and not self.all_roles_ready():
            missing_roles = [r for r, ok in (self._role_health or {}).items() if not ok]
            self.last_route = {
                "role": "",
                "model": "",
                "task_type": classify_task(request).task_type,
                "complexity": int(score_prompt(request)),
                "success": False,
                "reason": "roles_not_ready",
                "missing_roles": list(missing_roles),
            }
            return {
                "response": "Primary generation route is temporarily unavailable.",
                "confidence": 0.0,
                "reasoning_used": True,
                "model": "router_unavailable",
                "execution_time_ms": 0.0,
            }

        selected = self.route(request, context=context)
        task = classify_task(request)
        complexity = score_prompt(request)

        if not bool((self._role_health or {}).get(selected, True)):
            selected = self._fallback_ready_role(preferred=selected)

        try:
            out = self.models[selected].generate(request, context=context)
            if (
                float(out.get("confidence", 0.0)) < self.confidence_floor
                and selected != "reasoning"
            ):
                out = self.models["reasoning"].generate(request, context=context)
                selected = "reasoning"
            model_name = str(
                out.get("model")
                or getattr(self.models[selected], "model_name", selected)
            )
            self.last_route = {
                "role": selected,
                "model": model_name,
                "task_type": task.task_type,
                "complexity": int(complexity),
                "success": True,
            }
            self._record_recent_role(selected)
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
                model_name = str(
                    out.get("model")
                    or getattr(self.models[fallback_key], "model_name", fallback_key)
                )
                self.last_route = {
                    "role": fallback_key,
                    "model": model_name,
                    "task_type": task.task_type,
                    "complexity": int(complexity),
                    "success": True,
                }
                self._record_recent_role(fallback_key)
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
                    "model": str(
                        getattr(
                            self.models.get(fallback_key), "model_name", fallback_key
                        )
                    ),
                    "task_type": task.task_type,
                    "complexity": int(complexity),
                    "success": False,
                }
                self._log_usage(
                    fallback_key,
                    str(
                        getattr(
                            self.models.get(fallback_key), "model_name", fallback_key
                        )
                    ),
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

    def _fallback_ready_role(self, *, preferred: str) -> str:
        order = [preferred, "reasoning", "fast", "coding"]
        for role in order:
            if role in self.models and bool((self._role_health or {}).get(role, False)):
                return role
        return self.fallback_key

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
