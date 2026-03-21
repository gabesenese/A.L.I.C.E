"""Simple causal inference heuristics for debugging and explanation queries."""

from __future__ import annotations

from typing import Dict, List


class CausalInferenceEngine:
    def infer(self, text: str) -> Dict[str, List[str]]:
        query = str(text or "").lower()
        causes: List[str] = []
        checks: List[str] = []

        if "timeout" in query:
            causes.extend(["slow dependency", "network instability", "resource contention"])
            checks.extend(["inspect latency metrics", "retry with shorter dependency chain"])
        if "import" in query and "error" in query:
            causes.extend(["missing package", "circular import", "wrong virtual environment"])
            checks.extend(["verify package install", "inspect module path", "check active environment"])
        if "memory" in query and ("high" in query or "leak" in query):
            causes.extend(["unbounded cache growth", "object retention", "large payload accumulation"])
            checks.extend(["profile object counts", "add cache eviction", "sample heap over time"])

        if not causes:
            causes = ["insufficient context", "multiple interacting factors"]
            checks = ["gather logs", "compare with last known good state"]

        return {"likely_causes": causes[:4], "recommended_checks": checks[:4]}
