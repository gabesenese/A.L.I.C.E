from __future__ import annotations

import threading
from typing import Any, Optional

from ai.runtime.boundaries.boundary_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline

from app.config import Settings


class AppContainer:
    """Owns the single ALICE instance shared by the HTTP API and the CLI.

    The instance is built on first use rather than at import time. Construction
    loads models and starts companion threads, so callers that never touch the
    runtime (health checks, schema generation, tooling) should not pay for it.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._alice: Optional[Any] = None
        self._pipeline: Optional[ContractPipeline] = None
        self._lock = threading.RLock()

    @property
    def alice(self) -> Any:
        if self._alice is None:
            with self._lock:
                if self._alice is None:
                    from app.main import ALICE

                    self._alice = ALICE(
                        voice_enabled=self.settings.enable_voice,
                        llm_model=self.settings.ollama_model,
                        runtime_mode=self.settings.runtime_mode,
                    )
        return self._alice

    @property
    def pipeline(self) -> ContractPipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    existing = getattr(self.alice, "contract_pipeline", None)
                    self._pipeline = existing or ContractPipeline(build_runtime_boundaries(self.alice))
        return self._pipeline

    @property
    def nlp(self) -> Any:
        return self.alice.nlp

    @property
    def llm(self) -> Any:
        return self.alice.llm


def build_container(settings: Settings) -> AppContainer:
    return AppContainer(settings=settings)
