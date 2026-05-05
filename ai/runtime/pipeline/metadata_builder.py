from __future__ import annotations

from typing import Any, Dict


class PipelineMetadataBuilder:
    @staticmethod
    def build(base: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        out = dict(base or {})
        out.update({k: v for k, v in extra.items()})
        return out

