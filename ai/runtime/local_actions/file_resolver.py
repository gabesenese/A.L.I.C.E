from __future__ import annotations

from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Dict, List


class FileResolver:
    def resolve_target(self, target: str, files: List[str]) -> Dict[str, Any]:
        target = str(target or "").strip().strip("`'\"").replace("\\", "/").lstrip("./")
        if not target:
            return {"file_exists": False, "resolved": "", "close_matches": files[:8]}
        exact = [f for f in files if f.lower() == target.lower()]
        if exact:
            return {"file_exists": True, "resolved": exact[0], "close_matches": []}
        suffix = [f for f in files if f.lower().endswith("/" + target.lower())]
        if suffix:
            return {"file_exists": True, "resolved": suffix[0], "close_matches": []}
        basename_exact = [f for f in files if Path(f).name.lower() == Path(target).name.lower()]
        if len(basename_exact) == 1:
            return {"file_exists": True, "resolved": basename_exact[0], "close_matches": []}
        if len(basename_exact) > 1:
            ranked_basename = sorted(
                basename_exact,
                key=lambda c: SequenceMatcher(None, target.lower(), c.lower()).ratio(),
                reverse=True,
            )
            return {"file_exists": False, "resolved": "", "close_matches": ranked_basename[:8], "ambiguous": True}
        basename = target.split("/")[-1].lower()
        contains = [f for f in files if basename and basename in Path(f).name.lower()]
        fuzzy_basename = get_close_matches(basename, [Path(f).name.lower() for f in files], n=8, cutoff=0.55)
        fuzzy_files: List[str] = []
        for item in files:
            file_name = Path(item).name.lower()
            if file_name in fuzzy_basename and item not in fuzzy_files:
                fuzzy_files.append(item)
        ranked = sorted(
            set(contains + fuzzy_files),
            key=lambda c: SequenceMatcher(None, target.lower(), c.lower()).ratio(),
            reverse=True,
        )
        return {"file_exists": False, "resolved": "", "close_matches": ranked[:8]}

