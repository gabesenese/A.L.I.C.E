from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime_modes import RuntimeModeConfig


def _python_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part.startswith(".") for part in p.parts):
            continue
        if "__pycache__" in p.parts:
            continue
        yield p


def _line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        return 0


def _imports(path: Path):
    items = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("import "):
                items.append(line.split()[1].split(".")[0])
            elif line.startswith("from "):
                items.append(line.split()[1].split(".")[0])
    except Exception:
        pass
    return items


def _top_level_imports(path: Path):
    out = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                out.append(s)
    except Exception:
        pass
    return out


def _warn_threshold(path: Path, threshold: int) -> str:
    lines = _line_count(path)
    return f"WARNING: {path.as_posix()} is {lines} lines (>{threshold})" if lines > threshold else ""


def main() -> None:
    root = Path.cwd()
    files = list(_python_files(root))
    sizes = [(str(p).replace("\\", "/"), _line_count(p)) for p in files]
    sizes.sort(key=lambda x: x[1], reverse=True)
    imports = Counter()
    for p in files:
        imports.update(_imports(p))

    mode = "minimal"
    cfg = RuntimeModeConfig.for_mode(mode)
    enabled = [k.replace("enable_", "") for k, v in vars(cfg).items() if k.startswith("enable_") and v]
    skipped = [k.replace("enable_", "") for k, v in vars(cfg).items() if k.startswith("enable_") and not v]
    print(f"Runtime mode: {mode}")
    print(f"Modules initialized (by config): {', '.join(sorted(enabled))}")
    print(f"Optional systems skipped: {', '.join(sorted(skipped))}")
    print("Background loops started: proactive/autonomous/continuous are mode-gated.")

    print("\nTop 20 Largest Python Files")
    for path, lines in sizes[:20]:
        print(f"- {path}: {lines} lines")

    print("\nTop 10 Most Imported Modules")
    for name, count in imports.most_common(10):
        print(f"- {name}: {count}")

    print("\nSuspected God Files (>800 lines)")
    for path, lines in sizes:
        if lines > 800:
            print(f"- {path}: {lines}")

    app_main = root / "app" / "main.py"
    factory = root / "ai" / "runtime" / "alice_contract_factory.py"
    pipeline = root / "ai" / "runtime" / "contract_pipeline.py"
    print("\napp/main.py imports:")
    for line in _top_level_imports(app_main)[:80]:
        print(f"- {line}")
    for warning in (
        _warn_threshold(app_main, 800),
        _warn_threshold(factory, 500),
        _warn_threshold(pipeline, 500),
    ):
        if warning:
            print(warning)

    pyproject = root / "pyproject.toml"
    print("\npytest testpaths:")
    try:
        text = pyproject.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"testpaths\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
        if m:
            print(m.group(0))
    except Exception:
        pass


if __name__ == "__main__":
    main()
