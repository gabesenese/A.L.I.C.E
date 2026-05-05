from __future__ import annotations

from collections import Counter
from pathlib import Path

from ai.runtime.runtime_modes import get_runtime_mode, list_runtime_modes


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


def main() -> None:
    root = Path.cwd()
    files = list(_python_files(root))
    sizes = [(str(p).replace("\\", "/"), _line_count(p)) for p in files]
    sizes.sort(key=lambda x: x[1], reverse=True)
    imports = Counter()
    for p in files:
        imports.update(_imports(p))

    print("Runtime Modes")
    for mode_name in list_runtime_modes():
        mode = get_runtime_mode(mode_name)
        print(f"- {mode.name}: {len(mode.enabled_groups)} groups")

    print("\nTop 10 Largest Python Files")
    for path, lines in sizes[:10]:
        print(f"- {path}: {lines} lines")

    print("\nTop 10 Most Imported Modules")
    for name, count in imports.most_common(10):
        print(f"- {name}: {count}")

    print("\nSuspected God Files (>800 lines)")
    for path, lines in sizes:
        if lines > 800:
            print(f"- {path}: {lines}")


if __name__ == "__main__":
    main()

