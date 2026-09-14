"""No plugin may call a method its collaborator does not have.

The memory plugin called three: `add_episodic_memory`, `get_recent_memories`,
`search_memories`. None exist on MemorySystem. Each raised AttributeError into a
blanket `except Exception` and came back as a polite failure string — "Failed to
store preference: 'MemorySystem' object has no attribute 'add_episodic_memory'"
— so "remember that I prefer coffee" had never once worked, and nothing in the
tests, the logs, or the type checker said so.

That combination is what makes the defect class invisible: a dynamic call, a
catch-all, and a message that reads like a runtime problem rather than a missing
method. pyright does not flag it because the collaborator is typed `Optional` and
constructed at runtime; a unit test does not catch it because the natural thing
to hand a plugin is a mock, and a mock has every method you name.

So this asserts it structurally: parse each plugin for `self.<x>.<y>(...)`,
build the plugin for real, and check `<y>` exists on whatever `<x>` turned out
to be.
"""

import ast
import importlib
import os
import pathlib

import pytest

PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[2] / "ai" / "plugins"

# A collaborator of one of these types is data, not a subsystem — `self.tags.get`
# is a dict call and has nothing to do with this.
SCALARS = (str, int, float, bool, list, dict, set, tuple, type(None))


def _plugin_modules():
    for path in sorted(PLUGIN_DIR.glob("*.py")):
        if path.name != "__init__.py":
            yield path


def _collaborator_calls(path: pathlib.Path):
    """{class name: {attribute: {method: line}}} for every self.<attr>.<method>() call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        calls: dict = {}
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Attribute)
                and isinstance(sub.func.value.value, ast.Name)
                and sub.func.value.value.id == "self"
            ):
                calls.setdefault(sub.func.value.attr, {}).setdefault(sub.func.attr, sub.lineno)
        if calls:
            out[node.name] = calls
    return out


@pytest.mark.parametrize("path", list(_plugin_modules()), ids=lambda p: p.stem)
def test_a_plugin_never_calls_a_method_its_collaborator_lacks(path, monkeypatch, tmp_path):
    monkeypatch.setenv("ALICE_MULTI_LLM_MOCK", "1")
    monkeypatch.chdir(os.getcwd())

    module_name = "ai.plugins." + path.stem
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # an optional integration dependency is not this test's business
        pytest.skip(f"{module_name} does not import here: {type(exc).__name__}")

    ghosts = []
    for class_name, calls in _collaborator_calls(path).items():
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        try:
            instance = cls()
        except Exception:
            continue  # needs constructor arguments; nothing to resolve against
        for attribute, methods in calls.items():
            collaborator = getattr(instance, attribute, None)
            if isinstance(collaborator, SCALARS):
                continue
            for method, lineno in sorted(methods.items()):
                if not hasattr(collaborator, method):
                    ghosts.append(
                        f"{path.name}:{lineno} {class_name}: self.{attribute}.{method}() "
                        f"— {type(collaborator).__name__} has no {method}"
                    )

    assert not ghosts, "calls to methods that do not exist:\n  " + "\n  ".join(ghosts)
