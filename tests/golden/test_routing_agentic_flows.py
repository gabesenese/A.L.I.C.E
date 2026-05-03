from pathlib import Path

from ai.runtime.local_action_executor import LocalActionExecutor
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


class _NoReflectionAlice(_FakeAlice):
    class _EmptyReflection:
        def list_codebase(self):
            return []

    def __init__(self):
        super().__init__()
        self.self_reflection = self._EmptyReflection()


def test_deep_analysis_detects_structure_and_risks(tmp_path: Path):
    sample = tmp_path / "sample_target.py"
    sample.write_text(
        "import os\n"
        "import re\n\n"
        "# TODO improve route logic\n"
        "class Sample:\n"
        "    pass\n\n"
        "def alpha():\n"
        "    text = 'please rephrase'\n"
        "    if 'route' in text:\n"
        "        return True\n"
        "    return False\n\n"
        "def beta():\n"
        "    return re.search('x', 'xyz')\n",
        encoding="utf-8",
    )

    alice = _NoReflectionAlice()
    alice.PROJECT_ROOT = str(tmp_path)
    exe = LocalActionExecutor(alice)
    out = exe.execute(action="code:analyze_file", query="analyze sample_target.py", context={"target_file": "sample_target.py"})

    assert out["success"] is True
    lx = dict(out.get("local_execution") or {})
    analysis = dict(lx.get("analysis") or {})
    assert int(analysis.get("line_count", 0)) > 0
    assert int(analysis.get("import_count", 0)) >= 2
    assert int(analysis.get("class_count", 0)) >= 1
    assert int(analysis.get("function_count", 0)) >= 2
    assert int(analysis.get("todo_count", 0)) >= 1
    assert int(analysis.get("fallback_phrase_count", 0)) >= 1
    assert "Top risks" in out.get("response", "")


def test_operator_continuation_capability_then_file_target():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))

    first = pipeline.run_turn(
        user_input="you have access to Alice's code right?",
        user_id="u1",
        turn_number=10,
    )
    assert first.metadata["route"] == "local"
    assert first.metadata["intent"] == "code:request"
    assert (first.metadata.get("operator_context") or {}).get("awaiting_target") is True

    second = pipeline.run_turn(
        user_input="what about app/main.py?",
        user_id="u1",
        turn_number=11,
    )
    assert second.metadata["route"] == "local"
    assert second.metadata["intent"] == "code:analyze_file"
    op = dict(second.metadata.get("operator_context") or {})
    assert op.get("continuation_from_previous_turn") is True
