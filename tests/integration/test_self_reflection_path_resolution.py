"""Regression tests for self-reflection file path resolution."""

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.learning.self_reflection import SelfReflectionSystem


def test_basename_lookup_resolves_models_file():
    system = SelfReflectionSystem(str(project_root))
    code_file = system.read_file("fast_model.py")

    assert code_file is not None
    assert code_file.path.replace("\\", "/") == "models/fast_model.py"


def test_basename_lookup_resolves_brain_file():
    system = SelfReflectionSystem(str(project_root))
    code_file = system.read_file("model_router.py")

    assert code_file is not None
    assert code_file.path.replace("\\", "/") == "brain/model_router.py"
