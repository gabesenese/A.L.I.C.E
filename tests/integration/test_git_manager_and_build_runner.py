from ai.integration.build_runner import BuildRunner
from ai.integration.git_manager import GitManager


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_git_manager_status_and_branch_with_mock(monkeypatch):
    manager = GitManager(repo_root=".")

    def fake_run(args, cwd=None, capture_output=True, text=True, timeout=15):
        if args[:3] == ["git", "status", "--short"]:
            return _FakeCompleted(
                0, " M app/main.py\n?? ai/integration/git_manager.py\n", ""
            )
        if args[:4] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return _FakeCompleted(0, "main\n", "")
        return _FakeCompleted(1, "", "unexpected")

    monkeypatch.setattr("subprocess.run", fake_run)

    status = manager.status_short()
    assert status.success is True
    assert "app/main.py" in status.output

    branch = manager.current_branch()
    assert branch.success is True
    assert branch.output == "main"


def test_git_manager_handles_failures(monkeypatch):
    manager = GitManager(repo_root=".")

    def fake_run(args, cwd=None, capture_output=True, text=True, timeout=15):
        return _FakeCompleted(128, "", "fatal: not a git repository")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = manager.status_short()
    assert result.success is False
    assert "fatal" in result.error


def test_build_runner_executes_python_commands(monkeypatch):
    runner = BuildRunner(project_root=".")

    def fake_run(
        command, cwd=None, capture_output=True, text=True, timeout=120, shell=True
    ):
        if "pytest" in command:
            return _FakeCompleted(0, "8 passed in 0.50s\n", "")
        if "compileall" in command:
            return _FakeCompleted(0, "", "")
        return _FakeCompleted(1, "", "unknown command")

    monkeypatch.setattr("subprocess.run", fake_run)

    tests = runner.run_python_tests()
    assert tests.success is True
    assert "passed" in tests.output

    build = runner.run_python_build()
    assert build.success is True


def test_build_runner_failure_path(monkeypatch):
    runner = BuildRunner(project_root=".")

    def fake_run(
        command, cwd=None, capture_output=True, text=True, timeout=120, shell=True
    ):
        return _FakeCompleted(2, "", "build failed")

    monkeypatch.setattr("subprocess.run", fake_run)

    build = runner.run_python_build()
    assert build.success is False
    assert build.exit_code == 2
    assert "failed" in build.error
