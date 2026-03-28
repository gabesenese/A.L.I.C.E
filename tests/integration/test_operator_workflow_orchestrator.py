from ai.integration.operator_workflow import OperatorWorkflowOrchestrator


class _GitOK:
    def current_branch(self):
        class R:
            success = True
            output = "main"
            error = ""
        return R()

    def status_short(self):
        class R:
            success = True
            output = ""
            error = ""
        return R()


class _BuildOK:
    def run_python_build(self):
        class R:
            success = True
            output = ""
            error = ""
            exit_code = 0
        return R()

    def run_python_tests(self):
        class R:
            success = True
            output = "12 passed"
            error = ""
            exit_code = 0
        return R()


class _BuildFail:
    def run_python_build(self):
        class R:
            success = False
            output = ""
            error = "compile failed"
            exit_code = 1
        return R()


class _BuildFlaky:
    def __init__(self):
        self.calls = 0

    def run_python_build(self):
        self.calls += 1
        class R:
            success = True
            output = ""
            error = ""
            exit_code = 0
        class F:
            success = False
            output = ""
            error = "transient"
            exit_code = 1
        return F() if self.calls == 1 else R()

    def run_python_tests(self):
        class R:
            success = True
            output = "ok"
            error = ""
            exit_code = 0
        return R()


class _GitWriteOK(_GitOK):
    def has_changes(self):
        class R:
            success = True
            output = " M app/main.py"
            error = ""
        return R()

    def create_checkpoint(self, label):
        class R:
            success = True
            output = "Saved working directory and index state"
            error = ""
        return R()

    def stage_all(self):
        class R:
            success = True
            output = ""
            error = ""
        return R()

    def commit(self, message):
        class R:
            success = True
            output = f"[main 1234567] {message}"
            error = ""
            exit_code = 0
        return R()

    def drop_checkpoint(self, checkpoint_ref="stash@{0}"):
        class R:
            success = True
            output = "Dropped"
            error = ""
        return R()


class _GitWriteCommitFail(_GitWriteOK):
    def commit(self, message):
        class R:
            success = False
            output = ""
            error = "nothing to commit"
            exit_code = 1
        return R()

    def rollback_from_checkpoint(self, checkpoint_ref="stash@{0}"):
        class R:
            success = True
            output = "Applied stash"
            error = ""
        return R()

    def run_python_tests(self):
        class R:
            success = False
            output = ""
            error = "tests failed"
            exit_code = 2
        return R()


def test_operator_workflow_success_with_tests():
    wf = OperatorWorkflowOrchestrator(_GitOK(), _BuildOK()).run_repo_health_workflow(include_tests=True)
    assert wf.success is True
    rendered = wf.render()
    assert "python_tests" in rendered
    assert "Overall: PASS" in rendered


def test_operator_workflow_failure_on_build_step():
    wf = OperatorWorkflowOrchestrator(_GitOK(), _BuildFail()).run_repo_health_workflow(include_tests=False)
    assert wf.success is False
    rendered = wf.render()
    assert "python_build_check" in rendered
    assert "Overall: FAIL" in rendered


def test_controlled_commit_workflow_success():
    wf = OperatorWorkflowOrchestrator(_GitWriteOK(), _BuildOK()).run_controlled_commit_workflow("checkpoint")
    assert wf.success is True
    assert wf.rollback_attempted is False
    assert "commit created successfully" in wf.summary


def test_controlled_commit_workflow_rolls_back_on_commit_failure():
    wf = OperatorWorkflowOrchestrator(_GitWriteCommitFail(), _BuildOK()).run_controlled_commit_workflow("checkpoint")
    assert wf.success is False
    assert wf.rollback_attempted is True
    assert wf.rollback_success is True


def test_repo_health_workflow_recovers_from_transient_build_failure():
    flaky = _BuildFlaky()
    wf = OperatorWorkflowOrchestrator(_GitOK(), flaky).run_repo_health_workflow(include_tests=False)
    assert wf.success is True
    assert flaky.calls >= 2
