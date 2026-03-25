"""Operator workflow orchestration for multi-step repo workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ai.integration.build_runner import BuildRunner
from ai.integration.git_manager import GitManager


@dataclass
class WorkflowStepResult:
    name: str
    success: bool
    summary: str
    details: str = ""


@dataclass
class WorkflowResult:
    success: bool
    steps: List[WorkflowStepResult]

    def render(self) -> str:
        lines: List[str] = ["Operator workflow result:"]
        for step in self.steps:
            status = "PASS" if step.success else "FAIL"
            lines.append(f"- [{status}] {step.name}: {step.summary}")
            if step.details:
                lines.extend(f"  {line}" for line in step.details.splitlines()[:25])
        lines.append(f"Overall: {'PASS' if self.success else 'FAIL'}")
        return "\n".join(lines)


@dataclass
class ControlledWriteResult:
    success: bool
    summary: str
    checkpoint_ref: str
    rollback_attempted: bool
    rollback_success: bool
    details: str = ""

    def render(self) -> str:
        lines = [
            "Controlled write workflow result:",
            f"- success: {self.success}",
            f"- summary: {self.summary}",
            f"- checkpoint: {self.checkpoint_ref}",
            f"- rollback_attempted: {self.rollback_attempted}",
            f"- rollback_success: {self.rollback_success}",
        ]
        if self.details:
            lines.append("- details:")
            lines.extend(f"  {line}" for line in self.details.splitlines()[:30])
        return "\n".join(lines)


class OperatorWorkflowOrchestrator:
    def __init__(self, git_manager: GitManager, build_runner: BuildRunner) -> None:
        self.git = git_manager
        self.build = build_runner

    def run_repo_health_workflow(self, include_tests: bool = False) -> WorkflowResult:
        steps: List[WorkflowStepResult] = []

        branch = self.git.current_branch()
        steps.append(
            WorkflowStepResult(
                name="current_branch",
                success=branch.success,
                summary=branch.output or branch.error or "unknown",
            )
        )

        status = self.git.status_short()
        status_summary = "clean working tree" if status.success and not status.output else (status.output or status.error)
        steps.append(
            WorkflowStepResult(
                name="working_tree_status",
                success=status.success,
                summary=status_summary,
            )
        )

        build = self.build.run_python_build()
        steps.append(
            WorkflowStepResult(
                name="python_build_check",
                success=build.success,
                summary="build checks passed" if build.success else f"build failed (exit={build.exit_code})",
                details=build.error or build.output,
            )
        )

        if include_tests:
            tests = self.build.run_python_tests()
            steps.append(
                WorkflowStepResult(
                    name="python_tests",
                    success=tests.success,
                    summary="tests passed" if tests.success else f"tests failed (exit={tests.exit_code})",
                    details=tests.error or tests.output,
                )
            )

        overall = all(s.success for s in steps)
        return WorkflowResult(success=overall, steps=steps)

    def run_controlled_commit_workflow(self, commit_message: str) -> ControlledWriteResult:
        has_changes = self.git.has_changes()
        if not has_changes.success:
            return ControlledWriteResult(
                success=False,
                summary="unable to inspect repository changes",
                checkpoint_ref="",
                rollback_attempted=False,
                rollback_success=False,
                details=has_changes.error or has_changes.output,
            )
        if not has_changes.output.strip():
            return ControlledWriteResult(
                success=False,
                summary="no changes to commit",
                checkpoint_ref="",
                rollback_attempted=False,
                rollback_success=False,
                details="working tree is clean",
            )

        checkpoint = self.git.create_checkpoint("alice-operator-commit")
        checkpoint_ref = "stash@{0}"
        if not checkpoint.success:
            return ControlledWriteResult(
                success=False,
                summary="failed to create checkpoint",
                checkpoint_ref=checkpoint_ref,
                rollback_attempted=False,
                rollback_success=False,
                details=checkpoint.error or checkpoint.output,
            )

        stage = self.git.stage_all()
        if not stage.success:
            rollback = self.git.rollback_from_checkpoint(checkpoint_ref)
            return ControlledWriteResult(
                success=False,
                summary="failed to stage changes",
                checkpoint_ref=checkpoint_ref,
                rollback_attempted=True,
                rollback_success=rollback.success,
                details=(stage.error or stage.output) + "\n" + (rollback.error or rollback.output),
            )

        commit = self.git.commit(commit_message)
        if not commit.success:
            rollback = self.git.rollback_from_checkpoint(checkpoint_ref)
            return ControlledWriteResult(
                success=False,
                summary=f"commit failed (exit={commit.exit_code})",
                checkpoint_ref=checkpoint_ref,
                rollback_attempted=True,
                rollback_success=rollback.success,
                details=(commit.error or commit.output) + "\n" + (rollback.error or rollback.output),
            )

        drop = self.git.drop_checkpoint(checkpoint_ref)
        return ControlledWriteResult(
            success=True,
            summary="commit created successfully",
            checkpoint_ref=checkpoint_ref,
            rollback_attempted=False,
            rollback_success=drop.success,
            details=commit.output or commit.error,
        )
