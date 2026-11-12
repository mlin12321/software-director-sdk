import subprocess
from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    AgentRef,
    Boundary,
    Escalation,
    EscalationHandoff,
    EscalationKind,
    NodeHealth,
    OrgNodeInput,
)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository with one commit on ``main`` and a tracked file."""
    path = tmp_path / "repo"
    path.mkdir()
    run_git(path, "init", "--quiet", "--initial-branch=main")
    run_git(path, "config", "user.name", "Test")
    run_git(path, "config", "user.email", "test@example.invalid")
    (path / "README.md").write_text("hello\n")
    run_git(path, "add", "README.md")
    run_git(path, "commit", "--quiet", "-m", "init")
    return path


def make_node(
    node_id: str,
    parent_id: str | None,
    *,
    health: NodeHealth = "converging",
    path: str | None = None,
    escalations: list[Escalation] | None = None,
) -> OrgNodeInput:
    return OrgNodeInput(
        id=node_id,
        parent_id=parent_id,
        title=node_id,
        boundary=Boundary(path=node_id if path is None else path),
        isolation="worktree",
        worktree_path=".",
        branch="main",
        agents=[AgentRef(role="lead_ic", status="idle")],
        health=health,
        escalations=escalations or [],
    )


def make_escalation(
    escalation_id: str,
    *,
    kind: EscalationKind = "design_decision",
    raised_at: str = "2026-11-10T00:00:00Z",
    passed_up_by: tuple[str, ...] = (),
    held_by: str | None = None,
) -> Escalation:
    """An escalation that ``passed_up_by`` (nearest node first) each passed up.

    Leave ``held_by`` unset when that is every node above: it then waits on
    the human.
    """
    return Escalation(
        id=escalation_id,
        kind=kind,
        title=escalation_id,
        raised_at=raised_at,
        held_by=held_by,
        handoffs=[
            EscalationHandoff(
                node_id=node_id, reason=f"{node_id} could not", at=raised_at
            )
            for node_id in passed_up_by
        ],
    )


def small_tree() -> list[OrgNodeInput]:
    """root -> (a -> (a1, a2), b)."""
    return [
        make_node("root", None, path=""),
        make_node("a", "root", health="collision", path="svc/a"),
        make_node("b", "root", health="done", path="svc/b"),
        make_node("a1", "a", health="done", path="svc/a/one"),
        make_node("a2", "a", health="needs_attention", path="svc/a/two"),
    ]
