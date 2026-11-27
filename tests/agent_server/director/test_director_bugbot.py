"""What a team's IC is asked about its own work before it is reviewed."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from openhands.agent_server.director import bugbot
from openhands.agent_server.director.environment.git import GitCheckouts
from openhands.agent_server.director.models import SessionTeam
from openhands.sdk.skills import KeywordTrigger, Skill
from tests.agent_server.director.conftest import run_git


def _listed(repo: Path, base: str) -> str | None:
    team = SimpleNamespace(
        worktree_path=str(repo),
        base_commit=base,
        boundary=SimpleNamespace(path="services/api"),
    )
    return GitCheckouts().changes_listed(cast(SessionTeam, team))


def test_the_write_up_carries_gits_list_of_every_changed_file(repo: Path):
    base = run_git(repo, "rev-parse", "HEAD").strip()
    for path in ("services/api/app.py", "services/api/big.py", "apps/web/page.tsx"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("x = 1\n" * 20_000)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "--quiet", "-m", "work")

    files = _listed(repo, base)

    assert files is not None
    assert "services/api/app.py" in files and "services/api/big.py" in files
    assert "apps/web" not in files and "40000 insertions" in files
    assert files in bugbot.write_up_question(files)


def test_no_change_and_no_such_commit_are_told_apart(repo: Path):
    head = run_git(repo, "rev-parse", "HEAD").strip()

    assert _listed(repo, head) == ""
    assert _listed(repo, "0" * 40) is None


def test_the_instructions_are_the_public_code_review_skill(
    monkeypatch: pytest.MonkeyPatch,
):
    skills = [
        Skill(name="docker", content="Use docker."),
        Skill(
            name="review",
            content="Find real bugs.",
            trigger=KeywordTrigger(keywords=["/codereview"]),
        ),
    ]
    monkeypatch.setattr(bugbot, "load_public_skills", lambda **_: skills)

    assert bugbot.review_instructions() == "Find real bugs."

    monkeypatch.setattr(bugbot, "load_public_skills", lambda **_: skills[:1])

    assert bugbot.review_instructions() is None
