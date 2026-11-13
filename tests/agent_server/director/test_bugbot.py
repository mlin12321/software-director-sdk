"""The model's own pass over a team's diff."""

import json
from pathlib import Path

import pytest

from openhands.agent_server.director import bugbot
from openhands.agent_server.director.models import ReviewFinding
from openhands.sdk.skills import KeywordTrigger, Skill
from tests.agent_server.director.conftest import run_git


def test_findings_are_read_out_of_whatever_surrounds_them():
    answer = (
        "<think>hm</think>Here you go:\n```json\n"
        + json.dumps(
            {"findings": [{"path": "a.py", "line": 3, "issue": "Off by one."}]}
        )
        + "\n```"
    )

    assert bugbot.parse_findings(answer) == [
        ReviewFinding(path="a.py", line=3, issue="Off by one.")
    ]


def test_an_empty_list_of_findings_is_an_answer():
    assert bugbot.parse_findings('{"findings": []}') == []


@pytest.mark.parametrize(
    "answer",
    ["Looks fine to me.", "{not json}", '{"findings": [{"path": "a.py"}]}', "[]"],
)
def test_an_answer_that_is_not_findings_is_no_answer(answer: str):
    assert bugbot.parse_findings(answer) is None


def test_a_long_diff_is_cut_and_the_model_is_told():
    diff = "+x\n" * bugbot.MAX_DIFF_LENGTH

    system, user = bugbot.review_prompt("Be strict.", None, [], diff)

    assert system.startswith("Be strict.")
    assert f"the first {bugbot.MAX_DIFF_LENGTH} of {len(diff)}" in user
    assert len(user) < bugbot.MAX_DIFF_LENGTH + 500
    assert "(the team wrote none)" in user


def test_the_diff_is_of_the_teams_directory_since_it_started(repo: Path):
    base = run_git(repo, "rev-parse", "HEAD").strip()
    for path in ("services/api/app.py", "apps/web/page.tsx"):
        (repo / path).parent.mkdir(parents=True)
        (repo / path).write_text("new\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "--quiet", "-m", "work")

    diff = bugbot.team_diff(repo, base, "services/api")

    assert "services/api/app.py" in diff and "apps/web" not in diff
    assert bugbot.diff_or_none(repo, "0" * 40, "services/api") is None


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
