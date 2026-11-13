"""What a team's IC is asked about its own work before it is reviewed."""

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


def test_the_questions_carry_gits_list_of_every_changed_file(repo: Path):
    base = run_git(repo, "rev-parse", "HEAD").strip()
    for path in ("services/api/app.py", "services/api/big.py", "apps/web/page.tsx"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("x = 1\n" * 20_000)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "--quiet", "-m", "work")

    files = bugbot.changed_files(repo, base, "services/api")

    assert files is not None
    assert "services/api/app.py" in files and "services/api/big.py" in files
    assert "apps/web" not in files and "40000 insertions" in files
    assert files in bugbot.write_up_question(files)
    review = bugbot.review_question("Be strict.", None, ["POST /pay"], files)
    assert "Be strict." in review and "POST /pay" in review and files in review
    assert "(the team wrote none)" in review


def test_no_change_and_no_such_commit_are_told_apart(repo: Path):
    head = run_git(repo, "rev-parse", "HEAD").strip()

    assert bugbot.changed_files(repo, head, "services/api") == ""
    assert bugbot.changed_files(repo, "0" * 40, "services/api") is None


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
