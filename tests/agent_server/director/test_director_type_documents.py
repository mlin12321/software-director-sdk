"""The documents a type says its teams keep: the director's three, the
type's own, and the defaults every type starts from."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.agent_types import (
    DIRECTOR_DOCUMENTS,
    SHIPPED_TYPES,
    InvalidAgentType,
    load_types,
    read_type,
)
from openhands.agent_server.director.context import changed, steady
from openhands.agent_server.director.models import SplitChild
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import DIRECTOR_TOOL_NAMES
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    ProposeTeamsTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from tests.agent_server.director.conftest import (
    APPROVE,
    DONE,
    FakeLauncher,
    accept,
    an_org,
    commit,
    ic,
    manager,
    members_yaml,
    node,
    proposing,
    run_git,
    tasked,
    use,
)


FINDINGS = """\
documents:
  findings:
    path: notes/findings.md
    holds: One claim to a line.
"""
# A team whose own members read what its IC keeps.
LAB = members_yaml() + "description: Runs experiments.\n" + FINDINGS
# A team whose sub-teams read what it keeps.
BOARD = (
    members_yaml()
    + """\
description: Says what the labs below build on.
documents:
  baseline:
    path: baseline.md
    readers: [below]
    holds: What every lab compares against.
"""
)
LABS = proposing(
    [
        SplitChild(
            title="Runs",
            path="runs",
            task="Run it",
            agent_type="lab",
            isolation="worktree",
        ),
        SplitChild(title="Probe", path="probe", task="Probe it", agent_type="lab"),
        SplitChild(title="Tools", path="tools", task="Build the tools"),
    ]
)


def a_type(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(body)
    return path


@pytest.fixture
async def service(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
) -> AsyncIterator[DirectorService]:
    config = tmp_path / "config" / "director.yaml"
    a_type(config.parent / "types", "lab", LAB)
    a_type(config.parent / "types", "board", BOARD)
    config.write_text("")
    service = await an_org(repo, launcher, config, agent_type="board")
    await use(service, ic(service), ProposeTeamsTool, LABS)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)
    yield service
    await service.aclose()


def documents(service: DirectorService, team: str, role: str = "ic") -> dict[str, str]:
    """The documents kept as files a member of a team is given, by their ids."""
    context = service.context_of(node(service, team), role)
    return {
        document["document_id"]: document["text"]
        for document in context.get("documents_kept_as_files", [])
    }


def test_every_shipped_type_keeps_the_directors_documents_and_none_of_its_own():
    types = load_types([SHIPPED_TYPES], DIRECTOR_TOOL_NAMES)

    assert "_defaults" not in types
    for kind in types.values():
        assert tuple(kind.documents) == tuple(DIRECTOR_DOCUMENTS)
        assert kind.own_documents == {}
        # The spec, the contracts and the reports are documents as any other:
        # each says what it is written with and who reads it.
        assert {
            name: (document.written_with, tuple(document.readers))
            for name, document in kind.documents.items()
        } == DIRECTOR_DOCUMENTS
    # The default wording is the one of a team that builds software, and a
    # type says only what it changes.
    assert "- spec: How your team works inside its folder." in types[
        "software-engineer"
    ].text_for("ic")
    engineer = types["research-engineer"].text_for("ic")
    assert "- spec: What the team's code measures or produces" in engineer
    assert "- report: A decision and why, a setback, what failed." in engineer


def test_a_type_starts_from_the_defaults_of_the_org_over_the_shipped_ones(
    tmp_path: Path,
):
    (tmp_path / "_defaults.yaml").write_text(
        "documents:\n  report:\n    holds: What the org wants reported.\n"
        "review:\n  checklist: The org's checklist.\n"
    )
    a_type(tmp_path, "lab", LAB + "review:\n  write_up: The lab's write-up.\n")

    types = load_types([SHIPPED_TYPES, tmp_path], DIRECTOR_TOOL_NAMES)

    lab = types["lab"]
    assert list(lab.documents) == [*DIRECTOR_DOCUMENTS, "findings"]
    assert lab.documents["spec"].holds == "How your team works inside its folder."
    assert lab.documents["report"].holds == "What the org wants reported."
    # A mapping is merged key by key: the type's part and the defaults' both hold.
    assert lab.review.checklist == "The org's checklist."
    assert lab.review.write_up == "The lab's write-up."
    # The org's defaults hold for the shipped types of that org too.
    shipped = types["software-engineer"]
    assert shipped.documents["report"].holds == "What the org wants reported."
    assert "_defaults" not in types


def test_defaults_that_cannot_be_read_take_no_type_with_them(tmp_path: Path):
    (tmp_path / "_defaults.yaml").write_text("- not a mapping\n")
    a_type(tmp_path, "lab", LAB)

    types = load_types([SHIPPED_TYPES, tmp_path], DIRECTOR_TOOL_NAMES)

    assert list(types["lab"].documents) == [*DIRECTOR_DOCUMENTS, "findings"]


def test_a_team_is_told_where_a_document_of_its_types_own_is_and_who_reads_it(
    tmp_path: Path,
):
    lab = read_type(a_type(tmp_path, "lab", LAB), DIRECTOR_TOOL_NAMES)

    # A document that names no reader is the team's own.
    assert lab.documents["findings"].readers == ["team"]
    assert (
        "- findings (`notes/findings.md` in your team's folder; commit it with "
        "your work, and the others of your team read it as you last committed "
        "it): One claim to a line."
    ) in lab.text_for("ic")
    board = read_type(a_type(tmp_path, "board", BOARD), DIRECTOR_TOOL_NAMES)
    assert (
        "work, and your sub-teams read it as you last committed it): What every"
        in board.text_for("ic")
    )


def test_a_type_that_says_of_the_directors_documents_what_the_director_does_is_read(
    tmp_path: Path,
):
    said = (
        "documents:\n  contracts:\n    holds: x\n"
        "    written_with: edit_contracts\n    readers: [below, team]\n"
    )
    path = a_type(tmp_path, "lab", members_yaml() + "description: x\n" + said)

    lab = read_type(path, DIRECTOR_TOOL_NAMES)

    assert lab.documents["contracts"].readers == ["team", "below"]


@pytest.mark.parametrize(
    "said",
    [
        # What the director's own are written with and who reads them is
        # what the director does: a type cannot say otherwise.
        "  spec:\n    holds: x\n    path: spec.md\n",
        "  spec:\n    holds: x\n    readers: [team, below]\n",
        "  contracts:\n    holds: x\n    readers: [team]\n",
        "  report:\n    holds: x\n    written_with: write_spec\n",
        # A document of the type's own is a file in the team's folder.
        "  findings:\n    holds: x\n",
        "  findings:\n    holds: x\n    written_with: report\n",
        "  findings:\n    path: findings.md\n",
        "  findings:\n    holds: x\n    path: ../findings.md\n",
        "  findings:\n    holds: x\n    path: /findings.md\n",
        "  findings:\n    holds: x\n    path: .git/config\n",
        "  Findings:\n    holds: x\n    path: f.md\n",
        "  one:\n    holds: x\n    path: f.md\n  two:\n    holds: x\n    path: f.md\n",
        # It goes to the team and the teams below, never up or beside.
        "  findings:\n    holds: x\n    path: f.md\n    readers: [above]\n",
        "  findings:\n    holds: x\n    path: f.md\n    readers: [beside]\n",
        "  findings:\n    holds: x\n    path: f.md\n    readers: []\n",
    ],
)
def test_a_document_no_type_may_keep_is_refused(tmp_path: Path, said: str):
    path = a_type(
        tmp_path, "lab", members_yaml() + "description: x\ndocuments:\n" + said
    )

    with pytest.raises(InvalidAgentType):
        read_type(path, DIRECTOR_TOOL_NAMES)


async def test_a_document_goes_to_the_teams_below_and_to_no_other(
    service: DirectorService, repo: Path, tmp_path: Path
):
    root, runs = node(service, "root"), node(service, "runs")
    # The worktree the conversation of Runs would have made.
    worktree = tmp_path / "runs"
    run_git(repo, "worktree", "add", "--quiet", "-b", "runs", str(worktree))
    service.org.update_place(runs.id, worktree_path=str(worktree), branch="runs")
    before = steady(service.context_of(runs, "ic"))

    commit(repo, "baseline.md", "Compare against v1.\n")
    commit(worktree, "runs/notes/findings.md", "Checked: it holds.\n")

    baseline = {f"{root.id}/baseline": "Compare against v1.\n"}
    # The top team's document goes to every team below, of whatever type,
    # as soon as it is committed.
    for team in ("runs", "probe", "tools"):
        assert documents(service, team) == baseline
    # What Runs keeps goes neither up nor to the teams beside it, also once
    # its work is accepted.
    await use(service, ic(service, "runs"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept("runs"))
    await service.settled()
    assert documents(service, "root") == {}
    assert documents(service, "probe") == baseline
    # A team is told of it with the next message it gets, and when it
    # changes only the document that did is told again.
    now = steady(service.context_of(runs, "ic"))
    [told] = changed(before, now, "ic", works=True)["documents_kept_as_files"]
    assert (told["document"], told["path"]) == ("baseline", "baseline.md")
    commit(repo, "baseline.md", "Compare against v2.\n")
    later = steady(service.context_of(runs, "ic"))
    again = changed(now, later, "ic", works=True)["documents_kept_as_files"]
    assert [document["text"] for document in again["changed"]] == [
        "Compare against v2.\n"
    ]


async def test_a_teams_own_document_is_given_to_its_members_that_do_not_write_it(
    service: DirectorService, repo: Path
):
    probe = node(service, "probe")
    commit(repo, "probe/notes/findings.md", "Unchecked: maybe.\n")

    found = {f"{probe.id}/findings": "Unchecked: maybe.\n"}
    assert found.items() <= documents(service, "probe", "manager").items()
    # The IC wrote it and has the file.
    assert f"{probe.id}/findings" not in documents(service, "probe")
    # Its type names no reader but the team: the top team's Manager has none
    # of it, and the top team's own is for the teams below only.
    assert documents(service, "root", "manager") == {}
