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
    readers: [above, beside]
    holds: One claim to a line.
"""
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


def documents(service: DirectorService, team: str) -> dict[str, str]:
    """The documents of other teams a team's IC is given, by their ids."""
    context = service.context_of(node(service, team), "ic")
    return {
        document["document_id"]: document["text"]
        for document in context.get("documents_of_other_teams", [])
    }


def test_every_shipped_type_keeps_the_directors_documents_and_none_of_its_own():
    types = load_types([SHIPPED_TYPES], DIRECTOR_TOOL_NAMES)

    assert "_defaults" not in types
    for kind in types.values():
        assert tuple(kind.documents) == DIRECTOR_DOCUMENTS
        assert kind.own_documents == {}
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

    assert (
        "- findings (`notes/findings.md` in your team's folder; commit it with "
        "your work, and the team above and the teams beside yours read what of "
        "it was accepted): One claim to a line."
    ) in lab.text_for("ic")
    # What a team keeps for its own sub-teams waits for no review.
    board = read_type(a_type(tmp_path, "board", BOARD), DIRECTOR_TOOL_NAMES)
    assert (
        "work, and your sub-teams read it as you last committed it): What every"
        in board.text_for("ic")
    )


@pytest.mark.parametrize(
    "said",
    [
        # Where the director's own are kept and who reads them is not a type's.
        "  spec:\n    holds: x\n    path: spec.md\n",
        "  contracts:\n    holds: x\n    readers: [beside]\n",
        # A document of the type's own says where it is and who reads it.
        "  findings:\n    holds: x\n    readers: [above]\n",
        "  findings:\n    holds: x\n    path: findings.md\n",
        "  findings:\n    path: findings.md\n    readers: [above]\n",
        # It stays in the team's folder.
        "  findings:\n    holds: x\n    path: ../findings.md\n    readers: [above]\n",
        "  findings:\n    holds: x\n    path: /findings.md\n    readers: [above]\n",
        "  findings:\n    holds: x\n    path: .git/config\n    readers: [above]\n",
        "  findings:\n    holds: x\n    path: f.md\n    readers: [everyone]\n",
        "  Findings:\n    holds: x\n    path: f.md\n    readers: [above]\n",
        "  one:\n    holds: x\n    path: f.md\n    readers: [above]\n"
        "  two:\n    holds: x\n    path: f.md\n    readers: [above]\n",
    ],
)
def test_a_document_no_type_may_keep_is_refused(tmp_path: Path, said: str):
    path = a_type(
        tmp_path, "lab", members_yaml() + "description: x\ndocuments:\n" + said
    )

    with pytest.raises(InvalidAgentType):
        read_type(path, DIRECTOR_TOOL_NAMES)


async def test_a_document_reaches_its_readers_once_the_work_is_accepted(
    service: DirectorService, repo: Path, tmp_path: Path
):
    runs = node(service, "runs")
    # The worktree its conversation would have made.
    worktree = tmp_path / "runs"
    run_git(repo, "worktree", "add", "--quiet", "-b", "runs", str(worktree))
    service.org.update_place(runs.id, worktree_path=str(worktree), branch="runs")
    before = steady(service.context_of(node(service, "probe"), "ic"))
    commit(worktree, "runs/notes/findings.md", "Checked: it holds.\n")

    # What Runs wrote in its own checkout is nobody else's yet.
    assert documents(service, "root") == {}
    assert documents(service, "probe") == {}

    await use(service, ic(service, "runs"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept("runs"))
    await service.settled()

    found = {f"{runs.id}/findings": "Checked: it holds.\n"}
    # The team above and the team beside it read it; a team reads none of its own.
    assert documents(service, "root") == found
    assert documents(service, "probe") == found | documents(service, "probe")
    assert f"{runs.id}/findings" not in documents(service, "runs")
    # A team is told of it with the next message it gets, and only of it.
    now = steady(service.context_of(node(service, "probe"), "ic"))
    [told] = changed(before, now, "ic", works=True)["documents_of_other_teams"]
    assert (told["document"], told["path"]) == ("findings", "runs/notes/findings.md")
    # When it changes, only the document that did is told again.
    commit(repo, "runs/notes/findings.md", "Checked: it holds twice.\n")
    later = steady(service.context_of(node(service, "probe"), "ic"))
    again = changed(now, later, "ic", works=True)["documents_of_other_teams"]
    assert [document["text"] for document in again["changed"]] == [
        "Checked: it holds twice.\n"
    ]


async def test_a_document_is_read_only_by_the_teams_its_type_names(
    service: DirectorService, repo: Path
):
    root = node(service, "root")
    commit(repo, "baseline.md", "Compare against v1.\n")
    commit(repo, "probe/notes/findings.md", "Unchecked: maybe.\n")

    baseline = {f"{root.id}/baseline": "Compare against v1.\n"}
    # The top team's document goes to the teams below it, of whatever type.
    assert baseline.items() <= documents(service, "tools").items()
    assert baseline.items() <= documents(service, "runs").items()
    assert f"{root.id}/baseline" not in documents(service, "root")
    # Probe works in the top team's directory, so what it commits is there.
    probe = node(service, "probe")
    assert documents(service, "root") == {f"{probe.id}/findings": "Unchecked: maybe.\n"}
    # Tools is beside Probe, and its own type keeps no document for it to
    # give; it reads Probe's as any team beside it does.
    assert f"{probe.id}/findings" in documents(service, "tools")
