"""The documents a type says its teams keep: the director's three, the
type's own, and the defaults every type starts from."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director import compaction
from openhands.agent_server.director.agent_types import (
    DIRECTOR_DOCUMENTS,
    SHIPPED_TYPES,
    InvalidAgentType,
    load_types,
    read_type,
)
from openhands.agent_server.director.context import changed, steady
from openhands.agent_server.director.models import Report, SplitChild
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import DIRECTOR_TOOL_NAMES
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    ProposeTeamsTool,
    ReportTool,
    ReviewTeamTool,
    SubmitReviewTool,
    WriteSpecTool,
)
from openhands.agent_server.director.tools.definition import (
    ReportAction,
    WriteSpecAction,
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
    readers: [subteam]
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
        "    written_with: edit_contracts\n    readers: [subteam, team]\n"
    )
    path = a_type(tmp_path, "lab", members_yaml() + "description: x\n" + said)

    lab = read_type(path, DIRECTOR_TOOL_NAMES)

    assert lab.documents["contracts"].readers == ["subteam", "team"]


@pytest.mark.parametrize(
    "said",
    [
        # What the director's own are written with is how the director
        # keeps them: a type cannot say otherwise.
        "  spec:\n    holds: x\n    path: spec.md\n",
        "  report:\n    holds: x\n    written_with: write_spec\n",
        # A document of the type's own is a file in the team's folder.
        "  findings:\n    holds: x\n",
        # The director keeps one of the type's own only as a list of
        # entries, and a document is kept one way.
        "  findings:\n    holds: x\n    written_with: write_spec\n",
        "  findings:\n    holds: x\n    written_with: edit_contracts\n",
        "  findings:\n    holds: x\n    written_with: report\n    path: f.md\n",
        "  findings:\n    path: findings.md\n",
        "  findings:\n    holds: x\n    path: ../findings.md\n",
        "  findings:\n    holds: x\n    path: /findings.md\n",
        "  findings:\n    holds: x\n    path: .git/config\n",
        "  Findings:\n    holds: x\n    path: f.md\n",
        "  one:\n    holds: x\n    path: f.md\n  two:\n    holds: x\n    path: f.md\n",
        # It goes to the team and its sub-teams, never up or beside.
        "  spec:\n    holds: x\n    readers: [above]\n",
        "  findings:\n    holds: x\n    path: f.md\n    readers: [below]\n",
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


def test_who_reads_the_directors_documents_is_the_types_to_say(tmp_path: Path):
    said = (
        "documents:\n  spec:\n    holds: The plan.\n    readers: [team, subteam]\n"
        "  contracts:\n    holds: Promises.\n    readers: [team]\n"
        "  report:\n    holds: Decisions.\n"
    )
    path = a_type(tmp_path, "lab", members_yaml() + "description: x\n" + said)

    lab = read_type(path, DIRECTOR_TOOL_NAMES)

    assert lab.documents["spec"].readers == ["team", "subteam"]
    assert lab.reads("spec", "subteam") and not lab.reads("contracts", "subteam")
    # One the type does not say is read as the director has it.
    assert lab.documents["report"].readers == ["team"]
    # A member is told who reads one where that is not as the director
    # has it.
    told = lab.text_for("ic")
    assert "- spec (read by your team and your sub-teams): The plan." in told
    assert "- contracts (read by your team): Promises." in told
    assert "- report: Decisions." in told


# A team that calls its spec a protocol and keeps a list of findings with
# the director, which its sub-teams read too.
JOURNAL = (
    members_yaml()
    + """description: Keeps what it found.
documents:
  spec:
    title: protocol
    holds: The method.
  findings:
    written_with: report
    readers: [team, subteam]
    holds: One claim to an entry.
"""
)


def test_a_type_names_the_directors_documents_and_keeps_entries_of_its_own(
    tmp_path: Path,
):
    lab = read_type(a_type(tmp_path, "lab", JOURNAL), DIRECTOR_TOOL_NAMES)

    assert list(lab.logs) == ["findings"] and lab.own_documents == {}
    told = lab.text_for("ic")
    assert (
        "- protocol (what the director's tools and its own texts call your "
        "team's spec; written with `write_spec`): The method." in told
    )
    assert (
        "- findings (a list of entries the director keeps; add one with "
        "`report`, naming the document `findings`; read by your team and "
        "your sub-teams): One claim to an entry." in told
    )


async def test_the_entries_of_a_document_go_to_its_readers_and_are_no_reports(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    service = await an_org_of(JOURNAL, repo, launcher, tmp_path)
    try:
        root, runs = node(service, "root"), node(service, "runs")
        before = {
            role: steady(service.context_of(root, role)) for role in ("ic", "manager")
        }

        found = ReportAction(text="It holds for n < 10.", document="findings")
        said = await use(service, ic(service), ReportTool, found)
        assert "Added to findings" in said.text
        refused = await use(
            service, ic(service), ReportTool, ReportAction(text="x", document="notes")
        )
        assert "keeps no document 'notes'" in refused.text

        for role in ("ic", "manager"):
            now = service.context_of(root, role)
            assert [e["text"] for e in now["findings_kept_by_your_team"]] == [
                "It holds for n < 10."
            ]
            # An entry is none of the team's reports.
            assert [r["text"] for r in now["reports_of_your_team"]] == ["Chose A."]
        below = service.context_of(runs, "ic")
        assert [e["text"] for e in below["findings_kept_by_the_team_above"]] == [
            "It holds for n < 10."
        ]
        assert "findings_kept_by_your_team" not in below
        # The other member is told of the entry with the next message it
        # gets, and the one that wrote it is not told it back.
        told = changed(
            before["manager"],
            steady(service.context_of(root, "manager")),
            "manager",
            works=False,
        )
        assert [e["by"] for e in told["findings_kept_by_your_team"]] == ["ic"]
        seen = steady(service.context_of(root, "ic"))
        more = ReportAction(text="And for n = 10.", document="findings")
        await use(service, ic(service), ReportTool, more)
        now = steady(service.context_of(root, "ic"))
        assert len(now["findings_kept_by_your_team"]) == 2
        assert "findings_kept_by_your_team" not in changed(seen, now, "ic", works=True)
    finally:
        await service.aclose()


def test_no_model_rewrites_the_entries_of_a_document():
    kept = [
        Report(at="1", role="ic", text="a report"),
        Report(at="2", role="ic", text="an entry", document="findings"),
    ]

    assert [report.text for report in compaction.plain(kept)] == ["a report"]


# A team whose sub-teams read its spec and its reports and not its
# contracts, and whose Manager reads none of the three.
OPEN = (
    members_yaml()
    + """description: Works in the open.
documents:
  spec:
    holds: The plan.
    readers: [subteam]
  contracts:
    holds: Promises.
    readers: [subteam]
  report:
    holds: Decisions.
    readers: [subteam]
"""
)
CLOSED = OPEN.replace("[subteam]", "[team]")
PROMISED = proposing(
    [SplitChild(title="Runs", path="runs", task="Run it")], contracts="Runs: v1."
)


async def an_org_of(
    kind: str, repo: Path, launcher: FakeLauncher, tmp_path: Path
) -> DirectorService:
    config = tmp_path / "config" / "director.yaml"
    a_type(config.parent / "types", "open", kind)
    config.write_text("")
    service = await an_org(repo, launcher, config, agent_type="open")
    await use(service, ic(service), ProposeTeamsTool, PROMISED)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)
    await use(service, ic(service), WriteSpecTool, WriteSpecAction(file_text="Plan A."))
    await use(service, ic(service), ReportTool, ReportAction(text="Chose A."))
    return service


async def test_the_directors_documents_go_to_the_sub_teams_a_type_names(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    service = await an_org_of(OPEN, repo, launcher, tmp_path)
    try:
        root, runs = node(service, "root"), node(service, "runs")

        for role in ("ic", "manager"):
            below = service.context_of(runs, role)
            assert below["spec_of_the_team_above"] == "Plan A."
            assert [r["text"] for r in below["reports_of_the_team_above"]] == [
                "Chose A."
            ]
            assert below["contracts_with_the_teams_beside_you"] == ["Runs: v1.\n"]
        # The member that writes a document has it whoever else reads it,
        # and the others of the team have it only where the type names
        # the team.
        wrote = service.context_of(root, "ic")
        assert wrote["spec"] == "Plan A."
        assert wrote["contracts_between_your_sub_teams"] == ["Runs: v1.\n"]
        assert [r["text"] for r in wrote["reports_of_your_team"]] == ["Chose A."]
        other = service.context_of(root, "manager")
        assert other["spec"] is None
        assert other["contracts_between_your_sub_teams"] == []
        assert other["reports_of_your_team"] == []
        # Nothing of it goes up.
        assert "spec_of_the_team_above" not in wrote
    finally:
        await service.aclose()


async def test_the_directors_documents_stay_with_a_team_whose_type_names_no_sub_team(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    service = await an_org_of(CLOSED, repo, launcher, tmp_path)
    try:
        root, runs = node(service, "root"), node(service, "runs")

        below = service.context_of(runs, "ic")
        assert "spec_of_the_team_above" not in below
        assert "reports_of_the_team_above" not in below
        assert below["contracts_with_the_teams_beside_you"] == []
        other = service.context_of(root, "manager")
        assert other["spec"] == "Plan A."
        assert other["contracts_between_your_sub_teams"] == ["Runs: v1.\n"]
        assert [r["text"] for r in other["reports_of_your_team"]] == ["Chose A."]
    finally:
        await service.aclose()


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
