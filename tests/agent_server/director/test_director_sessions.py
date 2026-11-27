"""Tests for sessions: lines of work the human opens at a team beside the
org's own, each with agents and a checkout of its own."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.director.documents.store import DocumentRef
from openhands.agent_server.director.git import git
from openhands.agent_server.director.headline import TeamSummary
from openhands.agent_server.director.merge_back import merge_back
from openhands.agent_server.director.models import (
    Boundary,
    OpenedSession,
    OpenSessionRequest,
    RouteMessageRequest,
    SessionTeam,
    SplitChild,
)
from openhands.agent_server.director.org_persistence import create_org_board
from openhands.agent_server.director.org_store import (
    session_id_of,
)
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.run_state import TASK
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgTeamTool,
    ProposeTeamsTool,
    PruneTeamTool,
    ReadTeamContextTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgTeamAction,
    PruneTeamAction,
    ReadTeamContextAction,
    ReviewTeamAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import Conflict, NotFound, Unprocessable
from tests.agent_server.director.conftest import (
    API,
    APPROVE,
    DB,
    DONE,
    SPLIT,
    FakeLauncher,
    ServerEnded,
    accept,
    again,
    commit,
    cut_off,
    division_lead_ic,
    division_lead_of,
    ic,
    landed,
    manager,
    node,
    propose_and_approve,
    record_of,
    root_conversation,
    run_git,
    split_root,
    started,
    use,
    woken,
)
from tests.agent_server.director.document_tools import (
    ReportAction,
    ReportTool,
    WriteSpecAction,
    WriteSpecTool,
)


WEB = "apps.web"
REFUNDS = "Add refunds to the checkout API."


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    """An org of API (with DB below it) and Web, which the first session
    made and whose work is accepted."""
    service = await started(repo, launcher, stall_poll=3600)
    await split_root(service)
    await propose_and_approve(
        service,
        API,
        SplitChild(title="DB", path="services/api/db", task="Own the schema"),
    )
    await landed(service)
    yield service
    await service.aclose()


async def opened(
    service: DirectorService,
    at: str | None = API,
    text: str = REFUNDS,
    title: str = "Refunds",
) -> OpenedSession:
    session = await service.open_session(
        OpenSessionRequest(text=text, title=title, node_id=at)
    )
    await service.settled()
    return session


def own_checkout(
    service: DirectorService, repo: Path, tmp_path: Path, node_id: str
) -> Path:
    """Give a session's teams the worktree its conversation would have made."""
    name = node_id.replace("@", "-")
    worktree = tmp_path / name
    run_git(repo, "worktree", "add", "--quiet", "-b", name, str(worktree))
    for team in service.org.session(session_id_of(node(service, node_id))):
        if not team.division_lead:
            service.org.update_place(team.id, worktree_path=str(worktree), branch=name)
    return worktree


async def test_a_session_is_beside_the_org_tree_with_a_place_at_each_of_its_teams(
    service: DirectorService, launcher: FakeLauncher
):
    tree_before = [team.model_dump() for team in service.org.teams()]
    [first] = service.sessions()
    api = service.org.team(API)
    assert api is not None
    events: list[str] = []
    snapshot, unsubscribe = service.org.subscribe(
        "root", lambda event: events.append(event.type), full=True
    )

    session = await opened(service)
    unsubscribe()

    found = session.session
    assert (found.title, found.team_id, found.above_id) == ("Refunds", API, "root")
    assert (found.team, found.request) == ("API", REFUNDS)
    top, db, review = found.teams
    assert isinstance(top, SessionTeam) and top.id == f"{API}@{found.id}"
    # Above the team it was opened at is its Division Lead place,
    # where nobody stands until it submits.
    assert (review.id, review.team_id) == (f"root@{found.id}@lead", "root")
    assert review.division_lead and review.parent_id is None and review.agents == []
    assert (top.team_id, top.parent_id, top.child_ids) == (API, review.id, [db.id])
    assert top.boundary == api.boundary
    assert (db.team_id, db.parent_id, db.agents) == (DB, top.id, [])
    # The session's IC starts from the IC the org's first session began with.
    start = launcher.derived[-1]
    assert str(start.conversation_id) == ic(service, top.id) == found.conversation_id
    assert str(start.parent_id) == ic(service)
    assert REFUNDS in start.task and "session 'Refunds'" in start.task
    assert session.conversation_id == found.conversation_id
    # The tree is what it was: no node, no child and no count is the session's.
    assert [team.model_dump() for team in service.org.teams()] == tree_before
    assert service.org.team(top.id) is None and len(service.org) == 4
    assert [team.id for team in service.org.sub_teams("root")] == [API, WEB]
    now = service.org.snapshot("root", full=True)
    assert snapshot is not None and now is not None
    assert [team.id for team in now.nodes] == [team.id for team in snapshot.nodes]
    assert [open.id for open in snapshot.sessions] == [first.id]
    assert [open.id for open in now.sessions] == [first.id, found.id]
    assert "sessions" in events and "node_upsert" not in events
    assert [open.id for open in service.sessions()] == [first.id, found.id]
    [event] = [
        event
        for event in record_of(service, "root")
        if event.kind == "session_opened" and event.session == "Refunds"
    ]
    assert (event.team_id, event.detail) == (API, REFUNDS)


async def test_a_conversation_the_human_writes_in_is_marked_and_named(
    service: DirectorService, launcher: FakeLauncher
):
    session = await opened(service)
    await service.settled()
    top = ic(service, session.session.teams[0].id)
    assert top is not None

    routed = await service.route_message(
        RouteMessageRequest(text="Tidy the handlers.", path="services/api")
    )
    await service.settled()
    service.joined(top)
    await service.settled()

    # The IC of each session the human's words opened, once each; not one
    # only the director wrote to.
    assert launcher.joined == [
        UUID(ic(service) or ""),
        UUID(top),
        UUID(routed.conversation_id),
    ]
    assert launcher.titles[UUID(top)] == "API · Refunds (IC)"
    assert service.place_of(top).human_joined
    assert not service.place_of(ic(service, API) or "").human_joined
    with pytest.raises(NotFound):
        service.joined(str(uuid4()))


async def test_a_session_the_human_gave_no_name_is_called_what_it_is_for(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.summary = TeamSummary("Order Refunds", "Adds refunds to the API.")

    unnamed = await opened(service, title="")
    named = await opened(service, title="Refunds")

    titles = {s.id: s.title for s in service.org.sessions()}
    assert titles[unnamed.session.id] == "Order Refunds"
    assert titles[named.session.id] == "Refunds"
    # The team keeps its own name; its IC is called by both.
    found = service.org.find_by_conversation(unnamed.conversation_id)
    assert found is not None and found[0].title == "API"
    assert launcher.titles[UUID(unnamed.conversation_id)] == "API · Order Refunds (IC)"


async def test_a_conversation_links_to_the_ones_a_level_from_it(
    service: DirectorService,
):
    [first] = service.sessions()
    session = await opened(service)
    top = session.session.teams[0].id
    await use(
        service,
        ic(service, top),
        MsgTeamTool,
        MsgTeamAction(team=f"{DB}@{session.session.id}", text="Add a refunds table."),
    )
    await service.settled()

    def links(conversation_id: str | None) -> list[tuple[str, str, str, str]]:
        place = service.place_of(conversation_id or "")
        return [
            (link.relation, link.node_id, link.role, link.session)
            for link in place.links
        ]

    root = service.place_of(ic(service) or "")
    in_session = service.place_of(ic(service, top) or "")

    assert (root.node_id, root.role, root.session_id) == ("root", "ic", first.id)
    # The IC a session was opened with reaches the IC that reviewed it, its
    # Manager and its sub-teams, all of its own session.
    assert links(ic(service)) == [
        ("above", "root", "ic", "Checkout"),
        ("beside", "root", "manager", "Checkout"),
        ("below", API, "ic", "Checkout"),
        ("below", WEB, "ic", "Checkout"),
    ]
    assert (in_session.node_id, in_session.team) == (API, "API")
    assert (in_session.session_id, in_session.session) == (
        session.session.id,
        "Refunds",
    )
    # Nobody is above a session that has not submitted its work.
    assert links(ic(service, top)) == [("below", DB, "ic", "Refunds")]
    assert links(ic(service, API))[0] == ("above", "root", "ic", "Checkout")
    assert ("below", DB, "ic", "Checkout") in links(ic(service, API))

    await use(service, ic(service, top), SubmitReviewTool, DONE)

    assert links(ic(service, top))[0] == ("above", "root", "ic", "Refunds")
    assert links(division_lead_ic(service, top)) == [("below", API, "ic", "Refunds")]


async def test_a_session_at_the_top_team_is_reviewed_from_a_place_of_that_team(
    service: DirectorService,
):
    session = await opened(service, "root", "Rename the product everywhere.")

    found = session.session
    assert (found.team_id, found.above_id) == ("root", "root")
    assert found.teams[0].parent_id == f"root@{found.id}@lead"
    # It has a place at every team of the org and its Division Lead place,
    # and only the first started.
    assert [team.team_id for team in found.teams] == ["root", API, WEB, DB, "root"]
    assert [bool(team.agents) for team in found.teams] == [
        True,
        False,
        False,
        False,
        False,
    ]
    # The top team keeps its sub-teams and gains none.
    assert [team.id for team in service.org.sub_teams("root")] == [API, WEB]


async def test_a_session_is_refused_where_it_cannot_be_found_or_started(
    service: DirectorService, launcher: FakeLauncher
):
    with pytest.raises(NotFound):
        await opened(service, "no.such.team")

    del launcher.known[UUID(ic(service) or "")]
    with pytest.raises(Conflict, match="No conversation of the org"):
        await opened(service)
    assert [open.title for open in service.sessions()] == ["Checkout"]


async def test_the_director_chooses_the_team_when_none_is_named(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = f"Teams: {WEB}\nReason: It is the cart page."

    session = await opened(service, None, "Show the order id on the cart page.")

    assert session.session.team_id == WEB
    assert (session.chosen_by, session.reason) == ("model", "It is the cart page.")


async def test_a_sub_team_joins_a_session_when_its_ic_gives_it_a_task(
    service: DirectorService, launcher: FakeLauncher
):
    session = await opened(service)
    top = session.session.teams[0].id
    db = f"{DB}@{session.session.id}"
    to_db = len(launcher.messages_to(ic(service, DB)))

    await use(
        service,
        ic(service, top),
        MsgTeamTool,
        MsgTeamAction(team=db, text="Add a refunds table."),
    )

    start = launcher.derived[-1]
    assert str(start.conversation_id) == ic(service, db)
    assert str(start.parent_id) == ic(service, top)
    assert "Add a refunds table." in start.task and "Own the schema" in start.task
    # The DB of the other session hears nothing of it.
    assert len(launcher.messages_to(ic(service, DB))) == to_db


async def test_each_session_has_a_reviewer_of_its_own_and_the_second_meets_the_first(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    first = (await opened(service, title="Refunds")).session.teams[0].id
    second = (await opened(service, title="Vouchers")).session.teams[0].id
    to_root = len(launcher.messages_to(ic(service)))
    for team, content in ((first, "refunds\n"), (second, "vouchers\n")):
        worktree = own_checkout(service, repo, tmp_path, team)
        commit(worktree, "services/api/routes.ts", content)
        assert division_lead_ic(service, team) is None
        await use(service, ic(service, team), SubmitReviewTool, DONE)
        # Its reviewer starts for it, with the request.
        start = launcher.derived[-1]
        assert str(start.conversation_id) == division_lead_ic(service, team)
        assert "submits its work" in start.task and team in start.task
    assert division_lead_ic(service, first) != division_lead_ic(service, second)
    assert len(launcher.messages_to(ic(service))) == to_root

    await use(service, division_lead_ic(service, first), ReviewTeamTool, accept(first))
    await use(
        service, division_lead_ic(service, second), ReviewTeamTool, accept(second)
    )

    # What does not merge with the first is the second's reviewer's to
    # merge: its team is not sent back and keeps its turn.
    assert (repo / "services/api/routes.ts").read_text() == "refunds\n"
    reviews = [node(service, team).review for team in (first, second)]
    assert [review.state for review in reviews if review] == ["accepted", "pending"]
    merge = service.merge_checkout(second)
    assert merge is not None and merge.files == ["services/api/routes.ts"]
    assert merge.where in launcher.messages_to(division_lead_ic(service, second))[-1]
    waits = node(service, division_lead_of(service, second)).merge
    assert waits is not None
    assert [(item.kind, item.key, item.gate) for item in waits.items] == [
        ("code", "services/api/routes.ts", "ic")
    ]
    (Path(merge.where) / "services/api/routes.ts").write_text("refunds\nvouchers\n")
    await use(
        service, division_lead_ic(service, second), ReviewTeamTool, accept(second)
    )
    assert (repo / "services/api/routes.ts").read_text() == "refunds\nvouchers\n"
    review = node(service, second).review
    assert review is not None and review.state == "accepted"
    # An accepted session stays open, with its conversation, until the human
    # closes it.
    await service.settled()
    assert [open.title for open in service.sessions()] == [
        "Checkout",
        "Refunds",
        "Vouchers",
    ]
    assert launcher.deleted == []


async def test_work_uncommitted_beside_a_session_is_not_put_aside_for_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    """What is not committed in the checkout a session is merged into is
    somebody's work, not something in the way: the merge waits for it."""
    top = (await opened(service)).session.teams[0].id
    worktree = own_checkout(service, repo, tmp_path, top)
    commit(worktree, "services/api/routes.ts", "refunds\n")
    (repo / "services/api").mkdir(parents=True, exist_ok=True)
    (repo / "services/api/routes.ts").write_text("another's, not committed\n")
    await use(service, ic(service, top), SubmitReviewTool, DONE)

    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))

    assert (repo / "services/api/routes.ts").read_text() == (
        "another's, not committed\n"
    )
    review = node(service, top).review
    assert review is not None and review.state == "pending"
    told = launcher.messages_to(division_lead_ic(service, top))[-1]
    assert "uncommitted changes are in the way of the merge" in told


async def test_a_session_that_changed_nothing_itself_is_accepted_with_its_sub_teams(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    """Each change is reviewed once, by the IC above the team that made it.
    A session whose own IC changed nothing has no change left to review."""
    session = await opened(service)
    top, db = session.session.teams[0].id, f"{DB}@{session.session.id}"
    await use(service, ic(service, top), MsgTeamTool, MsgTeamAction(team=db, text="Go"))
    commit(repo, "services/api/db/schema.sql", "create table refunds;\n")
    await use(service, ic(service, db), SubmitReviewTool, DONE)

    await use(service, ic(service, top), ReviewTeamTool, accept(db))

    accepted = node(service, db).review
    assert accepted is not None and accepted.state == "accepted"
    passed = node(service, top).review
    assert passed is not None and (passed.state, passed.passed_up) == ("accepted", True)
    assert division_lead_ic(service, top) is None
    assert "without another review" in launcher.messages_to(ic(service, top))[-1]


async def test_sub_teams_a_session_changes_are_its_own_until_its_work_is_accepted(
    service: DirectorService,
):
    session = await opened(service)
    top = session.session.teams[0].id
    routes = SplitChild(title="Routes", path="services/api/routes", task="Route")

    await propose_and_approve(service, top, routes, keep=["DB"])
    by_another = await use(
        service,
        await woken(service),
        PruneTeamTool,
        PruneTeamAction(team=top, reason="Not needed"),
    )

    assert [team.title for team in service.org.below(top)] == ["DB", "Routes"]
    assert "not a team directly below yours" in by_another.text
    # Neither the org nor the other session at the team has the new one.
    assert [team.id for team in service.org.sub_teams(API)] == [DB]
    assert [team.title for team in service.org.below(node(service, API).id)] == ["DB"]

    await use(service, ic(service, top), SubmitReviewTool, DONE)
    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))

    assert [team.id for team in service.org.sub_teams(API)] == [
        DB,
        "services.api.routes",
    ]


async def test_the_sessions_of_a_team_know_of_each_other_and_share_accepted_reports(
    service: DirectorService,
):
    top = (await opened(service)).session.teams[0].id
    await use(
        service, ic(service, API), ReportTool, ReportAction(text="Routes are typed.")
    )
    await use(
        service, ic(service, top), ReportTool, ReportAction(text="Refunds are async.")
    )

    def reports(place: str) -> list[tuple[str, str]]:
        context = service.context_of(node(service, place), "ic")
        return [
            (report["by"], report["text"]) for report in context["reports_of_your_team"]
        ]

    other = service.context_of(node(service, API), "ic")
    session = service.context_of(node(service, top), "ic")

    # What a session records is its own until its work is accepted.
    assert reports(API) == [("ic", "Routes are typed.")]
    assert reports(top) == [("ic", "Refunds are async.")]
    assert session["you"]["session"] == "Refunds"
    assert [found["team_id"] for found in session["other_sessions_of_your_team"]] == [
        node(service, API).id
    ]
    assert [found["session"] for found in other["other_sessions_of_your_team"]] == [
        "Refunds"
    ]

    await use(service, ic(service, top), SubmitReviewTool, DONE)
    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))

    assert reports(API) == [
        ("ic", "Routes are typed."),
        ("ic in 'Refunds'", "Refunds are async."),
    ]


async def test_a_report_of_another_session_with_the_same_title_is_not_ones_own(
    service: DirectorService,
):
    first = (await opened(service)).session.teams[0].id
    await use(
        service, ic(service, first), ReportTool, ReportAction(text="Refunds are async.")
    )
    await use(service, ic(service, first), SubmitReviewTool, DONE)
    await use(service, division_lead_ic(service, first), ReviewTeamTool, accept(first))

    second = (await opened(service)).session.teams[0].id

    context = service.context_of(node(service, second), "ic")
    assert [report["by"] for report in context["reports_of_your_team"]] == [
        "ic in 'Refunds'"
    ]


async def test_a_session_opened_below_the_top_has_its_teams_contracts(
    service: DirectorService,
):
    # What the top team agreed between API and Web, as the org has it.
    service.team_memory.write("", "contracts.md", "API serves JSON to Web.")

    top = (await opened(service)).session.teams[0].id

    context = service.context_of(node(service, top), "ic")
    assert context["contracts_with_the_teams_beside_you"] == ["API serves JSON to Web."]


async def test_a_request_to_the_org_opens_a_session_and_reaches_no_other(
    service: DirectorService, launcher: FakeLauncher
):
    top = (await opened(service)).session.teams[0].id
    launcher.answers[USAGE_ID] = f"Teams: {API}\nReason: It is the API's."
    heard = {
        place.id: len(launcher.messages_to(ic(service, place.id)))
        for place in service.org.places()
        if place.agents
    }

    routed = await service.route_message(
        RouteMessageRequest(text="Make services/api faster.", path="", auto=True)
    )
    by_path = await service.route_message(
        RouteMessageRequest(text="Tidy the handlers.", path="services/api")
    )
    await service.settled()

    *_, faster, tidy = service.sessions()
    assert (routed.node_id, by_path.node_id) == (
        f"{API}@{faster.id}",
        f"{API}@{tidy.id}",
    )
    assert (faster.team_id, tidy.team_id) == (API, API)
    assert "Tidy the handlers." in launcher.derived[-1].task
    assert top in heard and heard == {
        place: len(launcher.messages_to(ic(service, place))) for place in heard
    }


async def test_an_agent_reads_the_other_sessions_with_its_context(
    service: DirectorService,
):
    await opened(service)

    observation = await use(
        service, ic(service, API), ReadTeamContextTool, ReadTeamContextAction()
    )

    assert "other_sessions_of_your_team" in observation.text


async def test_what_a_session_waits_on_the_human_for_is_told_with_its_team(
    service: DirectorService,
):
    session = await opened(service, "root", "Rename the product everywhere.")
    top = session.session.teams[0].id
    # Nobody stands above the team a session was opened at.
    raised = service.org.raise_escalation(
        top, "name", kind="design_decision", title="Which name?"
    )

    [waiting] = service.org.escalations()

    assert raised.held_by is None
    # It names the team of the org, which a client can show, and the session.
    assert (waiting.node_id, waiting.session_id, waiting.session) == (
        "root",
        session.session.id,
        "Refunds",
    )
    assert [step.id for step in waiting.path] == ["root"]


async def test_a_session_is_kept_on_its_machine_and_what_it_changes_with_the_org(
    service: DirectorService, repo: Path, launcher: FakeLauncher
):
    [first] = service.sessions()
    session = await opened(service)
    top = session.session.teams[0].id
    await use(
        service,
        ic(service, top),
        ReportTool,
        ReportAction(text="Refunds need the order id."),
    )
    await service.settled()
    await service.aclose()

    board = create_org_board(repo)
    saved = json.loads(board.read("org.json") or "{}")
    restarted = await again(repo, launcher)

    try:
        # Nothing of the session is in the org that is stored and exchanged.
        assert set(saved) == {"version", "nodes"}
        assert [team["id"] for team in saved["nodes"]] == ["root", API, WEB, DB]
        assert all("session" not in team for team in saved["nodes"])
        assert restarted.team_memory.reports(API, "ic") == []
        # That it was opened is with the team's memory.
        assert [
            event.session
            for event in record_of(restarted, "root")
            if event.kind == "session_opened"
        ] == ["Checkout", "Refunds"]
        # The sessions themselves come back from what this machine kept,
        # with what their agents recorded.
        assert [found.id for found in restarted.sessions()] == [
            first.id,
            session.session.id,
        ]
        found = restarted.sessions()[-1]
        assert [team.id for team in found.teams] == [
            team.id for team in session.session.teams
        ]
        assert found.conversation_id == session.conversation_id
        assert [
            report.text for report in restarted.own_reports(node(restarted, top))
        ] == ["Refunds need the order id."]
    finally:
        await restarted.aclose()


async def test_closing_a_session_removes_it_and_keeps_what_was_not_merged(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    session = await opened(service)
    top = session.session.teams[0].id
    conversation = ic(service, top)
    worktree = own_checkout(service, repo, tmp_path, top)
    commit(worktree, "services/api/routes.ts", "refunds\n")
    tip = run_git(worktree, "rev-parse", "HEAD").strip()
    (repo / "services/api").mkdir(parents=True, exist_ok=True)
    (repo / "services/api/own.ts").write_text("another's, not committed\n")

    service.close_session(session.session.id)
    await service.settled()

    assert [open.title for open in service.sessions()] == ["Checkout"]
    assert service.org.session(session.session.id) == []
    assert conversation is not None and UUID(conversation) in launcher.deleted
    kept = run_git(
        repo, "for-each-ref", "--format=%(objectname)", "refs/agents/sessions/"
    ).split()
    assert kept == [tip]
    assert (repo / "services/api/own.ts").exists()
    # None of its work was accepted, so it is not on the org's record.
    assert "session_closed" not in [e.kind for e in record_of(service, "root")]
    with pytest.raises(NotFound):
        service.close_session(session.session.id)


async def test_closing_a_session_that_never_started_touches_no_checkout(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    """Until its IC has started, a session's team names the checkout of
    the team above, where the work is somebody else's."""
    session = await opened(service)
    top = session.session.teams[0].id
    assert node(service, top).worktree_path == node(service, "root").worktree_path
    (repo / "services/api").mkdir(parents=True, exist_ok=True)
    (repo / "services/api/own.ts").write_text("the team's own, not committed\n")

    service.close_session(session.session.id)
    await service.settled()

    assert (repo / "services/api/own.ts").read_text() == (
        "the team's own, not committed\n"
    )
    assert run_git(repo, "for-each-ref", "refs/agents/pruned/") == ""


def test_a_merge_takes_nothing_the_team_may_only_read(repo: Path, tmp_path: Path):
    commit(repo, "api/contract.md", "v1\n")
    worktree = tmp_path / "session"
    run_git(repo, "worktree", "add", "--quiet", "-b", "session", str(worktree))
    commit(worktree, "api/contract.md", "changed by the team\n")

    held = Boundary(path="api", writable=["api/**"], readonly=["api/contract.md"])
    result = merge_back(repo, worktree, held, "API")

    assert (result.outcome, result.files) == ("outside", ("api/contract.md",))
    assert (repo / "api" / "contract.md").read_text() == "v1\n"


def test_a_merge_takes_nothing_from_a_team_that_may_write_nothing(
    repo: Path, tmp_path: Path
):
    commit(repo, "api/routes.ts", "v1\n")
    worktree = tmp_path / "session"
    run_git(repo, "worktree", "add", "--quiet", "-b", "session", str(worktree))
    commit(worktree, "api/routes.ts", "changed by the team\n")

    result = merge_back(repo, worktree, Boundary(path="api"), "API")

    assert (result.outcome, result.files) == ("outside", ("api/routes.ts",))
    assert (repo / "api" / "routes.ts").read_text() == "v1\n"


def test_a_merge_reaches_the_index_of_an_agent_that_owns_the_same_directory(
    repo: Path, tmp_path: Path
):
    """An agent's own index does not follow a merge into its checkout by
    itself. Left behind, its next commit would take the merged files out."""
    commit(repo, "api/routes.ts", "v1\n")
    commit(repo, "api/own.ts", "v1\n")
    worktree = tmp_path / "session"
    run_git(repo, "worktree", "add", "--quiet", "-b", "session", str(worktree))
    commit(worktree, "api/routes.ts", "from the session\n")
    commit(worktree, "api/new.ts", "from the session\n")
    git_dir = Path(run_git(repo, "rev-parse", "--absolute-git-dir").strip())
    own = {"GIT_INDEX_FILE": str(git_dir / "agent-index-abc")}
    git(repo, ["read-tree", "HEAD"], env=own)
    (repo / "api/own.ts").write_text("staged by the agent\n")
    git(repo, ["add", "api/own.ts"], env=own)

    result = merge_back(
        repo, worktree, Boundary(path="api", writable=["api/**"]), "API"
    )
    git(repo, ["commit", "--quiet", "-m", "The agent's own"], env=own)

    assert result.outcome == "merged"
    committed = {
        path: run_git(repo, "show", f"HEAD:{path}")
        for path in ("api/routes.ts", "api/new.ts", "api/own.ts")
    }
    assert committed == {
        "api/routes.ts": "from the session\n",
        "api/new.ts": "from the session\n",
        "api/own.ts": "staged by the agent\n",
    }


ROUTES = "services/api/routes.ts"


async def two_submitted(
    service: DirectorService, repo: Path, tmp_path: Path
) -> tuple[str, str]:
    """Two sessions at API that each changed another line of one file and
    submitted, the first before the second."""
    commit(repo, ROUTES, "one\ntwo\nthree\nfour\nfive\n")
    first = (await opened(service, title="Refunds")).session.teams[0].id
    second = (await opened(service, title="Vouchers")).session.teams[0].id
    for team, content, at in (
        (first, "refunds\ntwo\nthree\nfour\nfive\n", "2026-01-01T00:00:01+00:00"),
        (second, "one\ntwo\nthree\nfour\nvouchers\n", "2026-01-01T00:00:02+00:00"),
    ):
        commit(own_checkout(service, repo, tmp_path, team), ROUTES, content)
        await use(service, ic(service, team), SubmitReviewTool, DONE)
        review = node(service, team).review
        assert review is not None
        service.org.set_review(team, review.model_copy(update={"at": at}))
    return first, second


async def test_work_is_accepted_in_the_order_it_was_submitted(
    service: DirectorService, repo: Path, tmp_path: Path
):
    first, second = await two_submitted(service, repo, tmp_path)

    refused = await use(
        service, division_lead_ic(service, second), ReviewTeamTool, accept(second)
    )

    assert "submitted before" in refused.text
    pending = node(service, second).review
    assert pending is not None and pending.state == "pending"

    await use(service, division_lead_ic(service, first), ReviewTeamTool, accept(first))
    accepted = node(service, first).review
    assert accepted is not None and accepted.state == "accepted"


async def test_work_that_meets_merged_work_is_reconciled_before_it_is_merged(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    """git merges two changes to different lines of a file without a word.
    The reviewer is asked about them all the same."""
    first, second = await two_submitted(service, repo, tmp_path)
    await use(service, division_lead_ic(service, first), ReviewTeamTool, accept(first))

    asked = (
        await use(
            service, division_lead_ic(service, second), ReviewTeamTool, accept(second)
        )
    ).text

    assert "Not merged yet" in asked
    assert f"- {ROUTES}" in asked
    assert "merged meanwhile: API · Refunds" in asked
    assert (repo / ROUTES).read_text() == "refunds\ntwo\nthree\nfour\nfive\n"

    await use(
        service,
        division_lead_ic(service, second),
        ReviewTeamTool,
        ReviewTeamAction(team=second, decision="accept", reconciled="Both lines hold."),
    )

    assert (repo / ROUTES).read_text() == "refunds\ntwo\nthree\nfour\nvouchers\n"
    history = record_of(service, "root")
    assert [e.detail for e in history if e.kind == "reconciled"] == ["Both lines hold."]


async def test_a_spec_a_session_writes_is_its_own_until_its_work_is_accepted(
    service: DirectorService, repo: Path, tmp_path: Path
):
    top = (await opened(service)).session.teams[0].id
    before = service.team_memory.read("services/api/spec.md")

    await use(
        service, ic(service, top), WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )

    assert service.team_memory.read("services/api/spec.md") == before
    assert [spec.spec for spec in service.specs_of(node(service, top))] == ["# Refunds"]
    assert "# Refunds" not in [
        spec.spec for spec in service.specs_of(node(service, API))
    ]

    commit(own_checkout(service, repo, tmp_path, top), ROUTES, "refunds\n")
    await use(service, ic(service, top), SubmitReviewTool, DONE)
    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))

    assert service.team_memory.read("services/api/spec.md") == "# Refunds"
    api = service.org.team(API)
    assert api is not None and api.purpose == "Refunds"


async def test_any_document_a_session_writes_is_merged_with_its_accepted_work(
    service: DirectorService, repo: Path, tmp_path: Path
):
    # Nothing in merging is particular to the documents the shipped orgs
    # keep: one of any name goes the same way.
    top = (await opened(service)).session.teams[0].id
    glossary = DocumentRef("services/api", "glossary")

    service.documents.write(node(service, top).session, glossary, "A run is a job.")

    assert service.team_memory.read(glossary.path) is None
    assert service.documents.read(node(service, top).session, glossary) == (
        "A run is a job."
    )
    assert service.documents.read("", glossary) is None

    commit(own_checkout(service, repo, tmp_path, top), ROUTES, "refunds\n")
    await use(service, ic(service, top), SubmitReviewTool, DONE)
    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))

    assert service.team_memory.read(glossary.path) == "A run is a job."


async def test_a_spec_written_over_one_that_changed_meanwhile_is_reconciled(
    service: DirectorService, repo: Path, tmp_path: Path
):
    top = (await opened(service)).session.teams[0].id
    await use(
        service, ic(service, top), WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )
    other = (await opened(service, title="Vouchers")).session.teams[0].id
    await use(
        service,
        ic(service, other),
        WriteSpecTool,
        WriteSpecAction(file_text="# The API"),
    )
    await use(service, ic(service, other), SubmitReviewTool, DONE)
    await use(service, division_lead_ic(service, other), ReviewTeamTool, accept(other))
    commit(own_checkout(service, repo, tmp_path, top), ROUTES, "refunds\n")
    await use(service, ic(service, top), SubmitReviewTool, DONE)

    asked = (
        await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))
    ).text

    assert "It wrote a spec for services/api" in asked
    assert "# The API" in asked and "# Refunds" in asked
    assert service.team_memory.read("services/api/spec.md") == "# The API"

    await use(
        service,
        division_lead_ic(service, top),
        ReviewTeamTool,
        ReviewTeamAction(team=top, decision="accept", reconciled="Refunds holds."),
    )

    assert service.team_memory.read("services/api/spec.md") == "# Refunds"


async def test_a_spec_of_a_closed_session_is_dropped(service: DirectorService):
    session = (await opened(service)).session
    before = service.team_memory.read("services/api/spec.md")
    await use(
        service,
        ic(service, session.teams[0].id),
        WriteSpecTool,
        WriteSpecAction(file_text="# Refunds"),
    )

    service.close_session(session.id)
    await service.settled()

    assert service.team_memory.read("services/api/spec.md") == before
    assert service.spec_draft(session.teams[0]) is None


async def test_work_that_goes_up_by_itself_takes_its_place_after_what_waits(
    service: DirectorService, repo: Path, tmp_path: Path
):
    # Another session submitted since the first one's work was accepted.
    other = (await opened(service, title="Refunds")).session.teams[0].id
    commit(own_checkout(service, repo, tmp_path, other), "services/api/a.txt", "a\n")
    await use(service, ic(service, other), SubmitReviewTool, DONE)
    waits = node(service, other).review
    assert waits is not None

    # The first session's sub-teams do more; it changes nothing itself.
    commit(repo, "services/api/db/x.sql", "x\n")
    await use(service, ic(service, "services.api.db"), SubmitReviewTool, DONE)
    await use(service, ic(service, API), ReviewTeamTool, accept("services.api.db"))
    commit(repo, "apps/web/x.txt", "x\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))
    await service.settled()

    went_up = node(service, "root").review
    assert went_up is not None and went_up.passed_up
    assert went_up.seq > waits.seq


async def test_work_is_not_accepted_past_earlier_work_that_still_waits(
    service: DirectorService, repo: Path, tmp_path: Path
):
    commit(repo, "services/api/a.txt", "a\n")
    commit(repo, "services/api/b.txt", "b\n")
    first = (await opened(service, title="Refunds")).session.teams[0].id
    second = (await opened(service, title="Vouchers")).session.teams[0].id
    for team, file in ((first, "services/api/a.txt"), (second, "services/api/b.txt")):
        commit(own_checkout(service, repo, tmp_path, team), file, "changed\n")
        await use(service, ic(service, team), SubmitReviewTool, DONE)
    # The first cannot be merged yet: the human is asked, and it waits.
    (repo / "services" / "api" / "a.txt").write_text("in the way\n")

    # Both were checked before either was carried out.
    for team in (first, second):
        service.check_accept(team, "")
        service.accept_review(team, "")
    await service.settled()

    states = [node(service, team).review for team in (first, second)]
    assert [review.state for review in states if review] == ["pending", "pending"]


async def test_a_session_whose_ic_could_not_be_started_leaves_nothing_in_the_history(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    before = {event.id for event in service.org_history(100).events}
    tasks = service._runs().marked(TASK)
    launcher.root_error = Unprocessable("cannot start")

    with pytest.raises(Unprocessable):
        await service.open_session(
            OpenSessionRequest(text="Add refunds", node_id="root"),
            root_conversation(repo),
        )
    await service.settled()
    launcher.root_error = None

    assert [
        event for event in service.org_history(100).events if event.id not in before
    ] == []
    assert service._runs().marked(TASK) == tasks
    await service.aclose()


async def test_no_team_is_set_up_in_a_session_that_is_being_closed(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await woken(service, "root")
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    root_ic, root_manager = ic(service), manager(service)
    assert root_ic is not None and root_manager is not None
    # Both agents are in a turn as the human closes the session.
    launcher.report(root_ic, ConversationExecutionStatus.RUNNING)
    launcher.report(root_manager, ConversationExecutionStatus.RUNNING)
    await service.settled()
    [session] = service.org.sessions()
    places = {place.id for place in service.org.session(session.id)}
    service.close_session(session.id)
    await service.settled()

    # In the turn it is finishing, the Manager approves the sub-teams.
    await use(service, root_manager, ApproveTeamsTool, APPROVE)
    await service.settled()

    assert {place.id for place in service.org.session(session.id)} == places
    launcher.report(root_manager, ConversationExecutionStatus.FINISHED)
    launcher.report(root_ic, ConversationExecutionStatus.FINISHED)
    await service.settled()
    assert service.sessions() == []
    await service.aclose()


async def test_a_close_cut_off_records_no_end_of_a_session_the_org_never_knew(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    opened = await first.open_session(
        OpenSessionRequest(text="Add refunds", node_id="root")
    )
    await first.settled()
    ended = asyncio.Event()
    real = first.org.close_session

    def server_ends(session_id: str) -> bool:
        ended.set()
        raise ServerEnded

    first.org.close_session = server_ends  # type: ignore[method-assign]

    async def closing() -> None:
        first.close_session(opened.session.id)
        await first.settled()

    await cut_off(launcher, ended, closing())
    first.org.close_session = real  # type: ignore[method-assign]

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    assert all(session.id != opened.session.id for session in second.sessions())
    kinds = [event.kind for event in second.team_memory.history("root")]
    assert "session_closed" not in kinds
    await second.aclose()
