"""Tests for what a session's accepted work makes the org's: its
sub-teams, its specs and what its agents recorded."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    ContractsChoice,
    DirectorSettingsUpdate,
    OpenSessionRequest,
    OrgNodeInput,
    Report,
    RouteMessageRequest,
    SessionTeam,
    SessionTeamInput,
)
from openhands.agent_server.director.org_store import session_id_of, team_id_of
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import (
    MERGE_WAITS,
    agent_conversation,
)
from openhands.agent_server.director.tools import (
    PruneTeamTool,
    RaiseEscalationTool,
    ReportTool,
    ReviewTeamTool,
    SettleTeamsTool,
    SubmitReviewTool,
    WriteSpecTool,
)
from openhands.agent_server.director.tools.definition import (
    RaiseEscalationAction,
    ReportAction,
    ReviewTeamAction,
    SettleTeamsAction,
    WriteSpecAction,
)
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import (
    FakeLauncher,
    record_of,
    run_git,
    take_in,
)
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_review import DONE, accept, commit
from tests.agent_server.director.test_director_tools import (
    PRUNE_API,
    ic,
    manager,
    node,
    review_place_of,
    reviewer,
    split_root,
    use,
)


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    service = await started(repo, launcher)
    yield service
    await service.aclose()


async def landed(service: DirectorService) -> None:
    """Have the work of the session opened first reviewed and accepted."""
    await use(service, ic(service), SubmitReviewTool, DONE)
    await use(
        service, reviewer(service), ReviewTeamTool, accept(node(service, "root").id)
    )


async def test_the_org_has_a_sessions_sub_teams_once_its_work_is_accepted(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)

    assert [team.id for team in service.store.teams()] == ["root"]
    assert reviewer(service) is None

    await use(service, ic(service), SubmitReviewTool, DONE)

    # Its reviewer is the top team's IC in review mode, started for this.
    asked = reviewer(service)
    assert asked is not None
    start = launcher.derived[-1]
    assert str(start.conversation_id) == asked
    assert "submits its work" in start.task and "review mode" in start.suffixes[0]
    assert start.title == "Checkout (Reviewer)"
    assert start.checkout == node(service, review_place_of(service)).worktree_path

    await use(service, asked, ReviewTeamTool, accept(node(service, "root").id))

    assert [team.id for team in service.store.teams()] == [
        "root",
        "services.api",
        "apps.web",
    ]
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"
    assert service.store.place(review_place_of(service)) is not None


async def test_what_a_session_records_is_the_teams_once_its_work_is_accepted(
    service: DirectorService,
):
    await use(service, ic(service), ReportTool, ReportAction(text="Chose REST."))

    assert [r.text for r in service.reports_of(node(service, "root"))] == [
        "Chose REST."
    ]
    assert service.team_memory.reports("root", "ic") == []

    await landed(service)

    assert [r.text for r in service.team_memory.reports("root", "ic")] == [
        "Chose REST."
    ]
    assert service.own_reports(node(service, "root")) == []


async def test_what_became_of_a_sessions_teams_is_on_record_once_it_is_accepted(
    service: DirectorService,
):
    await split_root(service)
    top = node(service, "root")

    own = service.history_of(top.id)
    assert "set_up" in [event.kind for event in own]
    assert all(event.unaccepted for event in own)
    assert service.team_memory.history("root") == []

    await landed(service)

    kept = service.team_memory.history("root")
    assert {"set_up", "accepted"} <= {event.kind for event in kept}
    assert not any(event.unaccepted for event in kept)
    assert service.own_events(top.session, "root") == []


async def test_what_a_reviewer_records_is_the_teams_once_the_work_is_accepted(
    service: DirectorService,
):
    await use(service, ic(service), SubmitReviewTool, DONE)
    asked = reviewer(service)
    assert asked is not None
    await use(service, asked, ReportTool, ReportAction(text="Read the diff twice."))

    assert service.team_memory.reports("root", "ic") == []

    await use(service, asked, ReviewTeamTool, accept(node(service, "root").id))

    assert [r.text for r in service.team_memory.reports("root", "ic")] == [
        "Read the diff twice."
    ]


async def test_what_agents_write_to_each_other_is_never_the_orgs(
    service: DirectorService, repo: Path
):
    top = node(service, "root")
    service.message(top.id, top.id, "ic", "Split in two?")

    await landed(service)

    [thread] = service.threads_of(node(service, "root"))
    assert [message.text for message in thread.messages] == ["Split in two?"]
    kept = run_git(repo, "ls-tree", "-r", "--name-only", "refs/agents/memory")
    assert ".threads" not in kept


async def test_the_end_of_a_session_the_org_knows_is_on_its_record(
    service: DirectorService,
):
    await landed(service)

    service.close_session(node(service, "root").session)
    await service.settled()

    kinds = [event.kind for event in service.team_memory.history("root")]
    assert kinds.count("session_opened") == kinds.count("session_closed") == 1


def own_worktree(service: DirectorService, repo: Path, tmp_path: Path) -> Path:
    """Give the session opened first the worktree its conversation would
    have made."""
    worktree = tmp_path / "session"
    run_git(repo, "worktree", "add", "--quiet", "-b", "session", str(worktree))
    for place in service.store.session(session_id_of(node(service, "root"))):
        if not place.review_only:
            service.store.update_place(
                place.id, worktree_path=str(worktree), branch="session"
            )
    return worktree


async def test_accepted_work_is_merged_into_the_repositorys_checkout(
    service: DirectorService, repo: Path, tmp_path: Path
):
    worktree = own_worktree(service, repo, tmp_path)
    commit(worktree, "app.py", "refunds = True\n")

    await landed(service)

    assert (repo / "app.py").read_text() == "refunds = True\n"
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"


async def test_the_human_is_asked_when_their_uncommitted_work_is_in_the_way(
    service: DirectorService, repo: Path, tmp_path: Path
):
    commit(repo, "app.py", "refunds = False\n")
    worktree = own_worktree(service, repo, tmp_path)
    commit(worktree, "app.py", "refunds = True\n")
    (repo / "app.py").write_text("refunds = None\n")

    await landed(service)

    top = node(service, "root")
    assert top.review is not None and top.review.state == "pending"
    [asked] = top.escalations
    assert (asked.id, asked.held_by) == (MERGE_WAITS, None)
    assert (repo / "app.py").read_text() == "refunds = None\n"

    run_git(repo, "checkout", "--", "app.py")
    service.answer_escalation(top.id, MERGE_WAITS, "Cleared.")
    await service.settled()

    assert (repo / "app.py").read_text() == "refunds = True\n"
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"


async def test_a_team_a_session_set_up_is_asked_within_that_session_only(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    launcher.answers[USAGE_ID] = "Teams: apps.web\nReason: The cart page is Web's."
    need = RaiseEscalationAction(
        category="design_decision",
        title="The cart must send the order id",
        detail="POST /checkout needs it.",
        team_for="The cart page should send order_id with the form.",
    )

    await use(service, ic(service, "services.api"), RaiseEscalationTool, need)
    await service.settled()

    # The org does not have Web yet; the session that set it up does.
    assert service.store.team("apps.web") is None
    [raised] = node(service, "services.api").escalations
    assert raised.asked_of == node(service, "apps.web").id
    assert (
        "asks your team directly" in launcher.messages_to(ic(service, "apps.web"))[-1]
    )

    # A new request of the human's is for the org's own teams: it cannot
    # go to a team only a session has.
    sent = await service.route_message(
        RouteMessageRequest(text="Make the cart page faster.", auto=True)
    )
    assert team_id_of(node(service, sent.node_id)) == "root"


async def test_what_the_human_says_in_a_session_is_routed_among_its_own_teams(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    launcher.answers[USAGE_ID] = "Teams: apps.web\nReason: The cart page is Web's."
    [session] = service.store.sessions()

    said = await service.route_message(
        RouteMessageRequest(text="Make the cart page faster.", session_id=session.id)
    )
    await service.settled()

    # Web is a team only this session has. No session was opened for it.
    assert [open_now.id for open_now in service.store.sessions()] == [session.id]
    assert (said.node_id, said.chosen_by) == (node(service, "apps.web").id, "model")
    assert said.conversation_id == ic(service, "apps.web")
    assert launcher.messages_to(said.conversation_id)[-1] == (
        "Make the cart page faster."
    )
    # Its IC may say it is not Web's, which the IC above then has.
    handed = await use(
        service,
        said.conversation_id,
        RaiseEscalationTool,
        RaiseEscalationAction(category="misrouted", title="Not ours", detail="API's."),
    )
    assert "Handed to the IC of the team above" in handed.text
    assert "Make the cart page faster." in launcher.messages_to(ic(service))[-1]

    service.update_settings(DirectorSettingsUpdate(route_requests=False))
    kept = await service.route_message(
        RouteMessageRequest(text="Make the cart page lighter.", session_id=session.id)
    )
    assert (kept.node_id, kept.conversation_id) == (
        node(service, "root").id,
        ic(service),
    )
    with pytest.raises(NotFound):
        await service.route_message(
            RouteMessageRequest(text="Anyone?", session_id="s000000")
        )


async def at_api(service: DirectorService) -> tuple[str, str]:
    """A session opened at API once the org has it.

    Returns:
        The place it was opened at and its IC's conversation.
    """
    await split_root(service)
    await landed(service)
    opened = await service.open_session(
        OpenSessionRequest(text="Add refunds.", node_id="services.api")
    )
    await service.settled()
    return opened.session.teams[0].id, opened.conversation_id


async def test_work_is_accepted_and_a_team_the_org_lost_meanwhile_stays_gone(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    await use(
        service,
        conversation,
        ReportTool,
        ReportAction(text="Refunds are in."),
    )
    service.store.remove("services.api")

    await use(service, conversation, SubmitReviewTool, DONE)
    accepted = await use(service, reviewer(service, top), ReviewTeamTool, accept(top))

    # The session set up no teams of its own, so nothing is asked and the
    # team stays gone; its code is merged all the same.
    assert "Accepting the work of 'API'" in accepted.text
    assert service.store.team("services.api") is None
    place = service.store.place(top)
    assert place is not None and place.review is not None
    assert place.review.state == "accepted"
    assert service.own_reports(place) == []
    # What it recorded is not dropped with the team: it is that team's
    # record, with the nearest team that holds its directory.
    [left] = service.team_memory.former("services.api")
    assert (left.ended, left.above) == ("not_taken", "root")
    assert [report.text for report in left.reports] == ["Refunds are in."]
    assert service.team_memory.reports("services.api", "ic") == []
    assert "not_taken" in [e.kind for e in record_of(service, "root")]


JOBS = "services.api.jobs"


def sets_up_jobs(service: DirectorService, top: str, path: str = "services/api/jobs"):
    """Have a session set up a team for a directory under the team it was
    opened at."""
    place = service.store.place(top)
    assert isinstance(place, SessionTeam)
    service.store.upsert(
        SessionTeamInput(
            **place.model_dump(include={"session", "title", "isolation", "status"}),
            worktree_path=place.worktree_path,
            branch=place.branch,
            id=f"{JOBS}@{place.session}",
            team_id=JOBS,
            parent_id=top,
            agents=[],
            boundary=place.boundary.model_copy(update={"path": path}),
            since="set up by the session",
        )
    )


def org_gains(service: DirectorService, team_id: str, path: str) -> None:
    """Have other accepted work give the org a team under API."""
    api = service.store.team("services.api")
    assert api is not None
    service.store.upsert(
        OrgNodeInput(
            **api.model_dump(include={"title", "isolation"}),
            id=team_id,
            parent_id=api.id,
            boundary=api.boundary.model_copy(update={"path": path}),
            since="set up by other work",
        )
    )


def waiting(service: DirectorService, top: str) -> list[str]:
    """The teams of a session that wait to be settled."""
    place = service.store.place(review_place_of(service, top))
    assert isinstance(place, SessionTeam)
    return [team.team_id for team in place.org_change.teams] if place.org_change else []


def review_manager(service: DirectorService, top: str) -> str | None:
    place = service.store.place(review_place_of(service, top))
    assert place is not None
    return agent_conversation(place, "manager")


def path_of(service: DirectorService, team_id: str) -> str | None:
    team = service.store.team(team_id)
    return team.boundary.path if team else None


async def accepted(service: DirectorService, top: str, conversation: str) -> str:
    """Have a session's work submitted and accepted.

    Returns:
        What its reviewer was answered.
    """
    await use(service, conversation, SubmitReviewTool, DONE)
    answer = await use(service, reviewer(service, top), ReviewTeamTool, accept(top))
    await service.settled()
    return answer.text


async def test_a_stale_session_adds_its_team_and_keeps_what_the_org_gained(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, "services.api.cache", "services/api/cache")

    answer = await accepted(service, top, conversation)

    # Each team is settled on its own, and nobody is asked.
    assert "Accepting the work of 'API'" in answer
    assert path_of(service, JOBS) == "services/api/jobs"
    assert path_of(service, "services.api.cache") == "services/api/cache"
    assert waiting(service, top) == []
    assert review_manager(service, top) is None

    # What the session did not change is not brought up again either.
    service.store.remove("services.api.cache")
    await use(service, conversation, SubmitReviewTool, DONE)
    await use(service, reviewer(service, top), ReviewTeamTool, accept(top))
    await service.settled()
    assert path_of(service, JOBS) == "services/api/jobs"
    assert service.store.team("services.api.cache") is None


@pytest.mark.parametrize("org_holds", [True, False])
async def test_a_team_both_made_something_else_of_waits_for_the_manager_above(
    service: DirectorService, launcher: FakeLauncher, repo: Path, org_holds: bool
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    commit(repo, "services/api/refunds.py", "refunds = True\n")
    place = service.store.place(f"{JOBS}@{top.split('@')[1]}")
    assert place is not None
    service.write_spec(place, "Run the refund jobs.")

    answer = await accepted(service, top, conversation)

    # The code is merged whatever becomes of the team; the org's stays.
    assert "Accepting the work of 'API'" in answer
    review = node(service, top).review
    assert review is not None and review.state == "accepted"
    assert path_of(service, JOBS) == "services/api/workers"
    assert waiting(service, top) == [JOBS]
    asked = review_manager(service, top)
    assert asked is not None
    start = launcher.derived[-1]
    assert str(start.conversation_id) == asked
    assert "Manager of your team in review mode" in "".join(start.suffixes or [])
    told = launcher.messages_to(asked)[-1]
    assert "the session: 'API' in services/api/jobs, under services.api" in told
    assert "the org now: 'API' in services/api/workers, under services.api" in told
    # What the session wrote for it waits with it.
    assert service.spec_draft(place) is not None

    by_ic = await use(
        service, reviewer(service, top), SettleTeamsTool, SettleTeamsAction()
    )
    unknown = await use(
        service, asked, SettleTeamsTool, SettleTeamsAction(org_teams=["apps.web"])
    )
    settled = await use(
        service,
        asked,
        SettleTeamsTool,
        SettleTeamsAction(org_teams=[JOBS] if org_holds else [], note="One queue."),
    )
    again = await use(service, asked, SettleTeamsTool, SettleTeamsAction())

    assert by_ic.is_error and unknown.is_error and again.is_error
    assert f"The teams that wait: {JOBS}" in unknown.text
    assert not settled.is_error
    assert waiting(service, top) == []
    assert path_of(service, JOBS) == (
        "services/api/workers" if org_holds else "services/api/jobs"
    )
    assert service.spec_draft(place) is None
    said = launcher.messages_to(conversation)[-1]
    assert "One queue." in said
    assert ("The org's teams stay" in said) is org_holds
    # What the session wrote for the team is never dropped: it is the
    # team's spec where the org took the team, and that team's record with
    # the team above where it did not.
    written = service.team_memory.read("services/api/jobs/spec.md")
    assert written == (None if org_holds else "Run the refund jobs.")
    kept = service.team_memory.former(JOBS)
    assert [(left.ended, left.above, left.spec) for left in kept] == (
        [("not_taken", "services.api", "Run the refund jobs.")] if org_holds else []
    )


async def test_a_team_set_up_under_one_the_org_lost_goes_under_the_nearest_team(
    service: DirectorService, launcher: FakeLauncher
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    service.hand_over(api_place(service, top), "", IDEMPOTENT)
    service.store.remove("services.api")

    answer = await accepted(service, top, conversation)

    assert "Accepting the work of 'API'" in answer
    assert service.store.team("services.api") is None
    assert waiting(service, top) == [JOBS]
    asked = review_manager(service, top)
    assert asked is not None
    told = launcher.messages_to(asked)[-1]
    assert "it goes under team root" in told and "is not put back" in told
    # The contracts the lost team set for it come with it, and nothing of
    # them is the org's before it is settled.
    assert f"they come with team {JOBS}" in told and IDEMPOTENT in told
    before = service.team_memory.read("contracts.md")
    assert before is not None and IDEMPOTENT not in before

    await use(service, asked, SettleTeamsTool, SettleTeamsAction())

    # The team the org removed stays removed.
    assert service.store.team("services.api") is None
    jobs = service.store.team(JOBS)
    assert jobs is not None and jobs.parent_id == "root"
    assert path_of(service, JOBS) == "services/api/jobs"
    assert "moved" in [event.kind for event in record_of(service, "root")]
    # They are added to the contracts of the team it went under.
    assert service.team_memory.read("contracts.md") == (
        f"{before.rstrip()}\n\n{IDEMPOTENT}\n"
    )


API_CONTRACTS = "services/api/contracts.md"
API_SPEC = "services/api/spec.md"
IDEMPOTENT = "Jobs are idempotent."


def api_place(service: DirectorService, top: str) -> SessionTeam:
    place = service.store.place(top)
    assert isinstance(place, SessionTeam)
    return place


async def test_contracts_and_notes_a_session_hands_over_are_the_orgs_on_accept(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    place = api_place(service, top)

    service.hand_over(
        place, "The old jobs team retried forever.", "Jobs are idempotent."
    )

    # Until its work is accepted they are the session's alone.
    assert service.team_memory.read(API_CONTRACTS) is None
    assert service.team_memory.reports("services.api", "ic") == []
    assert [c.contracts for c in service.contracts_of(place) if c.own] == [
        "Jobs are idempotent.\n"
    ]
    assert [r.text for r in service.reports_of(place)] == [
        "The old jobs team retried forever."
    ]

    await accepted(service, top, conversation)

    assert service.team_memory.read(API_CONTRACTS) == "Jobs are idempotent.\n"
    [note] = service.team_memory.reports("services.api", "ic")
    assert (note.text, note.handed_over) == ("The old jobs team retried forever.", True)
    assert waiting(service, top) == []


@pytest.mark.parametrize(
    ("take", "holds"),
    [
        ("org", "Orders and carts go through Jobs.\n"),
        ("session", "Refunds go through Jobs.\n"),
        ("text", "Orders, carts and refunds go through Jobs.\n"),
    ],
)
async def test_contracts_both_changed_wait_for_the_manager_above(
    service: DirectorService, launcher: FakeLauncher, take: str, holds: str
):
    top, conversation = await at_api(service)
    service.hand_over(api_place(service, top), "", "Refunds go through Jobs.")
    # Other accepted work set the contracts of the same directory meanwhile.
    service.team_memory.write(
        "services/api", "contracts.md", "Orders go through Jobs.\n"
    )

    answer = await accepted(service, top, conversation)

    # The code is merged; of the contracts the org's stay until they are
    # settled, and nothing is overwritten.
    assert "Accepting the work of 'API'" in answer
    assert service.team_memory.read(API_CONTRACTS) == "Orders go through Jobs.\n"
    asked = review_manager(service, top)
    assert asked is not None
    told = launcher.messages_to(asked)[-1]
    assert "Refunds go through Jobs." in told and "Orders go through Jobs." in told
    assert "the contracts between the sub-teams of services/api" in told

    # What the Manager decided on is held to what it was shown.
    service.team_memory.write(
        "services/api", "contracts.md", "Orders and carts go through Jobs.\n"
    )
    choice = ContractsChoice.model_validate(
        {
            "path": "services/api",
            "take": take,
            "text": "Orders, carts and refunds go through Jobs.",
        }
    )
    unknown = await use(
        service,
        asked,
        SettleTeamsTool,
        SettleTeamsAction(contracts=[ContractsChoice(path="apps/web", take="org")]),
    )
    stale = await use(
        service, asked, SettleTeamsTool, SettleTeamsAction(contracts=[choice])
    )
    assert unknown.is_error and "services/api" in unknown.text
    assert stale.is_error and "changed since you were shown" in stale.text
    assert "Orders and carts go through Jobs." in stale.text
    assert service.team_memory.read(API_CONTRACTS) == (
        "Orders and carts go through Jobs.\n"
    )

    settled = await use(
        service, asked, SettleTeamsTool, SettleTeamsAction(contracts=[choice])
    )

    assert not settled.is_error
    assert service.team_memory.read(API_CONTRACTS) == holds
    place = service.store.place(review_place_of(service, top))
    assert place is not None and place.org_change is None
    # The session reads what holds now, not what it wrote.
    assert [c.contracts for c in service.contracts_of(api_place(service, top))] == [
        holds
    ]


async def test_what_a_reviewer_reconciled_holds_only_for_what_it_was_shown(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    await use(service, conversation, WriteSpecTool, WriteSpecAction(text="# Refunds"))
    service.team_memory.write("services/api", "spec.md", "# Orders")
    await use(service, conversation, SubmitReviewTool, DONE)
    decided = ReviewTeamAction(node_id=top, decision="accept", reconciled="Refunds.")

    # Saying it is reconciled before being shown anything settles nothing.
    unseen = await use(service, reviewer(service, top), ReviewTeamTool, decided)
    assert unseen.is_error and "# Orders" in unseen.text

    # The org's spec changes again before the reviewer answers.
    service.team_memory.write("services/api", "spec.md", "# Orders and carts")
    stale = await use(service, reviewer(service, top), ReviewTeamTool, decided)

    assert stale.is_error
    assert "changed since you were shown it" in stale.text
    assert "# Orders and carts" in stale.text
    assert service.team_memory.read(API_SPEC) == "# Orders and carts"

    again = await use(service, reviewer(service, top), ReviewTeamTool, decided)
    await service.settled()

    assert not again.is_error
    assert service.team_memory.read(API_SPEC) == "# Refunds"


async def test_what_a_session_wrote_for_a_team_that_was_replaced_is_not_the_new_teams(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    await use(service, conversation, WriteSpecTool, WriteSpecAction(text="# Refunds"))
    await use(service, conversation, ReportTool, ReportAction(text="Chose a ledger."))
    # Other accepted work put another team in the same directory, which has
    # the same id.
    service.store.update_team("services.api", since="set up by other work")

    await accepted(service, top, conversation)

    assert service.team_memory.read(API_SPEC) is None
    assert service.team_memory.reports("services.api", "ic") == []
    [left] = service.team_memory.former("services.api")
    # It is kept for the team that has its directory now.
    assert (left.ended, left.above) == ("not_taken", "services.api")
    assert left.spec == "# Refunds"
    assert [report.text for report in left.reports] == ["Chose a ledger."]
    # The session no longer reads the other team's records as its own.
    assert service.reports_of(api_place(service, top)) == []


async def test_a_team_only_the_session_renamed_is_renamed_in_the_org(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    service.store.update_place(top, title="Payments API")

    await accepted(service, top, conversation)

    team = service.store.team("services.api")
    assert team is not None and team.title == "Payments API"
    assert waiting(service, top) == []


async def test_closing_a_session_leaves_the_org_nothing_it_never_had_accepted(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    await use(service, conversation, WriteSpecTool, WriteSpecAction(text="# Refunds"))
    await use(service, conversation, ReportTool, ReportAction(text="Chose a ledger."))
    place = api_place(service, top)

    service.close_session(place.session)
    await service.settled()

    assert service.team_memory.read(API_SPEC) is None
    assert service.team_memory.reports("services.api", "ic") == []
    # Nor is it on the org's record that it was closed.
    assert "session_closed" not in [e.kind for e in record_of(service, "root")]
    assert service.drafts(place.session) == {}


async def test_closing_a_session_keeps_what_waited_of_its_accepted_work(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    jobs = service.store.place(f"{JOBS}@{api_place(service, top).session}")
    assert jobs is not None
    service.write_spec(jobs, "Run the refund jobs.")
    await accepted(service, top, conversation)
    assert waiting(service, top) == [JOBS]

    service.close_session(jobs.session)
    await service.settled()

    # Nobody settled it, so the org's team stays and what the session wrote
    # for its own is that team's record.
    assert path_of(service, JOBS) == "services/api/workers"
    [left] = service.team_memory.former(JOBS)
    assert (left.ended, left.spec) == ("not_taken", "Run the refund jobs.")


async def test_a_team_a_session_prunes_is_the_orgs_until_its_work_is_accepted(
    service: DirectorService,
):
    await split_root(service)
    await landed(service)
    await service.settled()
    kept = Report(at="2026-11-01T00:00:00+00:00", role="ic", text="Chose REST.")
    take_in(service.team_memory, reports={"services.api": [kept]})

    await use(service, manager(service), PruneTeamTool, PRUNE_API)
    await service.settled()

    # Other sessions still have the team and all it recorded.
    assert service.store.team("services.api") is not None
    assert service.team_memory.reports("services.api", "ic") == [kept]
    assert service.team_memory.former("services.api") == []

    await landed(service)
    await service.settled()

    assert service.store.team("services.api") is None
    [record] = service.team_memory.former("services.api")
    assert (record.ended, record.reports) == ("pruned", [kept])
    assert service.team_memory.reports("services.api", "ic") == []
