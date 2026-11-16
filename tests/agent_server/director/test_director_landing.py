"""Tests for what a session's accepted work makes the org's: its
sub-teams, its specs and what its agents recorded."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    DirectorSettingsUpdate,
    OpenSessionRequest,
    RouteMessageRequest,
    SessionTeam,
    SessionTeamInput,
)
from openhands.agent_server.director.org_store import session_id_of, team_id_of
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import MERGE_WAITS
from openhands.agent_server.director.tools import (
    RaiseEscalationTool,
    ReportTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    RaiseEscalationAction,
    ReportAction,
    ReviewTeamAction,
)
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import FakeLauncher, run_git
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_review import DONE, accept, commit
from tests.agent_server.director.test_director_tools import (
    ic,
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
    assert [
        (team.id, team.agents) for team in service.store.teams() if team.agents
    ] == []
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"
    assert service.store.get(review_place_of(service)) is not None


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


def own_worktree(service: DirectorService, repo: Path, tmp_path: Path) -> Path:
    """Give the session opened first the worktree its conversation would
    have made."""
    worktree = tmp_path / "session"
    run_git(repo, "worktree", "add", "--quiet", "-b", "session", str(worktree))
    for place in service.store.session(session_id_of(node(service, "root"))):
        if not place.review_only:
            service.store.update_node(
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

    # The session set up no teams of its own, so nothing is asked and what
    # it has of the org's teams is dropped; its code is merged all the same.
    assert "Accepting the work of 'API'" in accepted.text
    assert service.store.team("services.api") is None
    place = service.store.get(top)
    assert place is not None and place.review is not None
    assert place.review.state == "accepted"
    assert service.own_reports(place) == []


async def test_a_team_the_org_lost_comes_back_with_a_session_that_set_up_teams(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    place = service.store.get(top)
    assert isinstance(place, SessionTeam)
    # The session sets up a sub-team of its own.
    service.store.upsert(
        SessionTeamInput(
            **place.model_dump(include={"session", "title", "isolation", "status"}),
            worktree_path=place.worktree_path,
            branch=place.branch,
            id=f"services.api.jobs@{place.session}",
            team_id="services.api.jobs",
            parent_id=top,
            agents=[],
            boundary=place.boundary.model_copy(update={"path": "services/api/jobs"}),
        )
    )
    service.store.remove("services.api")

    await use(service, conversation, SubmitReviewTool, DONE)
    asked = reviewer(service, top)
    refused = await use(service, asked, ReviewTeamTool, accept(top))

    assert "the org no longer has 'API'" in refused.text
    assert service.store.team("services.api") is None

    await use(
        service,
        asked,
        ReviewTeamTool,
        ReviewTeamAction(node_id=top, decision="accept", reconciled="API stays."),
    )

    back = service.store.team("services.api")
    assert back is not None and (back.parent_id, back.agents) == ("root", [])
    assert service.store.team("services.api.jobs") is not None
