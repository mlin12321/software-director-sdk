"""Tests for what a session's accepted work makes the org's: its
sub-teams, its specs and what its agents recorded."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.org_store import session_id_of
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import MERGE_WAITS
from openhands.agent_server.director.tools import (
    ReportTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import ReportAction
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
