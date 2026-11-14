"""What an agent is told of its team's context without asking: all of it in
its briefing, and what changed with the next message it gets anyway."""

import json
from pathlib import Path
from typing import Any

import pytest

from openhands.agent_server.director.context import (
    UPDATE_HEADING,
    changed,
    steady,
)
from openhands.agent_server.director.models import RouteMessageRequest
from openhands.agent_server.director.run_state import TOLD
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    MsgIcTool,
    PruneTeamTool,
    RaiseEscalationTool,
    ReportTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgIcAction,
    PruneTeamAction,
    RaiseEscalationAction,
    ReportAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import FakeLauncher
from tests.agent_server.director.test_director_pause import again
from tests.agent_server.director.test_director_review import DONE, commit
from tests.agent_server.director.test_director_teams import started
from tests.agent_server.director.test_director_tools import (
    ic,
    manager,
    split_root,
    use,
)


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


def context(**changes: Any) -> dict[str, Any]:
    return {
        "status": "idle",
        "sub_teams": [
            {"node_id": "api", "status": "working", "headline": "Writes the API."}
        ],
        "reports_of_your_team": [],
        "escalations_you_raised": [],
        "review_of_your_work": None,
        **changes,
    }


def update_of(message: str) -> dict[str, Any] | None:
    """What a message says changed in the context, if it says so."""
    _, heading, block = message.partition(UPDATE_HEADING)
    return json.loads(block[block.index("{") :]) if heading else None


async def tell(service: DirectorService, launcher: FakeLauncher, to: str) -> str:
    """Send the IC of a team a message from the human; what it got."""
    await service.route_message(RouteMessageRequest(text="Go on.", path=to))
    await service.settled()
    return launcher.messages_to(ic(service, "apps.web" if to else "root"))[-1]


def test_a_team_that_starts_or_ends_a_turn_has_not_changed():
    working = context(status="working")
    idle = context(
        sub_teams=[{"node_id": "api", "status": "idle", "headline": "Waits."}]
    )

    assert steady(working) == steady(idle)
    assert "headline" not in steady(idle)["sub_teams"][0]
    blocked = context(sub_teams=[{"node_id": "api", "status": "blocked"}])
    assert changed(steady(idle), steady(blocked), "ic") == {
        "sub_teams": [{"node_id": "api", "status": "blocked"}]
    }


def test_only_what_differs_is_told_and_of_a_list_that_grew_only_what_was_added():
    first = {"by": "manager", "text": "Schema first."}
    before = context(reports_of_your_team=[first])
    now = context(
        reports_of_your_team=[first, {"by": "manager", "text": "Then the API."}],
        review_of_your_work={"state": "accepted"},
    )

    assert changed(before, before, "ic") == {}
    assert changed(before, now, "ic") == {
        "reports_of_your_team": {"added": [{"by": "manager", "text": "Then the API."}]},
        "review_of_your_work": {"state": "accepted"},
    }
    # Rewritten, not added to: all of it.
    rewritten = context(reports_of_your_team=[{"by": "ic", "text": "In short."}])
    assert changed(before, rewritten, "manager") == {
        "reports_of_your_team": [{"by": "ic", "text": "In short."}]
    }


def test_an_agent_is_not_told_back_what_it_added_itself():
    before = context()
    now = context(
        reports_of_your_team=[
            {"by": "ic", "text": "Mine."},
            {"by": "manager", "text": "Theirs."},
        ],
        escalations_you_raised=[{"escalation_id": "jwt"}],
    )

    assert changed(before, now, "ic") == {
        "reports_of_your_team": {"added": [{"by": "manager", "text": "Theirs."}]}
    }


async def test_a_briefing_carries_the_teams_whole_context(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)

    [start] = [s for s in launcher.derived if "IC of the team 'Web'" in s.task]

    assert "Contracts your team must keep" in start.task
    assert "read_node_context` is not needed in normal work" in start.task
    block = start.task[start.task.index("lost track.\n") :]
    rest = json.loads(block[block.index("{") : block.rindex("}") + 1])
    assert rest["you"] == {"role": "ic", "node_id": "apps.web", "team": "Web"}
    assert rest["team_above"] == "root" and rest["sub_teams"] == []
    # Said in the briefing's own words above, so not a second time.
    assert "spec" not in rest and "boundary" not in rest
    await service.aclose()


async def test_a_change_sends_nothing_and_comes_with_the_next_message(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await tell(service, launcher, "apps/web")
    heard = len(launcher.messages_to(ic(service, "apps.web")))

    await use(
        service,
        manager(service, "apps.web"),
        ReportTool,
        ReportAction(text="The router is server-side."),
    )

    assert len(launcher.messages_to(ic(service, "apps.web"))) == heard

    first = await tell(service, launcher, "apps/web")
    second = await tell(service, launcher, "apps/web")

    assert first.startswith("Go on.\n\n[director] context update")
    told = update_of(first)
    assert told is not None and list(told) == ["reports_of_your_team"]
    [report] = told["reports_of_your_team"]["added"]
    assert (report["by"], report["text"]) == ("manager", "The router is server-side.")
    assert second == "Go on."
    await service.aclose()


async def test_teams_that_start_and_end_turns_cause_no_update(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await tell(service, launcher, "")
    web = ic(service, "apps.web")
    assert web is not None

    launcher.report(web, ConversationExecutionStatus.RUNNING)
    working = await tell(service, launcher, "")
    launcher.report(web, ConversationExecutionStatus.FINISHED)
    idle = await tell(service, launcher, "")

    assert (working, idle) == ("Go on.", "Go on.")
    await service.aclose()


async def test_reviews_and_escalations_below_come_with_what_wakes_the_ic(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await tell(service, launcher, "")
    commit(repo, "services/api/app.py", "order_id = 1\n")
    await use(service, ic(service, "services.api"), SubmitReviewTool, DONE)

    await use(
        service,
        ic(service, "apps.web"),
        RaiseEscalationTool,
        RaiseEscalationAction(category="design_decision", title="Which router?"),
    )

    escalation = launcher.messages_to(ic(service))[-2]
    told = update_of(escalation)
    assert told is not None
    [to_settle] = told["escalations_for_you_to_settle"]["added"]
    assert to_settle["title"] == "Which router?"
    by_team = {team["node_id"]: team for team in told["sub_teams"]}
    assert by_team["apps.web"]["status"] == "blocked"
    assert by_team["services.api"]["review"] == "pending"
    # The Web IC raised it itself and is not told so.
    web = await tell(service, launcher, "apps/web")
    assert "escalations_you_raised" not in (update_of(web) or {})
    await service.aclose()


async def test_what_an_agent_was_told_outlasts_the_server(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    await tell(first, launcher, "")
    await first.aclose()

    second = await again(repo, launcher)
    await second.resume()

    assert await tell(second, launcher, "") == "Go on."
    await second.aclose()


async def test_an_agent_never_told_its_context_gets_all_of_it_once(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    root = ic(service)
    assert root is not None
    service._runs().unmark(TOLD, root)

    first = await tell(service, launcher, "")
    second = await tell(service, launcher, "")

    told = update_of(first)
    assert told is not None and told["you"]["node_id"] == "root"
    assert "boundary" in told and second == "Go on."
    await service.aclose()


async def test_what_the_agents_of_a_removed_team_were_told_is_forgotten(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    gone = {ic(service, "services.api"), manager(service, "services.api")}
    assert gone <= set(service._runs().marked(TOLD))

    await use(
        service,
        manager(service),
        PruneTeamTool,
        PruneTeamAction(node_id="services.api", reason="No way forward."),
    )

    assert not gone & set(service._runs().marked(TOLD))
    assert ic(service, "apps.web") in service._runs().marked(TOLD)
    await service.aclose()


async def test_the_directors_own_notices_bring_what_changed_too(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)

    await use(service, manager(service), MsgIcTool, MsgIcAction(text="Carry on."))

    approved, written = launcher.messages_to(ic(service))[-2:]
    assert approved.startswith("[director] Your Manager approved your sub-teams")
    told = update_of(approved)
    assert told is not None
    assert [team["node_id"] for team in told["sub_teams"]["added"]] == [
        "services.api",
        "apps.web",
    ]
    # Nothing changed between the two, so the second is the message alone.
    assert written == "[director] Your Manager writes:\nCarry on."
    await service.aclose()
