"""Tests for what an IC asks of a team that is not above it, directly."""

from pathlib import Path
from typing import Literal

import pytest

from openhands.agent_server.director.models import SplitChild
from openhands.agent_server.director.prompts import NEED_ASKED
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    RaiseEscalationTool,
    ResolveEscalationTool,
    StopTeamTool,
)
from openhands.agent_server.director.tools.definition import (
    RaiseEscalationAction,
    ResolveEscalationAction,
    StopTeamAction,
)
from tests.agent_server.director.conftest import FakeLauncher
from tests.agent_server.director.test_director_landing import landed
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_stalls import Clock
from tests.agent_server.director.test_director_teams import (
    API,
    DB,
    propose_and_approve,
)
from tests.agent_server.director.test_director_tools import (
    ic,
    node,
    split_root,
    use,
)


WEB = "apps.web"
TO_WEB = f"Teams: {WEB}\nReason: The cart page is Web's."
TO_DB = f"Teams: {DB}\nReason: It is about the schema."
TO_API = f"Teams: {API}\nReason: It is the API's."


def need(title: str = "The cart must send the order id") -> RaiseEscalationAction:
    return RaiseEscalationAction(
        category="design_decision",
        title=title,
        detail="POST /checkout needs it.",
        team_for="The cart page should send order_id with the form.",
    )


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher, clock: Clock):
    """An org of API (with DB below it) and Web, which the first session
    made and whose work is accepted, every team started."""
    service = await started(repo, launcher, clock=clock, stall_poll=3600)
    await split_root(service)
    await propose_and_approve(
        service,
        API,
        SplitChild(title="DB", path="services/api/db", task="Own the schema"),
    )
    await landed(service)
    yield service
    await service.aclose()


async def test_a_request_in_the_words_of_the_asking_team_still_goes_direct(
    service: DirectorService, launcher: FakeLauncher
):
    """A team asks in the words of its own work, so its own card shares
    most with what it asks. That is no doubt about whose the request is."""
    launcher.answers[USAGE_ID] = TO_WEB
    asked = RaiseEscalationAction(
        category="design_decision",
        title="Which schema column holds it?",
        team_for="Own the schema: the schema needs the order id from the form.",
    )

    await use(service, ic(service, DB), RaiseEscalationTool, asked)

    assert the_one(service).asked_of == node(service, WEB).id


def the_one(service: DirectorService, node_id: str = DB):
    [escalation] = node(service, node_id).escalations
    return escalation


def kinds(service: DirectorService, above: str = "root") -> list[str]:
    return [event.kind for event in service.team_memory.history(above)]


async def answer(
    service: DirectorService,
    by: str,
    action: Literal["settle", "pass_up"],
    text: str,
):
    return await use(
        service,
        ic(service, by),
        ResolveEscalationTool,
        ResolveEscalationAction(
            node_id=node(service, DB).id,
            escalation_id=the_one(service).id,
            action=action,
            text=text,
        ),
    )


async def test_what_an_ic_needs_of_another_team_goes_straight_to_its_ic(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_WEB
    above = {
        team: len(launcher.messages_to(ic(service, team))) for team in ("root", API)
    }

    raised = await use(service, ic(service, DB), RaiseEscalationTool, need())

    assert "takes it to the IC of the team it concerns" in raised.text
    escalation = the_one(service)
    assert (escalation.asked_of, escalation.held_by) == (
        node(service, WEB).id,
        node(service, API).id,
    )
    assert node(service, DB).status == "blocked"
    asked = launcher.messages_to(ic(service, WEB))[-1]
    assert "asks your team directly 'The cart must send the order id'" in asked
    assert "The cart page should send order_id with the form." in asked
    assert "It is a request, not a task" in asked
    # Neither IC above took a turn for it.
    for team, before in above.items():
        assert len(launcher.messages_to(ic(service, team))) == before
    # It is with Web's IC and no longer with the IC by the chain.
    [mine] = service.context_of(node(service, WEB), "ic")[
        "escalations_for_you_to_settle"
    ]
    assert mine["asked_of_you_directly"] is True
    assert (
        service.context_of(node(service, API), "ic")["escalations_for_you_to_settle"]
        == []
    )

    settled = await answer(service, WEB, "settle", "The form sends order_id now.")

    assert "Settled" in settled.text
    assert node(service, DB).escalations == []
    told = launcher.messages_to(ic(service, DB))[-1]
    assert "The IC of 'Web', which the runtime asked directly, settled" in told
    assert "The form sends order_id now." in told


async def test_the_team_above_both_keeps_sight_of_what_they_asked_of_each_other(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_WEB
    await use(service, ic(service, DB), RaiseEscalationTool, need())

    assert kinds(service)[-2:] == ["asked_team", "needed"]
    counted = service.team_memory.history("root")[-1]
    # Between the two teams directly below, which is what a merge would join.
    assert (counted.team_id, counted.other_id, counted.detail) == (API, WEB, NEED_ASKED)
    [row] = service.context_of(node(service, "root"), "ic")[
        "what_teams_below_asked_of_each_other_directly"
    ]
    assert (row["from"], row["to"], row["asked"], row["open"]) == ("DB", "Web", 1, 1)

    await answer(service, WEB, "settle", "Done.")

    assert kinds(service)[-1] == "answered"
    [row] = service.context_of(node(service, "root"), "ic")[
        "what_teams_below_asked_of_each_other_directly"
    ]
    assert (row["settled"], row["open"]) == (1, 0)


async def test_a_team_that_cannot_settle_it_gives_it_to_the_chain(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_WEB
    await use(service, ic(service, DB), RaiseEscalationTool, need())

    by_api = await answer(service, API, "settle", "Mine.")
    passed = await answer(service, WEB, "pass_up", "The contract has no order id.")

    assert by_api.is_error and "is not with your team" in by_api.text
    assert "now with the IC of the team above the one that asked" in passed.text
    escalation = the_one(service)
    assert (escalation.asked_of, escalation.held_by) == (None, node(service, API).id)
    told = launcher.messages_to(ic(service, API))[-1]
    assert "escalates 'The cart must send the order id'" in told
    assert (
        "Passed up by Web, which was asked directly (passed up): "
        "The contract has no order id."
    ) in told
    [row] = service.context_of(node(service, "root"), "ic")[
        "what_teams_below_asked_of_each_other_directly"
    ]
    assert (row["passed_up"], row["open"]) == (1, 0)


async def test_an_unsure_choice_goes_up_the_chain_as_before(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = [TO_WEB, TO_DB]

    await use(service, ic(service, DB), RaiseEscalationTool, need())

    assert the_one(service).asked_of is None
    assert (
        "escalates 'The cart must send the order id'"
        in (launcher.messages_to(ic(service, API))[-1])
    )
    assert "asked_team" not in kinds(service)


async def test_a_team_above_the_asking_one_is_reached_by_the_chain(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_API

    await use(service, ic(service, DB), RaiseEscalationTool, need())

    assert the_one(service).asked_of is None
    assert "escalates" in launcher.messages_to(ic(service, API))[-1]


async def test_two_teams_never_wait_on_each_other(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = [TO_WEB, TO_WEB, TO_DB, TO_DB]
    await use(service, ic(service, DB), RaiseEscalationTool, need())

    await use(service, ic(service, WEB), RaiseEscalationTool, need("Need a column"))

    assert the_one(service, WEB).asked_of is None
    assert "escalates 'Need a column'" in launcher.messages_to(ic(service))[-1]


async def test_a_stopped_team_is_not_asked(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_WEB
    await use(
        service,
        ic(service),
        StopTeamTool,
        StopTeamAction(team=WEB, reason="Wait for the design."),
    )

    await use(service, ic(service, DB), RaiseEscalationTool, need())

    assert the_one(service).asked_of is None


async def test_no_more_than_three_requests_are_open_with_one_team(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_WEB

    for number in range(4):
        await use(service, ic(service, DB), RaiseEscalationTool, need(f"Need {number}"))

    asked = [e.asked_of for e in node(service, DB).escalations]
    web = node(service, WEB).id
    assert asked == [web, web, web, None]
    # At the org's setbacks number the IC above both is woken, once.
    woken = [
        text
        for text in launcher.messages_to(ic(service))
        if "has now asked team 'Web' directly 3 times" in text
    ]
    assert len(woken) == 1


async def test_a_request_nobody_answers_goes_to_the_chain_by_itself(
    service: DirectorService, launcher: FakeLauncher, clock: Clock
):
    launcher.answers[USAGE_ID] = TO_WEB
    await use(service, ic(service, DB), RaiseEscalationTool, need())

    clock.passes(119)
    await service._check_asks()
    await service.settled()
    assert the_one(service).asked_of == node(service, WEB).id

    clock.passes(1)
    await service._check_asks()
    await service.settled()

    assert the_one(service).asked_of is None
    told = launcher.messages_to(ic(service, API))[-1]
    assert "Web, which was asked directly (not answered): no answer in 120 min" in told
    [row] = service.context_of(node(service, "root"), "ic")[
        "what_teams_below_asked_of_each_other_directly"
    ]
    assert (row["not_answered"], row["open"]) == (1, 0)


async def test_without_team_for_or_with_routing_off_nothing_changes(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_WEB
    plain = need().model_copy(update={"team_for": ""})

    await use(service, ic(service, DB), RaiseEscalationTool, plain)

    assert the_one(service).asked_of is None
    assert launcher.asked == []
