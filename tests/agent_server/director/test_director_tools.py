"""The tools an org's agents use, called as the agents call them."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.director.models import (
    CreateOrgRequest,
    OrgNode,
    SplitChild,
)
from openhands.agent_server.director.prompts import MANAGER_ROLE
from openhands.agent_server.director.service import (
    DirectorService,
    agent_conversation,
)
from openhands.agent_server.director.tools import (
    DecideSplitTool,
    ProposeBoundarySplitTool,
    RaiseEscalationTool,
    ReadNodeContextTool,
    ResolveEscalationTool,
    SetStatusTool,
    WriteSpecTool,
)
from openhands.agent_server.director.tools.definition import (
    DecideSplitAction,
    ProposeBoundarySplitAction,
    RaiseEscalationAction,
    ReadNodeContextAction,
    ResolveEscalationAction,
    SetStatusAction,
    WriteSpecAction,
)
from openhands.agent_server.director.tools.impl import DirectorObservation
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.tool.tool import Action, ToolDefinition
from tests.agent_server.director.conftest import FakeLauncher, root_conversation


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    service = DirectorService(repo, save_delay=0, run_limit_retry=0)
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout",
            task="Build the checkout flow",
            conversation=root_conversation(repo),
        )
    )
    await service.settled()
    yield service
    await service.aclose()


async def use(
    service: DirectorService,
    conversation_id: str | None,
    tool: type[ToolDefinition[Any, Any]],
    action: Action,
) -> DirectorObservation:
    """Call ``tool`` as the agent of ``conversation_id`` does: off the loop."""
    assert conversation_id is not None
    state = cast(ConversationState, SimpleNamespace(id=UUID(conversation_id)))
    executor = tool.create(state)[0].executor
    assert executor is not None
    observation = await asyncio.to_thread(executor, action)
    await service.settled()
    assert isinstance(observation, DirectorObservation)
    return observation


def node(service: DirectorService, node_id: str) -> OrgNode:
    found = service.store.get(node_id)
    assert found is not None
    return found


def ic(service: DirectorService, node_id: str = "root") -> str | None:
    return agent_conversation(node(service, node_id), "ic")


def manager(service: DirectorService, node_id: str = "root") -> str | None:
    return agent_conversation(node(service, node_id), "manager")


SPLIT = ProposeBoundarySplitAction(
    children=[
        SplitChild(title="API", path="services/api", task="Serve the checkout API"),
        SplitChild(
            title="Web",
            path="apps/web",
            task="Build the checkout pages",
            isolation="worktree",
        ),
    ],
    contracts="POST /checkout returns {order_id}",
)


async def split_root(service: DirectorService) -> None:
    await use(service, ic(service), ProposeBoundarySplitTool, SPLIT)
    await use(
        service,
        manager(service),
        DecideSplitTool,
        DecideSplitAction(decision="approve"),
    )


async def test_an_agent_reads_who_it_is_and_what_it_inherits(service: DirectorService):
    service.blackboard.teach("", "Never log card numbers")

    observation = await use(
        service, manager(service), ReadNodeContextTool, ReadNodeContextAction()
    )

    context = json.loads(observation.text)
    assert context["you"] == {"role": "manager", "node_id": "root", "team": "Checkout"}
    assert context["rules"] == ["Never log card numbers"]
    assert context["sub_teams"] == []


async def test_a_conversation_outside_the_org_is_refused(service: DirectorService):
    observation = await use(
        service, str(uuid4()), ReadNodeContextTool, ReadNodeContextAction()
    )

    assert observation.is_error


async def test_the_ic_writes_the_spec_its_sub_teams_inherit(service: DirectorService):
    written = await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(content="# Checkout\n\nAll prices are integer cents."),
    )
    refused = await use(
        service, manager(service), WriteSpecTool, WriteSpecAction(content="# Mine")
    )

    assert not written.is_error and refused.is_error
    assert node(service, "root").spec_summary == "Checkout"
    [spec] = service.blackboard.inherited("services/api")
    assert spec.readonly and "integer cents" in spec.spec


async def test_a_proposed_split_waits_for_the_manager(
    service: DirectorService, launcher: FakeLauncher
):
    observation = await use(service, ic(service), ProposeBoundarySplitTool, SPLIT)

    assert not observation.is_error
    proposal = node(service, "root").pending_split
    assert proposal is not None and proposal.status == "proposed"
    assert [child.writable for child in proposal.children] == [
        ["services/api/**"],
        ["apps/web/**"],
    ]
    assert service.store.children("root") == []
    told = launcher.messages_to(manager(service))[-1]
    assert "services/api" in told and "POST /checkout" in told


async def test_only_the_ic_proposes_and_only_the_manager_decides(
    service: DirectorService,
):
    by_manager = await use(service, manager(service), ProposeBoundarySplitTool, SPLIT)
    await use(service, ic(service), ProposeBoundarySplitTool, SPLIT)
    by_ic = await use(
        service, ic(service), DecideSplitTool, DecideSplitAction(decision="approve")
    )

    assert by_manager.is_error and by_ic.is_error
    assert service.store.children("root") == []


async def test_an_overlapping_split_is_refused_and_nothing_waits(
    service: DirectorService,
):
    overlapping = ProposeBoundarySplitAction(
        children=[
            SplitChild(title="API", path="services/api", task="a"),
            SplitChild(title="Auth", path="services/api/auth", task="b"),
        ],
        contracts="",
    )

    observation = await use(service, ic(service), ProposeBoundarySplitTool, overlapping)

    assert observation.is_error and "overlap" in observation.text
    assert node(service, "root").pending_split is None


async def test_the_manager_can_send_a_split_back_and_the_ic_can_propose_again(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, ic(service), ProposeBoundarySplitTool, SPLIT)
    again_too_early = await use(service, ic(service), ProposeBoundarySplitTool, SPLIT)
    without_feedback = await use(
        service, manager(service), DecideSplitTool, DecideSplitAction(decision="refine")
    )

    await use(
        service,
        manager(service),
        DecideSplitTool,
        DecideSplitAction(decision="refine", feedback="Web and API share a schema."),
    )

    assert again_too_early.is_error and without_feedback.is_error
    proposal = node(service, "root").pending_split
    assert proposal is not None and proposal.status == "refine_requested"
    assert "share a schema" in launcher.messages_to(ic(service))[-1]
    again = await use(service, ic(service), ProposeBoundarySplitTool, SPLIT)
    assert not again.is_error


async def test_an_approved_split_starts_a_pod_for_each_sub_team(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)

    api, web = service.store.children("root")
    assert (api.id, api.title, api.boundary.path) == (
        "services.api",
        "API",
        "services/api",
    )
    assert (web.id, web.isolation) == ("apps.web", "worktree")
    starts = {str(start.conversation_id): start for start in launcher.derived}
    api_ic, web_ic = starts[str(ic(service, api.id))], starts[str(ic(service, web.id))]
    # Each IC continues from the IC above; its Manager shares its directory.
    assert str(api_ic.parent_id) == str(web_ic.parent_id) == ic(service)
    assert (api_ic.isolation, web_ic.isolation) == ("shared", "worktree")
    assert api_ic.suffixes == [] and "Serve the checkout API" in api_ic.task
    web_manager = starts[str(manager(service, web.id))]
    assert str(web_manager.parent_id) == ic(service, web.id)
    assert (web_manager.isolation, web_manager.suffixes) == ("shared", [MANAGER_ROLE])

    assert node(service, "root").pending_split is None
    assert "API (services/api)" in launcher.messages_to(ic(service))[-1]
    [contracts] = service.blackboard.inherited("services/api")
    assert contracts.readonly and "POST /checkout" in contracts.spec


async def test_a_sub_team_that_cannot_start_is_shown_as_blocked(
    service: DirectorService, launcher: FakeLauncher
):
    root_ic = ic(service)
    assert root_ic is not None
    await use(service, root_ic, ProposeBoundarySplitTool, SPLIT)
    # The IC's conversation is deleted before its Manager approves.
    del launcher.known[UUID(root_ic)]

    await use(
        service,
        manager(service),
        DecideSplitTool,
        DecideSplitAction(decision="approve"),
    )

    api, web = service.store.children("root")
    assert (api.status, web.status) == ("blocked", "blocked")
    assert [agent.status for agent in api.agents] == ["lost", "idle"]


async def test_an_escalation_climbs_through_the_managers_to_the_human(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api = "services.api"

    raised = await use(
        service,
        ic(service, api),
        RaiseEscalationTool,
        RaiseEscalationAction(
            category="design_decision", title="Cents or decimals?", detail="For totals."
        ),
    )
    [escalation] = node(service, api).escalations
    assert not raised.is_error and escalation.held_by == "root"
    assert "Cents or decimals?" in launcher.messages_to(manager(service))[-1]
    assert service.store.escalations("root") == []

    not_theirs = await use(
        service,
        manager(service, "apps.web"),
        ResolveEscalationTool,
        ResolveEscalationAction(
            node_id=api, escalation_id=escalation.id, action="settle", text="Cents."
        ),
    )
    passed = await use(
        service,
        manager(service),
        ResolveEscalationTool,
        ResolveEscalationAction(
            node_id=api,
            escalation_id=escalation.id,
            action="pass_up",
            text="Finance has to choose.",
        ),
    )

    assert not_theirs.is_error and not passed.is_error
    [waiting] = service.store.escalations("root")
    assert waiting.escalation.held_by is None
    assert waiting.escalation.handoffs[0].reason == "Finance has to choose."


async def test_a_manager_settles_what_it_can_and_the_team_below_is_told(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api = "services.api"
    await use(
        service,
        ic(service, api),
        RaiseEscalationTool,
        RaiseEscalationAction(category="design_decision", title="Cents or decimals?"),
    )
    [escalation] = node(service, api).escalations

    await use(
        service,
        manager(service),
        ResolveEscalationTool,
        ResolveEscalationAction(
            node_id=api, escalation_id=escalation.id, action="settle", text="Cents."
        ),
    )

    assert node(service, api).escalations == []
    assert "Cents." in launcher.messages_to(ic(service, api))[-1]


async def test_a_team_declares_itself_done_and_the_manager_above_hears(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)

    await use(
        service,
        ic(service, "apps.web"),
        SetStatusTool,
        SetStatusAction(status="done", summary="Pages shipped."),
    )

    assert node(service, "apps.web").status == "done"
    assert node(service, "root").descendant_counts.done == 1
    told = launcher.messages_to(manager(service))[-1]
    assert "done" in told and "Pages shipped." in told
