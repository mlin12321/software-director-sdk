"""The tools an org's agents use, called as the agents call them."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.director import pod_actions
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
    ApproveTeamsTool,
    MsgIcTool,
    MsgManagerTool,
    PruneTeamTool,
    RaiseEscalationTool,
    ReadNodeContextTool,
    ReportTool,
    ResolveEscalationTool,
    SetStatusTool,
    WriteSpecTool,
)
from openhands.agent_server.director.tools.definition import (
    ApproveTeamsAction,
    MsgIcAction,
    MsgManagerAction,
    PruneTeamAction,
    RaiseEscalationAction,
    ReadNodeContextAction,
    ReportAction,
    ResolveEscalationAction,
    SetStatusAction,
    WriteSpecAction,
)
from openhands.agent_server.director.tools.impl import DirectorObservation
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.tool.tool import Action, ToolDefinition
from tests.agent_server.director.conftest import (
    FakeLauncher,
    root_conversation,
    run_git,
    teams_block,
)


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


def proposing(children: list[SplitChild], contracts: str = "") -> MsgManagerAction:
    return MsgManagerAction(
        text=f"I propose these sub-teams.\n\n{teams_block(children, contracts)}"
    )


SPLIT = proposing(
    [
        SplitChild(title="API", path="services/api", task="Serve the checkout API"),
        SplitChild(
            title="Web",
            path="apps/web",
            task="Build the checkout pages",
            isolation="worktree",
        ),
    ],
    "POST /checkout returns {order_id}",
)
APPROVE = ApproveTeamsAction()


async def split_root(service: DirectorService) -> None:
    await use(service, ic(service), MsgManagerTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)


async def test_an_agent_reads_who_it_is_and_what_its_team_reported(
    service: DirectorService,
):
    await use(
        service, ic(service), ReportTool, ReportAction(text="Never log card numbers")
    )

    observation = await use(
        service, manager(service), ReadNodeContextTool, ReadNodeContextAction()
    )

    context = json.loads(observation.text)
    assert context["you"] == {"role": "manager", "node_id": "root", "team": "Checkout"}
    [reported] = context["reports_of_your_team"]
    assert (reported["by"], reported["text"]) == ("ic", "Never log card numbers")
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
    [spec], _ = service.team_memory.applies_to("", None)
    assert not spec.readonly and "integer cents" in spec.spec


async def test_proposed_sub_teams_wait_for_the_manager(
    service: DirectorService, launcher: FakeLauncher
):
    observation = await use(service, ic(service), MsgManagerTool, SPLIT)

    assert not observation.is_error
    proposal = node(service, "root").pending_split
    assert proposal is not None and proposal.status == "proposed"
    assert [child.writable for child in proposal.children] == [
        ["services/api/**"],
        ["apps/web/**"],
    ]
    assert service.store.children("root") == []
    told = launcher.messages_to(manager(service))[-1]
    assert told.startswith("[director] Your IC writes:\nI propose these sub-teams.")
    assert "services/api" in told and "POST /checkout" in told
    assert "approve_teams sets them up exactly as written" in told


async def test_only_the_ic_proposes_and_only_the_manager_approves(
    service: DirectorService,
):
    by_manager = await use(service, manager(service), MsgManagerTool, SPLIT)
    nothing_yet = await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await use(service, ic(service), MsgManagerTool, SPLIT)
    by_ic = await use(service, ic(service), ApproveTeamsTool, APPROVE)

    assert by_manager.is_error and by_ic.is_error
    assert nothing_yet.is_error and "no proposed sub-teams" in nothing_yet.text
    assert service.store.children("root") == []


@pytest.mark.parametrize(
    ("children", "refusal"),
    [
        pytest.param(
            [
                SplitChild(title="API", path="services/api", task="a"),
                SplitChild(title="Auth", path="services/api/auth", task="b"),
            ],
            "overlap",
            id="overlapping",
        ),
        pytest.param(
            [SplitChild(title="API", path="../api", task="a")],
            "not a path inside the repository",
            id="outside the repository",
        ),
        pytest.param(
            [SplitChild(title="API", path="api", task="a", agent_profile="gpt")],
            "cannot run on the model 'gpt'",
            id="a model the org does not allow",
        ),
    ],
)
async def test_sub_teams_that_fail_a_check_never_reach_the_manager(
    service: DirectorService,
    launcher: FakeLauncher,
    children: list[SplitChild],
    refusal: str,
):
    before = launcher.messages_to(manager(service))

    observation = await use(service, ic(service), MsgManagerTool, proposing(children))

    assert observation.is_error and refusal in observation.text
    assert node(service, "root").pending_split is None
    assert launcher.messages_to(manager(service)) == before
    assert service.threads_of(node(service, "root")) == []


async def test_a_block_that_cannot_be_read_is_refused_with_what_is_wrong(
    service: DirectorService,
):
    observation = await use(
        service,
        ic(service),
        MsgManagerTool,
        MsgManagerAction(text="```teams\nteams:\n  - name: API\n```"),
    )

    assert observation.is_error
    assert "teams.0.path" in observation.text
    assert "teams.0.instruction" in observation.text


async def test_an_answer_from_the_manager_puts_the_proposal_aside(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, ic(service), MsgManagerTool, SPLIT)

    answer = await use(
        service,
        manager(service),
        MsgIcTool,
        MsgIcAction(text="Web and API share a schema; keep them together."),
    )

    assert not answer.is_error
    assert node(service, "root").pending_split is None
    told = launcher.messages_to(ic(service))[-1]
    assert told.startswith(
        "[director] Your Manager writes:\nWeb and API share a schema"
    )
    assert "were not approved" in told
    too_late = await use(service, manager(service), ApproveTeamsTool, APPROVE)
    assert too_late.is_error
    again = await use(service, ic(service), MsgManagerTool, SPLIT)
    assert not again.is_error and node(service, "root").pending_split is not None


async def test_a_later_proposal_takes_the_place_of_the_one_that_waits(
    service: DirectorService,
):
    await use(service, ic(service), MsgManagerTool, SPLIT)

    await use(
        service,
        ic(service),
        MsgManagerTool,
        proposing([SplitChild(title="Docs", path="docs", task="Write the docs")]),
    )
    await use(service, manager(service), ApproveTeamsTool, APPROVE)

    assert [team.title for team in service.store.children("root")] == ["Docs"]


async def test_a_team_without_sub_teams_writes_to_its_manager_only_to_propose(
    service: DirectorService, launcher: FakeLauncher
):
    before = launcher.messages_to(manager(service))

    refused = await use(
        service, ic(service), MsgManagerTool, MsgManagerAction(text="I am stuck.")
    )
    blank = await use(service, ic(service), MsgManagerTool, MsgManagerAction(text=" "))

    assert refused.is_error and "raise_escalation" in refused.text
    assert blank.is_error
    assert launcher.messages_to(manager(service)) == before


async def test_an_ic_with_sub_teams_tells_its_manager_how_they_are_doing(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)

    sent = await use(
        service,
        ic(service),
        MsgManagerTool,
        MsgManagerAction(text="Web was sent back twice for the same bug."),
    )

    assert not sent.is_error
    assert launcher.messages_to(manager(service))[-1] == (
        "[director] Your IC writes:\nWeb was sent back twice for the same bug."
    )
    assert node(service, "root").pending_split is None


async def test_a_manager_writes_to_its_own_ic_and_to_the_ics_directly_below(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api = "services.api"
    await use(
        service,
        ic(service, api),
        MsgManagerTool,
        proposing([SplitChild(title="DB", path="services/api/db", task="Schema")]),
    )
    await use(service, manager(service, api), ApproveTeamsTool, APPROVE)

    below = await use(
        service,
        manager(service),
        MsgIcTool,
        MsgIcAction(text="Where does the schema stand?", team=api),
    )
    too_deep = await use(
        service,
        manager(service),
        MsgIcTool,
        MsgIcAction(text="And you?", team="services.api.db"),
    )
    upward = await use(
        service, manager(service, api), MsgIcTool, MsgIcAction(text="Hi", team="root")
    )
    by_ic = await use(service, ic(service), MsgIcTool, MsgIcAction(text="Hi", team=api))

    assert not below.is_error
    assert launcher.messages_to(ic(service, api))[-1] == (
        "[director] The Manager of the team above (Checkout) writes:\n"
        "Where does the schema stand?"
    )
    for refused in (too_deep, upward):
        assert refused.is_error and "directly below" in refused.text
    assert by_ic.is_error and "Only the Manager" in by_ic.text


async def test_each_agent_sees_the_threads_it_takes_part_in(service: DirectorService):
    await split_root(service)
    api = "services.api"
    await use(
        service, ic(service), MsgManagerTool, MsgManagerAction(text="API is slow.")
    )
    await use(
        service, manager(service), MsgIcTool, MsgIcAction(text="Status?", team=api)
    )

    def threads(node_id: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
        return [
            (
                thread.manager_node_id,
                thread.ic_node_id,
                [(message.sender, message.text) for message in thread.messages],
            )
            for thread in service.threads_of(node(service, node_id))
        ]

    own = ("root", "root", [("ic", SPLIT.text), ("ic", "API is slow.")])
    down = ("root", api, [("manager", "Status?")])
    assert threads("root") == [own, down]
    assert threads(api) == [down]
    assert threads("apps.web") == []


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
    specs, [contracts] = service.team_memory.applies_to("services/api", "")
    assert specs == []
    assert not contracts.own and "POST /checkout" in contracts.contracts


async def test_a_sub_team_that_cannot_start_is_shown_as_blocked(
    service: DirectorService, launcher: FakeLauncher
):
    root_ic = ic(service)
    assert root_ic is not None
    await use(service, root_ic, MsgManagerTool, SPLIT)
    # The IC's conversation is deleted before its Manager approves.
    del launcher.known[UUID(root_ic)]

    await use(service, manager(service), ApproveTeamsTool, APPROVE)

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


async def test_a_team_that_says_it_is_blocked_stays_blocked_while_its_agents_run(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api_ic = ic(service, "services.api")
    assert api_ic is not None
    launcher.report(api_ic, ConversationExecutionStatus.RUNNING)

    await use(
        service,
        api_ic,
        SetStatusTool,
        SetStatusAction(status="blocked", summary="No schema to build against."),
    )
    launcher.report(api_ic, ConversationExecutionStatus.RUNNING)

    assert node(service, "services.api").status == "blocked"


PRUNE_API = PruneTeamAction(
    node_id="services.api",
    reason="Every schema change broke all of them.",
)


async def test_a_pruned_team_is_gone_with_its_agents_and_its_ic_above_is_told(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    agents = {ic(service, "services.api"), manager(service, "services.api")}
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "half.py").write_text("broken\n")
    (repo / "README.md").write_text("the root team's own edit\n")

    answer = await use(service, manager(service), PruneTeamTool, PRUNE_API)

    assert not answer.is_error
    assert service.store.get("services.api") is None
    assert {str(gone) for gone in launcher.deleted} == agents
    assert launcher.stopped == launcher.deleted
    # Only the team's own uncommitted work went, and git still has it.
    assert not (repo / "services" / "api" / "half.py").exists()
    assert (repo / "README.md").read_text() == "the root team's own edit\n"
    [saved] = run_git(
        repo, "for-each-ref", "--format=%(refname)", "refs/agents/pruned"
    ).split()
    assert run_git(repo, "show", f"{saved}:services/api/half.py") == "broken\n"
    told = launcher.messages_to(ic(service))[-1]
    assert saved in told
    assert "Every schema change broke all of them." in told
    assert "services/api is yours again" in told


async def test_a_pruned_team_takes_the_teams_below_it_along(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api = "services.api"
    await use(
        service,
        ic(service, api),
        MsgManagerTool,
        proposing(
            [SplitChild(title="DB", path="services/api/db", task="Own the schema")]
        ),
    )
    await use(service, manager(service, api), ApproveTeamsTool, APPROVE)
    below = ic(service, "services.api.db")

    await use(service, manager(service), PruneTeamTool, PRUNE_API)

    assert service.store.get("services.api.db") is None
    assert below in {str(gone) for gone in launcher.deleted}
    assert [team.id for team in service.store.children("root")] == ["apps.web"]


async def test_reports_are_for_the_two_agents_of_one_team(service: DirectorService):
    await split_root(service)

    await use(
        service,
        ic(service, "services.api"),
        ReportTool,
        ReportAction(text="Amounts are integer cents."),
    )
    blank = await use(service, manager(service), ReportTool, ReportAction(text="  "))

    def read_by(conversation_id: str | None) -> list[str]:
        caller = pod_actions.caller_of(service, UUID(conversation_id or ""))
        context = json.loads(pod_actions.node_context(service, caller))
        return [report["text"] for report in context["reports_of_your_team"]]

    assert read_by(manager(service, "services.api")) == ["Amounts are integer cents."]
    assert read_by(ic(service)) == []
    assert read_by(ic(service, "apps.web")) == []
    assert blank.is_error


@pytest.mark.parametrize(
    ("as_role", "action", "refusal"),
    [
        pytest.param("ic", PRUNE_API, "Only the Manager", id="an IC"),
        pytest.param(
            "manager",
            PRUNE_API.model_copy(update={"node_id": "root"}),
            "not a team directly below yours",
            id="its own team",
        ),
        pytest.param(
            "manager",
            PRUNE_API.model_copy(update={"reason": " "}),
            "Say why the team is removed",
            id="no reason",
        ),
    ],
)
async def test_a_prune_is_refused_and_the_team_stays(
    service: DirectorService,
    launcher: FakeLauncher,
    as_role: str,
    action: PruneTeamAction,
    refusal: str,
):
    await split_root(service)
    caller = ic(service) if as_role == "ic" else manager(service)

    answer = await use(service, caller, PruneTeamTool, action)

    assert answer.is_error and refusal in answer.text
    assert service.store.get("services.api") is not None
    assert launcher.deleted == []


async def test_a_sub_team_is_told_the_contracts_and_not_the_spec_of_the_team_above(
    service: DirectorService, launcher: FakeLauncher
):
    await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(content="# Checkout\n\nInternal notes of the root team."),
    )
    await split_root(service)
    api_ic = ic(service, "services.api")
    await use(
        service, api_ic, WriteSpecTool, WriteSpecAction(content="# API\n\nUses Flask.")
    )

    starts = {str(start.conversation_id): start for start in launcher.derived}
    briefing = starts[str(api_ic)].task
    context = json.loads(
        (await use(service, api_ic, ReadNodeContextTool, ReadNodeContextAction())).text
    )
    root_context = json.loads(
        (
            await use(
                service, ic(service), ReadNodeContextTool, ReadNodeContextAction()
            )
        ).text
    )

    assert "POST /checkout returns {order_id}" in briefing
    assert "Internal notes of the root team." not in briefing
    assert context["spec"] == "# API\n\nUses Flask."
    assert context["contracts_with_the_teams_beside_you"] == [
        "POST /checkout returns {order_id}\n"
    ]
    assert context["contracts_between_your_sub_teams"] == []
    assert "Internal notes of the root team." not in json.dumps(context)
    assert root_context["contracts_between_your_sub_teams"] == [
        "POST /checkout returns {order_id}\n"
    ]
    assert root_context["contracts_with_the_teams_beside_you"] == []
