"""The tools an org's agents use, called as the agents call them."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from openhands.agent_server.director import agent_actions
from openhands.agent_server.director.models import (
    CreateOrgRequest,
    SplitChild,
)
from openhands.agent_server.director.org_store import (
    team_id_of,
)
from openhands.agent_server.director.prompts import MANAGER_ROLE
from openhands.agent_server.director.service import (
    DirectorService,
)
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    DeclineTeamsTool,
    MsgIcTool,
    MsgManagerTool,
    ProposeTeamsTool,
    PruneTeamTool,
    RaiseEscalationTool,
    ReadTeamContextTool,
    ResolveEscalationTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    DeclineTeamsAction,
    MsgIcAction,
    MsgManagerAction,
    PruneTeamAction,
    ReadTeamContextAction,
    ResolveEscalationAction,
    SubmitReviewAction,
)
from openhands.agent_server.director.tools.remote import call_tool
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
)
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import (
    APPROVE,
    CENTS,
    ORG_PROTOCOL,
    PRUNE_API,
    SPLIT,
    SPLIT_CONTRACTS,
    FakeLauncher,
    agent_conversation,
    division_lead_ic,
    has_place,
    ic,
    manager,
    node,
    proposing,
    record_of,
    root_conversation,
    run_git,
    shipped,
    split_root,
    sub_teams,
    tasked,
    use,
    woken,
    write_contracts,
)
from tests.agent_server.director.document_tools import (
    ProposeTeamsAction,
    ReportAction,
    ReportTool,
    ReviewTeamAction,
    WriteSpecAction,
    WriteSpecTool,
)


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


async def test_an_agent_reads_who_it_is_and_what_its_team_reported(
    service: DirectorService,
):
    await use(
        service, ic(service), ReportTool, ReportAction(text="Never log card numbers")
    )

    observation = await use(
        service, await woken(service), ReadTeamContextTool, ReadTeamContextAction()
    )

    context = json.loads(observation.text)
    assert context["you"] == {
        "role": "manager",
        "team_id": node(service, "root").id,
        "team": "Checkout",
        "type": "software-engineer",
        "session": "Checkout",
    }
    [reported] = context["reports_of_your_team"]
    assert (reported["by"], reported["text"]) == ("ic", "Never log card numbers")
    assert context["sub_teams"] == []


async def test_a_conversation_outside_the_org_is_refused(service: DirectorService):
    observation = await use(
        service, str(uuid4()), ReadTeamContextTool, ReadTeamContextAction()
    )

    assert observation.is_error


async def test_the_ic_writes_the_spec_its_sub_teams_inherit(service: DirectorService):
    written = await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(file_text="# Checkout\n\nAll prices are integer cents."),
    )
    refused = await use(
        service,
        await woken(service),
        WriteSpecTool,
        WriteSpecAction(file_text="# Mine"),
    )

    assert not written.is_error and refused.is_error
    assert node(service, "root").summary == "Checkout"
    read = await use(service, ic(service), ReadTeamContextTool, ReadTeamContextAction())
    assert "integer cents" in json.loads(read.text)["spec"]
    # It is the team's once the work it was written with is accepted.
    assert service.team_memory.read("spec.md") is None

    await use(
        service, ic(service), SubmitReviewTool, SubmitReviewAction(summary="Done.")
    )
    await use(
        service,
        division_lead_ic(service),
        ReviewTeamTool,
        ReviewTeamAction(team=node(service, "root").id, decision="accept"),
    )

    assert "integer cents" in (service.team_memory.read("spec.md") or "")


async def test_proposed_sub_teams_wait_for_the_manager(
    service: DirectorService, launcher: FakeLauncher
):
    await write_contracts(service, SPLIT_CONTRACTS)
    observation = await use(service, ic(service), ProposeTeamsTool, SPLIT)

    assert not observation.is_error
    proposal = node(service, "root").pending_split
    assert proposal is not None and proposal.status == "proposed"
    assert [child.writable for child in proposal.children] == [
        ["services/api/**"],
        ["apps/web/**"],
    ]
    assert sub_teams(service) == []
    told = launcher.messages_to(manager(service))[-1]
    assert told.startswith("[director] Your IC writes:\nI propose these sub-teams.")
    assert "services/api" in told
    # The Manager reads what the sub-teams will have to keep between them.
    assert (
        "Your team's contracts, which each sub-team reads:\n"
        "POST /checkout returns {order_id}" in told
    )
    assert "approve_teams sets them up exactly as written" in told


async def test_sub_teams_are_not_proposed_before_their_contracts_are_written(
    service: DirectorService,
):
    conversation = ic(service)
    assert conversation is not None
    teams = {"teams": [team.model_dump() for team in SPLIT.teams]}

    refused = call_tool(service, UUID(conversation), "propose_teams", teams)

    # The contracts say so themselves: the tool asks for no document.
    assert refused.is_error and "no contracts yet" in refused.text
    assert "`write_contracts`, then propose again" in refused.text
    assert node(service, "root").pending_split is None

    await write_contracts(service, SPLIT_CONTRACTS)
    proposed = call_tool(service, UUID(conversation), "propose_teams", teams)

    assert not proposed.is_error
    assert node(service, "root").pending_split is not None


async def test_only_the_ic_proposes_and_only_the_manager_approves(
    service: DirectorService,
):
    by_manager = await use(service, await woken(service), ProposeTeamsTool, SPLIT)
    nothing_yet = await use(service, await woken(service), ApproveTeamsTool, APPROVE)
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    by_ic = await use(service, ic(service), ApproveTeamsTool, APPROVE)

    assert by_manager.is_error and by_ic.is_error
    assert nothing_yet.is_error and "no proposed sub-teams" in nothing_yet.text
    assert sub_teams(service) == []


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

    observation = await use(service, ic(service), ProposeTeamsTool, proposing(children))

    assert observation.is_error and refusal in observation.text
    assert node(service, "root").pending_split is None
    assert launcher.messages_to(manager(service)) == before
    assert service.threads_of(node(service, "root")) == []


@pytest.mark.parametrize(
    ("written", "problem"),
    [
        pytest.param({"teams": []}, "teams", id="no teams"),
        pytest.param(
            {"teams": [{"name": "API", "path": "a"}]},
            "instruction",
            id="no instruction",
        ),
        pytest.param(
            {
                "teams": [
                    {"name": "API", "path": "a", "instruction": "x", "manager": "y"}
                ]
            },
            "manager",
            id="a field teams do not have",
        ),
        pytest.param(
            {
                "teams": [
                    {
                        "name": "API",
                        "path": "a",
                        "instruction": "x",
                        "isolation": "container",
                    }
                ]
            },
            "isolation",
            id="an isolation there is not",
        ),
    ],
)
def test_a_proposal_the_schema_does_not_allow_is_refused(
    written: dict[str, object], problem: str
):
    with pytest.raises(ValidationError, match=problem):
        ProposeTeamsAction.model_validate(written)


async def test_a_team_without_sub_teams_tells_its_manager_only_a_proposal(
    service: DirectorService, launcher: FakeLauncher
):
    said = await use(
        service, ic(service), MsgManagerTool, MsgManagerAction(text="Hello.")
    )

    assert said.is_error and "propose_teams" in said.text
    assert launcher.messages_to(manager(service)) == []


async def test_a_message_from_the_manager_leaves_the_proposal_waiting(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, ic(service), ProposeTeamsTool, SPLIT)

    answer = await use(
        service,
        manager(service),
        MsgIcTool,
        MsgIcAction(text="Do Web and API share a schema?"),
    )

    assert not answer.is_error
    assert "still wait on you: approve_teams or decline_teams" in answer.text
    told = launcher.messages_to(ic(service))[-1]
    assert told.startswith("[director] Your Manager writes:\nDo Web and API share")
    assert "still wait on your Manager" in told
    approved = await use(service, manager(service), ApproveTeamsTool, APPROVE)
    assert not approved.is_error
    assert [team_id_of(team) for team in sub_teams(service)] == [
        "services.api",
        "apps.web",
    ]


DECLINE = DeclineTeamsAction(reason="Web and API share a schema; keep them together.")


async def test_a_manager_declines_proposed_teams_and_the_reason_is_kept(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, ic(service), ProposeTeamsTool, SPLIT)

    declined = await use(service, manager(service), DeclineTeamsTool, DECLINE)

    assert not declined.is_error
    assert node(service, "root").pending_split is None
    assert node(service, "root").child_ids == []
    told = launcher.messages_to(ic(service))[-1]
    assert told.startswith(
        "[director] Your Manager writes:\nWeb and API share a schema"
    )
    assert "Your Manager declined the sub-teams you proposed (API (services/api)" in (
        told
    )
    last = record_of(service, "root")[-1]
    assert (last.kind, last.paths) == ("declined", ["services/api", "apps/web"])
    assert last.detail.endswith(
        "Reason: Web and API share a schema; keep them together."
    )
    too_late = await use(service, manager(service), ApproveTeamsTool, APPROVE)
    assert too_late.is_error


async def test_a_proposal_for_a_directory_says_what_was_declined_for_it_before(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), DeclineTeamsTool, DECLINE)

    await use(service, ic(service), ProposeTeamsTool, SPLIT)

    told = launcher.messages_to(manager(service))[-1]
    assert (
        "- Sub-teams for apps/web, services/api were proposed before and "
        "declined: API (services/api), Web (apps/web). Reason: Web and API "
        "share a schema; keep them together."
    ) in told
    assert "decline_teams declines them" in told


async def test_only_a_manager_with_a_waiting_proposal_declines_and_says_why(
    service: DirectorService,
):
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    by_ic = await use(service, ic(service), DeclineTeamsTool, DECLINE)
    no_reason = await use(
        service, manager(service), DeclineTeamsTool, DeclineTeamsAction(reason=" ")
    )

    assert by_ic.is_error and "Only the Manager" in by_ic.text
    assert no_reason.is_error and "Say why" in no_reason.text
    assert node(service, "root").pending_split is not None
    await use(service, manager(service), DeclineTeamsTool, DECLINE)
    nothing = await use(service, manager(service), DeclineTeamsTool, DECLINE)
    assert nothing.is_error and "no proposed sub-teams" in nothing.text


async def test_a_later_proposal_takes_the_place_of_the_one_that_waits(
    service: DirectorService,
):
    await use(service, ic(service), ProposeTeamsTool, SPLIT)

    await use(
        service,
        ic(service),
        ProposeTeamsTool,
        proposing([SplitChild(title="Docs", path="docs", task="Write the docs")]),
    )
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)

    assert [team.title for team in sub_teams(service)] == ["Docs"]


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
    message, _, update = launcher.messages_to(manager(service))[-1].partition("\n\n")
    assert message == (
        "[director] Your IC writes:\nWeb was sent back twice for the same bug."
    )
    # Its team got sub-teams since it last heard: that comes with the message.
    assert update.startswith("[director] context update") and "apps.web" in update
    assert node(service, "root").pending_split is None


async def test_a_manager_writes_to_its_own_ic_and_to_the_ics_directly_below(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api = "services.api"
    await use(
        service,
        ic(service, api),
        ProposeTeamsTool,
        proposing([SplitChild(title="DB", path="services/api/db", task="Schema")]),
    )
    await use(service, manager(service, api), ApproveTeamsTool, APPROVE)
    await tasked(service)

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
    # With it comes that the team below it has started since.
    assert launcher.messages_to(ic(service, api))[-1].startswith(
        "[director] The Manager of the team above (Checkout) writes:\n"
        "Where does the schema stand?\n"
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

    top = node(service, "root").id
    [proposal, said] = service.threads_of(node(service, "root"))[0].messages
    assert proposal.text.startswith("I propose these sub-teams.\n\nProposed sub-teams:")
    own = (top, top, [("ic", proposal.text), ("ic", said.text)])
    assert said.text == "API is slow."
    down = (top, node(service, api).id, [("manager", "Status?")])
    assert threads("root") == [own, down]
    assert threads(api) == [down]
    assert threads("apps.web") == []


async def test_an_approved_split_sets_up_each_sub_team(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)

    api, web = sub_teams(service)
    assert (team_id_of(api), api.title, api.boundary.path) == (
        "services.api",
        "API",
        "services/api",
    )
    assert (team_id_of(web), web.isolation) == ("apps.web", "worktree")
    # They are the session's until its work is accepted.
    assert [team.id for team in service.org.teams()] == ["root"]
    starts = {str(start.conversation_id): start for start in launcher.derived}
    api_ic, web_ic = starts[str(ic(service, api.id))], starts[str(ic(service, web.id))]
    # Each IC continues from the IC above.
    assert str(api_ic.parent_id) == str(web_ic.parent_id) == ic(service)
    assert (api_ic.isolation, web_ic.isolation) == ("shared", "worktree")
    assert "Serve the checkout API" in api_ic.task
    assert ORG_PROTOCOL in api_ic.task and shipped(MANAGER_ROLE) not in api_ic.task
    # A team has no Manager until something is sent to it; the Manager then
    # continues from the team's IC and shares its directory.
    assert manager(service, web.id) is None and len(starts) == 3
    woken_id = await woken(service, web.id)
    [web_manager] = [
        start for start in launcher.derived if str(start.conversation_id) == woken_id
    ]
    assert "Build the checkout pages" in web_manager.task
    assert "Look in on your team." in launcher.messages_to(woken_id)[-1]
    assert str(web_manager.parent_id) == ic(service, web.id)
    assert web_manager.isolation == "shared"
    assert shipped(MANAGER_ROLE) in web_manager.task

    assert node(service, "root").pending_split is None
    assert "API (services/api)" in launcher.messages_to(ic(service))[-1]
    [contracts] = [
        shown
        for shown in service.documents_shown(node(service, "services.api"))
        if shown.name == "contracts"
    ]
    assert not contracts.own and "POST /checkout" in contracts.text
    # They are the session's alone until its work is accepted.
    assert service.team_memory.read("contracts.md") is None


async def test_a_sub_team_that_cannot_start_is_shown_as_blocked(
    service: DirectorService, launcher: FakeLauncher
):
    root_ic = ic(service)
    assert root_ic is not None
    await use(service, root_ic, ProposeTeamsTool, SPLIT)
    # The IC's conversation is deleted before its Manager approves.
    del launcher.known[UUID(root_ic)]

    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)

    api, web = sub_teams(service)
    assert (api.status, web.status) == ("blocked", "blocked")
    assert [agent.status for agent in api.agents] == ["lost"]


READ = ReadTeamContextAction()


def resolving(
    node_id: str, escalation_id: str, action: Literal["settle", "pass_up"], text: str
) -> ResolveEscalationAction:
    return ResolveEscalationAction(
        team=node_id, escalation_id=escalation_id, action=action, text=text
    )


async def test_what_an_ic_escalates_goes_to_the_ic_above_and_then_to_the_human(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api = "services.api"
    to_manager = len(launcher.messages_to(manager(service)))

    raised = await use(service, ic(service, api), RaiseEscalationTool, CENTS)

    [escalation] = node(service, api).escalations
    assert not raised.is_error
    assert (escalation.held_by, escalation.raised_by) == (
        node(service, "root").id,
        "ic",
    )
    told = launcher.messages_to(ic(service))[-1]
    assert "The IC of team 'API'" in told and "Cents or decimals?" in told
    assert len(launcher.messages_to(manager(service))) == to_manager
    assert service.org.escalations() == []
    seen = {
        role: json.loads((await use(service, agent, ReadTeamContextTool, READ)).text)
        for role, agent in (("ic", ic(service)), ("manager", manager(service)))
    }
    [to_settle] = seen["ic"]["escalations_for_you_to_settle"]
    assert to_settle["escalation_id"] == escalation.id
    raiser = to_settle["team_id"]
    assert raiser == node(service, api).id
    assert seen["manager"]["escalations_for_you_to_settle"] == []

    by_a_peer = await use(
        service,
        ic(service, "apps.web"),
        ResolveEscalationTool,
        resolving(raiser, escalation.id, "settle", "Cents."),
    )
    by_the_manager = await use(
        service,
        manager(service),
        ResolveEscalationTool,
        resolving(raiser, escalation.id, "settle", "Cents."),
    )
    passed = await use(
        service,
        ic(service),
        ResolveEscalationTool,
        resolving(raiser, escalation.id, "pass_up", "Finance has to choose."),
    )

    assert by_a_peer.is_error and "not with your team" in by_a_peer.text
    assert (
        by_the_manager.is_error
        and "raised by the IC of its team" in by_the_manager.text
    )
    assert not passed.is_error and "waits on the human" in passed.text
    [waiting] = service.org.escalations()
    assert waiting.escalation.held_by is None
    assert waiting.escalation.handoffs[0].reason == "Finance has to choose."

    service.answer_escalation(node(service, api).id, escalation.id, "Cents.")
    await service.settled()

    assert "Cents." in launcher.messages_to(ic(service, api))[-1]
    assert node(service, api).escalations == []


async def test_a_team_is_blocked_while_its_ic_waits_on_an_escalation(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api = "services.api"
    api_ic = ic(service, api)
    assert api_ic is not None

    await use(service, api_ic, RaiseEscalationTool, CENTS)
    launcher.report(api_ic, ConversationExecutionStatus.RUNNING)

    [escalation] = node(service, api).escalations
    raiser = node(service, api).id
    assert node(service, api).status == "blocked"

    await use(
        service,
        ic(service),
        ResolveEscalationTool,
        resolving(raiser, escalation.id, "settle", "Cents."),
    )

    assert node(service, api).escalations == []
    assert node(service, api).status == "working"
    told = launcher.messages_to(api_ic)[-1]
    assert "The IC of 'Checkout' settled" in told and "Cents." in told


async def test_what_a_manager_escalates_goes_to_the_manager_above(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    api = "services.api"
    to_ic = len(launcher.messages_to(ic(service)))

    raised = await use(service, await woken(service, api), RaiseEscalationTool, CENTS)

    [escalation] = node(service, api).escalations
    raiser = node(service, api).id
    assert not raised.is_error and "Manager of the team above" in raised.text
    assert (escalation.held_by, escalation.raised_by) == (
        node(service, "root").id,
        "manager",
    )
    assert node(service, api).status != "blocked"
    told = launcher.messages_to(await woken(service))[-1]
    assert "The Manager of team 'API'" in told and "For totals." in told
    assert len(launcher.messages_to(ic(service))) == to_ic

    by_the_ic = await use(
        service,
        ic(service),
        ResolveEscalationTool,
        resolving(raiser, escalation.id, "settle", "Cents."),
    )
    settled = await use(
        service,
        await woken(service),
        ResolveEscalationTool,
        resolving(raiser, escalation.id, "settle", "Cents."),
    )

    assert by_the_ic.is_error and "raised by the Manager of its team" in by_the_ic.text
    assert not settled.is_error and node(service, api).escalations == []
    assert "Cents." in launcher.messages_to(await woken(service, api))[-1]
    assert launcher.messages_to(ic(service, api)) == []


@pytest.mark.parametrize("role", ["ic", "manager"])
async def test_what_the_top_team_escalates_waits_on_the_human(
    service: DirectorService, launcher: FakeLauncher, role: str
):
    await woken(service)
    raiser = agent_conversation(node(service, "root"), role)
    other = agent_conversation(
        node(service, "root"), "manager" if role == "ic" else "ic"
    )
    heard = len(launcher.messages_to(other))

    raised = await use(service, raiser, RaiseEscalationTool, CENTS)

    assert "waits on the human" in raised.text
    [waiting] = service.org.escalations()
    assert waiting.escalation.held_by is None
    assert node(service, "root").status == ("blocked" if role == "ic" else "idle")

    service.answer_escalation(node(service, "root").id, waiting.escalation.id, "Cents.")
    await service.settled()

    assert "The human answered" in launcher.messages_to(raiser)[-1]
    assert len(launcher.messages_to(other)) == heard
    assert node(service, "root").status == "idle"


async def test_a_pruned_team_is_gone_with_its_agents_and_its_ic_above_is_told(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    agents = {ic(service, "services.api"), await woken(service, "services.api")}
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "half.py").write_text("broken\n")
    (repo / "README.md").write_text("the root team's own edit\n")

    answer = await use(service, await woken(service), PruneTeamTool, PRUNE_API)

    assert not answer.is_error
    assert not has_place(service, "services.api")
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
        ProposeTeamsTool,
        proposing(
            [SplitChild(title="DB", path="services/api/db", task="Own the schema")]
        ),
    )
    await use(service, manager(service, api), ApproveTeamsTool, APPROVE)
    await tasked(service)
    below = ic(service, "services.api.db")

    await use(service, manager(service), PruneTeamTool, PRUNE_API)

    assert not has_place(service, "services.api.db")
    assert below in {str(gone) for gone in launcher.deleted}
    assert [team_id_of(team) for team in sub_teams(service)] == ["apps.web"]


async def test_reports_are_for_the_two_agents_of_one_team(service: DirectorService):
    await split_root(service)

    await use(
        service,
        ic(service, "services.api"),
        ReportTool,
        ReportAction(text="Amounts are integer cents."),
    )
    blank = await use(
        service, await woken(service), ReportTool, ReportAction(text="  ")
    )

    def read_by(conversation_id: str | None) -> list[str]:
        caller = agent_actions.caller_of(service, UUID(conversation_id or ""))
        context = json.loads(agent_actions.node_context(service, caller))
        return [report["text"] for report in context["reports_of_your_team"]]

    assert read_by(await woken(service, "services.api")) == [
        "Amounts are integer cents."
    ]
    assert read_by(ic(service)) == []
    assert read_by(ic(service, "apps.web")) == []
    assert blank.is_error


@pytest.mark.parametrize(
    ("as_role", "action", "refusal"),
    [
        pytest.param("ic", PRUNE_API, "Only the Manager", id="an IC"),
        pytest.param(
            "manager",
            PRUNE_API.model_copy(update={"team": "root"}),
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
    assert has_place(service, "services.api")
    assert launcher.deleted == []


async def test_a_sub_team_is_told_the_contracts_and_not_the_spec_of_the_team_above(
    service: DirectorService, launcher: FakeLauncher
):
    await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(file_text="# Checkout\n\nInternal notes of the root team."),
    )
    await split_root(service)
    api_ic = ic(service, "services.api")
    await use(
        service,
        api_ic,
        WriteSpecTool,
        WriteSpecAction(file_text="# API\n\nUses Flask."),
    )

    starts = {str(start.conversation_id): start for start in launcher.derived}
    briefing = starts[str(api_ic)].task
    context = json.loads(
        (await use(service, api_ic, ReadTeamContextTool, ReadTeamContextAction())).text
    )
    root_context = json.loads(
        (
            await use(
                service, ic(service), ReadTeamContextTool, ReadTeamContextAction()
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


async def test_every_message_between_agents_goes_one_way(
    service: DirectorService, launcher: FakeLauncher
):
    top = node(service, "root").id

    stopped = service.send_msg(top, "ic", top, "manager", "Split in two?")
    await service.settled()
    service.send_msg(top, "manager", top, "ic", "Not yet.")
    await service.settled()

    assert stopped is False
    assert launcher.messages_to(manager(service))[-1].startswith(
        "[director] Your IC writes:\nSplit in two?"
    )
    assert launcher.messages_to(ic(service))[-1].startswith(
        "[director] Your Manager writes:\nNot yet."
    )
    [thread] = service.threads_of(node(service, "root"))
    assert [(said.sender, said.text) for said in thread.messages] == [
        ("ic", "Split in two?"),
        ("manager", "Not yet."),
    ]


async def test_a_message_to_a_team_that_is_gone_is_refused(service: DirectorService):
    top = node(service, "root").id

    with pytest.raises(NotFound):
        service.send_msg(top, "ic", "gone@nowhere", "ic", "Anyone?")


async def test_an_agent_reads_how_the_org_works_again_with_its_context(
    service: DirectorService,
):
    its_manager = await woken(service)

    def read(conversation_id: str | None) -> str:
        caller = agent_actions.caller_of(service, UUID(conversation_id or ""))
        return json.loads(agent_actions.node_context(service, caller))[
            "how_the_org_works"
        ]

    # No agent has it in its system prompt, so it can be asked for again.
    assert read(ic(service)) == ORG_PROTOCOL
    assert read(its_manager).startswith(ORG_PROTOCOL)
    assert shipped(MANAGER_ROLE) in read(its_manager)
