"""A team whose type names its own members runs as any team does."""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.models import SplitChild
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    ProposeTeamsTool,
    RaiseEscalationTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    RaiseEscalationAction,
    SubmitReviewAction,
)
from openhands.agent_server.director.tools.remote import call_tool, list_tools
from tests.agent_server.director.conftest import (
    FakeLauncher,
    agent_conversation,
    commit,
    tasked,
)
from tests.agent_server.director.test_director_agent_types import an_org
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    node,
    proposing,
    use,
)


BENCH = [
    "read_team_context",
    "raise_escalation",
    "report",
    "write_spec",
    "submit_review",
    "propose_teams",
    "msg_team",
]
LAB = f"""\
description: A lab.
agents:
  bench:
    writes: true
    text: Runs the experiments.
    tools: [{", ".join(BENCH)}]
    lead_tools: [read_team_context]
  referee:
    text: Checks what the benches below found.
    tools: [read_team_context, review_team]
    lead_tools: [read_team_context, review_team]
  chair:
    tools: [read_team_context, approve_teams, decline_teams]
    lead_tools: [read_team_context, settle_teams]
"""
TEAMS = proposing(
    [
        SplitChild(title="Assay", path="assay", task="Run the assay"),
        SplitChild(title="Rig", path="rig", task="Build the rig", agent_type="lab"),
    ],
    contracts="Assay runs on what Rig builds.",
)


def member(service: DirectorService, name: str, node_id: str = "root") -> str | None:
    return agent_conversation(node(service, node_id), name)


@pytest.fixture
async def service(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
) -> AsyncIterator[DirectorService]:
    config = tmp_path / "config" / "director.yaml"
    (config.parent / "types").mkdir(parents=True)
    config.write_text("")
    (config.parent / "types" / "lab.yaml").write_text(LAB)
    service = await an_org(repo, launcher, config, agent_type="lab")
    yield service
    await service.aclose()


async def split(service: DirectorService) -> None:
    await use(service, member(service, "bench"), ProposeTeamsTool, TEAMS)
    await use(service, member(service, "chair"), ApproveTeamsTool, APPROVE)
    await tasked(service)


async def test_a_team_starts_with_the_member_that_works_in_its_folder(
    service: DirectorService, launcher: FakeLauncher
):
    root = node(service, "root")

    assert [agent.role for agent in root.agents] == ["bench"]
    assert launcher.root_tools == BENCH
    team = service.team(root)
    assert (team.writer, team.reviewer, team.overseer) == ("bench", "referee", "chair")
    assert team.members == ("bench", "referee", "chair")


async def test_each_member_is_offered_its_own_tools_and_refused_the_rest(
    service: DirectorService,
):
    bench = member(service, "bench")
    assert bench is not None

    offered = [tool.name for tool in list_tools(service, UUID(bench))]
    refused = call_tool(service, UUID(bench), "approve_teams", {})

    assert sorted(offered) == sorted(BENCH)
    assert refused.is_error and "not one of your tools" in refused.text


async def test_a_proposal_goes_to_the_member_that_approves_and_starts_it(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, member(service, "bench"), ProposeTeamsTool, TEAMS)

    chair = member(service, "chair")
    assert chair is not None and member(service, "referee") is None
    assert "Proposed sub-teams" in "\n".join(launcher.messages_to(chair))
    by_bench = await use(service, member(service, "bench"), ApproveTeamsTool, APPROVE)
    assert by_bench.is_error and "Only the chair" in by_bench.text

    await use(service, chair, ApproveTeamsTool, APPROVE)
    await tasked(service)

    # Each sub-team has the members of its own type.
    assert [agent.role for agent in node(service, "assay").agents] == ["ic"]
    assert [agent.role for agent in node(service, "rig").agents] == ["bench"]


async def test_submitted_work_reaches_the_member_that_reviews_above(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split(service)

    commit(repo, "rig/frame.md", "A frame.\n")
    await use(
        service,
        member(service, "bench", "rig"),
        SubmitReviewTool,
        SubmitReviewAction(summary="The rig stands."),
    )

    referee = member(service, "referee")
    assert referee is not None
    told = "\n".join(launcher.messages_to(referee))
    assert "Review request" in told and "Team 'Rig'" in told
    bench = member(service, "bench")
    assert "Review request" not in "\n".join(launcher.messages_to(bench))
    assert node(service, "rig").review is not None


async def test_each_member_is_told_of_the_org_in_the_names_its_type_gives(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split(service)
    commit(repo, "rig/frame.md", "A frame.\n")
    await use(
        service,
        member(service, "bench", "rig"),
        SubmitReviewTool,
        SubmitReviewAction(summary="The rig stands."),
    )
    await service.settled()

    [to_bench, *_] = launcher.messages_to(member(service, "bench"))
    assert "You are the bench of" in to_bench
    assert "Propose sub-teams to your chair" in to_bench
    assert "Manager" not in to_bench and "your IC" not in to_bench
    # A member is told its part by the tools it has, not as a Manager.
    [referee] = [start for start in launcher.derived if "(referee)" in start.title]
    assert "You are the referee of your team" in referee.task
    assert "`review_team`" in referee.task
    assert "You are your team's" not in referee.task
    assert "Wait until something is sent to you." in referee.task
    [chair] = [start for start in launcher.derived if "(chair)" in start.title]
    assert "You are your team's chair, not its bench." in chair.task


async def test_an_escalation_from_below_reaches_the_member_of_the_same_name(
    service: DirectorService, launcher: FakeLauncher
):
    await split(service)

    raised = await use(
        service,
        member(service, "bench", "rig"),
        RaiseEscalationTool,
        RaiseEscalationAction(
            category="design_decision", title="Which frame?", detail="Two would fit."
        ),
    )

    assert not raised.is_error
    assert "It is with the bench of the team above" in raised.text
    told = "\n".join(launcher.messages_to(member(service, "bench")))
    assert "The bench of team 'Rig'" in told and "Which frame?" in told
