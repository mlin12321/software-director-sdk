"""What a team that is replaced or pruned leaves, and what the teams that
come after it are given of it."""

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from openhands.agent_server.director import compaction
from openhands.agent_server.director.launch import StartedAgent
from openhands.agent_server.director.models import OldSpec, SplitChild
from openhands.agent_server.director.org_store import team_id_of
from openhands.agent_server.director.prompts import (
    HAND_OVER_ROLE,
    left_behind,
)
from openhands.agent_server.director.run_state import TASK
from openhands.agent_server.director.service import (
    DirectorService,
    hand_over_of,
)
from openhands.agent_server.director.split import ProposedTeam
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    HandOverTool,
    MsgManagerTool,
    MsgTeamTool,
    ProposeTeamsTool,
    PruneTeamTool,
    ReadTeamContextTool,
    StopTeamTool,
)
from openhands.agent_server.director.tools.definition import (
    HandOverAction,
    MsgManagerAction,
    MsgTeamAction,
    ProposeTeamsAction,
    PruneTeamAction,
    ReadTeamContextAction,
    StopTeamAction,
)
from openhands.agent_server.director_router import get_node_memory
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    APPROVE,
    ORG_PROTOCOL,
    PRUNE_API,
    FakeLauncher,
    ServerEnded,
    again,
    attempt,
    create,
    cut_off,
    end_server_at,
    former_of,
    has_place,
    ic,
    manager,
    memory_of,
    node,
    proposing,
    shipped,
    split_api,
    started,
    sub_teams,
    tasked,
    use,
)
from tests.agent_server.director.document_tools import (
    ReportAction,
    ReportTool,
    WriteSpecAction,
    WriteSpecTool,
)


API = "services.api"
WEB = "services.web"
TWO = proposing(
    [
        SplitChild(title="API", path="services/api", task="Serve the checkout API"),
        SplitChild(title="Web", path="services/web", task="Build the pages"),
    ],
    "POST /checkout returns {order_id}",
)


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher):
    """An org of API and Web. API has a spec, two reports and messages with
    its Manager and with the IC above."""
    service = await started(repo, launcher)
    await use(service, ic(service), ProposeTeamsTool, TWO)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)
    api_ic = ic(service, API)
    await use(
        service, api_ic, WriteSpecTool, WriteSpecAction(file_text="# API by hand")
    )
    await use(service, api_ic, ReportTool, ReportAction(text="Chose SQLite."))
    await use(service, api_ic, ReportTool, ReportAction(text="Dropped the cache."))
    await use(
        service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text="Mind auth.")
    )
    yield service
    await service.aclose()


async def replace(
    service: DirectorService, *teams: SplitChild, keep: list[str], contracts: str = ""
):
    proposal = proposing(list(teams), contracts, keep)
    await use(service, ic(service), ProposeTeamsTool, proposal)
    return await use(service, manager(service), ApproveTeamsTool, APPROVE)


ORDERS = SplitChild(title="Orders", path="services/api", task="Keep the orders")


def aside(service: DirectorService, node_id: str = "root") -> str | None:
    """The conversation of a team's IC in hand-over mode."""
    mode = hand_over_of(node(service, node_id))
    return mode.conversation_id if mode else None


def given_aside(service: DirectorService, launcher: FakeLauncher) -> str:
    """What the top team's hand-over conversation was started with."""
    conversation = aside(service)
    [start] = [
        start
        for start in launcher.derived
        if str(start.conversation_id) == conversation
    ]
    return start.task


async def test_a_team_put_in_another_teams_directory_gets_none_of_its_memory(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[compaction.USAGE_ID] = "SQLite, no cache."

    await replace(service, ORDERS, keep=["Web"])

    orders = node(service, API)
    assert orders.title == "Orders" and orders.agents == []
    assert service.reports_of(orders) == []
    assert service.threads_of(orders) == []
    assert service._runs().marked(TASK)[orders.id] == "Keep the orders"
    [former] = former_of(service, API)
    assert (former.title, former.ended, former.path) == (
        "API",
        "replaced",
        "services/api",
    )
    # What it was for, without the message that started it.
    assert former.purpose == "Serve the checkout API"
    assert [report.text for report in former.reports] == [
        "Chose SQLite.",
        "Dropped the cache.",
    ]
    assert former.compacted_report == "SQLite, no cache."
    assert [thread.file for thread in former.threads] == [
        f".threads-ic/{node(service, 'root').id}/{orders.id}.json"
    ]


async def test_the_spec_of_the_team_before_stays_and_is_shown_as_old(
    service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"])

    orders = node(service, API)
    assert orders.old_specs == [OldSpec(path="services/api", team="API")]
    [spec] = service.specs_of(orders)
    assert (spec.spec, spec.old, spec.left_by) == ("# API by hand", True, "API")

    await tasked(service)
    briefing = launcher.derived[-1].task
    assert "A spec left by a team before you (API), in services/api" in briefing
    assert "# API by hand" in briefing and "Chose SQLite." not in briefing
    context = service.context_of(node(service, API), "ic")
    assert context["spec"] is None
    assert context["specs_left_by_teams_before_you"] == [
        {"directory": "services/api", "team": "API", "spec": "# API by hand"}
    ]

    orders_ic = ic(service, API)
    await use(service, orders_ic, WriteSpecTool, WriteSpecAction(file_text="# Orders"))

    orders = node(service, API)
    assert orders.old_specs == []
    [own] = service.specs_of(orders)
    assert (own.spec, own.old) == ("# Orders", False)


async def test_teams_that_split_or_merge_directories_are_shown_the_specs_left_there(
    service: DirectorService,
):
    await use(
        service, ic(service, WEB), WriteSpecTool, WriteSpecAction(file_text="# Web")
    )

    # A merge: one team over both directories.
    await replace(
        service, SplitChild(title="Services", path="services", task="Serve"), keep=[]
    )

    merged = node(service, "services")
    assert sorted(merged.old_specs, key=lambda old: old.path) == [
        OldSpec(path="services/api", team="API"),
        OldSpec(path="services/web", team="Web"),
    ]
    assert {spec.spec for spec in service.specs_of(merged)} == {
        "# API by hand",
        "# Web",
    }

    # A split: two teams inside the directory of one, which never wrote a
    # spec of its own, so what it was left is passed on.
    await replace(
        service,
        SplitChild(title="HTTP", path="services/api/http", task="Routes"),
        SplitChild(title="Pages", path="services/web", task="Pages"),
        keep=[],
    )

    assert node(service, "services.api.http").old_specs == [
        OldSpec(path="services/api", team="API")
    ]
    assert node(service, WEB).old_specs == [OldSpec(path="services/web", team="Web")]
    assert len(former_of(service, "services")) == 1


async def test_the_ic_is_told_who_has_each_replaced_teams_directory_now(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[compaction.USAGE_ID] = "Flask, SQLite"

    # A split: two teams inside API's directory, and the rest of it to no team.
    await replace(
        service,
        SplitChild(title="HTTP", path="services/api/http", task="Routes"),
        SplitChild(title="Store", path="services/api/db", task="Orders"),
        keep=["Web"],
    )

    told = given_aside(service, launcher)
    assert (
        "Its folder is now divided: HTTP (services/api/http), Store "
        "(services/api/db). What of services/api is in none of those is yours again."
    ) in told
    assert "Its reports, as one:\nFlask, SQLite" in told
    assert "give each the part that concerns its folder" in told
    # Like a merged team, each is set up and not started, and is shown the
    # spec of the team it was cut from as old.
    for team_id in ("services.api.http", "services.api.db"):
        team = node(service, team_id)
        assert team.agents == [] and service.reports_of(team) == []
        assert team.old_specs == [OldSpec(path="services/api", team="API")]
    below = service.context_of(node(service, "root"), "ic")["sub_teams"]
    assert {team["team"]: team["status"] for team in below} == {
        "Web": "idle",
        "HTTP": "not_started",
        "Store": "not_started",
    }

    # A merge: one team over the directories of all three.
    await replace(
        service, SplitChild(title="Services", path="services", task="Serve"), keep=[]
    )

    told = given_aside(service, launcher)
    assert told.count("Its folder is now in the folder of Services (services).") == 3

    # Teams that take none of a removed team's directory.
    [api] = former_of(service, API)[-1:]
    elsewhere = left_behind([api], [SplitChild(title="Apps", path="apps", task="Apps")])
    assert "No new team has its folder: services/api is yours again." in elsewhere


async def test_reports_that_no_model_shortens_are_kept_in_full(
    service: DirectorService,
):
    await replace(service, ORDERS, keep=["Web"])

    [former] = former_of(service, API)
    assert former.compacted_report == "Chose SQLite.\n\nDropped the cache."


async def test_a_pruned_team_is_recorded_and_its_spec_is_old_for_the_team_above(
    service: DirectorService,
):
    await use(service, manager(service), PruneTeamTool, PRUNE_API)

    assert not has_place(service, API)
    [former] = former_of(service, API)
    assert (former.ended, len(former.reports)) == ("pruned", 2)
    assert node(service, "root").old_specs == [OldSpec(path="services/api", team="API")]
    context = service.context_of(node(service, "root"), "ic")
    [left] = context["specs_left_by_teams_before_you"]
    assert left["spec"] == "# API by hand"
    # The record is the session's own until its work is accepted.
    assert service.team_memory.former(API) == []


async def test_sub_teams_of_a_stopped_team_that_is_replaced_go_on(
    service: DirectorService, launcher: FakeLauncher
):
    db = SplitChild(title="DB", path="services/api/db", task="Own the schema")
    await use(service, ic(service, API), ProposeTeamsTool, proposing([db]))
    await use(service, manager(service, API), ApproveTeamsTool, APPROVE)
    await tasked(service)
    db_ic = str(ic(service, "services.api.db"))
    launcher.report(db_ic, ConversationExecutionStatus.RUNNING)
    await use(
        service, ic(service), StopTeamTool, StopTeamAction(team=API, reason="Stuck.")
    )
    await service.settled()
    launcher.report(db_ic, ConversationExecutionStatus.IDLE)
    before = len(launcher.resumed)

    await replace(
        service, SplitChild(title="Services", path="services", task="Serve"), keep=[]
    )
    await service.settled()

    assert node(service, "services.api.db").parent_id == node(service, "services").id
    assert db_ic in [str(conversation) for conversation, _ in launcher.resumed[before:]]


async def test_what_was_said_to_a_manager_is_archived_with_its_team(
    service: DirectorService,
):
    await use(
        service,
        ic(service, API),
        ProposeTeamsTool,
        ProposeTeamsAction(
            teams=[
                ProposedTeam(name="DB", path="services/api/db", instruction="Schema")
            ],
            text="Splitting.",
        ),
    )

    api = node(service, API).id

    await replace(service, ORDERS, keep=["Web"])

    [former] = former_of(service, API)
    assert f".threads/{api}/{api}.json" in [thread.file for thread in former.threads]
    assert service.threads_of(node(service, API)) == []


async def test_the_ic_that_proposed_is_given_what_the_replaced_team_left(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[compaction.USAGE_ID] = "SQLite, no cache."

    approved = await replace(service, ORDERS, keep=["Web"])

    told = given_aside(service, launcher)
    assert "Team 'API' (id services.api, services/api), replaced." in told
    assert "It was for: Serve the checkout API" in told
    assert "Its reports, as one:\nSQLite, no cache." in told
    assert "reaches a new team unless you hand it over" in told
    for tool in ("`hand_over`", "`write_spec`"):
        assert tool in told
    assert "`msg_team`" in launcher.messages_to(ic(service))[-1]
    assert "pass on what the replaced teams left" in approved.text
    assert "msg_ic" in approved.text


async def test_a_first_split_reminds_of_the_spec_and_hands_nothing_over(
    service: DirectorService, launcher: FakeLauncher
):
    [told] = [
        text
        for text in launcher.messages_to(ic(service))
        if "your Manager approved" in text
    ]

    assert "Rewrite your own spec with `write_spec`" in told
    assert "hand_over" not in told


async def test_what_the_ic_hands_over_is_in_the_new_teams_briefing(
    service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"])

    handed = await use(
        service,
        ic(service),
        HandOverTool,
        HandOverAction(team=API, notes="API chose SQLite; keep it."),
    )

    assert "It has not started" in handed.text and node(service, API).agents == []
    [report] = service.reports_of(node(service, API))
    assert (report.text, report.handed_over) == ("API chose SQLite; keep it.", True)

    await tasked(service)

    briefing = launcher.derived[-1].task
    assert "Handed over by the IC of the team above" in briefing
    assert "API chose SQLite; keep it." in briefing
    context = service.context_of(node(service, API), "ic")
    [told] = context["reports_of_your_team"]
    assert told["by"] == "ic_of_the_team_above"


async def test_what_is_handed_to_a_team_at_work_reaches_it_with_its_next_message(
    service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"])

    handed = await use(
        service,
        ic(service),
        HandOverTool,
        HandOverAction(team=WEB, notes="API dropped its cache."),
    )
    await use(service, ic(service), MsgTeamTool, MsgTeamAction(team=WEB, text="Go on."))

    assert "with the next message they get" in handed.text
    heard = launcher.messages_to(ic(service, WEB))[-1]
    assert "Go on." in heard and "API dropped its cache." in heard


async def test_only_an_ic_hands_over_and_only_to_a_team_directly_below(
    service: DirectorService,
):
    notes = HandOverAction(team=API, notes="Keep SQLite.")

    by_manager = await use(service, manager(service), HandOverTool, notes)
    sideways = await use(service, ic(service, WEB), HandOverTool, notes)
    nothing = await use(service, ic(service), HandOverTool, HandOverAction(team=API))

    assert by_manager.is_error and "Only the IC" in by_manager.text
    assert sideways.is_error and "not a team directly below yours" in sideways.text
    assert nothing.is_error and "nothing to hand over" in nothing.text


async def test_contracts_are_handed_to_a_team_for_its_own_sub_teams(
    service: DirectorService,
):
    for text in ("DB serves HTTP.", "DB serves HTTP over a socket."):
        await use(
            service, ic(service), HandOverTool, HandOverAction(team=WEB, contracts=text)
        )

    kept = memory_of(service, "services/web/contracts.md")
    assert kept == "DB serves HTTP over a socket.\n"


async def test_the_contracts_of_a_block_that_replaces_teams_take_the_place_of_the_old(
    service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"], contracts="Orders serves Web.")

    assert memory_of(service, "contracts.md") == "Orders serves Web.\n"
    told = launcher.messages_to(ic(service))[-1]
    assert "are now the ones of this proposal alone" in told

    await replace(
        service,
        SplitChild(title="Ledger", path="services/api", task="Keep the ledger"),
        keep=["Web"],
    )

    assert memory_of(service, "contracts.md") == "Orders serves Web.\n"
    told = launcher.messages_to(ic(service))[-1]
    assert "If they name teams that are gone, restate them" in told


async def test_what_a_removed_team_kept_of_its_documents_is_in_its_record(
    service: DirectorService, launcher: FakeLauncher
):
    await split_api(service)

    await use(service, manager(service), PruneTeamTool, PRUNE_API)

    [former] = former_of(service, API)
    assert former.documents == {"contracts": "Order has an id.\n"}
    [told] = [
        team
        for team in service.context_of(node(service, "root"), "ic", in_full=True)[
            "former_teams"
        ]
        if team["team_id"] == API
    ]
    assert told["documents"] == former.documents
    assert "Its contracts:\nOrder has an id." in "\n".join(left_behind([former]))


async def test_the_ic_above_is_given_what_a_pruned_team_left(
    service: DirectorService, launcher: FakeLauncher
):
    pruned = await use(service, manager(service), PruneTeamTool, PRUNE_API)

    assert "Your Manager pruned the team 'API'" in launcher.messages_to(ic(service))[-1]
    told = given_aside(service, launcher)
    assert "your Manager pruned the team 'API'" in told
    assert "Team 'API' (id services.api, services/api), pruned." in told
    assert "Chose SQLite.\n\nDropped the cache." in told
    assert "Rewrite your own spec with `write_spec`" in told
    assert "`hand_over`" in told
    assert "msg_ic" in pruned.text


async def test_an_ic_can_read_again_what_removed_teams_left(
    service: DirectorService,
):
    await replace(service, ORDERS, keep=["Web"])

    read = await use(service, ic(service), ReadTeamContextTool, ReadTeamContextAction())

    [former] = json.loads(read.text)["former_teams"]
    assert (former["team"], former["ended"]) == ("API", "replaced")
    assert former["reports_as_one"] == "Chose SQLite.\n\nDropped the cache."
    # What is told with every message names the team and no more.
    [named] = service.context_of(node(service, "root"), "ic")["former_teams"]
    assert sorted(named) == ["at", "directory", "ended", "team", "team_id"]
    assert "former_teams" not in service.context_of(node(service, "root"), "manager")


async def test_a_hand_over_is_a_side_conversation_of_the_ic_with_its_own_briefing(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[compaction.USAGE_ID] = "SQLite, no cache."
    await use(service, ic(service), WriteSpecTool, WriteSpecAction(file_text="# Shop"))
    before = len(launcher.derived)

    await replace(service, ORDERS, keep=["Web"], contracts="Orders serves Web.")

    # One conversation more, derived from the IC's and not started by a task.
    [start] = launcher.derived[before:]
    assert str(start.conversation_id) == aside(service)
    assert str(start.parent_id) == ic(service)
    assert start.title.endswith("(IC, hand-over)")
    # What a hand-over is for is part of what it is told, as the org's ways are.
    assert shipped(HAND_OVER_ROLE) in start.task and ORG_PROTOCOL in start.task
    assert node(service, API).agents == []
    # It is told who it is, what changed, what was left and what its team
    # works to; none of it depends on what the IC's own conversation holds.
    told = start.task
    assert "You are the IC of the team" in told and "in hand-over mode" in told
    # The team still has its two agents and no third.
    assert [agent.role for agent in node(service, "root").agents] == ["ic", "manager"]
    orders = node(service, API).id
    assert f"- Orders (id {orders}, services/api): Keep the orders" in told
    assert "Its folder is now in the folder of Orders (services/api)." in told
    assert "Its reports, as one:\nSQLite, no cache." in told
    assert "The spec it wrote (old; it stays in services/api):\n# API by hand" in told
    assert "Your team's spec:\n# Shop" in told
    assert "Contracts between your sub-teams:\nOrders serves Web." in told
    assert '"sub_teams"' in told
    # The IC's own conversation is told of it and is given none of the material.
    main = launcher.messages_to(ic(service))[-1]
    assert "in hand-over mode: a side conversation of yours" in main
    assert "API (services/api)" in main
    assert "SQLite, no cache." not in main and "`hand_over`: notes" not in main


async def test_a_hand_over_conversation_hands_over_and_writes_the_spec_and_no_more(
    service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"])
    side = aside(service)

    handed = await use(
        service, side, HandOverTool, HandOverAction(team=API, notes="Keep SQLite.")
    )
    spec = await use(service, side, WriteSpecTool, WriteSpecAction(file_text="# Shop"))
    read = await use(service, side, ReadTeamContextTool, ReadTeamContextAction())
    refused = [
        await use(service, side, ReportTool, ReportAction(text="Done.")),
        await use(service, side, MsgTeamTool, MsgTeamAction(team=API, text="Go.")),
        await use(service, side, MsgManagerTool, MsgManagerAction(text="Hello.")),
    ]

    assert not handed.is_error and not spec.is_error
    [report] = service.reports_of(node(service, API))
    assert (report.text, report.handed_over) == ("Keep SQLite.", True)
    assert node(service, "root").purpose == "Shop"
    assert json.loads(read.text)["former_teams"][0]["reports_as_one"]
    for answer in refused:
        assert answer.is_error and "for the hand-over alone" in answer.text
    assert node(service, API).agents == []
    refusal = attempt(service, side, "file_editor", create("README.md"))
    assert refusal is not None and "In hand-over mode you do not edit files" in refusal


async def test_the_ic_is_told_what_its_hand_over_conversation_did_when_it_ends(
    service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"])
    side = aside(service)
    assert side is not None
    was = node(service, "root").status

    launcher.report(side, ConversationExecutionStatus.RUNNING)
    # A hand-over beside the team's work does not change how the team stands.
    assert node(service, "root").status == was
    await use(service, side, HandOverTool, HandOverAction(team=API, notes="Keep it."))
    await use(service, side, WriteSpecTool, WriteSpecAction(file_text="# Shop"))
    launcher.report(side, ConversationExecutionStatus.FINISHED)
    await service.settled()

    told = launcher.messages_to(ic(service))[-1]
    assert "Your hand-over mode has ended." in told
    assert "It gave notes or contracts to: Orders." in told
    assert "It rewrote your team's spec." in told
    # Told once.
    launcher.report(side, ConversationExecutionStatus.RUNNING)
    launcher.report(side, ConversationExecutionStatus.FINISHED)
    await service.settled()
    ended = [
        text
        for text in launcher.messages_to(ic(service))
        if "Your hand-over mode has ended." in text
    ]
    assert len(ended) == 1


async def test_a_hand_over_conversation_that_did_nothing_leaves_it_to_the_ic(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, manager(service), PruneTeamTool, PRUNE_API)
    side = aside(service)
    assert side is not None

    launcher.report(side, ConversationExecutionStatus.RUNNING)
    launcher.report(side, ConversationExecutionStatus.FINISHED)
    await service.settled()

    told = launcher.messages_to(ic(service))[-1]
    assert "It gave nothing to any team." in told
    assert "It did not rewrite your team's spec: do that with `write_spec`." in told


async def test_each_hand_over_gets_a_conversation_of_its_own(
    service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"])
    first = aside(service)
    assert first is not None

    await use(service, manager(service), PruneTeamTool, PRUNE_API)

    second = aside(service)
    assert second is not None and second != first
    mode = hand_over_of(node(service, "root"))
    assert mode is not None and mode.status == "idle"
    # The first is no agent of the org any more.
    late = await use(
        service, first, HandOverTool, HandOverAction(team=WEB, notes="Too late.")
    )
    assert late.is_error and "not an agent of the org" in late.text
    assert "your Manager pruned the team 'Orders'" in given_aside(service, launcher)


async def test_a_hand_over_mode_that_cannot_start_is_tried_once_more_and_then_lost(
    service: DirectorService, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    tried: list[object] = []

    async def fails(*args: object, **kwargs: object) -> None:
        tried.append(args)
        raise RuntimeError("no room for it")

    monkeypatch.setattr(launcher, "start_derived", fails)

    await replace(service, ORDERS, keep=["Web"])

    assert len(tried) == 2
    # It stays on the IC as lost, for whoever looks at the team.
    mode = hand_over_of(node(service, "root"))
    assert mode is not None and mode.status == "lost"
    told = launcher.messages_to(ic(service))[-1]
    assert "Your hand-over mode could not be started" in told
    # It is told so and no more: the job is not moved to its main conversation.
    assert "Team 'API'" not in told and "`hand_over`" not in told


async def test_a_hand_over_mode_that_starts_at_the_second_try_is_not_reported(
    service: DirectorService, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    start_derived = launcher.start_derived
    tried: list[object] = []

    async def fails_once(*args: Any, **kwargs: Any) -> StartedAgent:
        tried.append(args)
        if len(tried) == 1:
            raise RuntimeError("no room for it")
        return await start_derived(*args, **kwargs)

    monkeypatch.setattr(launcher, "start_derived", fails_once)

    await replace(service, ORDERS, keep=["Web"])

    mode = hand_over_of(node(service, "root"))
    assert mode is not None and mode.status != "lost"
    assert "Team 'API'" in given_aside(service, launcher)
    assert "could not be started" not in "".join(launcher.messages_to(ic(service)))


def ended_notices(service: DirectorService, launcher: FakeLauncher) -> list[str]:
    return [
        text
        for text in launcher.messages_to(ic(service))
        if "Your hand-over mode has ended." in text
    ]


async def test_a_hand_over_that_ended_while_the_server_was_down_is_told_at_its_start(
    repo: Path, service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"])
    side = aside(service)
    assert side is not None
    launcher.report(side, ConversationExecutionStatus.RUNNING)
    await use(service, side, HandOverTool, HandOverAction(team=API, notes="Keep it."))
    await service.aclose()
    launcher.known[UUID(side)] = ConversationExecutionStatus.FINISHED

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    mode = hand_over_of(node(second, "root"))
    assert mode is not None and mode.conversation_id == side
    assert mode.status != "working"
    [told] = ended_notices(second, launcher)
    assert "It gave notes or contracts to: Orders." in told
    await second.aclose()


async def test_a_hand_over_conversation_gone_with_the_server_is_shown_as_lost(
    repo: Path, service: DirectorService, launcher: FakeLauncher
):
    await use(service, manager(service), PruneTeamTool, PRUNE_API)
    side = aside(service)
    assert side is not None
    launcher.report(side, ConversationExecutionStatus.RUNNING)
    await service.aclose()
    del launcher.known[UUID(side)]

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    mode = hand_over_of(node(second, "root"))
    assert mode is not None and mode.status == "lost"
    [told] = ended_notices(second, launcher)
    assert "It gave nothing to any team." in told
    await second.aclose()


async def test_a_server_that_ends_as_a_hand_over_starts_starts_it_at_its_next_start(
    repo: Path, service: DirectorService, launcher: FakeLauncher
):
    ended = end_server_at(launcher, "derived")
    await use(service, ic(service), ProposeTeamsTool, proposing([ORDERS], "", ["Web"]))
    with suppress(ServerEnded):
        await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await ended.wait()
    await service.aclose()

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    assert "Team 'API'" in given_aside(second, launcher)
    mode = hand_over_of(node(second, "root"))
    assert mode is not None and mode.status != "lost"
    assert "could not be started" not in "".join(launcher.messages_to(ic(second)))
    await second.aclose()


async def test_a_paused_org_pauses_a_hand_over_and_it_goes_on_when_resumed(
    service: DirectorService, launcher: FakeLauncher
):
    await replace(service, ORDERS, keep=["Web"])
    side = aside(service)
    assert side is not None
    launcher.report(side, ConversationExecutionStatus.RUNNING)

    await service.pause()
    await service.settled()

    assert UUID(side) in launcher.paused
    assert ended_notices(service, launcher) == []
    # What resuming does names the hand-over, whose conversation is there.
    told = await service.recovery()
    assert [(item.kind, item.detail) for item in told.items] == [
        ("run_again", "Its IC in hand-over mode runs again.")
    ]

    await service.resume()
    await service.settled()

    assert UUID(side) in [conversation for conversation, _ in launcher.resumed]
    launcher.report(side, ConversationExecutionStatus.RUNNING)
    launcher.report(side, ConversationExecutionStatus.FINISHED)
    await service.settled()
    assert len(ended_notices(service, launcher)) == 1


DB = SplitChild(title="DB", path="services/api/db", task="Keep the tables")
HTTP = SplitChild(title="HTTP", path="services/api/http", task="Serve the routes")


async def api_in_hand_over(service: DirectorService, launcher: FakeLauncher) -> str:
    """Have API's IC in hand-over mode, at work on a team its Manager pruned."""
    await use(service, ic(service, API), ProposeTeamsTool, proposing([DB, HTTP]))
    await use(service, manager(service, API), ApproveTeamsTool, APPROVE)
    await use(
        service,
        manager(service, API),
        PruneTeamTool,
        PruneTeamAction(team="services.api.db", reason="Not needed."),
    )
    side = aside(service, API)
    assert side is not None
    launcher.report(side, ConversationExecutionStatus.RUNNING)
    return side


async def test_a_stopped_teams_hand_over_stops_with_it(
    service: DirectorService, launcher: FakeLauncher
):
    side = await api_in_hand_over(service, launcher)

    await use(
        service,
        ic(service),
        StopTeamTool,
        StopTeamAction(team=API, reason="Wrong direction."),
    )

    assert UUID(side) in launcher.paused


async def test_a_pruned_teams_hand_over_goes_with_it(
    service: DirectorService, launcher: FakeLauncher
):
    side = await api_in_hand_over(service, launcher)

    await use(service, manager(service), PruneTeamTool, PRUNE_API)

    assert [team_id_of(team) for team in sub_teams(service)] == [WEB]
    assert service.org.find_hand_over(side) is None
    late = await use(
        service, side, HandOverTool, HandOverAction(team="services.api.http", notes="x")
    )
    assert late.is_error and "not an agent of the org" in late.text
    # The team above hands over what API left, in a conversation of its own.
    above = aside(service)
    assert above is not None and above != side
    assert "your Manager pruned the team 'API'" in given_aside(service, launcher)


async def test_a_client_is_shown_the_teams_removed_from_below_a_team_and_their_record(
    service: DirectorService,
):
    await replace(service, ORDERS, keep=["Web"])

    shown = get_node_memory(node(service, "root").id, service)

    [former] = shown.former_teams
    assert (former.title, former.path, former.ended) == (
        "API",
        "services/api",
        "replaced",
    )
    assert "Chose SQLite." in former.compacted_report
    assert [(event.kind, event.team) for event in shown.history][-2:] == [
        ("replaced", "API"),
        ("set_up", "Orders"),
    ]
    # They are the team above's to see, not the new team's.
    assert get_node_memory(node(service, API).id, service).former_teams == []


async def test_a_split_finished_again_after_a_restart_hands_over_once(
    repo: Path,
    service: DirectorService,
    launcher: FakeLauncher,
    monkeypatch: pytest.MonkeyPatch,
):
    """The server ends after the hand-over was begun and before the end of
    the split was saved, so the split is finished a second time."""
    save_tree = service._save_tree
    ended = asyncio.Event()

    def save(store) -> None:
        # Nothing is saved once the server has ended.
        if ended.is_set():
            return
        if any(i.kind == "hand_over" for i in service._runs().open_intents()):
            ended.set()
            raise ServerEnded
        save_tree(store)

    monkeypatch.setattr(service, "_save_tree", save)
    await use(service, ic(service), ProposeTeamsTool, proposing([ORDERS], "", ["Web"]))
    await cut_off(
        launcher, ended, use(service, manager(service), ApproveTeamsTool, APPROVE)
    )
    await service.aclose()

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    asides = [s for s in launcher.derived if shipped(HAND_OVER_ROLE) in s.task]
    assert len(asides) == 1
    await second.aclose()
