"""What a team that is replaced or pruned leaves, and what the teams that
come after it are given of it."""

import json
from pathlib import Path

import pytest

from openhands.agent_server.director import compaction
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.models import OldSpec, Report, SplitChild
from openhands.agent_server.director.run_state import TASK
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgManagerTool,
    MsgTeamTool,
    PruneTeamTool,
    ReportTool,
    StopTeamTool,
    WriteSpecTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgManagerAction,
    MsgTeamAction,
    ReportAction,
    StopTeamAction,
    WriteSpecAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import FakeLauncher, tasked
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    PRUNE_API,
    ic,
    manager,
    node,
    proposing,
    use,
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
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher):
    """An org of API and Web. API has a spec, two reports and messages with
    its Manager and with the IC above."""
    service = await started(repo, launcher)
    await use(service, ic(service), MsgManagerTool, TWO)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)
    api_ic = ic(service, API)
    await use(service, api_ic, WriteSpecTool, WriteSpecAction(content="# API by hand"))
    await use(service, api_ic, ReportTool, ReportAction(text="Chose SQLite."))
    await use(service, api_ic, ReportTool, ReportAction(text="Dropped the cache."))
    await use(
        service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text="Mind auth.")
    )
    yield service
    await service.aclose()


async def replace(service: DirectorService, *teams: SplitChild, keep: list[str]):
    await use(service, ic(service), MsgManagerTool, proposing(list(teams), keep=keep))
    await use(service, manager(service), ApproveTeamsTool, APPROVE)


ORDERS = SplitChild(title="Orders", path="services/api", task="Keep the orders")


async def test_a_team_put_in_another_teams_directory_gets_none_of_its_memory(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[compaction.USAGE_ID] = "SQLite, no cache."

    await replace(service, ORDERS, keep=["Web"])

    orders = node(service, API)
    assert orders.title == "Orders" and orders.agents == []
    assert service.reports_of(orders) == []
    assert service.threads_of(orders) == []
    assert service._runs().marked(TASK)[API] == "Keep the orders"
    [former] = service.team_memory.former(API)
    assert (former.title, former.ended, former.path) == (
        "API",
        "replaced",
        "services/api",
    )
    assert former.purpose.startswith("Serve the checkout API")
    assert [report.text for report in former.reports] == [
        "Chose SQLite.",
        "Dropped the cache.",
    ]
    assert former.compacted_report == "SQLite, no cache."
    assert [thread.file for thread in former.threads] == [
        ".threads-ic/root/services.api.json"
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
    await use(service, orders_ic, WriteSpecTool, WriteSpecAction(content="# Orders"))

    orders = node(service, API)
    assert orders.old_specs == []
    [own] = service.specs_of(orders)
    assert (own.spec, own.old) == ("# Orders", False)


async def test_teams_that_split_or_merge_directories_are_shown_the_specs_left_there(
    service: DirectorService,
):
    await use(
        service, ic(service, WEB), WriteSpecTool, WriteSpecAction(content="# Web")
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
    assert len(service.team_memory.former("services")) == 1


async def test_reports_that_no_model_shortens_are_kept_in_full(
    service: DirectorService,
):
    await replace(service, ORDERS, keep=["Web"])

    [former] = service.team_memory.former(API)
    assert former.compacted_report == "Chose SQLite.\n\nDropped the cache."


async def test_a_pruned_team_is_recorded_and_its_spec_is_old_for_the_team_above(
    service: DirectorService,
):
    await use(service, manager(service), PruneTeamTool, PRUNE_API)

    assert service.store.get(API) is None
    [former] = service.team_memory.former(API)
    assert (former.ended, len(former.reports)) == ("pruned", 2)
    assert node(service, "root").old_specs == [OldSpec(path="services/api", team="API")]
    context = service.context_of(node(service, "root"), "ic")
    [left] = context["specs_left_by_teams_before_you"]
    assert left["spec"] == "# API by hand"
    # Recording the same removal again changes nothing.
    again = service.team_memory.record_former(
        former.model_copy(update={"reports": [], "compacted_report": ""})
    )
    assert again == former


async def test_an_agent_reads_its_teams_reports_whatever_conversation_wrote_them(
    repo: Path, service: DirectorService
):
    api = node(service, API)
    board = GitStore(repo)
    kept = Report(at="2026-01-01T00:00:00+00:00", role="ic", text="Kept by name.")
    # As reports were kept before: under the agent's conversation.
    board.write(
        "", f".reports/{ic(service, API)}.json", json.dumps([kept.model_dump()])
    )
    board.write("", ".reports/someone-else.json", json.dumps([kept.model_dump()]))

    read = service.reports_of(api)

    assert [report.text for report in read] == [
        "Kept by name.",
        "Chose SQLite.",
        "Dropped the cache.",
    ]
    assert board.read(f".reports/{ic(service, API)}.json") is None
    assert board.read(".reports/someone-else.json") is not None
    assert len(board.reports(API, "ic")) == 3


async def test_sub_teams_of_a_stopped_team_that_is_replaced_go_on(
    service: DirectorService, launcher: FakeLauncher
):
    db = SplitChild(title="DB", path="services/api/db", task="Own the schema")
    await use(service, ic(service, API), MsgManagerTool, proposing([db]))
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

    assert node(service, "services.api.db").parent_id == "services"
    assert db_ic in [str(conversation) for conversation, _ in launcher.resumed[before:]]


async def test_what_was_said_to_a_manager_is_archived_with_its_team(
    service: DirectorService,
):
    await use(
        service,
        ic(service, API),
        MsgManagerTool,
        MsgManagerAction(
            text="Splitting.\n\n```teams\nteams:\n"
            "  - {name: DB, path: services/api/db, instruction: Schema}\n```"
        ),
    )

    await replace(service, ORDERS, keep=["Web"])

    [former] = service.team_memory.former(API)
    assert ".threads/services.api/services.api.json" in [
        thread.file for thread in former.threads
    ]
    assert service.threads_of(node(service, API)) == []
