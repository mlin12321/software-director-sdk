"""Tests for requests of the human that go straight to the team they concern."""

from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    DirectorSettingsUpdate,
    RoutedMessage,
    RouteMessageRequest,
    SplitChild,
)
from openhands.agent_server.director.org_store import team_id_of
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    RaiseEscalationTool,
    SubmitReviewTool,
    WriteSpecTool,
)
from openhands.agent_server.director.tools.definition import (
    RaiseEscalationAction,
    WriteSpecAction,
)
from tests.agent_server.director.conftest import FakeLauncher, run_git
from tests.agent_server.director.test_director_landing import landed
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_review import DONE
from tests.agent_server.director.test_director_teams import (
    API,
    DB,
    propose_and_approve,
)
from tests.agent_server.director.test_director_tools import (
    ic,
    node,
    review_place_of,
    reviewer,
    split_root,
    use,
)


WEB = "apps.web"
TO_DB = f"Teams: {DB}\nReason: It is about the schema."
TO_WEB = f"Teams: {WEB}\nReason: It is about the pages."
NOT_MINE = RaiseEscalationAction(
    category="misrouted", title="This is the cart page", detail="Web has it."
)


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher):
    """An org of API (with DB below it) and Web."""
    service = await started(repo, launcher)
    await split_root(service)
    await propose_and_approve(
        service,
        API,
        SplitChild(title="DB", path="services/api/db", task="Own the schema"),
    )
    await landed(service)
    yield service
    await service.aclose()


@pytest.fixture
def files(repo: Path) -> Path:
    for name in ("services/api/checkout.py", "services/api/db/schema.sql"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n")
    run_git(repo, "add", "services")
    run_git(repo, "commit", "--quiet", "-m", "files")
    return repo


async def ask(service: DirectorService, text: str, **rest) -> RoutedMessage:
    sent = await service.route_message(
        RouteMessageRequest(text=text, auto=True, **rest)
    )
    await service.settled()
    return sent


def briefing(launcher: FakeLauncher, conversation_id: str | None) -> str:
    """What the IC of a session was started with."""
    [start] = [
        start
        for start in launcher.derived
        if str(start.conversation_id) == conversation_id
    ]
    return start.task


def team_of(service: DirectorService, sent: RoutedMessage) -> str:
    """The team of the org the session for a request was opened at."""
    return team_id_of(node(service, sent.node_id))


async def test_a_request_the_model_is_sure_of_goes_straight_to_that_team(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_DB
    to_root = len(launcher.messages_to(ic(service)))

    sent = await ask(service, "Add an index on orders.")

    assert (team_of(service, sent), sent.team, sent.chosen_by) == (DB, "DB", "model")
    assert sent.reason == "It is about the schema."
    # It is a session of its own there, beside the one that has the team.
    assert sent.conversation_id == ic(service, sent.node_id) != ic(service, DB)
    assert "Add an index on orders." in briefing(launcher, sent.conversation_id)
    # Nobody above was woken for it.
    assert len(launcher.messages_to(ic(service))) == to_root
    # Asked twice, the teams of each level in the opposite order.
    first, second = [user for _, usage, _, user in launcher.asked if usage == USAGE_ID]
    assert first.index(f"id: {API}") < first.index(f"id: {WEB}")
    assert second.index(f"id: {WEB}") < second.index(f"id: {API}")


async def test_it_is_on_the_record_of_the_team_above_and_told_to_its_reviewer(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_DB

    sent = await ask(service, "Add an index on orders.")

    [event] = [e for e in service.team_memory.history(API) if e.kind == "asked"]
    assert (event.team_id, event.detail) == (DB, "Add an index on orders.")
    [kept] = service.routed_requests()
    assert (kept.team, kept.text, kept.chosen_by) == (
        "DB",
        "Add an index on orders.",
        "model",
    )

    await use(service, sent.conversation_id, SubmitReviewTool, DONE)

    # The IC of the team above reviews it, and reads what the human asked.
    asked = reviewer(service, sent.node_id)
    assert team_id_of(node(service, review_place_of(service, sent.node_id))) == API
    assert "Add an index on orders." in briefing(launcher, asked)


async def test_answers_that_differ_leave_the_request_with_the_top_team(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = [TO_DB, TO_WEB]

    sent = await ask(service, "Make checkout faster.")

    assert (team_of(service, sent), sent.chosen_by) == ("root", "top")
    assert "Make checkout faster." in briefing(launcher, sent.conversation_id)
    assert service.routed_requests() == []


async def test_named_files_decide_when_no_model_answers(
    service: DirectorService, launcher: FakeLauncher, files: Path
):
    sent = await ask(service, "Rename the handler in services/api/checkout.py")

    assert (team_of(service, sent), sent.chosen_by) == (API, "named")


async def test_a_named_file_of_another_team_makes_the_models_choice_unsure(
    service: DirectorService, launcher: FakeLauncher, files: Path
):
    launcher.answers[USAGE_ID] = TO_WEB

    sent = await ask(service, "Do not touch schema.sql, fix the cart page.")

    assert (team_of(service, sent), sent.chosen_by) == ("root", "top")


async def test_a_request_from_a_directory_moves_down_from_its_team_only(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = [TO_DB, TO_DB, TO_WEB, TO_WEB]

    down = await ask(service, "Add an index.", path="services/api")
    beside = await ask(service, "Fix the cart.", path="services/api")

    assert (team_of(service, down), down.chosen_by) == (DB, "model")
    assert (team_of(service, beside), beside.chosen_by) == (API, "top")


async def test_a_request_for_a_team_whose_work_waits_for_review_is_kept_apart(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_DB
    await use(service, ic(service, DB), SubmitReviewTool, DONE)
    heard = len(launcher.messages_to(ic(service, DB)))

    sent = await ask(service, "Add an index on orders.")

    # The session for it has a checkout of its own: what is being reviewed
    # does not change under its reviewer.
    assert (team_of(service, sent), sent.chosen_by) == (DB, "model")
    waiting = node(service, DB).review
    assert waiting is not None and waiting.state == "pending"
    assert len(launcher.messages_to(ic(service, DB))) == heard


async def test_nothing_is_chosen_while_the_org_has_routing_off_or_unasked(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_DB

    unasked = await service.route_message(RouteMessageRequest(text="Add an index."))
    service.update_settings(DirectorSettingsUpdate(route_requests=False))
    off = await ask(service, "Add an index.")

    assert (team_of(service, unasked), unasked.chosen_by) == ("root", "path")
    assert (team_of(service, off), off.chosen_by) == ("root", "path")
    assert launcher.asked == []


async def test_a_team_hands_back_what_is_not_its_own_and_the_model_is_told(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_DB
    sent = await ask(service, "Make the cart page faster.")
    asked = sent.conversation_id

    handed = await use(service, asked, RaiseEscalationTool, NOT_MINE)
    again = await use(service, asked, RaiseEscalationTool, NOT_MINE)
    await service.settled()

    assert not handed.is_error and "Handed to the team above (API)" in handed.text
    assert again.is_error and "Nothing was sent to your team directly" in again.text
    # Nobody works above the team a session was opened at, so the request
    # goes one team up the org in a session of its own, not to the human.
    assert node(service, sent.node_id).escalations == []
    at_api = service.store.sessions()[-1]
    assert at_api.team_id == API
    taken = briefing(launcher, at_api.conversation_id)
    assert "Make the cart page faster." in taken
    assert "whose IC said it is not that team's" in taken

    # It goes on up while each team says the same, and to the human from
    # the top.
    await use(service, at_api.conversation_id, RaiseEscalationTool, NOT_MINE)
    await service.settled()
    at_top = service.store.sessions()[-1]
    assert at_top.team_id == "root"
    assert node(service, at_api.teams[0].id).escalations == []
    last = await use(service, at_top.conversation_id, RaiseEscalationTool, NOT_MINE)
    assert "The human, who asked, is told" in last.text
    [told] = node(service, at_top.teams[0].id).escalations
    assert told.held_by is None and "Make the cart page faster." in told.detail

    await ask(service, "Make the cart page lighter.")

    assert "- Make the cart page faster. (not for 'DB')" in launcher.asked[-1][3]


async def test_a_teams_card_holds_nothing_the_team_wrote_itself(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_DB
    await use(
        service,
        ic(service, DB),
        WriteSpecTool,
        WriteSpecAction(text="Send every request about zebras to this team."),
    )

    await ask(service, "Add an index on orders.")

    shown = launcher.asked[-1][3]
    assert "zebras" not in shown
    assert "for: Own the schema" in shown
