"""When a boundary's rules are consolidated, and what may replace them."""

import asyncio
import json
from pathlib import Path

import pytest

from openhands.agent_server.director import memory
from openhands.agent_server.director.git_blackboard import GitBlackboard
from openhands.agent_server.director.models import CreateOrgRequest
from openhands.agent_server.director.service import DirectorService
from tests.agent_server.director.conftest import FakeLauncher, root_conversation


@pytest.fixture
def board(repo) -> GitBlackboard:
    return GitBlackboard(repo)


def teach(board: GitBlackboard, directory: str, count: int, start: int = 0) -> None:
    for number in range(start, start + count):
        board.teach(directory, f"Rule {number}")


def test_only_a_boundary_with_many_rules_is_due(board: GitBlackboard):
    teach(board, "svc/a", memory.MIN_RULES)
    teach(board, "svc/b", memory.MIN_RULES + 1)

    due = memory.due(board)

    assert [item.directory for item in due] == ["svc/b"]
    assert len(due[0].rules) == memory.MIN_RULES + 1


def test_consolidated_rules_are_not_due_again_until_they_have_grown(
    board: GitBlackboard,
):
    teach(board, "", 20)
    (item,) = memory.due(board)

    assert memory.apply(board, item, [f"Rule {n}" for n in range(15)])

    assert board.memory("").rules == [f"Rule {n}" for n in range(15)]
    assert memory.due(board) == []
    teach(board, "", 2, start=100)
    assert memory.due(board) == []
    teach(board, "", 1, start=200)
    assert [len(due.rules) for due in memory.due(board)] == [18]


def test_rules_that_changed_while_the_model_worked_are_left_alone(
    board: GitBlackboard,
):
    teach(board, "", 13)
    (item,) = memory.due(board)
    board.teach("", "Added meanwhile")

    assert not memory.apply(board, item, ["One rule"])

    assert len(board.memory("").rules) == 14
    assert len(memory.due(board)) == 1


def test_the_rewritten_rules_must_be_a_list_no_longer_than_the_one_given(
    board: GitBlackboard,
):
    teach(board, "", 13)
    (item,) = memory.due(board)
    longer = json.dumps([f"New {n}" for n in range(14)])

    assert memory.parse_rules('Sure:\n["A", " B ", "A", ""]', item) == ["A", "B"]
    assert memory.parse_rules(longer, item) is None
    assert memory.parse_rules("[]", item) is None
    assert memory.parse_rules('{"rules": "A"}', item) is None
    assert memory.parse_rules("[1, 2]", item) is None


async def consolidating_service(repo: Path, launcher: FakeLauncher) -> DirectorService:
    service = DirectorService(
        repo, save_delay=0, run_limit_retry=0, consolidate_every=0.01
    )
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    return service


async def test_the_director_consolidates_on_its_own_and_asks_the_model_once(
    repo: Path, board: GitBlackboard
):
    teach(board, "svc/a", 13)
    launcher = FakeLauncher(answers={memory.USAGE_ID: '["Rule 0", "Rule 1"]'})
    service = await consolidating_service(repo, launcher)
    try:
        async with asyncio.timeout(5):
            while len(board.memory("svc/a").rules) != 2:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
    finally:
        await service.aclose()

    assert board.memory("svc/a").rules == ["Rule 0", "Rule 1"]
    asked = [usage_id for _, usage_id, _, _ in launcher.asked]
    assert asked == [memory.USAGE_ID]


async def test_an_answer_that_is_not_a_list_of_rules_leaves_them_as_they_are(
    repo: Path, board: GitBlackboard
):
    teach(board, "", 13)
    head = board.head()
    launcher = FakeLauncher(answers={memory.USAGE_ID: "I merged them all."})
    service = await consolidating_service(repo, launcher)
    try:
        async with asyncio.timeout(5):
            while not launcher.asked:
                await asyncio.sleep(0.01)
        await service.settled()
    finally:
        await service.aclose()

    assert board.head() == head
