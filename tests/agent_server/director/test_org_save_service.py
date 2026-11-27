"""Saving the tree by itself: a store, a repository and a run state in
memory, and no director."""

import asyncio
from pathlib import Path

import pytest

from openhands.agent_server.director.git_store import GitStore, Landing
from openhands.agent_server.director.org_persistence import (
    ORG_REF,
    create_org_board,
    load_org,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import SESSION, RunState
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.errors import Unavailable
from tests.agent_server.director.conftest import (
    make_team,
    open_session_in,
    run_git,
    shipped_type,
    small_tree,
)


@pytest.fixture
def store() -> OrgStore:
    store = OrgStore(shipped_type)
    for node in small_tree():
        store.upsert(node)
    return store


@pytest.fixture
def run() -> RunState:
    return RunState(None)


def saver(
    store: OrgStore, repo: Path, run: RunState, queue: WorkQueueService, delay: float
) -> OrgSaveService:
    org_save = OrgSaveService(
        org=lambda: store,
        repo_dir=lambda: repo,
        team_memory=lambda: GitStore(repo),
        run_state=lambda: run,
        work_queue=queue,
        delay=delay,
    )
    store.on_change(org_save.tree_changed)
    return org_save


def saved(repo: Path) -> OrgStore:
    loaded = OrgStore(shipped_type)
    load_org(create_org_board(repo), loaded)
    return loaded


def landing(file: str, text: str) -> Landing:
    return Landing(
        files={file: (text, None)},
        seen={},
        reports={},
        former=[],
        forget=[],
        forget_reports=[],
        events={},
        routed=[],
    )


async def test_changes_of_the_tree_are_gathered_into_one_save(
    store: OrgStore, repo: Path, run: RunState
):
    board = create_org_board(repo)
    async with WorkQueueService(default_lane="org") as queue:
        async with saver(store, repo, run, queue, delay=0.05):
            store.upsert(make_team("c", "root", path="svc/c"))
            store.upsert(make_team("d", "root", path="svc/d"))
            await asyncio.sleep(0.01)
            await queue.settled()
            assert board.head() is None

            await asyncio.sleep(0.1)
            await queue.settled()

    assert run_git(repo, "rev-list", "--count", ORG_REF).strip() == "1"
    assert {"c", "d"} <= {team.id for team in saved(repo).teams()}


async def test_what_a_session_has_is_kept_at_once(
    store: OrgStore, repo: Path, run: RunState
):
    async with WorkQueueService(default_lane="org") as queue:
        async with saver(store, repo, run, queue, delay=60):
            open_session_in(store, "root", "s1")
            await asyncio.sleep(0.01)
            await queue.settled()

            assert run.marked(SESSION)
            assert create_org_board(repo).head() is None


async def test_exiting_saves_what_still_waited(
    store: OrgStore, repo: Path, run: RunState
):
    async with WorkQueueService(default_lane="org") as queue:
        async with saver(store, repo, run, queue, delay=60):
            store.upsert(make_team("c", "root", path="svc/c"))
            await asyncio.sleep(0.01)

    assert "c" in {team.id for team in saved(repo).teams()}


async def test_nothing_is_saved_on_a_timer_once_it_is_exited(
    store: OrgStore, repo: Path, run: RunState
):
    async with WorkQueueService(default_lane="org") as queue:
        org_save = saver(store, repo, run, queue, delay=0)
        async with org_save:
            pass
        store.upsert(make_team("c", "root", path="svc/c"))
        await asyncio.sleep(0.01)
        await queue.settled()

    assert create_org_board(repo).head() is None


async def test_a_save_asked_for_twice_before_it_is_made_is_made_once(
    store: OrgStore, repo: Path, run: RunState
):
    saves: list[OrgStore] = []
    async with WorkQueueService(default_lane="org") as queue:
        async with saver(store, repo, run, queue, delay=60) as org_save:
            write = org_save.save_tree
            org_save.save_tree = lambda store: (saves.append(store), write(store))[1]
            org_save.save_soon()
            org_save.save_soon()
            await queue.settled()

    assert len(saves) == 1


async def test_a_save_cannot_be_asked_for_while_nothing_is_carried_out(
    store: OrgStore, repo: Path, run: RunState
):
    org_save = saver(store, repo, run, WorkQueueService(default_lane="org"), delay=60)

    with pytest.raises(Unavailable):
        org_save.save_soon()


def test_a_landing_moves_the_tree_and_team_memory_together(
    store: OrgStore, repo: Path, run: RunState
):
    org_save = saver(store, repo, run, WorkQueueService(default_lane="org"), delay=60)
    store.upsert(make_team("c", "root", path="svc/c"))

    left = org_save.commit_landing(landing("svc/c/notes.md", "kept"), "land", True)

    assert left == {}
    assert GitStore(repo).read("svc/c/notes.md") == "kept"
    assert "c" in {team.id for team in saved(repo).teams()}


def test_a_landing_without_the_tree_leaves_the_saved_tree_as_it_was(
    store: OrgStore, repo: Path, run: RunState
):
    org_save = saver(store, repo, run, WorkQueueService(default_lane="org"), delay=60)

    org_save.commit_landing(landing("notes.md", "kept"), "land", False)

    assert GitStore(repo).read("notes.md") == "kept"
    assert create_org_board(repo).head() is None
