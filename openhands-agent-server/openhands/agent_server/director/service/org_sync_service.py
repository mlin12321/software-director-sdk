"""The org on more than one machine: its tree and team memory pushed to
a remote, pulled from it, and merged where both sides changed them."""

import asyncio
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openhands.agent_server.director.environment import CodingEnvironment
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    MemorySync,
    OrgConflict,
    OrgMerge,
    OrgNodeInput,
    OrgSync,
    ResolveMergeRequest,
    Side,
    SyncResult,
)
from openhands.agent_server.director.org_merge import (
    Nodes,
    by_id,
    merge_orgs,
)
from openhands.agent_server.director.org_persistence import (
    org_file,
    save_org,
    saved_nodes,
    stored_nodes,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
)
from openhands.agent_server.director.service.org_load_service import OrgLoadService
from openhands.agent_server.director.service.org_run_service import (
    PAUSE_LANE,
    OrgRunService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.work_queue_service import (
    BASE_LANE,
    WorkQueueService,
)
from openhands.sdk.errors import (
    Conflict,
    Unprocessable,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# How often a merge is made again because the org was saved meanwhile.
_MERGE_ATTEMPTS = 4


@dataclass
class OrgSyncService:
    """Pushes the org to a remote, pulls it from one and merges the two.

    Only the org's tree and team memory travel; sessions are a machine's
    own. A pull pauses the org, and a merge that needs the human's
    choices waits in the run state until they are made or it is
    abandoned. One pull, push or step of a merge runs at a time.
    """

    environment: CodingEnvironment
    """Where the org's records are kept."""
    org_load: OrgLoadService
    """The org that is synced and the repository it is in."""
    org_state: OrgStateService
    org_save: OrgSaveService
    work_queue: WorkQueueService
    org_run: OrgRunService
    """Pauses the org for a pull and resumes it after."""
    launcher: Callable[[], TeamLauncher]
    """What reaches the conversations; raises where there is none."""
    _sync_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def pull_org(self, remote: str) -> OrgSync:
        """Take in the org a remote of the repository has, as a merge.

        The org is paused while it is merged and, unless the merge stops on
        conflicts, resumed if it was running. What only the remote changed
        is taken; what both changed in the same place is left as a conflict
        and nothing is written until each has a choice (:meth:`resolve_merge`)
        or the merge is abandoned (:meth:`abort_merge`).

        Raises:
            Conflict: A merge waits for choices.
            NotFound: The repository has no such remote.
            UpstreamFailed: The remote could not be reached.
            Unavailable: The director is not running, or has no repository.
        """
        self.launcher()
        store = await asyncio.to_thread(lambda: self.org_load.org)
        async with self._sync_lock:
            if self.org_state.merging() is not None:
                raise Conflict(
                    "A merge waits for choices. Resolve it or abandon it first."
                )
            repo = await asyncio.to_thread(self.org_load.repository)
            await asyncio.to_thread(self.environment.remote.fetch, repo, remote)
            theirs = await asyncio.to_thread(
                self.environment.remote.org_tree_of, repo, remote
            )
            if store.root_id is None:
                return await self.work_queue.result_in(
                    BASE_LANE, lambda: self._take_org(repo, remote, theirs)
                )
            resume = self.org_state.paused is None
            if resume:
                await self.org_run.pause()
                await self.work_queue.after(PAUSE_LANE)
            state = {
                "remote": remote,
                "theirs": theirs,
                "choices": {},
                "resume": resume,
            }
            # What it takes in changes what the org shares.
            return await self.work_queue.result_in(
                BASE_LANE, lambda: self._merge(state)
            )

    async def push_org(self, remote: str) -> OrgSync:
        """Give a remote of the repository the org tree and the team
        memory. Sessions stay on this machine.

        Raises:
            Conflict: There is no org, a merge waits for choices, or the
                remote has changes that were not pulled.
            NotFound: The repository has no such remote.
            UpstreamFailed: The remote could not be reached or refused.
        """
        store = await asyncio.to_thread(lambda: self.org_load.org)
        if store.root_id is None:
            raise Conflict("There is no org to push")
        async with self._sync_lock:
            if self.org_state.merging() is not None:
                raise Conflict(
                    "A merge waits for choices. Resolve it or abandon it first."
                )
            repo = self.org_load.repo_dir_now()

            def push() -> OrgSync:
                self.org_save.save_tree(store)
                self.environment.remote.fetch(repo, remote)
                if self.environment.remote.behind(repo, remote):
                    raise Conflict(
                        f"{remote} has changes of the org that are not here. "
                        "Pull first."
                    )
                self.environment.remote.push(repo, remote)
                return OrgSync(
                    remote=remote,
                    result="pushed",
                    head=self.environment.org_tree(repo).head(),
                )

            return await asyncio.to_thread(push)

    async def merge(self) -> OrgMerge | None:
        """The merge that waits for choices, if there is one."""
        state = self.org_state.merging()
        if state is None:
            return None
        return (await asyncio.to_thread(self._weigh, state)).merge

    async def resolve_merge(self, request: ResolveMergeRequest) -> OrgSync:
        """Choose a side for conflicts of the merge that waits. When every
        conflict has a choice the merge is finished.

        Raises:
            Conflict: No merge waits for choices.
            Unprocessable: A choice names no conflict of the merge.
        """
        self.launcher()
        async with self._sync_lock:
            state = self.org_state.merging()
            if state is None:
                raise Conflict("No merge waits for choices")
            open_ = (await asyncio.to_thread(self._weigh, state)).merge
            known = {conflict.id for conflict in open_.conflicts} if open_ else set()
            unknown = sorted(set(request.choices) - known)
            if unknown:
                raise Unprocessable(
                    f"The merge has no conflict named {', '.join(unknown)}"
                )
            choices: dict[str, str] = {**state["choices"], **request.choices}
            if request.rest is not None:
                choices = {each: request.rest for each in known} | choices
            return await self.work_queue.result_in(
                BASE_LANE, lambda: self._merge({**state, "choices": choices})
            )

    async def abort_merge(self) -> OrgSync:
        """Abandon the merge that waits; the org stays as it was.

        Raises:
            Conflict: No merge waits for choices.
        """
        async with self._sync_lock:
            state = self.org_state.merging()
            if state is None:
                raise Conflict("No merge waits for choices")
            self.org_state.set_merge(None)
            if state["resume"]:
                await self.org_run.resume()
            return OrgSync(
                remote=state["remote"],
                result="up_to_date",
                head=await asyncio.to_thread(
                    self.environment.org_tree(self.org_load.repo_dir_now()).head
                ),
            )

    async def _take_org(self, repo: Path, remote: str, theirs: str | None) -> OrgSync:
        """Take the remote's org as the org of a repository that has none."""
        if theirs is None:
            return OrgSync(remote=remote, result="up_to_date", head=None)

        def take() -> OrgSync:
            board = self.environment.org_tree(repo)
            if not board.move_to(theirs):
                raise Conflict("The org was changed meanwhile. Pull once more.")
            pulled = self._pull_memory(repo, remote, ())
            self.org_load.join(repo)
            return OrgSync(
                remote=remote,
                result="fast_forward",
                head=board.head(),
                memory=pulled,
            )

        outcome = await asyncio.to_thread(take)
        await asyncio.to_thread(self.org_load.working)
        await self.org_run.came_back()
        return outcome

    async def _merge(self, state: dict[str, Any]) -> OrgSync:
        """Carry a merge as far as its choices allow."""
        try:
            outcome = await asyncio.to_thread(self._weigh, state, True)
        except Exception:
            # Nothing waits to be resolved: the org the pull paused goes on.
            if state["resume"] and self.org_state.merging() is None:
                await self.org_run.resume()
            raise
        if outcome.result == "conflicts":
            self.org_state.set_merge(state)
            return outcome
        self.org_state.set_merge(None)
        await self.org_run.resync()
        if state["resume"]:
            await self.org_run.resume()
        return outcome

    def _weigh(self, state: dict[str, Any], land: bool = False) -> OrgSync:
        """Work a merge out and, when ``land`` is set and no conflict is
        open, write it."""
        store = self.org_load.org
        repo, remote = self.org_load.repo_dir_now(), state["remote"]
        theirs: str | None = state["theirs"]
        choices: dict[str, Side] = state["choices"]
        board = self.environment.org_tree(repo)
        team_memory = self.environment.team_memory(repo)
        their_board = self.environment.remote.team_memory_of(repo, remote)
        files = [
            OrgConflict(
                id=f"file:{clash.path}",
                kind="file",
                node_id="",
                title="",
                field=clash.path,
                mine=clash.ours,
                theirs=clash.theirs,
                choice=choices.get(f"file:{clash.path}"),
            )
            for clash in (team_memory.clashes(their_board) if their_board else [])
        ]
        for _ in range(_MERGE_ATTEMPTS):
            save_org(board, store)
            head = board.head()
            mine = by_id(saved_nodes(board, head))
            merged = None
            other: Nodes = {}
            if theirs is not None and not board.contains(theirs):
                other = by_id(saved_nodes(board, theirs))
                base = by_id(saved_nodes(board, board.base_with(theirs)))
                merged = merge_orgs(base, mine, other, choices)
            conflicts = [*(merged.conflicts if merged else []), *files]
            if any(conflict.choice is None for conflict in conflicts):
                return OrgSync(
                    remote=remote,
                    result="conflicts",
                    head=head,
                    merge=OrgMerge(
                        remote=remote, theirs=theirs or "", conflicts=conflicts
                    ),
                )
            if not land:
                return OrgSync(remote=remote, result="up_to_date", head=head)
            result: SyncResult = "up_to_date"
            if merged is not None and theirs is not None:
                nodes = merged.nodes
                if by_id(nodes) == other and board.move_to(theirs):
                    landed, result = True, "fast_forward"
                else:
                    landed = board.land(
                        [org_file(nodes)],
                        f"org: merge {remote}",
                        merging=theirs,
                    )
                    result = "merged"
                if not landed:
                    continue
                self._apply(store, nodes)
            break
        else:
            raise Conflict("The org kept changing while it was merged. Pull once more.")
        taken = [conflict.field for conflict in files if conflict.choice == "theirs"]
        synced = self._pull_memory(repo, remote, taken)
        if result == "up_to_date" and synced is not None and synced.changed:
            result = "merged"
        self.org_save.save_tree(store)
        return OrgSync(
            remote=remote,
            result=result,
            head=board.head(),
            memory=synced,
        )

    def _pull_memory(
        self, repo: Path, remote: str, take_theirs: Collection[str]
    ) -> MemorySync | None:
        ref = self.environment.remote.team_memory_name(remote)
        if self.environment.remote.team_memory_of(repo, remote) is None:
            return None
        board = self.environment.team_memory(repo)
        before = board.head()
        merged = board.merge(ref, take_theirs)
        return MemorySync(
            remote=remote,
            head=merged.head,
            changed=merged.head != before,
            kept_local=merged.kept_local,
        )

    def _apply(self, store: OrgStore, nodes: list[dict[str, Any]]) -> None:
        """Make ``store`` hold the merged tree, announcing what changed."""
        wanted = by_id(nodes)
        # Only the org's own teams: what a session has stays as it is.
        for node in store.teams():
            moved = node.id in wanted and wanted[node.id]["parent_id"] != node.parent_id
            if (node.id not in wanted or moved) and store.team(node.id) is not None:
                store.remove(node.id)
        current = by_id(stored_nodes(store))
        for node in nodes:
            if current.get(node["id"]) != node:
                store.upsert(OrgNodeInput.model_validate(node))
        # Teams may have changed place, which no one team's change tells.
        store.show_again()
