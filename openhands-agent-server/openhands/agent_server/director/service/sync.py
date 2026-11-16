"""An org that is paused, comes back, is pulled, pushed or merged."""

import asyncio
import json
import logging
from collections import deque
from collections.abc import Collection
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from openhands.agent_server.director import org_sync
from openhands.agent_server.director.changes import require_editable, write_by_hand
from openhands.agent_server.director.git_store import (
    GitStore,
)
from openhands.agent_server.director.models import (
    AgentRole,
    FileEdit,
    MemorySync,
    OrgConflict,
    OrgMerge,
    OrgNode,
    OrgNodeInput,
    OrgSync,
    Recovery,
    ResolveMergeRequest,
    Side,
    StartAgainRequest,
    SyncResult,
)
from openhands.agent_server.director.org_merge import (
    Nodes,
    by_id,
    merge_orgs,
    rehomed,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    org_file,
    save_org,
    saved_nodes,
    stored_nodes,
)
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    OrgStore,
)
from openhands.agent_server.director.prompts import (
    POD_PROTOCOL,
)
from openhands.agent_server.director.recovery import (
    STARTING,
    describe,
    fingerprint,
    has_work,
    put_aside_stale,
    restore_worktrees,
)
from openhands.agent_server.director.run_state import (
    HELD,
    RUNNING,
    TEAM,
    WAITING,
)
from openhands.agent_server.director.service.sessions import Sessions
from openhands.agent_server.director.service.state import (
    _MERGE_ATTEMPTS,
    AGENT_PAUSED,
    ORG_LANE,
    OTHER_WORK_NOTE,
    PAUSE_LANE,
    RESTART_NOTE,
    TOOLS_OF,
    _agent_title,
    _lane_for,
    _new_pod,
    _shared_root,
    agent_conversation,
    apart,
    hand_over_of,
)
from openhands.sdk.conversation.request import AgentLaunchAdditions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    Conflict,
    NotFound,
    Unprocessable,
)
from openhands.sdk.tool.spec import Tool


logger = logging.getLogger(__name__)


class Sync(Sessions):
    """Holds an org, takes it up again, and moves it between clones."""

    async def pause(self) -> Recovery:
        """Hold the org: its agents stop after the step they are in, and
        the director's own steps and messages wait.

        What agents still report is recorded and acted on after
        :meth:`resume`.

        Raises:
            Conflict: There is no org, or it is the fixture.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        if store.root_id is None or self._fixture:
            raise Conflict("There is no org to pause")
        if self._paused is None:
            self._hold("human")
            self._enqueue(self._pause_agents, PAUSE_LANE)
        return await self.recovery()

    async def resume(self) -> Recovery:
        """Take a paused org up again.

        Run state that no longer fits the org is put aside first. Then the
        teams that waited to start are started in the order they waited,
        the steps the director had begun are finished, held messages are
        delivered and the agents that were running run again.

        Returns:
            What was put aside, and nothing else.

        Raises:
            Unavailable: The director is not running.
        """
        launcher = self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        if self._paused is None:
            return await self.recovery()
        run = self._runs()
        await asyncio.to_thread(self._put_aside_stale)
        await asyncio.to_thread(
            restore_worktrees, self._repo_dir, store.all(), self._home()
        )
        dropped = self._dropped()
        was_running = run.marked(RUNNING)
        waited = list(run.marked(WAITING))
        statuses = await launcher.statuses([UUID(agent) for agent in was_running])
        again: dict[str, str | None] = {}
        for conversation_id in was_running:
            status = statuses.get(UUID(conversation_id))
            if status == ConversationExecutionStatus.ERROR:
                again[conversation_id] = RESTART_NOTE
            elif status == ConversationExecutionStatus.PAUSED:
                again[conversation_id] = None
        self._resuming = set(again)
        self._release()
        await self._resync()
        for node_id in waited:
            run.unmark(WAITING, node_id)
        intents = run.open_intents()
        starting = {intent.node_id for intent in intents}
        self._line = deque(node_id for node_id in waited if node_id in starting)
        for intent in intents:
            self._enqueue(
                lambda intent=intent: self._carry_out(intent.id), _lane_for(intent)
            )
        for held in run.held():
            self._enqueue(
                lambda held=held: self._deliver(
                    held.seq, held.conversation_id, held.text
                ),
                held.node_id or ORG_LANE,
            )
        for node_id in run.marked(HELD):
            submitted = store.get(node_id)
            self.release_reviews(submitted.parent_id if submitted else None)
        for conversation_id, note in again.items():
            self._enqueue(
                lambda conversation_id=conversation_id, note=note: self._run_again(
                    conversation_id, note
                ),
                was_running[conversation_id],
            )
        return Recovery(paused=None, items=dropped, messages_held=0)

    async def recovery(self) -> Recovery:
        """What resuming the org does, and what was put aside as stale."""
        store = await asyncio.to_thread(lambda: self.store)
        run = self._runs()
        if self._paused is None:
            return Recovery(paused=None, items=[], messages_held=0)
        launcher = self._launcher
        statuses = (
            await launcher.statuses(
                [
                    UUID(conversation_id)
                    for node in store.all()
                    for agent in node.agents
                    for conversation_id in (
                        agent.conversation_id,
                        # A hand-over is paused and run again as an agent is.
                        agent.hand_over.conversation_id if agent.hand_over else None,
                    )
                    if conversation_id is not None
                ]
            )
            if launcher is not None
            else {}
        )
        items = await asyncio.to_thread(describe, run, store, statuses, self._repo_dir)
        return Recovery(
            paused=self._paused,
            items=[*items, *self._dropped()],
            messages_held=len(run.held()),
        )

    async def pull_org(self, remote: str) -> OrgSync:
        """Take in the org a remote of the repository has, as a merge.

        The org is paused while it is merged and, unless the merge stops on
        conflicts, resumed if it was running. What only the remote changed
        is taken; what both changed in the same place is left as a conflict
        and nothing is written until each has a choice (:meth:`resolve_merge`)
        or the merge is abandoned (:meth:`abort_merge`).

        Raises:
            Conflict: A merge waits for choices, or this is the fixture.
            NotFound: The repository has no such remote.
            UpstreamFailed: The remote could not be reached.
            Unavailable: The director is not running, or has no repository.
        """
        self._require_launcher()
        if self._fixture:
            raise Conflict("The director serves a fixture tree; it has no remote")
        store = await asyncio.to_thread(lambda: self.store)
        async with self._sync_lock:
            if self._merging() is not None:
                raise Conflict(
                    "A merge waits for choices. Resolve it or abandon it first."
                )
            repo = await asyncio.to_thread(self._repository)
            await asyncio.to_thread(org_sync.fetch, repo, remote)
            theirs = await asyncio.to_thread(
                org_sync.commit_at, repo, org_sync.remote_org(remote)
            )
            if store.root_id is None:
                return await self._take_org(repo, remote, theirs)
            resume = self._paused is None
            if resume:
                await self.pause()
                await self._after(PAUSE_LANE)
            return await self._merge(
                {"remote": remote, "theirs": theirs, "choices": {}, "resume": resume}
            )

    async def push_org(self, remote: str) -> OrgSync:
        """Give a remote of the repository the org, the team memory and the
        branches of the teams' work.

        Raises:
            Conflict: There is no org, a merge waits for choices, or the
                remote has changes that were not pulled.
            NotFound: The repository has no such remote.
            UpstreamFailed: The remote could not be reached or refused.
        """
        store = await asyncio.to_thread(lambda: self.store)
        if store.root_id is None or self._fixture:
            raise Conflict("There is no org to push")
        async with self._sync_lock:
            if self._merging() is not None:
                raise Conflict(
                    "A merge waits for choices. Resolve it or abandon it first."
                )
            repo = self._repo_dir

            def push() -> OrgSync:
                self._save_tree(store)
                org_sync.fetch(repo, remote)
                if org_sync.behind(repo, remote):
                    raise Conflict(
                        f"{remote} has changes of the org that are not here. "
                        "Pull first."
                    )
                branches = org_sync.push(repo, remote, store.all())
                return OrgSync(
                    remote=remote,
                    result="pushed",
                    head=create_org_board(repo).head(),
                    branches=branches,
                )

            return await asyncio.to_thread(push)

    async def merge(self) -> OrgMerge | None:
        """The merge that waits for choices, if there is one."""
        state = self._merging()
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
        self._require_launcher()
        async with self._sync_lock:
            state = self._merging()
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
            return await self._merge({**state, "choices": choices})

    async def abort_merge(self) -> OrgSync:
        """Abandon the merge that waits; the org stays as it was.

        Raises:
            Conflict: No merge waits for choices.
        """
        async with self._sync_lock:
            state = self._merging()
            if state is None:
                raise Conflict("No merge waits for choices")
            self._set_merge(None)
            if state["resume"]:
                await self.resume()
            return OrgSync(
                remote=state["remote"],
                result="up_to_date",
                head=await asyncio.to_thread(create_org_board(self._repo_dir).head),
            )

    async def edit_file(self, node_id: str, path: str, content: str) -> FileEdit:
        """Write one file of a team by hand, commit it and tell its IC.

        Raises:
            NotFound: There is no such team, or its checkout is not here.
            Conflict: An agent of the team is in the middle of a turn.
            Forbidden: The team's agents may not write the file either.
            Invalid: The path does not name a file that can be committed.
        """
        store = await asyncio.to_thread(lambda: self.store)
        node = store.get(node_id)
        if node is None:
            raise NotFound(f"Unknown node: {node_id}")
        sub_teams = store.children(node_id)
        require_editable(node, sub_teams, path)
        agents = [
            UUID(agent.conversation_id)
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        if agents and self._launcher is not None:
            statuses = await self._launcher.statuses(agents)
            if ConversationExecutionStatus.RUNNING in statuses.values():
                raise Conflict(
                    f"An agent of team {node.title!r} is working. Pause the "
                    "org or wait for it to finish, then save again."
                )
        edit = await asyncio.to_thread(write_by_hand, node, sub_teams, path, content)
        ic = agent_conversation(node, "ic")
        if edit.commit is not None and ic is not None and self._loop is not None:
            self.notify(
                ic,
                f"The human changed {edit.path} in your team's checkout by hand "
                f"and committed it as {edit.commit[:12]}. Read the file again "
                "before you change it, and keep what they wrote unless they "
                "tell you otherwise.",
            )
        return edit

    async def start_again(self, node_id: str, request: StartAgainRequest) -> OrgNode:
        """Start afresh a team whose agents have no conversation on this
        server, on the work its branch has.

        The team's IC starts in the team's own checkout and is briefed from
        the team's spec and last headline; its Manager is derived from it.
        The IC starts as the given conversation, or without one like the IC
        of the team above (from the team's own agent profile if it has one).

        Raises:
            NotFound: There is no such team.
            Conflict: The team is done, its IC has a conversation here, or
                no conversation was given and the team above has no IC here
                to start it like.
            Unavailable: The director is not running.
        """
        launcher = self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        node = store.get(node_id)
        if node is None:
            raise NotFound(f"There is no team {node_id}")
        if node.status == "done":
            raise Conflict(f"Team {node.title} is done")
        ic = agent_conversation(node, "ic")
        if ic is not None and await launcher.started(UUID(ic)) is not None:
            raise Conflict(f"The IC of team {node.title} has a conversation here")
        conversation = request.conversation
        above = store.get(node.parent_id) if node.parent_id else None
        # A session's IC that never started has no checkout yet: it gets
        # one of its own, never the one of the team it works beside.
        own_checkout = not (
            above is not None and store.opens_session(node) and not apart(node, above)
        )
        if conversation is not None and not own_checkout:
            raise Conflict(
                f"The session {store.session_title(node)!r} has no checkout yet. "
                "Start it again without a conversation."
            )
        if conversation is None:
            if above is None:
                raise Conflict(
                    f"Team {node.title} is the top of the org: it is started "
                    "again by a conversation started in its folder"
                )
            above_ic = agent_conversation(above, "ic")
            if above_ic is None or await launcher.started(UUID(above_ic)) is None:
                raise Conflict(
                    f"Team {above.title} above it has no IC here. Start that "
                    "team again first."
                )
        await asyncio.to_thread(restore_worktrees, self._repo_dir, [node], self._home())
        ic_id = uuid4()
        task = await asyncio.to_thread(self._task_of, node)
        if conversation is None:
            store.update_node(node_id, agents=_new_pod(ic_id), status="idle")
            self._begin(
                "start_team",
                node_id,
                task=task,
                parent=node.parent_id,
                checkout=node.worktree_path if own_checkout else None,
            )
            return store.get(node_id) or node
        additions = conversation.agent_launch_additions or AgentLaunchAdditions()
        suffix = "\n\n".join(
            part
            for part in (additions.system_message_suffix_append, POD_PROTOCOL)
            if part
        )
        store.update_node(node_id, agents=_new_pod(ic_id), status="idle")
        try:
            started = await launcher.start_root(
                conversation.model_copy(
                    update={
                        "conversation_id": ic_id,
                        "initial_message": None,
                        "autotitle": False,
                        "worktree": False,
                        "workspace": conversation.workspace.model_copy(
                            update={"working_dir": node.worktree_path}
                        ),
                        "agent_launch_additions": additions.model_copy(
                            update={"system_message_suffix_append": suffix}
                        ),
                    }
                ),
                [Tool(name=name) for name in TOOLS_OF["ic"]],
                title=_agent_title(node.title, "ic"),
                tool_params=await asyncio.to_thread(self._confine, node_id, "ic"),
                adapt_agent=await asyncio.to_thread(
                    self._adapter, str(ic_id), TOOLS_OF["ic"]
                ),
            )
        except BaseException:
            store.update_node(node_id, agents=node.agents, status=node.status)
            raise
        store.set_agent(node_id, "ic", kind=started.kind, acp_server=started.acp_server)
        self._begin("brief", node_id, role="ic", task=task)
        return store.get(node_id) or node

    async def _after(self, lane: str) -> None:
        """Wait until what ``lane`` was asked to do so far is done."""
        done = asyncio.Event()

        async def reached() -> None:
            done.set()

        self._enqueue(reached, lane)
        await done.wait()

    async def _take_org(self, repo: Path, remote: str, theirs: str | None) -> OrgSync:
        """Take the remote's org as the org of a repository that has none."""
        if theirs is None:
            return OrgSync(remote=remote, result="up_to_date", head=None)

        def take() -> OrgSync:
            board = create_org_board(repo)
            if not board.move_to(theirs):
                raise Conflict("The org was changed meanwhile. Pull once more.")
            nodes = saved_nodes(board, theirs)
            here = rehomed(nodes, {}, _shared_root(nodes), str(repo))
            if here != nodes:
                board.land(
                    [org_file(here)],
                    f"org: at home in {repo.name}",
                )
            pulled = self._pull_memory(repo, remote, ())
            self._join(repo)
            return OrgSync(
                remote=remote,
                result="fast_forward",
                head=board.head(),
                memory=pulled,
                branches=org_sync.take_in_branches(repo, remote, self.store.all()),
            )

        outcome = await asyncio.to_thread(take)
        await asyncio.to_thread(self._working)
        await self._came_back()
        return outcome

    async def _merge(self, state: dict[str, Any]) -> OrgSync:
        """Carry a merge as far as its choices allow."""
        outcome = await asyncio.to_thread(self._weigh, state, True)
        if outcome.result == "conflicts":
            self._set_merge(state)
            return outcome
        self._set_merge(None)
        await self._resync()
        store = self.store
        for branch in outcome.branches:
            node = store.get(branch.node_id)
            ic = agent_conversation(node, "ic") if node else None
            if branch.result == "diverged" and ic is not None and branch.ref:
                self._post(
                    ic,
                    OTHER_WORK_NOTE.format(
                        branch=branch.branch, remote=outcome.remote, ref=branch.ref
                    ),
                )
        if state["resume"]:
            await self.resume()
        return outcome

    def _weigh(self, state: dict[str, Any], land: bool = False) -> OrgSync:
        """Work a merge out and, when ``land`` is set and no conflict is
        open, write it."""
        store = self.store
        repo, remote = self._repo_dir, state["remote"]
        theirs: str | None = state["theirs"]
        choices: dict[str, Side] = state["choices"]
        board = create_org_board(repo)
        team_memory = GitStore(repo)
        their_board = org_sync.commit_at(repo, org_sync.remote_memory(remote))
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
                nodes = rehomed(
                    merged.nodes, mine, _shared_root(list(other.values())), str(repo)
                )
                if by_id(nodes) == other:
                    landed, result = board.move_to(theirs), "fast_forward"
                else:
                    landed = board.land(
                        [org_file(nodes)],
                        f"org: merge {remote}",
                        merging=theirs,
                    )
                    result = "merged"
                if not landed:
                    continue
                before = {node.id: fingerprint(node) for node in store.all()}
                self._apply(store, nodes)
                run = self._runs()
                # What the run state was written for, so that what the merge
                # changed under it is found.
                run.replace_marks(TEAM, before)
                run.put("org_commit", head)
            break
        else:
            raise Conflict("The org kept changing while it was merged. Pull once more.")
        taken = [conflict.field for conflict in files if conflict.choice == "theirs"]
        synced = self._pull_memory(repo, remote, taken)
        if result == "up_to_date" and synced is not None and synced.changed:
            result = "merged"
        branches = org_sync.take_in_branches(repo, remote, store.all())
        already = len(self._dropped())
        self._put_aside_stale()
        self._save_tree(store)
        return OrgSync(
            remote=remote,
            result=result,
            head=board.head(),
            memory=synced,
            branches=branches,
            put_aside=self._dropped()[already:],
        )

    def _pull_memory(
        self, repo: Path, remote: str, take_theirs: Collection[str]
    ) -> MemorySync | None:
        ref = org_sync.remote_memory(remote)
        if org_sync.commit_at(repo, ref) is None:
            return None
        board = GitStore(repo)
        before = board.head()
        merged = board.merge(ref, take_theirs)
        return MemorySync(
            remote=remote,
            head=merged.head,
            changed=merged.head != before,
            kept_local=merged.kept_local,
            pushed=False,
        )

    def _apply(self, store: OrgStore, nodes: list[dict[str, Any]]) -> None:
        """Make ``store`` hold the merged tree, announcing what changed."""
        wanted = by_id(nodes)
        for node in store.all():
            moved = node.id in wanted and wanted[node.id]["parent_id"] != node.parent_id
            if (node.id not in wanted or moved) and store.get(node.id) is not None:
                store.remove(node.id)
        current = by_id(stored_nodes(store))
        for node in nodes:
            if current.get(node["id"]) != node:
                store.upsert(OrgNodeInput.model_validate(node))

    def _put_aside_stale(self) -> None:
        run = self._runs()
        head = create_org_board(self._repo_dir).head()
        dropped = put_aside_stale(
            run, self.store, instance=self._instance(), org_commit=head
        )
        if run.get("paused") is None:
            # Nothing that was left is this server's to take up.
            self._paused = None
            self._announce_state()
        if dropped:
            run.put(
                "dropped",
                json.dumps(
                    [item.model_dump() for item in [*self._dropped(), *dropped]]
                ),
            )

    async def _came_back(self) -> None:
        """Look at what an org that was just loaded was left in the middle
        of. A server that ended without stopping its org leaves it paused."""
        await asyncio.to_thread(self._look_at_what_was_left)
        await self._resync()
        if self._paused in ("shutdown", "crash") and self.settings.resume_on_start:
            await self.resume()

    def _look_at_what_was_left(self) -> None:
        if self._fixture or self.store.root_id is None:
            return
        run = self._runs()
        self._put_aside_stale()
        if self._paused is None and run.get("clean") == "0" and has_work(run):
            self._hold("crash")
        if self._paused is None:
            for item in self._dropped():
                logger.warning("Run state put aside: %s", item.detail)
            run.put("dropped", None)
        self._working()

    async def _pause_agents(self) -> None:
        store = self.store
        run = self._runs()
        launcher = self._require_launcher()
        agents: list[tuple[str, AgentRole, UUID]] = [
            (node.id, agent.role, UUID(agent.conversation_id))
            for node in store.all()
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        statuses = await launcher.statuses([agent for _, _, agent in agents])
        stopping: list[UUID] = []
        for node_id, role, conversation_id in agents:
            if statuses.get(conversation_id) != ConversationExecutionStatus.RUNNING:
                continue
            run.mark(RUNNING, str(conversation_id), node_id)
            self._running.add(str(conversation_id))
            with suppress(NotFound):
                store.set_agent(node_id, role, status=AGENT_PAUSED)
            stopping.append(conversation_id)
        # An IC in hand-over mode is paused in that conversation too.
        modes = [
            (node.id, mode)
            for node in store.all()
            if (mode := hand_over_of(node)) is not None
        ]
        running = await launcher.statuses(
            [UUID(mode.conversation_id) for _, mode in modes]
        )
        for node_id, mode in modes:
            aside = UUID(mode.conversation_id)
            if running.get(aside) != ConversationExecutionStatus.RUNNING:
                continue
            run.mark(RUNNING, mode.conversation_id, node_id)
            self._running.add(mode.conversation_id)
            with suppress(NotFound):
                store.set_hand_over(
                    node_id, mode.model_copy(update={"status": AGENT_PAUSED})
                )
            stopping.append(aside)
        await asyncio.gather(*(launcher.pause(agent) for agent in stopping))

    def _on_status(
        self, conversation_id: UUID, status: ConversationExecutionStatus
    ) -> None:
        if status != ConversationExecutionStatus.RUNNING and self._loop is not None:
            self._loop.call_soon_threadsafe(self._room.set)
        store = self._store
        if store is None:
            return
        agent = str(conversation_id)
        found = store.find_by_conversation(agent)
        if found is None:
            self._on_hand_over_status(store, agent, status)
            return
        node, role = found
        if self._paused is None:
            self._note_running(
                agent, node.id, status == ConversationExecutionStatus.RUNNING
            )
        was = node.status
        now = store.set_agent(node.id, role, status=self._shown(agent, status))
        if self._paused is None and self._loop is not None:
            if now.status == "blocked" and was != "blocked":
                # A blocked team is no longer waited for.
                self.release_reviews(node.parent_id)
            if role == "ic" and status == ConversationExecutionStatus.RUNNING:
                self.release_reviews(node.id, now=True)

    def _on_hand_over_status(
        self, store: OrgStore, conversation_id: str, status: ConversationExecutionStatus
    ) -> None:
        """Follow the hand-over conversation of an IC. When it stops
        working, the IC is told in its main conversation what it did."""
        node = store.find_hand_over(conversation_id)
        mode = hand_over_of(node) if node else None
        if node is None or mode is None:
            return
        if self._paused is None:
            self._note_running(
                conversation_id, node.id, status == ConversationExecutionStatus.RUNNING
            )
        shown = self._shown(conversation_id, status)
        store.set_hand_over(node.id, mode.model_copy(update={"status": shown}))
        # One that was held while it worked may have ended while it was held.
        at_work = ("working", AGENT_PAUSED)
        if mode.status in at_work and shown not in at_work:
            self._hand_over_ended(node.id)

    async def _resync(self) -> None:
        """Read every agent's status back from its conversation."""
        store = self.store
        agents: list[tuple[str, AgentRole, UUID]] = [
            (node.id, agent.role, UUID(agent.conversation_id))
            for node in store.all()
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        if not agents:
            return
        # Shells that were lost with the git directory come back for the
        # agents that still start theirs.
        await asyncio.to_thread(self._confine_all)
        statuses = await self._require_launcher().statuses(
            [conversation_id for _, _, conversation_id in agents]
        )
        starting = {
            intent.node_id
            for intent in self._runs().open_intents()
            if intent.kind in STARTING
        }
        for node_id, role, conversation_id in agents:
            status = statuses.get(conversation_id)
            agent = str(conversation_id)
            if self._paused is None:
                self._note_running(
                    agent, node_id, status == ConversationExecutionStatus.RUNNING
                )
            if status is not None:
                shown = self._shown(agent, status)
            else:
                shown = "idle" if node_id in starting else AGENT_LOST
            with suppress(NotFound):
                store.set_agent(node_id, role, status=shown)
        await self._resync_hand_overs()

    async def _resync_hand_overs(self) -> None:
        """Read back how each IC's hand-over conversation stands. One that
        ended or was lost while nothing followed it is ended here."""
        store = self.store
        modes = [
            (node.id, mode)
            for node in store.all()
            if (mode := hand_over_of(node)) is not None
        ]
        if not modes:
            return
        statuses = await self._require_launcher().statuses(
            [UUID(mode.conversation_id) for _, mode in modes]
        )
        starting = {
            intent.node_id
            for intent in self._runs().open_intents()
            if intent.kind == "hand_over"
        }
        for node_id, mode in modes:
            status = statuses.get(UUID(mode.conversation_id))
            if status is not None:
                self._on_hand_over_status(store, mode.conversation_id, status)
            elif node_id not in starting and mode.status != AGENT_LOST:
                store.set_hand_over(
                    node_id, mode.model_copy(update={"status": AGENT_LOST})
                )
                self._hand_over_ended(node_id)
