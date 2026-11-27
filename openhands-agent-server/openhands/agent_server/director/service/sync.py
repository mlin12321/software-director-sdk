"""An org that is paused, comes back, is pulled, pushed or merged."""

import asyncio
import json
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
from openhands.agent_server.director.intents import Brief, StartTeam
from openhands.agent_server.director.models import (
    FileEdit,
    MemorySync,
    OrgConflict,
    OrgMerge,
    OrgNodeInput,
    OrgSync,
    Recovery,
    ResolveMergeRequest,
    SessionTeam,
    Side,
    StartAgainRequest,
    SyncResult,
)
from openhands.agent_server.director.org_merge import (
    Nodes,
    by_id,
    merge_orgs,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    org_file,
    save_org,
    saved_nodes,
    stored_nodes,
)
from openhands.agent_server.director.org_store import (
    AGENT_BLOCKED,
    AGENT_LOST,
    OrgStore,
    is_division_lead,
    new_agents,
)
from openhands.agent_server.director.prompts import (
    changed_since_accept,
)
from openhands.agent_server.director.recovery import (
    STARTING,
    describe,
    has_work,
    put_aside_stale,
    restore_worktrees,
)
from openhands.agent_server.director.run_state import (
    HELD,
    RUNNING,
    WAITING,
)
from openhands.agent_server.director.service.agent_start_service import agent_title
from openhands.agent_server.director.service.sessions import Sessions
from openhands.agent_server.director.service.stall_watch_service import (
    STALL_WAITS,
)
from openhands.agent_server.director.service.state import (
    _MERGE_ATTEMPTS,
    AGENT_PAUSED,
    PAUSE_LANE,
    RESTART_NOTE,
    apart,
    hand_over_of,
    lane_for,
)
from openhands.agent_server.director.service.team_messaging_service import (
    STARTED_WITH,
    failure_waits,
)
from openhands.agent_server.director.service.work_queue_service import (
    BASE_LANE,
    ORG_LANE,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    Conflict,
    NotFound,
    Unprocessable,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.spec import Tool


logger = get_logger(__name__)


class Sync(Sessions):
    """Holds an org, takes it up again, and moves it between clones."""

    async def pause(self) -> Recovery:
        """Hold the org: its agents stop after the step they are in, and
        the director's own steps and messages wait.

        What agents still report is recorded and acted on after
        :meth:`resume`.

        Raises:
            Conflict: There is no org.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.org)
        if store.root_id is None:
            raise Conflict("There is no org to pause")
        if self.org_state.paused is None:
            self.org_state.hold("human")
            self.start_line.org_held()
            self.work_queue.enqueue(self._pause_agents, PAUSE_LANE)
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
        store = await asyncio.to_thread(lambda: self.org)
        # One resume at a time: a second would run the same agents again.
        async with self._resume_lock:
            if self.org_state.paused is None:
                return await self.recovery()
            # A pause that is still stopping agents would stop them after this
            # has set them going again.
            await self.work_queue.after(PAUSE_LANE)
            run = self.org_state.run_state()
            await asyncio.to_thread(self._put_aside_stale)
            await asyncio.to_thread(
                restore_worktrees, self._repo_dir, store.places(), self._home()
            )
            dropped = self.org_state.dropped()
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
            self.org_state.resuming = set(again)
            self.org_state.release()
        await self._resync()
        for conversation_id in again:
            # Seen stopped, they were just taken for agents that do not run:
            # a server that ends before they are run again takes them up.
            run.mark(RUNNING, conversation_id, was_running[conversation_id])
        for place_id in waited:
            run.unmark(WAITING, place_id)
        intents = run.open_intents()
        starting = {intent.node_id for intent in intents}
        self.start_line.line_up(place_id for place_id in waited if place_id in starting)
        for intent in intents:
            self.resumable_work.take_up(intent.id, lane_for(intent))
        for held in run.held():
            self.work_queue.enqueue(
                lambda held=held: self.message_delivery.deliver(
                    held.seq, held.conversation_id, held.text
                ),
                held.node_id or ORG_LANE,
            )
        for place_id in run.marked(HELD):
            submitted = store.place(place_id)
            self.release_reviews(submitted.parent_id if submitted else None)
        for conversation_id, note in again.items():
            self.work_queue.enqueue(
                lambda conversation_id=conversation_id, note=note: (
                    self.team_messaging.run_again(conversation_id, note)
                ),
                was_running[conversation_id],
            )
        return Recovery(paused=None, items=dropped, messages_held=0)

    async def recovery(self) -> Recovery:
        """What resuming the org does, and what was put aside as stale."""
        store = await asyncio.to_thread(lambda: self.org)
        run = self.org_state.run_state()
        if self.org_state.paused is None:
            return Recovery(paused=None, items=[], messages_held=0)
        launcher = self._launcher
        statuses = (
            await launcher.statuses(
                [
                    UUID(conversation_id)
                    for node in store.places()
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
            paused=self.org_state.paused,
            items=[*items, *self.org_state.dropped()],
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
            Conflict: A merge waits for choices.
            NotFound: The repository has no such remote.
            UpstreamFailed: The remote could not be reached.
            Unavailable: The director is not running, or has no repository.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.org)
        async with self._sync_lock:
            if self.org_state.merging() is not None:
                raise Conflict(
                    "A merge waits for choices. Resolve it or abandon it first."
                )
            repo = await asyncio.to_thread(self._repository)
            await asyncio.to_thread(org_sync.fetch, repo, remote)
            theirs = await asyncio.to_thread(
                org_sync.commit_at, repo, org_sync.remote_org(remote)
            )
            if store.root_id is None:
                return await self.work_queue.result_in(
                    BASE_LANE, lambda: self._take_org(repo, remote, theirs)
                )
            resume = self.org_state.paused is None
            if resume:
                await self.pause()
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
        store = await asyncio.to_thread(lambda: self.org)
        if store.root_id is None:
            raise Conflict("There is no org to push")
        async with self._sync_lock:
            if self.org_state.merging() is not None:
                raise Conflict(
                    "A merge waits for choices. Resolve it or abandon it first."
                )
            repo = self._repo_dir

            def push() -> OrgSync:
                self.org_save.save_tree(store)
                org_sync.fetch(repo, remote)
                if org_sync.behind(repo, remote):
                    raise Conflict(
                        f"{remote} has changes of the org that are not here. "
                        "Pull first."
                    )
                org_sync.push(repo, remote)
                return OrgSync(
                    remote=remote,
                    result="pushed",
                    head=create_org_board(repo).head(),
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
        self._require_launcher()
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
                await self.resume()
            return OrgSync(
                remote=state["remote"],
                result="up_to_date",
                head=await asyncio.to_thread(create_org_board(self._repo_dir).head),
            )

    async def edit_file(self, place_id: str, path: str, content: str) -> FileEdit:
        """Write one file of a team by hand, commit it and tell its IC.

        Raises:
            NotFound: There is no such team, or its checkout is not here.
            Conflict: An agent of the team is in the middle of a turn.
            Forbidden: The team's agents may not write the file either.
            Invalid: The path does not name a file that can be committed.
        """
        store = await asyncio.to_thread(lambda: self.org)
        place = store.place(place_id)
        if place is None:
            raise NotFound(f"Unknown team: {place_id}")
        sub_teams = store.below(place_id)
        require_editable(place, sub_teams, path)
        agents = [
            UUID(agent.conversation_id)
            for agent in place.agents
            if agent.conversation_id is not None
        ]
        if agents and self._launcher is not None:
            statuses = await self._launcher.statuses(agents)
            if ConversationExecutionStatus.RUNNING in statuses.values():
                raise Conflict(
                    f"An agent of team {place.title!r} is working. Pause the "
                    "org or wait for it to finish, then save again."
                )
        edit = await asyncio.to_thread(write_by_hand, place, sub_teams, path, content)
        ic = self.team_types.writer_conversation(place)
        if edit.commit is not None and ic is not None and self._loop is not None:
            self.message_delivery.notify(
                ic,
                f"The human changed {edit.path} in your team's checkout by hand "
                f"and committed it as {edit.commit[:12]}. Read the file again "
                "before you change it, and keep what they wrote unless they "
                "tell you otherwise.",
            )
        return edit

    async def start_again(
        self, node_id: str, request: StartAgainRequest, said: str | None = None
    ) -> SessionTeam:
        """Start afresh a team whose agents have no conversation on this
        server, on the work its branch has.

        The team's IC starts in the team's own checkout and is briefed
        from what the org kept of the team and its last headline; its
        Manager is derived from it.
        The IC starts as the given conversation, or without one like the IC
        of the team above (from the team's own agent profile if it has one).
        What the human ``said`` when asked about the lost IC ends its briefing.

        Raises:
            NotFound: There is no such team.
            Conflict: The team is done, its IC has a conversation here, or
                no conversation was given and the team above has no IC here
                to start it like.
            Unavailable: The director is not running.
        """
        names = self.team_types.names_at(node_id)
        launcher = self._require_launcher()
        store = await asyncio.to_thread(lambda: self.org)
        place = store.place(node_id)
        if place is None:
            team = store.team(node_id)
            if team is None:
                raise NotFound(f"There is no team {node_id}")
            raise Conflict(
                f"Nobody works at {team.title!r} itself: open a session there."
            )
        # A session's reviewer starts like the first time: from the IC the
        # session works with, in the checkout it reviews in.
        leads = is_division_lead(place)
        if place.status == "done":
            raise Conflict(f"Team {place.title} is done")
        ic = self.team_types.writer_conversation(place)
        if ic is not None and await launcher.started(UUID(ic)) is not None:
            raise Conflict(
                f"The {names.writer} of team {place.title} has a conversation here"
            )
        conversation = request.conversation
        like = self.agent_start.derived_from(place) if leads else place.parent_id
        above = store.place(like) if like else None
        # A session's IC that never started has no checkout yet: it gets
        # one of its own, never the one of the team it works beside.
        own_checkout = not leads and not (
            above is not None and store.opens_session(place) and not apart(place, above)
        )
        if conversation is not None and not own_checkout:
            raise Conflict(
                f"The session {store.session_title(place)!r} has no checkout yet. "
                "Start it again without a conversation."
            )
        if (
            conversation is None
            and await self.agent_start.started_ic(above) is None
            and await self.agent_start.template() is None
        ):
            raise Conflict(
                "No conversation of the org runs on this server to start "
                f"the {names.writer} of {place.title!r} from. Start it again with a "
                "conversation."
            )
        await asyncio.to_thread(
            restore_worktrees, self._repo_dir, [place], self._home()
        )
        ic_id = uuid4()
        task = await asyncio.to_thread(
            lambda: (
                self.team_messaging.purpose(place) if leads else self._task_of(place)
            )
        )
        if said:
            task = (
                f"{task}{STARTED_WITH}The human was told that the {names.writer} "
                "before "
                f"you was lost, and says:\n{said}"
            )
        # The team's other agents stay its own: only the IC is new.
        writer = self.team_types.team(place).writer
        tools = self.team_types.tools_for(place, writer)
        agents = [
            *new_agents(ic_id, writer),
            *(a for a in place.agents if a.role != writer),
        ]
        if conversation is None:
            store.update_place(node_id, agents=agents, status="idle")
            self.resumable_work.begin(
                node_id,
                StartTeam(
                    task=task,
                    parent=like,
                    checkout=place.worktree_path
                    if own_checkout or self.agent_start.in_place(place)
                    else None,
                ),
            )
            return store.place(node_id) or place
        store.update_place(node_id, agents=agents, status="idle")
        try:
            started = await launcher.start_root(
                conversation.model_copy(
                    update={
                        "conversation_id": ic_id,
                        "initial_message": None,
                        "autotitle": False,
                        "worktree": False,
                        "workspace": conversation.workspace.model_copy(
                            update={"working_dir": place.worktree_path}
                        ),
                    }
                ),
                [Tool(name=name) for name in tools],
                title=agent_title(
                    place.title,
                    self.team_types.team(place).title(writer),
                    store.session_title(place),
                ),
                tool_params=await asyncio.to_thread(
                    self.confinement.confine, node_id, writer
                ),
                adapt_agent=await asyncio.to_thread(
                    self.confinement.adapter, str(ic_id), tools
                ),
            )
        except BaseException:
            store.update_place(node_id, agents=place.agents, status=place.status)
            raise
        store.set_agent(
            node_id,
            writer,
            kind=started.kind,
            acp_server=started.acp_server,
            model=started.model,
        )
        self.resumable_work.begin(node_id, Brief(role=writer, task=task))
        return store.place(node_id) or place

    async def _take_org(self, repo: Path, remote: str, theirs: str | None) -> OrgSync:
        """Take the remote's org as the org of a repository that has none."""
        if theirs is None:
            return OrgSync(remote=remote, result="up_to_date", head=None)

        def take() -> OrgSync:
            board = create_org_board(repo)
            if not board.move_to(theirs):
                raise Conflict("The org was changed meanwhile. Pull once more.")
            pulled = self._pull_memory(repo, remote, ())
            self._join(repo)
            return OrgSync(
                remote=remote,
                result="fast_forward",
                head=board.head(),
                memory=pulled,
            )

        outcome = await asyncio.to_thread(take)
        await asyncio.to_thread(self._working)
        await self._came_back()
        return outcome

    async def _merge(self, state: dict[str, Any]) -> OrgSync:
        """Carry a merge as far as its choices allow."""
        try:
            outcome = await asyncio.to_thread(self._weigh, state, True)
        except Exception:
            # Nothing waits to be resolved: the org the pull paused goes on.
            if state["resume"] and self.org_state.merging() is None:
                await self.resume()
            raise
        if outcome.result == "conflicts":
            self.org_state.set_merge(state)
            return outcome
        self.org_state.set_merge(None)
        await self._resync()
        if state["resume"]:
            await self.resume()
        return outcome

    def _weigh(self, state: dict[str, Any], land: bool = False) -> OrgSync:
        """Work a merge out and, when ``land`` is set and no conflict is
        open, write it."""
        store = self.org
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

    def _put_aside_stale(self) -> None:
        run = self.org_state.run_state()
        dropped = put_aside_stale(run, self.org, instance=self._instance())
        if run.get("paused") is None:
            # Nothing that was left is this server's to take up.
            self.org_state.refresh()
        if dropped:
            run.put(
                "dropped",
                json.dumps(
                    [
                        item.model_dump()
                        for item in [*self.org_state.dropped(), *dropped]
                    ]
                ),
            )

    async def _came_back(self) -> None:
        """Look at what an org that was just loaded was left in the middle
        of. A server that ended without stopping its org leaves it paused."""
        await asyncio.to_thread(self._look_at_what_was_left)
        await self._resync()
        if (
            self.org_state.paused in ("shutdown", "crash")
            and self.org_settings.settings.resume_on_start
        ):
            await self.resume()

    def _look_at_what_was_left(self) -> None:
        if self.org.root_id is None:
            return
        run = self.org_state.run_state()
        self._put_aside_stale()
        if self.org_state.paused is None and run.get("clean") == "0" and has_work(run):
            self.org_state.hold("crash")
            self.start_line.org_held()
        if self.org_state.paused is None:
            for item in self.org_state.dropped():
                logger.warning("Run state put aside: %s", item.detail)
            run.put("dropped", None)
        self._working()

    async def _pause_agents(self) -> None:
        store = self.org
        run = self.org_state.run_state()
        launcher = self._require_launcher()
        agents: list[tuple[str, str, UUID]] = [
            (node.id, agent.role, UUID(agent.conversation_id))
            for node in store.places()
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        statuses = await launcher.statuses([agent for _, _, agent in agents])
        stopping: list[UUID] = []
        for place_id, role, conversation_id in agents:
            if statuses.get(conversation_id) != ConversationExecutionStatus.RUNNING:
                continue
            run.mark(RUNNING, str(conversation_id), place_id)
            self.org_state.running.add(str(conversation_id))
            with suppress(NotFound):
                store.set_agent(place_id, role, status=AGENT_PAUSED)
            stopping.append(conversation_id)
        # An IC in hand-over mode is paused in that conversation too.
        modes = [
            (node.id, mode)
            for node in store.places()
            if (mode := hand_over_of(node)) is not None
        ]
        running = await launcher.statuses(
            [UUID(mode.conversation_id) for _, mode in modes]
        )
        for place_id, mode in modes:
            aside = UUID(mode.conversation_id)
            if running.get(aside) != ConversationExecutionStatus.RUNNING:
                continue
            run.mark(RUNNING, mode.conversation_id, place_id)
            self.org_state.running.add(mode.conversation_id)
            with suppress(NotFound):
                store.set_hand_over(
                    place_id, mode.model_copy(update={"status": AGENT_PAUSED})
                )
            stopping.append(aside)
        await asyncio.gather(*(launcher.pause(agent) for agent in stopping))

    def _on_status(
        self, conversation_id: UUID, status: ConversationExecutionStatus
    ) -> None:
        if status != ConversationExecutionStatus.RUNNING and self._loop is not None:
            self._loop.call_soon_threadsafe(self.start_line.room_made)
        store = self._store
        if store is None:
            return
        agent = str(conversation_id)
        found = store.find_by_conversation(agent)
        if found is None:
            self._on_hand_over_status(store, agent, status)
            return
        node, role = found
        if status == ConversationExecutionStatus.RUNNING:
            self.org_state.resuming.discard(agent)
        if self.org_state.paused is None:
            self.message_delivery.note_running(
                agent, node.id, status == ConversationExecutionStatus.RUNNING
            )
        was = node.status
        shown = self._shown(agent, status)
        failed = shown == AGENT_BLOCKED and all(
            a.status != AGENT_BLOCKED for a in node.agents if a.role == role
        )
        now = store.set_agent(node.id, role, status=shown)
        if self.org_state.paused is None and self._loop is not None:
            if failed:
                self.work_queue.enqueue(
                    lambda: self._failed(node.id, role, conversation_id, status),
                    node.id,
                )
            if status == ConversationExecutionStatus.RUNNING:
                self._runs_again(now, role)
            if now.status == "blocked" and was != "blocked":
                # A blocked team is no longer waited for.
                self.release_reviews(node.parent_id)
            if (
                role == self.team_types.team(node).writer
                and status == ConversationExecutionStatus.RUNNING
            ):
                self.release_reviews(node.id)
            if status != ConversationExecutionStatus.RUNNING:
                self._remind_to_submit(node.session)

    async def _failed(
        self,
        place_id: str,
        role: str,
        conversation_id: UUID,
        status: ConversationExecutionStatus,
    ) -> None:
        """Find why an agent's conversation failed, keep it and tell of it."""
        said = await self._require_launcher().failure(conversation_id)
        if said is None:
            said = (
                "It repeated the same step without getting anywhere."
                if status == ConversationExecutionStatus.STUCK
                else "Its conversation ended with an error it gave no account of."
            )
        team = self.org.place(place_id)
        if team is None or all(
            a.status != AGENT_BLOCKED for a in team.agents if a.role == role
        ):
            # It runs again already.
            return
        self.team_messaging.tell_of_failure(place_id, role, said)

    def acts(self, place_id: str, role: str) -> None:
        """An agent that uses one of the director's tools is at work,
        whatever its conversation last said of itself. Safe from any thread.

        A conversation that fails and is started again by what runs it may
        never say that it runs: the failure the human was told of would
        then stay with them after the agent has gone on.
        """
        team = self.org.place(place_id)
        if team is None or all(
            a.status != AGENT_BLOCKED for a in team.agents if a.role == role
        ):
            return
        with suppress(NotFound):
            self._runs_again(self.org.set_agent(place_id, role, status="working"), role)

    def _runs_again(self, team: SessionTeam, role: str) -> None:
        """An agent that runs is no longer something the human is asked of."""
        waits = {failure_waits(role)} | (
            {STALL_WAITS} if role == self.team_types.team(team).writer else set()
        )
        settled = [e.id for e in team.escalations if e.id in waits]
        for escalation_id in settled:
            with suppress(NotFound):
                self.org.settle_escalation(team.id, escalation_id)
        if settled:
            self.org_save.save_soon()

    def _on_hand_over_status(
        self, store: OrgStore, conversation_id: str, status: ConversationExecutionStatus
    ) -> None:
        """Follow the hand-over conversation of an IC. When it stops
        working, the IC is told in its main conversation what it did."""
        node = store.find_hand_over(conversation_id)
        mode = hand_over_of(node) if node else None
        if node is None or mode is None:
            return
        if status == ConversationExecutionStatus.RUNNING:
            self.org_state.resuming.discard(conversation_id)
        if self.org_state.paused is None:
            self.message_delivery.note_running(
                conversation_id, node.id, status == ConversationExecutionStatus.RUNNING
            )
        shown = self._shown(conversation_id, status)
        store.set_hand_over(node.id, mode.model_copy(update={"status": shown}))
        # One that was held while it worked may have ended while it was held.
        at_work = ("working", AGENT_PAUSED)
        if mode.status in at_work and shown not in at_work:
            self._hand_over_ended(node.id)
            self._remind_to_submit(node.session)

    def _remind_to_submit(self, session: str) -> None:
        """Tell the IC of a session at rest that what it changed since its
        work was accepted is not the org's until it submits again. Once for
        each accept. Safe from any thread."""
        if (
            self._changed_since_accept.get(session, True)
            or self.org_state.paused is not None
        ):
            return
        places = self.org.session(session)
        at_work = ("working", AGENT_PAUSED)
        if any(
            place.status == "working"
            or ((mode := hand_over_of(place)) is not None and mode.status in at_work)
            for place in places
        ):
            return
        self._changed_since_accept[session] = True
        top = places[0] if places else None
        if top is None or top.review is None or top.review.state != "accepted":
            return
        self.message_delivery.notify(
            self.team_types.writer_conversation(top),
            changed_since_accept(self.team_types.kept_documents(top)),
        )

    async def _resync(self) -> None:
        """Read every agent's status back from its conversation."""
        store = self.org
        agents: list[tuple[str, str, UUID]] = [
            (node.id, agent.role, UUID(agent.conversation_id))
            for node in store.places()
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        if not agents:
            return
        # Shells that were lost with the git directory come back for the
        # agents that still start theirs.
        await asyncio.to_thread(self.confinement.confine_all)
        statuses = await self._require_launcher().statuses(
            [conversation_id for _, _, conversation_id in agents]
        )
        starting = {
            intent.node_id
            for intent in self.org_state.run_state().open_intents()
            if intent.kind in STARTING
        }
        for place_id, role, conversation_id in agents:
            status = statuses.get(conversation_id)
            agent = str(conversation_id)
            if self.org_state.paused is None:
                self.message_delivery.note_running(
                    agent, place_id, status == ConversationExecutionStatus.RUNNING
                )
            if status is not None:
                shown = self._shown(agent, status)
            else:
                shown = "idle" if place_id in starting else AGENT_LOST
            with suppress(NotFound):
                store.set_agent(place_id, role, status=shown)
        await self._resync_hand_overs()

    async def _resync_hand_overs(self) -> None:
        """Read back how each IC's hand-over conversation stands. One that
        ended or was lost while nothing followed it is ended here."""
        store = self.org
        modes = [
            (node.id, mode)
            for node in store.places()
            if (mode := hand_over_of(node)) is not None
        ]
        if not modes:
            return
        statuses = await self._require_launcher().statuses(
            [UUID(mode.conversation_id) for _, mode in modes]
        )
        starting = {
            intent.node_id
            for intent in self.org_state.run_state().open_intents()
            if intent.kind == "hand_over"
        }
        for place_id, mode in modes:
            status = statuses.get(UUID(mode.conversation_id))
            if status is not None:
                self._on_hand_over_status(store, mode.conversation_id, status)
            elif place_id not in starting and mode.status != AGENT_LOST:
                store.set_hand_over(
                    place_id, mode.model_copy(update={"status": AGENT_LOST})
                )
                self._hand_over_ended(place_id)
