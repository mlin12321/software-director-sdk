"""Starting a team's agents and what is sent to them."""

import asyncio
import json
import logging
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from openhands.agent_server.director.context import (
    briefing_block,
    steady,
)
from openhands.agent_server.director.git import (
    GitError,
    bring_over,
    head_of,
    uncommitted,
)
from openhands.agent_server.director.git_store import (
    CONTRACTS_FILE,
)
from openhands.agent_server.director.models import (
    AgentRef,
    AgentRole,
    OrgNode,
    Report,
    ThreadMessage,
)
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    reviews_only,
    session_id_of,
    session_team_id,
)
from openhands.agent_server.director.pods import StartedAgent
from openhands.agent_server.director.prompts import (
    HAND_OVER_ROLE,
    MANAGER_ROLE,
    POD_PROTOCOL,
    REVIEW_MANAGER_ROLE,
    REVIEW_ROLE,
    brief,
    hand_over_brief,
    left_for_team,
    manager_check,
)
from openhands.agent_server.director.run_state import (
    CHECKS,
    HANDING,
    PROGRESS,
    TASK,
    TOLD,
    Intent,
)
from openhands.agent_server.director.service.delivery import Delivery
from openhands.agent_server.director.service.state import (
    _STARTED_WITH,
    HAND_OVER_TOOLS,
    LEFT_REFS,
    REVIEW_MANAGER_TOOLS,
    REVIEW_TOOLS,
    TOOLS_OF,
    _agent_title,
    _base_of,
    _branch_of,
    _checkout_root,
    _Held,
    _instant,
    _new_pod,
    _progress_of,
    agent_conversation,
    team_of,
)
from openhands.sdk.errors import (
    NotFound,
)
from openhands.sdk.tool.spec import Tool


logger = logging.getLogger(__name__)


class Teams(Delivery):
    """Starts agents and carries what is said between teams."""

    def message(
        self,
        manager_node_id: str,
        ic_node_id: str,
        sender: AgentRole,
        text: str,
        note: str = "",
    ) -> None:
        """Keep and deliver a message between a Manager and an IC. Safe
        from any thread.

        Args:
            manager_node_id: The Manager's team.
            ic_node_id: The IC's team: the same one, or one directly below.
            note: What the director adds for the reader; it is not kept.
        """
        store = self.store
        manager_node, ic_node = store.get(manager_node_id), store.get(ic_node_id)
        if manager_node is None or ic_node is None:
            raise NotFound("That team is no longer in the org")
        self._memory().append_message(
            manager_node_id,
            ic_node_id,
            ThreadMessage(at=_instant(), sender=sender, text=text),
        )
        if sender == "ic":
            self.tell_manager(manager_node_id, f"Your IC writes:\n{text}{note}")
            return
        heading = (
            "Your Manager writes:"
            if manager_node_id == ic_node_id
            else f"The Manager of the team above ({manager_node.title}) writes:"
        )
        self.tell_ic(ic_node_id, f"{heading}\n{text}{note}")

    def tell_manager(self, node_id: str, text: str, *, review: bool = False) -> None:
        """Send a team's Manager a message from the director. A team has no
        Manager until something is sent to it: the first message starts
        one, and follows its briefing. Safe from any thread.

        Args:
            review: The text asks it to settle a session's sub-teams.
                Nothing else reaches the Manager of the place a session's
                work is reviewed from.
        """
        store = self.store
        with self._adding_manager:
            node = store.get(node_id)
            # Nobody stands at a team of the org itself: agents are at its
            # places in the sessions.
            if node is None or not session_id_of(node):
                return
            if reviews_only(node) and not review:
                return
            if agent_conversation(node, "manager") is None:
                manager = AgentRef(
                    role="manager", conversation_id=str(uuid4()), status="idle"
                )
                node = store.update_node(node_id, agents=[*node.agents, manager])
                task = self._runs().marked(TASK).get(node_id) or node.spec_summary
                self._begin("launch_manager", node_id, task=task)
        self.notify(agent_conversation(node, "manager"), text)

    def tell_ic(
        self, node_id: str, text: str, *, human: bool = False, review: bool = False
    ) -> str | None:
        """Send a team's IC a message. A team that was only set up has no IC
        until something is sent to it: the first message starts one and is
        part of its briefing. Safe from any thread.

        Args:
            human: The human's own words, which reach an agent at once and
                are not marked as the director's.
            review: The text asks for a review. Nothing else starts the IC
                of the place a session's work is reviewed from.

        Returns:
            The IC's conversation; None when the team is gone.
        """
        store = self.store
        with self._adding_manager:
            node = store.get(node_id)
            if node is None:
                return None
            if not session_id_of(node):
                # Nobody stands at a team of the org itself.
                return None
            conversation_id = agent_conversation(node, "ic")
            reviewing = reviews_only(node)
            if conversation_id is None and reviewing and not review:
                return None
            if conversation_id is None:
                [ic] = _new_pod(uuid4())
                store.update_node(node_id, agents=[ic, *node.agents])
                purpose = self._purpose(node)
                self._begin(
                    "start_team",
                    node_id,
                    task=f"{purpose}{_STARTED_WITH}{text}",
                    purpose=purpose,
                    parent=self._derived_from(node),
                    **(
                        {"checkout": node.worktree_path} if self._in_place(node) else {}
                    ),
                )
                if human and ic.conversation_id:
                    self.joined(ic.conversation_id)
                return ic.conversation_id
        if human:
            self._post(conversation_id, text, at_once=True)
            self.joined(conversation_id)
        else:
            self.notify(conversation_id, text)
        return conversation_id

    def _in_place(self, node: OrgNode) -> bool:
        """Whether a team that starts works in the checkout it has: the
        place a session is reviewed from, and the team a session is opened
        at when sessions get no worktree or the session has one already."""
        return reviews_only(node) or (
            self.store.opens_session(node)
            and (
                node.isolation != "worktree"
                or _checkout_root(node.worktree_path)
                != _checkout_root(str(self._repo_dir))
            )
        )

    def _derived_from(self, node: OrgNode) -> str | None:
        """The place whose IC the IC of ``node`` is derived from when it
        starts: the team above, or for the team a session is opened at and
        its reviewer, who of the session works already."""
        store = self.store
        opened = store.session_of(node)
        if reviews_only(node) and opened:
            return session_team_id(opened.team_id, opened.id)
        if store.opens_session(node):
            # A session that went a team higher has an IC below already.
            return next(
                (team.id for team in store.children(node.id) if team.agents),
                node.parent_id,
            )
        return node.parent_id

    def _tools_of(self, node: OrgNode, role: AgentRole) -> tuple[str, ...]:
        """The director's tools an agent of ``node`` is offered."""
        if reviews_only(node):
            return REVIEW_TOOLS if role == "ic" else REVIEW_MANAGER_TOOLS
        return TOOLS_OF[role]

    async def _template(self) -> str | None:
        """A conversation of the org to derive an agent from when nothing
        above it has started: the IC a session started with, oldest first."""
        store, launcher = self.store, self._require_launcher()
        for session in store.sessions():
            ic = session.conversation_id
            if ic is not None and await launcher.started(UUID(ic)) is not None:
                return ic
        return None

    def _purpose(self, node: OrgNode) -> str:
        """What a team is for, without the message it was started with."""
        task = self._runs().marked(TASK).get(node.id) or node.spec_summary
        return task.split(_STARTED_WITH)[0]

    def message_team(self, node_id: str, team_id: str, text: str) -> bool:
        """Keep and deliver what a team's IC writes to the IC of a team
        directly below. A team that was stopped goes on with it. Safe from
        any thread.

        Returns:
            Whether the team was stopped.
        """
        store = self.store
        node, team = store.get(node_id), store.get(team_id)
        if node is None or team is None:
            raise NotFound("That team is no longer in the org")
        self._memory().append_message(
            node_id,
            team_id,
            ThreadMessage(at=_instant(), sender="ic", text=text),
            ic_to_ic=True,
        )
        stopped = team.stopped
        if stopped is not None:
            store.set_stopped(team_id, None)
            self._enqueue(lambda: self._go_on(team_id, stopped.running), team_id)
        self.progressed(team_id)
        self.tell_ic(
            team_id,
            f"The IC of the team above ({node.title}) writes:\n{text}"
            + (
                "\n\nIt had stopped your team; with this message your team goes on."
                if stopped is not None
                else ""
            ),
        )
        return stopped is not None

    def count_accepted_review(self, node_id: str) -> None:
        """Count a review a team's IC accepted, and wake the team's Manager
        to look at its teams when that makes as many as the org set. Safe
        from any thread."""
        limit = self.settings.manager_check_accepted_reviews
        if not limit:
            return
        with self._counting:
            counted = self._counted(node_id)
            counted["accepted"] += 1
            due = counted["accepted"] >= limit
            if due:
                counted["accepted"] = 0
            self._runs().mark(CHECKS, node_id, json.dumps(counted))
        if due:
            self._wake_manager(
                node_id,
                f"your IC has accepted {limit} reviews since the last look "
                "of this kind.",
                counted,
                after_successes=True,
            )

    def count_ic_escalation(self, node_id: str, team_id: str) -> None:
        """Count an escalation the IC of a team directly below raised to a
        team's IC, and wake the team's Manager when that team has raised as
        many, on the work it has in hand, as the org set. Safe from any
        thread."""
        limit = self.settings.manager_check_ic_setbacks
        team = self.store.get(team_id)
        if not limit or team is None:
            return
        with self._counting:
            counted = self._counted(node_id)
            raised = counted["escalations"].get(team_id, 0) + 1
            due = raised >= limit
            told = dict(counted["escalations"], **{team_id: raised})
            counted["escalations"][team_id] = 0 if due else raised
            self._runs().mark(CHECKS, node_id, json.dumps(counted))
        if due:
            self._wake_manager(
                node_id,
                f"the IC of {team.title!r} has escalated {limit} times to "
                "your IC on the work it has in hand.",
                {**counted, "escalations": told},
                team=team,
            )

    def _wake_manager(
        self,
        node_id: str,
        reason: str,
        counted: dict[str, Any],
        *,
        team: OrgNode | None = None,
        after_successes: bool = False,
    ) -> None:
        """Wake a team's Manager to look at its teams, and keep that it was.

        Args:
            team: The team whose escalations reached the limit, when that
                is why; it is kept with the event, so that the record says
                on which of a team's tasks the limit was reached.
        """
        store = self.store
        now = self._clock()
        quiet = {}
        for team_id, kept in self._runs().marked(PROGRESS).items():
            at = _progress_of(kept).get("at")
            if at:
                since = now - datetime.fromisoformat(at)
                quiet[team_id] = int(since.total_seconds() // 60)
        teams = store.children(node_id)
        waiting = {}
        for below in teams:
            raised = [e.raised_at for e in below.escalations if e.raised_by == "ic"]
            if raised:
                since = now - datetime.fromisoformat(min(raised))
                waiting[below.id] = int(since.total_seconds() // 60)
        self.record_event(node_id, "woken", team, reason)
        self.tell_manager(
            node_id,
            manager_check(
                reason,
                teams,
                quiet,
                waiting,
                counted["escalations"],
                self.team_memory.history(node_id),
                after_successes=after_successes,
                escalation_limit=self.settings.manager_check_ic_setbacks,
            ),
        )

    async def _stop(self, node_id: str) -> None:
        team = self.store.get(node_id)
        if team is None or team.stopped is None or self._paused is not None:
            return
        launcher = self._require_launcher()
        for conversation_id in self._conversations_below(team):
            if conversation_id in self._running:
                with suppress(NotFound):
                    await launcher.pause(UUID(conversation_id))

    async def _go_on(
        self, node_id: str, running: list[str], *, own_too: bool = False
    ) -> None:
        """Set a team that was stopped going again: what waited for its
        agents is delivered, and those cut off in a turn run again.

        Args:
            own_too: Run the team's own IC again as well, when no message
                to it does.
        """
        team = self.store.get(node_id)
        if team is None or team.stopped is not None or self._paused is not None:
            return
        own = None if own_too else agent_conversation(team, "ic")
        conversations = self._conversations_below(team)
        # In the order it was sent, and before the message that set the
        # team going, which waits behind this in the team's lane.
        for held in self._runs().held():
            if held.conversation_id in conversations:
                self._at_once.add(held.seq)
                await self._deliver(held.seq, held.conversation_id, held.text)
        for conversation_id in conversations:
            # The team's own IC is run by the message that set it going.
            if conversation_id in running and conversation_id != own:
                await self._run_again(conversation_id, None)

    def progressed(self, node_id: str | None) -> None:
        """Note that a team got somewhere just now: the time it may show no
        progress starts again. Safe from any thread."""
        if node_id is None:
            return
        run = self._runs()
        seen = _progress_of(run.marked(PROGRESS).get(node_id))
        run.mark(
            PROGRESS,
            node_id,
            json.dumps({"at": self._clock().isoformat(), "head": seen.get("head")}),
        )

    def hand_over(self, team: OrgNode, notes: str, contracts: str) -> None:
        """Give a team what its IC above passes on of the teams that were
        there before it: notes, kept with the team's reports, and the
        contracts its own sub-teams must keep. Starts nothing. Safe from
        any thread."""
        team_memory = self._memory()
        if notes.strip():
            team_memory.append_report(
                team_of(team),
                "ic",
                Report(at=_instant(), role="ic", text=notes.strip(), handed_over=True),
            )
        if contracts.strip():
            team_memory.write(
                team.boundary.path, CONTRACTS_FILE, f"{contracts.strip()}\n"
            )

    def _task_of(self, node: OrgNode) -> str:
        """What a team that is started again is to do, from what the org
        kept of it."""
        above = self.store.get(node.parent_id) if node.parent_id else None
        specs, _ = self.team_memory.applies_to(
            node.boundary.path, above.boundary.path if above else None
        )
        task = specs[0].spec if specs else node.spec_summary
        if node.headline and node.headline not in task:
            task = f"{task}\n\nWhere the team last was: {node.headline}"
        return (
            f"{task}\n\nThis team was worked on before, on another machine. "
            "What it committed is in your checkout: look at it first and "
            "carry on from there."
        )

    async def _run_again(self, conversation_id: str, note: str | None) -> None:
        self._resuming.discard(conversation_id)
        if self._is_stopped(conversation_id):
            # Its team was stopped meanwhile and goes on when it is told to.
            return
        with suppress(NotFound):
            await self._require_launcher().resume(UUID(conversation_id), note)

    def _briefing(self, node: OrgNode, role: AgentRole, task: str) -> tuple[str, str]:
        """An agent's first message, and the context it tells, as kept to
        find later what changed."""
        team_memory = self._memory()
        above = self.store.get(node.parent_id) if node.parent_id else None
        _, contracts = team_memory.applies_to(
            node.boundary.path, above.boundary.path if above else None
        )
        context = self.context_of(node, role)
        text = brief(
            node,
            role,
            task,
            self.specs_of(node),
            contracts,
            self.reports_of(node),
            briefing_block(context),
        )
        return text, json.dumps(steady(context))

    def _hand_over_briefing(self, node: OrgNode, job: str) -> str:
        """The first message of an IC's hand-over conversation."""
        above = self.store.get(node.parent_id) if node.parent_id else None
        _, contracts = self._memory().applies_to(
            node.boundary.path, above.boundary.path if above else None
        )
        return hand_over_brief(
            node,
            job,
            self.specs_of(node),
            contracts,
            self.reports_of(node),
            briefing_block(self.context_of(node, "ic")),
        )

    async def _start_hand_over(
        self, node: OrgNode, conversation_id: str, job: str
    ) -> bool:
        """Start the hand-over conversation of a node's IC, derived from its
        main one.

        Returns:
            Whether it runs.
        """
        launcher = self._require_launcher()
        try:
            if await launcher.started(UUID(conversation_id)) is not None:
                return True
            parent = await self._started_ic(node)
            if parent is None:
                raise NotFound(f"Node {node.id} has no IC to hand over")
            text = await asyncio.to_thread(self._hand_over_briefing, node, job)
            tool_params = await asyncio.to_thread(
                self._confine, node.id, "ic", conversation_id
            )
            adapt = await asyncio.to_thread(
                self._adapter, conversation_id, HAND_OVER_TOOLS
            )

            async def start() -> StartedAgent:
                return await launcher.start_derived(
                    UUID(parent),
                    UUID(conversation_id),
                    title=f"{node.title} (IC, hand-over)",
                    task=text,
                    isolation="shared",
                    suffixes=[HAND_OVER_ROLE],
                    tool_params=tool_params,
                    adapt_agent=adapt,
                )

            await self._when_there_is_room(node.id, start)
            if self._paused is not None:
                self._note_running(conversation_id, node.id, True)
                await launcher.pause(UUID(conversation_id))
        except _Held:
            raise
        except Exception:
            logger.exception("Could not start the hand-over of node %s", node.id)
            return False
        return True

    async def _brief(self, node_id: str, role: AgentRole, task: str) -> None:
        node = self.store.get(node_id)
        if node is None:
            return
        conversation_id = agent_conversation(node, role)
        if conversation_id is not None:
            text, told = await asyncio.to_thread(self._briefing, node, role, task)
            self._runs().mark(TOLD, conversation_id, told)
            await self._send(conversation_id, text)

    async def _start_agent(
        self,
        node_id: str,
        role: AgentRole,
        parent_conversation: str | None,
        task: str,
        checkout: str | None = None,
    ) -> bool:
        """Start one agent of a node from ``parent_conversation``.

        Args:
            checkout: The team's checkout when it has one already and the
                agent is to work on in it.

        Returns:
            Whether it runs. A failure is recorded on the node.
        """
        store = self.store
        node = store.get(node_id)
        conversation_id = agent_conversation(node, role) if node else None
        if node is None or conversation_id is None:
            return False
        launcher = self._require_launcher()
        try:
            started = await launcher.started(UUID(conversation_id))
            if started is None:
                started = await self._start_new(
                    node, role, conversation_id, parent_conversation, task, checkout
                )
                if self._paused is not None:
                    self._note_running(conversation_id, node_id, True)
                    await launcher.pause(UUID(conversation_id))
        except _Held:
            raise
        except Exception:
            logger.exception("Could not start the %s of node %s", role, node_id)
            store.set_agent(node_id, role, status=AGENT_LOST)
            return False
        store.set_agent(node_id, role, kind=started.kind, acp_server=started.acp_server)
        if role == "ic":
            store.update_node(
                node_id,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(_branch_of, started.workspace),
                base_commit=node.base_commit
                or await asyncio.to_thread(_base_of, started.workspace),
                boundary_enforcement=self._enforcement(started.kind),
            )
        return True

    async def _start_new(
        self,
        node: OrgNode,
        role: AgentRole,
        conversation_id: str,
        parent_conversation: str | None,
        task: str,
        checkout: str | None = None,
    ) -> StartedAgent:
        """Start the conversation of an agent that has none yet.

        Raises:
            NotFound: There is nobody to derive the agent from.
            _Held: The org was paused while the team waited for a run.
        """
        node_id = node.id
        launcher = self._require_launcher()
        text, told = await asyncio.to_thread(self._briefing, node, role, task)
        tool_params = await asyncio.to_thread(self._confine, node_id, role)
        if parent_conversation is None:
            raise NotFound(f"Node {node_id} has nobody to derive its {role} from")
        parent_id = UUID(parent_conversation)
        # A team given a profile of its own starts its IC from it; its
        # Manager, like every Manager, is derived from the IC.
        profile = node.agent_profile if role == "ic" else None
        kind = await (
            launcher.profile_kind(profile)
            if profile
            else launcher.agent_kind(parent_id)
        )
        isolation = node.isolation if role == "ic" else "shared"
        reviewing = reviews_only(node)
        if kind == "acp" and role == "ic" and self._bwrap is None:
            # Nothing holds its process to the team's directory, so its
            # work is kept apart and checked when it is merged.
            isolation = "worktree"
        tools = self._tools_of(node, role)
        adapt = await asyncio.to_thread(self._adapter, conversation_id, tools)
        # A session starts from where the repository's checkout is.
        base = (
            await asyncio.to_thread(head_of, self._repo_dir)
            if role == "ic"
            and not checkout
            and isolation == "worktree"
            and self.store.opens_session(node)
            else None
        )
        title = _agent_title(
            node.title,
            role,
            self.store.session_title(node),
            review=reviews_only(node) and role == "ic",
        )

        async def start() -> StartedAgent:
            if profile:
                return await launcher.start_from_profile(
                    profile,
                    parent_id,
                    UUID(conversation_id),
                    [Tool(name=name) for name in tools],
                    title=title,
                    task=text,
                    isolation=isolation,
                    suffixes=[POD_PROTOCOL, *([REVIEW_ROLE] if reviewing else [])],
                    tool_params=tool_params,
                    adapt_agent=adapt,
                    checkout=checkout,
                    base=base,
                )
            return await launcher.start_derived(
                parent_id,
                UUID(conversation_id),
                title=title,
                task=text,
                isolation=isolation,
                suffixes=[REVIEW_ROLE if role == "ic" else REVIEW_MANAGER_ROLE]
                if reviewing
                else []
                if role == "ic"
                else [MANAGER_ROLE],
                tool_params=tool_params,
                adapt_agent=adapt,
                checkout=checkout,
                base=base,
            )

        started = await self._when_there_is_room(node_id, start)
        self._runs().mark(TOLD, conversation_id, told)
        return started

    async def _started_ic(self, node: OrgNode | None) -> str | None:
        """The IC of ``node`` or, while that team has not started, of the
        nearest team above that has: the conversation a new agent is
        derived from."""
        store, launcher = self.store, self._require_launcher()
        while node is not None:
            ic = agent_conversation(node, "ic")
            if ic is not None and await launcher.started(UUID(ic)) is not None:
                return ic
            node = store.get(node.parent_id) if node.parent_id else None
        return None

    async def _launch_manager(self, node_id: str, task: str) -> None:
        node = self.store.get(node_id)
        if node is not None:
            # Where a session is reviewed from, its IC may never have
            # started: the session's own then stands in.
            derived_from = await self._started_ic(node) or (
                await self._started_ic(self.store.get(self._derived_from(node) or ""))
                if reviews_only(node)
                else None
            )
            await self._start_agent(node_id, "manager", derived_from, task)

    async def _do_brief(self, intent: Intent) -> None:
        if intent.payload["role"] == "ic":
            self._runs().mark(TASK, intent.node_id, intent.payload["task"])
        await self._brief(
            intent.node_id, intent.payload["role"], intent.payload["task"]
        )

    async def _do_launch_manager(self, intent: Intent) -> None:
        await self._launch_manager(intent.node_id, intent.payload["task"])

    async def _do_start_team(self, intent: Intent) -> None:
        node_id, task = intent.node_id, intent.payload["task"]
        above = self.store.get(intent.payload["parent"] or "")
        parent_ic = await self._started_ic(above) or await self._template()
        checkout = intent.payload.get("checkout")
        self._runs().mark(TASK, node_id, task)
        self.progressed(node_id)
        started = self.store.get(node_id)
        left: list[str] = []
        # A session starts beside the team's own work: what is uncommitted
        # in the team's directory is not the session's.
        opens = started is not None and (
            self.store.opens_session(started) or reviews_only(started)
        )
        if started is not None:
            # A team of the same name before it left nothing to count on.
            self._work_accepted(started)
            if above is not None and not opens:
                with suppress(GitError, OSError):
                    left = await asyncio.to_thread(
                        uncommitted,
                        Path(above.worktree_path),
                        started.boundary.path,
                    )
        await self._start_agent(
            node_id,
            "ic",
            parent_ic,
            f"{task}\n\n{left_for_team(left)}" if left else task,
            checkout,
        )
        if above is not None and not opens:
            await asyncio.to_thread(self._pass_down, above, node_id)
        if opens and not reviews_only(started):
            self._checkout_for_session(node_id)
        purpose = intent.payload.get("purpose", task)
        self._enqueue(lambda: self._summarize(node_id, purpose, first=True), node_id)

    def _checkout_for_session(self, node_id: str) -> None:
        """Give the teams of a session that have not started the checkout
        of the team the session was opened at, as a team that is set up
        has the checkout of the team above."""
        store = self.store
        top = store.get(node_id)
        if top is None or not top.agents:
            return
        teams = store.children(node_id)
        for team in teams:
            if not team.agents:
                store.update_node(
                    team.id, worktree_path=top.worktree_path, branch=top.branch
                )
                teams.extend(store.children(team.id))

    def _pass_down(self, above: OrgNode, node_id: str) -> None:
        """Close a started team's directory to the IC above, and bring what
        that IC left uncommitted there into the team's own checkout."""
        team = self.store.get(node_id)
        if team is None or not team.worktree_path:
            return
        with suppress(GitError, OSError):
            # The shell the IC above starts next cannot write there.
            self._confine(above.id, "ic")
        source, target = Path(above.worktree_path), Path(team.worktree_path)
        if _checkout_root(str(source)) == _checkout_root(str(target)):
            # In one checkout the team finds the files where they are.
            return
        try:
            bring_over(source, target, team.boundary.path, f"{LEFT_REFS}/{team.id}")
        except (GitError, OSError):
            logger.exception("Could not pass what was left to node %s", team.id)

    def note_hand_over(
        self, node_id: str, *, team: str | None = None, spec: bool = False
    ) -> None:
        """Keep what a team's hand-over conversation did, for its IC to be
        told when it ends. Safe from any thread."""
        run = self._runs()
        kept = run.marked(HANDING).get(node_id)
        if kept is None:
            return
        done = json.loads(kept)
        if team is not None and team not in done["given"]:
            done["given"].append(team)
        done["spec"] = done["spec"] or spec
        run.mark(HANDING, node_id, json.dumps(done))
