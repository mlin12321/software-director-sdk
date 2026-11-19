"""How the service reaches agents: messages, the steps that wait for an
agent to be quiet, and the questions put to a model."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from uuid import UUID

from openhands.agent_server.director import compaction
from openhands.agent_server.director.context import (
    changed,
    steady,
    update_block,
)
from openhands.agent_server.director.git import (
    GitError,
)
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    AgentRole,
    Report,
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    division_lead_id,
)
from openhands.agent_server.director.prompts import (
    from_director,
)
from openhands.agent_server.director.run_state import (
    JOINED,
    RUNNING,
    TOLD,
    WAITING,
    Intent,
)
from openhands.agent_server.director.service.records import Records
from openhands.agent_server.director.service.state import (
    MODELS_LANE,
    ORG_LANE,
    _agent_title,
    _Held,
    _instant,
    _lane_for,
    agent_conversation,
)
from openhands.agent_server.event_service import ConversationRunLimitExceeded
from openhands.sdk.errors import (
    Invalid,
    NotFound,
    Unavailable,
)


logger = logging.getLogger(__name__)


# How often a message is sent before it is left held.
_SEND_ATTEMPTS = 3


class Delivery(Records):
    """Sends agents what the director has for them."""

    def refresh_headline(self, place_id: str, report: str) -> None:
        """Have a team's headline rewritten after it reported. Any thread."""
        place = self.store.place(place_id)
        if place is not None:
            task = f"{place.spec_summary}\n\nLatest report of the team: {report}"
            self._enqueue(
                lambda: self._summarize(place_id, task), f"headline:{place_id}"
            )

    def notify(
        self, conversation_id: str | None, text: str, *, at_once: bool = False
    ) -> None:
        """Send an agent a message from the director. Safe from any thread.

        Args:
            at_once: Do not wait for the end of the turn of an agent that
                runs its own tools; for what the human said.
        """
        if conversation_id is not None:
            self._post(conversation_id, from_director(text), at_once=at_once)

    def report(self, place: SessionTeam, role: AgentRole, text: str) -> None:
        """Record something for the two agents of a team. Safe from any
        thread.

        Raises:
            Invalid: There is nothing to record.
        """
        text = text.strip()
        if not text:
            raise Invalid("A report cannot be empty")
        report = Report(
            at=_instant(),
            role=role,
            text=text,
            session=self.store.session_title(place),
        )
        # Its session's alone until the session's work is accepted.
        self._keep_report(place, report)

    def reports_grew(self, node_id: str, role: AgentRole) -> None:
        """Compact what one agent of a team recorded, if it has grown past
        what the org set. Safe from any thread."""
        reports = self._memory().reports(node_id, role)
        if compaction.due(reports, self.settings.report_compaction_chars):
            self._enqueue(lambda: self._compact(node_id, role), MODELS_LANE)

    def joined(self, conversation_id: str) -> None:
        """Note that the human has written in an agent's conversation, so
        that it is listed with the conversations the human started. Safe
        from any thread, and to call again.

        Raises:
            NotFound: No agent of the org has this conversation.
        """
        found = self.store.find_by_conversation(conversation_id)
        if found is None:
            raise NotFound(f"No agent of the org has conversation {conversation_id}")
        run = self._runs()
        if conversation_id in run.marked(JOINED):
            return
        run.mark(JOINED, conversation_id)
        # After what starts the conversation, when it is not there yet.
        self._enqueue(lambda: self._mark_joined(conversation_id), found[0].id)

    async def _mark_joined(self, conversation_id: str) -> None:
        try:
            await self._require_launcher().mark_joined(UUID(conversation_id))
        except NotFound:
            # Not started yet, or gone: asked for again when it is written to.
            self._runs().unmark(JOINED, conversation_id)

    def _post(self, conversation_id: str, text: str, *, at_once: bool = False) -> None:
        """Record a message to an agent, then deliver it. Safe from any
        thread. While the org is paused the message waits.

        Args:
            at_once: Deliver it even in the middle of the agent's turn. A
                message to an agent that runs its own tools ends the turn
                it is in, so the director's own notices wait for that turn
                to end and only the human's words interrupt.
        """
        if self._loop is None:
            raise Unavailable("The director is not running agents on this server")
        found = self.store.find_by_conversation(conversation_id)
        place_id = found[0].id if found else ""
        seq = self._runs().hold(conversation_id, place_id, text)
        if at_once:
            self._at_once.add(seq)
        self._save_soon()
        self._enqueue(
            lambda: self._deliver(seq, conversation_id, text), place_id or ORG_LANE
        )

    async def _deliver(self, seq: int, conversation_id: str, text: str) -> None:
        run = self._runs()
        if self._paused is not None or seq in self._delivering:
            return
        if not run.is_held(seq):
            return
        if self._is_stopped(conversation_id):
            # It waits until the IC above sets the team going again.
            return
        if seq not in self._at_once and (
            self._in_a_turn_of_its_own(conversation_id)
            or self._settles(conversation_id)
        ):
            return
        self._delivering.add(seq)
        self._at_once.discard(seq)
        try:
            text, told = await asyncio.to_thread(
                self._with_what_changed, conversation_id, text
            )
            await self._send_again(UUID(conversation_id), text)
            if told is not None:
                run.mark(TOLD, conversation_id, told)
            self._turn_begun(conversation_id)
        except NotFound:
            store = self.store
            found = store.find_by_conversation(conversation_id)
            if found is not None:
                place, role = found
                known = any(a.role == role and a.reason for a in place.agents)
                store.set_agent(
                    place.id,
                    role,
                    status=AGENT_LOST,
                    reason=None
                    if known
                    else "Its conversation is no longer on this server.",
                )
            logger.warning("Agent conversation %s is gone", conversation_id)
        finally:
            self._delivering.discard(seq)
        run.delivered(seq)

    async def _send_again(self, conversation_id: UUID, text: str) -> None:
        """Send a message, trying again when the server could not take it.
        What is sent to the team after it waits behind it meanwhile."""
        launcher = self._require_launcher()
        for attempt in range(_SEND_ATTEMPTS):
            try:
                return await launcher.send(conversation_id, text)
            except NotFound:
                raise
            except Exception:
                if attempt == _SEND_ATTEMPTS - 1:
                    # It stays held, for the end of the agent's next turn.
                    raise
                logger.warning("A message to %s is sent again", conversation_id)
                await asyncio.sleep(self._run_limit_retry)

    def knows(self, place_id: str, role: AgentRole, *keys: str) -> None:
        """Take as told to an agent the parts of its context it has just
        written itself, so that they are not sent back to it. Safe from any
        thread."""
        place = self.store.place(place_id)
        conversation_id = agent_conversation(place, role) if place else None
        run = self._runs()
        before = run.marked(TOLD).get(conversation_id or "")
        if place is None or conversation_id is None or not before:
            return
        try:
            now = steady(self.context_of(place, role))
        except (GitError, OSError):
            return
        told = json.loads(before) | {key: now[key] for key in keys if key in now}
        run.mark(TOLD, conversation_id, json.dumps(told))

    def _with_what_changed(
        self, conversation_id: str, text: str
    ) -> tuple[str, str | None]:
        """A message to an agent with what changed in its team's context
        since it was last told, so that no message has to be sent for a
        change alone.

        Returns:
            The message, and the context to keep as told once it is sent;
            None when it tells nothing new.
        """
        found = self.store.find_by_conversation(conversation_id)
        if found is None:
            return text, None
        node, role = found
        try:
            now = steady(self.context_of(node, role))
        except (GitError, OSError):
            return text, None
        before = self._runs().marked(TOLD).get(conversation_id)
        # An agent briefed before contexts were kept is told all of it once.
        news = changed(json.loads(before) if before else {}, now, role)
        if not news:
            return text, None
        return f"{text}\n\n{update_block(news)}", json.dumps(now)

    def _turn_begun(self, conversation_id: str) -> None:
        """Count an agent that runs its own tools as in a turn from the
        moment something is sent to it. Its conversation says so only a
        moment later, and a second notice in between would end the turn the
        first one began."""
        found = self.store.find_by_conversation(conversation_id)
        if found is None:
            return
        node, role = found
        if any(agent.role == role and agent.kind == "acp" for agent in node.agents):
            self._note_running(conversation_id, node.id, True)

    def _in_a_turn_of_its_own(self, conversation_id: str) -> bool:
        """Whether a message now would cut short the turn of an agent that
        runs its own tools."""
        if conversation_id not in self._running:
            return False
        found = self.store.find_by_conversation(conversation_id)
        if found is None:
            return False
        node, role = found
        return any(agent.role == role and agent.kind == "acp" for agent in node.agents)

    def _settles(self, conversation_id: str) -> bool:
        """Whether the agent's team is about to be changed, so that nothing
        new is started in it."""
        found = self.store.find_by_conversation(conversation_id)
        return found is not None and any(
            found[0].id in teams for teams, _ in self._settling.values()
        )

    def _wait_until_quiet(self, intent: Intent, node_ids: list[str]) -> None:
        """Hold a step until no agent of the teams is in a turn. Messages to
        them wait meanwhile, so that no new turn begins.

        Raises:
            _Held: An agent of theirs is still running; the step is taken
                up again when it stops.
        """
        self._settling[intent.id] = (set(node_ids), _lane_for(intent))
        store = self.store
        for team in map(store.place, node_ids):
            for agent in team.agents if team else []:
                if agent.conversation_id in self._running:
                    raise _Held

    def _let_go(self, intent: Intent) -> None:
        """Deliver what waited for the teams a step has now changed."""
        teams, _ = self._settling.pop(intent.id, (set(), ""))
        store = self.store
        for team in map(store.place, teams):
            for agent in team.agents if team else []:
                if agent.conversation_id is not None:
                    self._deliver_what_waited(agent.conversation_id)

    def _deliver_what_waited(self, conversation_id: str) -> None:
        """Deliver the messages that waited for an agent's turn to end."""
        for held in self._runs().held():
            if held.conversation_id != conversation_id:
                continue
            self._at_once.add(held.seq)
            self._enqueue(
                lambda held=held: self._deliver(
                    held.seq, held.conversation_id, held.text
                ),
                held.node_id or ORG_LANE,
            )

    async def _when_there_is_room[T](
        self, place_id: str, start: Callable[[], Awaitable[T]]
    ) -> T:
        """Run ``start`` for a team; while the server is full, wait in line
        for a conversation to stop running and try again."""
        while place_id in self._line and self._line[0] != place_id:
            await self._line_moved.wait()
        try:
            return await start()
        except ConversationRunLimitExceeded:
            logger.info("Server is at its run limit; team %s waits", place_id)
        finally:
            # Nothing is awaited from here to the gate, so the team is in
            # line at the gate before the next one tries.
            self._leave_line(place_id)
        store = self.store
        run = self._runs()
        run.mark(WAITING, place_id)
        with suppress(NotFound):
            store.update_place(place_id, waiting_to_start=True)
        keeps_place = False
        try:
            async with self._start_gate:
                while True:
                    if self._paused is not None:
                        raise _Held
                    self._room.clear()
                    try:
                        return await start()
                    except ConversationRunLimitExceeded:
                        pass
                    # Also after a pause: not every conversation is the org's.
                    with suppress(TimeoutError):
                        await asyncio.wait_for(
                            self._room.wait(), timeout=self._run_limit_retry
                        )
        except (_Held, asyncio.CancelledError):
            # Stopped while it waited: it is still next when the org resumes.
            keeps_place = True
            raise
        finally:
            if not keeps_place:
                run.unmark(WAITING, place_id)
            with suppress(NotFound):
                store.update_place(place_id, waiting_to_start=False)

    def _note_running(self, conversation_id: str, place_id: str, running: bool) -> None:
        if running == (conversation_id in self._running):
            return
        run = self._runs()
        if running:
            self._running.add(conversation_id)
            run.mark(RUNNING, conversation_id, place_id)
        else:
            self._running.discard(conversation_id)
            run.unmark(RUNNING, conversation_id)
            if self._loop is not None:
                self._deliver_what_waited(conversation_id)
                for intent_id, (teams, lane) in list(self._settling.items()):
                    if place_id in teams:
                        self._enqueue(
                            lambda intent_id=intent_id: self._carry_out(intent_id),
                            lane,
                        )

    async def _send(self, conversation_id: str, text: str) -> None:
        """Deliver a message as part of a step, in its place among them."""
        found = self.store.find_by_conversation(conversation_id)
        seq = self._runs().hold(conversation_id, found[0].id if found else "", text)
        await self._deliver(seq, conversation_id, text)

    async def _ask_agent(self, conversation_id: str, question: str) -> str | None:
        try:
            return await asyncio.wait_for(
                self._require_launcher().ask_agent(UUID(conversation_id), question),
                timeout=self._model_call_timeout,
            )
        except (NotFound, TimeoutError):
            return None

    async def _ask(
        self, conversation_id: str, system: str, user: str, usage_id: str
    ) -> str | None:
        try:
            return await asyncio.wait_for(
                self._require_launcher().ask(
                    UUID(conversation_id), system, user, usage_id=usage_id
                ),
                timeout=self._model_call_timeout,
            )
        except (NotFound, TimeoutError):
            return None

    def _asker(self) -> str | None:
        """The conversation the director's own model calls are made from."""
        # Nobody works at the top team itself: the IC a session started
        # with, oldest first.
        sessions = self._store.sessions() if self._store else []
        return next((s.conversation_id for s in sessions if s.conversation_id), None)

    async def _compact(self, node_id: str, role: AgentRole) -> None:
        """Have a model rewrite one agent's reports shorter, if they are
        still too long."""
        asker = self._asker()
        if asker is None or self._paused is not None:
            return
        team_memory = self._memory()
        reports = await asyncio.to_thread(team_memory.reports, node_id, role)
        if not compaction.due(reports, self.settings.report_compaction_chars):
            return
        answer = await self._ask(
            asker, *compaction.prompt(reports), compaction.USAGE_ID
        )
        shorter = compaction.compacted(answer or "", reports, _instant())
        if shorter is None:
            logger.warning("No usable rewrite of the reports of %s %s", node_id, role)
            return
        # Rewriting them changes what the org shares.
        await self._base(
            lambda: asyncio.to_thread(
                team_memory.replace_reports, node_id, role, reports, shorter
            )
        )

    async def _as_one_report(self, reports: list[Report]) -> str:
        """``reports`` as one report: rewritten shorter by a model when
        there are several, and in full when it gives no usable answer."""
        full = "\n\n".join(report.text for report in reports)
        asker = self._asker()
        if len(reports) < 2 or asker is None:
            return full
        answer = await self._ask(
            asker, *compaction.prompt(reports), compaction.USAGE_ID
        )
        shorter = compaction.compacted(answer or "", reports, _instant())
        return shorter[0].text if shorter else full

    async def _name_org(self, top_id: str) -> None:
        """Give the top team the name a model made of the first task. The
        Division Lead of that session is called what its team is."""
        store = self.store
        top = store.place(top_id)
        if top is None or store.root_id is None:
            return
        store.update_team(store.root_id, title=top.title, headline=top.headline)
        lead = store.place(division_lead_id(store.root_id, top.session))
        if lead is not None:
            store.update_place(lead.id, title=top.title)

    async def _summarize(
        self,
        place_id: str,
        task: str,
        *,
        rename: bool = False,
        first: bool = False,
    ) -> None:
        """Have a model write a team's headline and, with ``rename``, its title.

        Args:
            first: This is the summary of the task the team was made for. It
                is dropped once the team has reported, which may happen
                before the model was asked.
        """
        store = self.store

        def reported() -> bool:
            current = store.place(place_id)
            return (
                first
                and current is not None
                and current.headline != provisional_summary(task).headline
            )

        place = store.place(place_id)
        conversation_id = agent_conversation(place, "ic") if place else None
        if place is None or conversation_id is None or reported():
            return
        if self._paused is not None:
            return
        launcher = self._require_launcher()
        try:
            async with self._model_calls:
                summary = await asyncio.wait_for(
                    launcher.summarize(UUID(conversation_id), task),
                    timeout=self._summary_timeout,
                )
        except (NotFound, TimeoutError):
            return
        if summary is None or reported():
            return
        changes: dict[str, object] = {"headline": summary.headline}
        if rename:
            changes["title"] = summary.title
        try:
            place = store.update_place(place_id, **changes)
        except NotFound:
            return
        if not rename:
            return
        if store.opens_session(place):
            # A session the human gave no name is called what it is for.
            store.rename_session(place.session, summary.title)
        await self._rename_agents(place, launcher)

    async def _name_session(self, place_id: str, request: str) -> None:
        """Call a session the human gave no name by what a model makes of
        its request. The team it was opened at keeps its own name."""
        store = self.store
        place = store.place(place_id)
        conversation_id = agent_conversation(place, "ic") if place else None
        if place is None or conversation_id is None or self._paused is not None:
            return
        launcher = self._require_launcher()
        try:
            async with self._model_calls:
                summary = await asyncio.wait_for(
                    launcher.summarize(UUID(conversation_id), request),
                    timeout=self._summary_timeout,
                )
        except (NotFound, TimeoutError):
            return
        place = store.place(place_id)
        if summary is None or place is None:
            return
        store.rename_session(place.session, summary.title)
        await self._rename_agents(place, launcher)

    async def _rename_agents(self, place: SessionTeam, launcher: TeamLauncher) -> None:
        store = self.store
        for agent in place.agents:
            if agent.conversation_id is not None:
                with suppress(NotFound):
                    await launcher.rename(
                        UUID(agent.conversation_id),
                        _agent_title(
                            place.title, agent.role, store.session_title(place)
                        ),
                    )
