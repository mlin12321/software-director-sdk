"""Messages from the director to agents: each is held in the run state
until it is delivered, and waits where delivering it now would cut a turn
short or reach a team that is being changed."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from openhands.agent_server.director.context import (
    again_block,
    changed,
    steady,
    update_block,
)
from openhands.agent_server.director.git import GitError
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import SessionTeam
from openhands.agent_server.director.org_store import AGENT_LOST, OrgStore
from openhands.agent_server.director.prompts import from_director
from openhands.agent_server.director.run_state import CONDENSED, RUNNING, TOLD, Intent
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import Held
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.errors import NotFound, Unavailable
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# How often a message is sent before it is left held.
_SEND_ATTEMPTS = 3


@dataclass
class MessageDeliveryService:
    """Sends agents what the director has for them.

    A message is held in the run state first, so one that is not
    delivered before the server stops is delivered after. With a message
    goes what changed in the agent's context since it was last told. The
    service also knows which agents are in a turn, and holds the steps
    that wait for the agents of a team to be quiet.
    """

    org: Callable[[], OrgStore]
    """The org whose agents are written to."""
    org_state: OrgStateService
    """Where messages are held, and whether the org is paused."""
    org_save: OrgSaveService
    """Saves the tree a held message refers to."""
    work_queue: WorkQueueService
    """What carries deliveries out, in the lane of the agent's team."""
    team_types: TeamTypeService
    """Who the members of a team are and what each is told of its part."""
    launcher: Callable[[], TeamLauncher]
    """What reaches the conversations; raises where there is none."""
    is_running: Callable[[], bool]
    """Whether the director is running agents on this server."""
    context_of: Callable[[SessionTeam, str], dict[str, Any]]
    """What a member of a team reads of its place."""
    is_stopped: Callable[[str], bool]
    """Whether the team of a conversation, or one above it, is stopped."""
    lane_of: Callable[[Intent], str]
    """The lane a step is carried out in."""
    take_up: Callable[[int, str], None]
    """Takes up a step, by the id of its intent, in a lane."""
    default_lane: str
    """The lane of a message to an agent that has no team."""
    retry: float
    """How many seconds lie between two tries to send a message."""
    # The messages being delivered right now, so that resuming does not
    # deliver one a second time.
    _delivering: set[int] = field(default_factory=set, init=False)
    # Messages that do not wait for their agent's turn to end: what the
    # human said, and what waited for a turn that has now ended.
    _at_once: set[int] = field(default_factory=set, init=False)
    # The teams each step waits for the agents of to stop, by the id of
    # its intent, and the lane the step is carried out in.
    _settling: dict[int, tuple[set[str], str]] = field(default_factory=dict, init=False)

    def notify(
        self, conversation_id: str | None, text: str, *, at_once: bool = False
    ) -> None:
        """Send an agent a message from the director. Safe from any thread.

        Args:
            at_once: Do not wait for the end of the turn of an agent that
                runs its own tools; for what the human said.
        """
        if conversation_id is not None:
            self.post(conversation_id, from_director(text), at_once=at_once)

    def post(self, conversation_id: str, text: str, *, at_once: bool = False) -> None:
        """Record a message to an agent, then deliver it. Safe from any
        thread. While the org is paused the message waits.

        Args:
            at_once: Deliver it even in the middle of the agent's turn. A
                message to an agent that runs its own tools ends the turn
                it is in, so the director's own notices wait for that turn
                to end and only the human's words interrupt.
        """
        if not self.is_running():
            raise Unavailable("The director is not running agents on this server")
        found = self.org().find_by_conversation(conversation_id)
        place_id = found[0].id if found else ""
        seq = self.org_state.run_state().hold(conversation_id, place_id, text)
        if at_once:
            self._at_once.add(seq)
        self.org_save.save_soon()
        self.work_queue.enqueue(
            lambda: self.deliver(seq, conversation_id, text),
            place_id or self.default_lane,
        )

    async def deliver(
        self, seq: int, conversation_id: str, text: str, *, at_once: bool = False
    ) -> None:
        """Deliver a held message, unless it has to wait: for the org to
        resume, for its team to be set going again, for the turn of an
        agent that runs its own tools to end, or for a step that is
        changing its team.

        Args:
            at_once: Do not wait for the agent's turn to end.
        """
        if at_once:
            self._at_once.add(seq)
        run = self.org_state.run_state()
        if self.org_state.paused is not None or seq in self._delivering:
            return
        if not run.is_held(seq):
            return
        if self.is_stopped(conversation_id):
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
            condensed = await self._condensed_since_told(conversation_id)
            text, told = await asyncio.to_thread(
                self._with_what_changed, conversation_id, text, condensed is not None
            )
            await self.send_again(UUID(conversation_id), text)
            if told is not None:
                run.mark(TOLD, conversation_id, told)
            if condensed is not None:
                run.mark(CONDENSED, conversation_id, condensed)
            self._turn_begun(conversation_id)
        except NotFound:
            store = self.org()
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

    async def send_again(self, conversation_id: UUID, text: str) -> None:
        """Send a message, trying again when the server could not take it.
        What is sent to the team after it waits behind it meanwhile."""
        launcher = self.launcher()
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
                await asyncio.sleep(self.retry)

    def knows(self, place_id: str, role: str, *keys: str) -> None:
        """Take as told to an agent the parts of its context it has just
        written itself, so that they are not sent back to it. Safe from any
        thread."""
        place = self.org().place(place_id)
        conversation_id = (
            self.team_types.team(place).conversation(role) if place else None
        )
        run = self.org_state.run_state()
        before = run.marked(TOLD).get(conversation_id or "")
        if place is None or conversation_id is None or not before:
            return
        try:
            now = steady(self.context_of(place, role))
        except (GitError, OSError):
            return
        told = json.loads(before) | {key: now[key] for key in keys if key in now}
        run.mark(TOLD, conversation_id, json.dumps(told))

    async def _condensed_since_told(self, conversation_id: str) -> str | None:
        """The shortening of an agent's history that it has not been told
        again after, if there is one. What the director said to it before
        then may be gone from what its model reads."""
        launcher = self.launcher()
        try:
            last = await launcher.condensed(UUID(conversation_id))
        except NotFound:
            return None
        known = self.org_state.run_state().marked(CONDENSED).get(conversation_id)
        return last if last is not None and last != known else None

    def _with_what_changed(
        self, conversation_id: str, text: str, again: bool = False
    ) -> tuple[str, str | None]:
        """A message to an agent with what changed in its team's context
        since it was last told, so that no message has to be sent for a
        change alone.

        Args:
            again: Its history was shortened since it was last told: it is
                told how the org works and all of its team's context once
                more.

        Returns:
            The message, and the context to keep as told once it is sent;
            None when it tells nothing new.
        """
        found = self.org().find_by_conversation(conversation_id)
        if found is None:
            return text, None
        node, role = found
        try:
            now = steady(self.context_of(node, role))
        except (GitError, OSError):
            return text, None
        if again:
            said = again_block(self.team_types.role_text(node, role), now)
            return f"{text}\n\n{said}", json.dumps(now)
        before = self.org_state.run_state().marked(TOLD).get(conversation_id)
        news = changed(
            json.loads(before) if before else {},
            now,
            role,
            works=role == self.team_types.team(node).writer,
        )
        if not news:
            return text, None
        return f"{text}\n\n{update_block(news)}", json.dumps(now)

    def _turn_begun(self, conversation_id: str) -> None:
        """Count an agent that runs its own tools as in a turn from the
        moment something is sent to it. Its conversation says so only a
        moment later, and a second notice in between would end the turn the
        first one began."""
        found = self.org().find_by_conversation(conversation_id)
        if found is None:
            return
        node, role = found
        if any(agent.role == role and agent.kind == "acp" for agent in node.agents):
            self.note_running(conversation_id, node.id, True)

    def _in_a_turn_of_its_own(self, conversation_id: str) -> bool:
        """Whether a message now would cut short the turn of an agent that
        runs its own tools."""
        if conversation_id not in self.org_state.running:
            return False
        found = self.org().find_by_conversation(conversation_id)
        if found is None:
            return False
        node, role = found
        return any(agent.role == role and agent.kind == "acp" for agent in node.agents)

    def _settles(self, conversation_id: str) -> bool:
        """Whether the agent's team is about to be changed, so that nothing
        new is started in it."""
        found = self.org().find_by_conversation(conversation_id)
        return found is not None and self.is_settling(found[0].id)

    def is_settling(self, place_id: str) -> bool:
        """Whether a step waits for the agents of a team to stop."""
        return any(place_id in teams for teams, _ in self._settling.values())

    def stop_waiting(self, intent_id: int) -> None:
        """Forget the teams a step waited for, delivering nothing."""
        self._settling.pop(intent_id, None)

    def wait_until_quiet(self, intent: Intent, node_ids: list[str]) -> None:
        """Hold a step until no agent of the teams is in a turn. Messages to
        them wait meanwhile, so that no new turn begins.

        Raises:
            Held: An agent of theirs is still running; the step is taken
                up again when it stops.
        """
        self._settling[intent.id] = (set(node_ids), self.lane_of(intent))
        store = self.org()
        for team in map(store.place, node_ids):
            for agent in team.agents if team else []:
                if agent.conversation_id in self.org_state.running:
                    raise Held

    def let_go(self, intent: Intent) -> None:
        """Deliver what waited for the teams a step has now changed."""
        teams, _ = self._settling.pop(intent.id, (set(), ""))
        store = self.org()
        for team in map(store.place, teams):
            for agent in team.agents if team else []:
                if agent.conversation_id is not None:
                    self.deliver_what_waited(agent.conversation_id)

    def deliver_what_waited(self, conversation_id: str) -> None:
        """Deliver what waited for an agent's turn to end, as one message:
        each of several would end the turn the one before it began."""
        if self._settles(conversation_id):
            # A step is changing its team; the step delivers it when done.
            return
        run = self.org_state.run_state()
        waiting = [
            held
            for held in run.held()
            if held.conversation_id == conversation_id
            and held.seq not in self._delivering
        ]
        if not waiting:
            return
        seq, place_id, text = waiting[0].seq, waiting[0].node_id, waiting[0].text
        if len(waiting) > 1:
            text = "\n\n".join(held.text for held in waiting)
            # Held as one before the others are forgotten: a server that ends
            # in between says something twice, never loses it.
            seq = run.hold(conversation_id, place_id, text)
            for held in waiting:
                run.delivered(held.seq)
        self._at_once.add(seq)
        self.work_queue.enqueue(
            lambda: self.deliver(seq, conversation_id, text),
            place_id or self.default_lane,
        )

    def note_running(self, conversation_id: str, place_id: str, running: bool) -> None:
        """Record that an agent began or ended a turn. At the end of one,
        what waited for it is delivered and the steps that waited for its
        team are taken up."""
        if running == (conversation_id in self.org_state.running):
            return
        run = self.org_state.run_state()
        if running:
            self.org_state.running.add(conversation_id)
            run.mark(RUNNING, conversation_id, place_id)
        else:
            self.org_state.running.discard(conversation_id)
            run.unmark(RUNNING, conversation_id)
            if self.is_running():
                self.deliver_what_waited(conversation_id)
                for intent_id, (teams, lane) in list(self._settling.items()):
                    if place_id in teams:
                        self.take_up(intent_id, lane)

    async def send(self, conversation_id: str, text: str) -> None:
        """Deliver a message as part of a step, in its place among them."""
        found = self.org().find_by_conversation(conversation_id)
        seq = self.org_state.run_state().hold(
            conversation_id, found[0].id if found else "", text
        )
        await self.deliver(seq, conversation_id, text)
