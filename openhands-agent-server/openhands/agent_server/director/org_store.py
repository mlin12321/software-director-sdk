"""In-memory org tree.

The store owns the derived fields on every node (``child_ids``,
``descendant_counts`` and ``escalation_count``) and is the only thing that
emits org events, so every subscriber sees one consistent tree.
"""

import threading
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast, get_args

from openhands.agent_server.director.models import (
    AgentKind,
    AgentRole,
    Escalation,
    EscalationHandoff,
    EscalationKind,
    EscalationPathStep,
    EscalationsEvent,
    HandOverMode,
    NodeRemovedEvent,
    NodeStatus,
    NodeUpsertEvent,
    OpenEscalation,
    OrgNode,
    OrgNodeInput,
    Review,
    Session,
    SessionInput,
    SessionsEvent,
    SessionTeam,
    SessionTeamInput,
    SnapshotEvent,
    StatusCounts,
    Stopped,
)
from openhands.sdk.errors import Invalid, NotFound


NODE_STATUSES: tuple[NodeStatus, ...] = get_args(NodeStatus)
# An agent whose conversation failed, and one that has none on this server.
AGENT_FAILED = "blocked"
AGENT_LOST = "lost"


def status_of(node: OrgNode | OrgNodeInput) -> NodeStatus:
    """The status a team has by what is known of it; no team declares one.

    Blocked while its IC waits on an escalation, an agent of it failed or
    the IC above has stopped it, working while an agent of it is in a turn,
    done once it has submitted
    its work (until that is sent back), blocked when an agent has no
    conversation here, idle otherwise.
    """
    agents = {agent.status for agent in node.agents}
    if node.stopped is not None:
        return "blocked"
    if AGENT_FAILED in agents or any(
        escalation.raised_by == "ic" for escalation in node.escalations
    ):
        return "blocked"
    if "working" in agents:
        return "working"
    if node.review is not None and node.review.state != "changes_requested":
        # Submitted: it rests until its reviewer or something else wakes it.
        return "done"
    return "blocked" if AGENT_LOST in agents else "idle"


type OrgChange = NodeUpsertEvent | NodeRemovedEvent | EscalationsEvent | SessionsEvent
type OrgListener = Callable[[OrgChange], None]

_DERIVED_FIELDS = {"child_ids", "descendant_counts", "escalation_count"}


def session_team_id(team_id: str, session_id: str) -> str:
    """The id of what a session has at the team ``team_id``."""
    return f"{team_id}@{session_id}"


def review_place_id(team_id: str, session_id: str) -> str:
    """The id of the place a session's work is reviewed from, at the team
    ``team_id``."""
    return f"{team_id}@{session_id}@review"


def reviews_only(node: OrgNodeInput | None) -> bool:
    """Whether ``node`` is the place a session's work is reviewed from."""
    return isinstance(node, SessionTeamInput) and node.review_only


def session_id_of(node: OrgNodeInput) -> str:
    """The session ``node`` belongs to; '' for a team of the org."""
    return node.session if isinstance(node, SessionTeamInput) else ""


def team_id_of(node: OrgNodeInput) -> str:
    """The team of the org ``node`` is, or is the work of in a session."""
    return node.team_id if isinstance(node, SessionTeamInput) else node.id


class _Units(Mapping[str, OrgNode]):
    """The teams of the org and the teams of the sessions as one lookup."""

    def __init__(
        self, nodes: Mapping[str, OrgNode], seats: Mapping[str, OrgNode]
    ) -> None:
        self._parts = (nodes, seats)

    def __getitem__(self, key: str) -> OrgNode:
        for part in self._parts:
            if key in part:
                return part[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for part in self._parts:
            yield from part

    def __len__(self) -> int:
        return sum(len(part) for part in self._parts)


@dataclass
class _Subscription:
    focus_id: str
    # Nodes whose direct reports the subscriber also follows.
    expanded: frozenset[str]
    # Whether the subscriber follows every node below the focused one.
    full: bool
    # The nodes above the focused one. A node never changes parent.
    above: frozenset[str]
    # What the subscriber was last told, to announce only real changes.
    escalations: list[OpenEscalation] = field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class OrgStoreError(Invalid):
    """A change would leave the org tree malformed."""


class OrgStore:
    """A tree of :class:`OrgNode`, safe to use from several threads.

    The tree is the org's structure. Beside it the store holds the open
    sessions: each is a record of its own with, for the team it was opened
    at and every team below that one, a :class:`SessionTeam`. Those are in
    no node's ``child_ids``, in no rollup and in no snapshot's ``nodes``.
    Everything that reads or changes one team's work (``get``, the
    escalations, the review, the agents) takes a team of the org or a team
    of a session alike, by its id.

    Listeners are called while the store is locked, in the order changes
    happened. They must return quickly and must not call back into the store.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: dict[str, OrgNode] = {}
        self._sessions: dict[str, SessionInput] = {}
        self._seats: dict[str, SessionTeam] = {}
        self._root_id: str | None = None
        self._listeners: dict[OrgListener, _Subscription] = {}
        self._change_listeners: list[Callable[[], None]] = []
        self._agents: dict[str, tuple[str, AgentRole]] = {}
        # The hand-over conversation of each IC that has one -> its node.
        self._hand_overs: dict[str, str] = {}

    @property
    def root_id(self) -> str | None:
        return self._root_id

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> OrgNode | None:
        """A team of the org, or a team of a session, by its id."""
        return self._nodes.get(node_id) or self._seats.get(node_id)

    def team(self, node_id: str) -> OrgNode | None:
        """A node of the org tree; None for anything else."""
        return self._nodes.get(node_id)

    def children(self, node_id: str) -> list[OrgNode]:
        """The sub-teams of ``node_id``. Those of a team of a session are
        that session's."""
        with self._lock:
            node = self.get(node_id)
            if node is None:
                return []
            units = self._units()
            return [units[child_id] for child_id in node.child_ids]

    def opened_under(self, node_id: str) -> list[SessionTeam]:
        """The teams sessions were opened at whose work the IC of
        ``node_id`` reviews."""
        with self._lock:
            return [
                self._seats[session_team_id(session.team_id, session.id)]
                for session in self._sessions.values()
                if session.above_id == node_id
            ]

    def below(self, node_id: str) -> list[OrgNode]:
        """Everything whose work the IC of ``node_id`` reviews: its
        sub-teams, and the teams sessions were opened at under it."""
        with self._lock:
            return [*self.children(node_id), *self.opened_under(node_id)]

    def opens_session(self, node: OrgNode) -> bool:
        """Whether ``node`` is the team a session was opened at."""
        with self._lock:
            session = self.session_of(node)
            return (
                session is not None
                and not reviews_only(node)
                and session.team_id == team_id_of(node)
            )

    def teams(self) -> list[OrgNode]:
        """The org tree, parents before children."""
        with self._lock:
            return self._ordered(self._nodes, self._root_id)

    def session_of(self, node: OrgNodeInput) -> SessionInput | None:
        """The session ``node`` belongs to; None for a team of the org."""
        return self._sessions.get(session_id_of(node))

    def session_title(self, node: OrgNodeInput) -> str:
        """What the human called the session ``node`` belongs to; '' for a
        team of the org."""
        session = self.session_of(node)
        return session.title if session else ""

    def session(self, session_id: str) -> list[SessionTeam]:
        """The teams of a session, the one it was opened at first."""
        with self._lock:
            seats = [
                seat for seat in self._seats.values() if seat.session == session_id
            ]
            opened = self._sessions.get(session_id)
            top = session_team_id(opened.team_id, session_id) if opened else ""
            return sorted(seats, key=lambda seat: (seat.review_only, seat.id != top))

    def sessions(self) -> list[Session]:
        """The open sessions as a client reads them, oldest first."""
        with self._lock:
            # One that is being opened has its places only one by one.
            return [
                self._session_view(session)
                for session in self._sessions.values()
                if session_team_id(session.team_id, session.id) in self._seats
            ]

    def instance(self, team_id: str, session_id: str) -> OrgNode | None:
        """The team ``team_id`` as it is in a session; with '' the team of
        the org itself."""
        if not session_id:
            return self._nodes.get(team_id)
        return self._seats.get(session_team_id(team_id, session_id))

    def teams_below(self, node_id: str) -> int:
        """How many sub-teams ``node_id`` has, at any depth."""
        with self._lock:
            found = [node_id]
            for current in found:
                found.extend(team.id for team in self.children(current))
            return len(found) - 1

    def sessions_at(self, node_id: str, *, below: bool = True) -> list[str]:
        """The titles of the sessions that have a team at ``node_id`` and,
        with ``below``, at a team below it."""
        with self._lock:
            node = self.get(node_id)
            if node is None:
                return []
            subtree = [team_id_of(node)]
            if below:
                for current in subtree:
                    found = self._nodes.get(current)
                    subtree.extend(found.child_ids if found else [])
            return sorted(
                {
                    self._sessions[seat.session].title
                    for seat in self._seats.values()
                    if seat.team_id in subtree and seat.session != session_id_of(node)
                }
            )

    def open_session(
        self, session: SessionInput, teams: Iterable[SessionTeamInput]
    ) -> SessionTeam:
        """Record a session with what it has at each of its teams, the team
        it was opened at first.

        Raises:
            OrgStoreError: The session is already open, or its teams do not
                hang together.
        """
        with self._lock:
            if session.id in self._sessions:
                raise OrgStoreError(f"Session {session.id} is already open")
            self._sessions[session.id] = session
            try:
                for team in teams:
                    self.upsert(team)
                return self._seats[session_team_id(session.team_id, session.id)]
            except (OrgStoreError, KeyError) as error:
                self.close_session(session.id)
                raise OrgStoreError(
                    f"Session {session.id} does not fit: {error}"
                ) from error

    def lift_session(
        self,
        session: SessionInput,
        top: SessionTeamInput,
        review: SessionTeamInput,
        others: Iterable[SessionTeamInput],
    ) -> SessionTeam:
        """Make a session one opened at the team above the one it was
        opened at. The team it was opened at before becomes a sub-team of
        ``top``, and its work is reviewed from ``review``.

        Args:
            session: The session as it is from now on.
            others: What it has at the other teams below ``top`` from now
                on, parents before children.

        Raises:
            OrgStoreError: The session is not open, its work is being
                reviewed, or the places do not hang together.
        """
        with self._lock:
            was = self._sessions.get(session.id)
            if was is None:
                raise OrgStoreError(f"Session {session.id} is not open")
            before = self._seats[session_team_id(was.team_id, was.id)]
            reviewed_from = self._seats[before.parent_id or ""]
            if reviewed_from.agents or before.review is not None:
                raise OrgStoreError(f"The work of session {session.id} is in review")
            if (
                top.id != session_team_id(session.team_id, session.id)
                or top.id in self._seats
                or top.parent_id != review.id
                or review.id != review_place_id(session.above_id, session.id)
                or not review.review_only
            ):
                raise OrgStoreError(f"Session {session.id} cannot go to {top.id}")
            self._sessions[session.id] = session
            del self._seats[reviewed_from.id]
            self._seats[review.id] = SessionTeam(
                **review.model_dump(), child_ids=[top.id]
            )
            self._seats[top.id] = SessionTeam(**top.model_dump(), child_ids=[before.id])
            self._seats[before.id] = before.model_copy(update={"parent_id": top.id})
            moved = [before.id]
            for current in moved:
                moved.extend(self._seats[current].child_ids)
            home = cast(dict[str, OrgNode], self._seats)
            for current in moved:
                below = self._seats[current]
                chain = self._chain(home, below.parent_id)
                self._seats[current] = below.model_copy(
                    update={
                        "escalations": [
                            self._rerouted(escalation, chain)
                            for escalation in below.escalations
                        ]
                    }
                )
            for other in others:
                self._upsert_seat(other)
            self._announce_sessions()
            self._announce_escalations()
            self._changed()
            return self._seats[top.id]

    def set_shape(self, session_id: str, shape: list[tuple[str, str, str]]) -> None:
        """Keep the sub-teams the org has under a session's team, as the
        session last knew them. Nothing for a session that is not open."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(update={"shape": shape})
                self._changed()

    def rename_session(self, session_id: str, title: str) -> None:
        """Call a session something else. Nothing for one that is not open."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions[session_id] = session.model_copy(update={"title": title})
                self._announce_sessions()

    def close_session(self, session_id: str) -> bool:
        """Forget a session and everything it has at its teams.

        Returns:
            Whether the session was open.
        """
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                return False
            for seat in self.session(session_id):
                self._index_agents(seat, None)
                del self._seats[seat.id]
            self._announce_sessions()
            self._announce_escalations()
            self._changed()
            return True

    def ancestors(self, node_id: str) -> list[OrgNode]:
        """What is above ``node_id``, root first, excluding the node itself.
        Above the team a session was opened at is the team whose IC reviews
        the session's work."""
        with self._lock:
            chain: list[OrgNode] = []
            node = self.get(node_id)
            while node is not None and node.parent_id is not None:
                node = self.get(node.parent_id)
                # Nobody stands at the place a session is reviewed from.
                if node is not None and not reviews_only(node):
                    chain.append(node)
            chain.reverse()
            return chain

    def all(self) -> list[OrgNode]:
        """Everything that has agents: the org tree, parents before
        children, then the teams of each session."""
        with self._lock:
            return [*self.teams(), *self._seats.values()]

    def find_by_conversation(
        self, conversation_id: str
    ) -> tuple[OrgNode, AgentRole] | None:
        """The node an agent's conversation belongs to, and the agent's role."""
        with self._lock:
            found = self._agents.get(conversation_id)
            if found is None:
                return None
            node_id, role = found
            return self._units()[node_id], role

    def find_hand_over(self, conversation_id: str) -> OrgNode | None:
        """The node whose IC has ``conversation_id`` as its hand-over
        conversation."""
        with self._lock:
            node_id = self._hand_overs.get(conversation_id)
            return self._units()[node_id] if node_id is not None else None

    def set_hand_over(self, node_id: str, mode: HandOverMode | None) -> OrgNode:
        """Put a node's IC in hand-over mode, change how that stands, or
        end it. The node's status does not follow: a hand-over says nothing
        of how the team's work stands.

        Raises:
            NotFound: The node does not exist.
        """
        with self._lock:
            node = self._require(node_id)
            agents = [
                agent.model_copy(update={"hand_over": mode})
                if agent.role == "ic"
                else agent
                for agent in node.agents
            ]
            if agents == node.agents:
                return node
            return self.update_node(node_id, agents=agents)

    def on_change(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` after every change to the tree.

        Like stream listeners it runs while the store is locked.

        Returns:
            A function that stops the calls.
        """
        with self._lock:
            self._change_listeners.append(listener)

        def stop() -> None:
            with self._lock:
                if listener in self._change_listeners:
                    self._change_listeners.remove(listener)

        return stop

    def update_node(self, node_id: str, **changes: object) -> OrgNode:
        """Change fields of an existing node.

        Raises:
            NotFound: The node does not exist.
            OrgStoreError: The changed node would not fit into the tree.
        """
        with self._lock:
            node = self._require(node_id)
            current = node.model_dump(exclude=_DERIVED_FIELDS)
            kind = SessionTeamInput if isinstance(node, SessionTeam) else OrgNodeInput
            return self.upsert(kind.model_validate({**current, **changes}))

    def set_agent(
        self,
        node_id: str,
        role: AgentRole,
        *,
        conversation_id: str | None = None,
        status: str | None = None,
        kind: AgentKind | None = None,
        acp_server: str | None = None,
    ) -> OrgNode:
        """Change the conversation, the status or the kind of one agent of a
        node.

        The node's status follows (see :func:`status_of`).

        Raises:
            NotFound: The node does not exist or has no agent in that role.
        """
        with self._lock:
            node = self._require(node_id)
            if all(agent.role != role for agent in node.agents):
                raise NotFound(f"Node {node_id} has no {role} agent")
            agents = [
                agent
                if agent.role != role
                else agent.model_copy(
                    update={
                        "conversation_id": conversation_id or agent.conversation_id,
                        "status": status or agent.status,
                        "kind": kind or agent.kind,
                        "acp_server": acp_server or agent.acp_server,
                    }
                )
                for agent in node.agents
            ]
            if agents == node.agents:
                return node
            status = status_of(node.model_copy(update={"agents": agents}))
            return self.update_node(node_id, agents=agents, status=status)

    def set_review(
        self, node_id: str, review: Review | None, **changes: object
    ) -> OrgNode:
        """Change where the review of a node's work stands, and with it the
        node's status. ``changes`` are other fields to change with it.

        Raises:
            NotFound: The node does not exist.
        """
        with self._lock:
            node = self._require(node_id)
            status = status_of(node.model_copy(update={"review": review}))
            return self.update_node(node_id, review=review, status=status, **changes)

    def set_stopped(self, node_id: str, stopped: Stopped | None) -> OrgNode:
        """Record that a node was stopped, or that it goes on, and with it
        the node's status.

        Raises:
            NotFound: The node does not exist.
        """
        with self._lock:
            node = self._require(node_id)
            status = status_of(node.model_copy(update={"stopped": stopped}))
            return self.update_node(node_id, stopped=stopped, status=status)

    def upsert(self, node_input: OrgNodeInput) -> OrgNode:
        """Insert a node or update an existing one.

        A node cannot move to a different parent; remove it and insert it
        again instead.

        Raises:
            OrgStoreError: The node would not fit into the tree.
        """
        with self._lock:
            if isinstance(node_input, SessionTeamInput):
                return self._upsert_seat(node_input)
            existing = self._nodes.get(node_input.id)
            parent_id = node_input.parent_id
            if existing is not None:
                if existing.parent_id != parent_id:
                    raise OrgStoreError(
                        f"Node {node_input.id} cannot change parent from "
                        f"{existing.parent_id} to {parent_id}"
                    )
            elif parent_id is None:
                if self._root_id is not None:
                    raise OrgStoreError(
                        f"The tree already has a root ({self._root_id}); "
                        f"{node_input.id} needs a parent"
                    )
            elif parent_id not in self._nodes:
                raise OrgStoreError(
                    f"Parent {parent_id} of node {node_input.id} does not exist"
                )
            self._check_escalations(node_input, self._chain(self._nodes, parent_id))

            child_ids = existing.child_ids if existing else []
            node = OrgNode(
                **node_input.model_dump(),
                child_ids=child_ids,
                descendant_counts=(
                    existing.descendant_counts if existing else StatusCounts()
                ),
                escalation_count=self._waiting_on_human(node_input)
                + sum(self._nodes[c].escalation_count for c in child_ids),
            )
            self._nodes[node.id] = node
            self._index_agents(existing, node)
            if existing is None:
                if parent_id is None:
                    self._root_id = node.id
                else:
                    parent = self._nodes[parent_id]
                    self._nodes[parent_id] = parent.model_copy(
                        update={"child_ids": [*parent.child_ids, node.id]}
                    )

            self._emit(NodeUpsertEvent(node=node), node.id, parent_id)
            # A new child changes its parent's ``child_ids``, so the parent is
            # always re-announced on insert; above that, only changed rollups.
            self._refresh_ancestors(parent_id, always_announce_first=existing is None)
            # What waits on the human names the teams on the way to it.
            if (
                existing is None
                or existing.escalations != node.escalations
                or existing.title != node.title
            ):
                self._announce_escalations()
            self._changed()
            return node

    def _upsert_seat(self, seat_input: SessionTeamInput) -> SessionTeam:
        """Insert or change what a session has at one team."""
        existing = self._seats.get(seat_input.id)
        parent_id = seat_input.parent_id
        session = self._sessions.get(seat_input.session)
        if session is None:
            raise OrgStoreError(f"Session {seat_input.session} is not open")
        reviewing = review_place_id(session.above_id, session.id)
        above = self._seats.get(parent_id or "")
        if seat_input.review_only:
            fits = seat_input.id == reviewing and parent_id is None
        elif seat_input.id != session_team_id(seat_input.team_id, session.id):
            raise OrgStoreError(f"{seat_input.id} is no team of session {session.id}")
        elif seat_input.team_id == session.team_id:
            fits = parent_id == reviewing and above is not None
        else:
            fits = (
                above is not None
                and above.session == session.id
                and not above.review_only
            )
        if not fits:
            raise OrgStoreError(
                f"{seat_input.id} cannot be below {parent_id} in session {session.id}"
            )
        if existing is not None and existing.parent_id != parent_id:
            raise OrgStoreError(f"{seat_input.id} cannot change what is above it")
        units = self._units()
        self._check_escalations(seat_input, self._chain(units, parent_id))
        seat = SessionTeam(
            **seat_input.model_dump(),
            child_ids=existing.child_ids if existing else [],
        )
        self._seats[seat.id] = seat
        self._index_agents(existing, seat)
        if existing is None and isinstance(above, SessionTeam):
            self._seats[above.id] = above.model_copy(
                update={"child_ids": [*above.child_ids, seat.id]}
            )
        self._announce_sessions()
        if existing is None or existing.escalations != seat.escalations:
            self._announce_escalations()
        self._changed()
        return seat

    def move(self, node_id: str, parent_id: str) -> OrgNode:
        """Put a node, with everything below it, under another parent.

        An escalation of the node or of a node below it keeps the handoffs
        of the teams that are still above it in the same order, and is held
        by the next team above those.

        Raises:
            NotFound: The node or the new parent does not exist.
            OrgStoreError: The node is the root, or the new parent is the
                node itself or below it.
        """
        with self._lock:
            node = self._require(node_id)
            self._require(parent_id)
            former = node.parent_id
            home = self._home_of(node_id)
            if (
                former is None
                or parent_id not in home
                or session_id_of(home[parent_id]) != session_id_of(node)
                or reviews_only(home[parent_id])
                or node_id in [parent_id, *self._chain(home, parent_id)]
            ):
                raise OrgStoreError(f"Node {node_id} cannot move under {parent_id}")
            if former == parent_id:
                return node
            old_parent = home[former]
            home[former] = old_parent.model_copy(
                update={"child_ids": [c for c in old_parent.child_ids if c != node_id]}
            )
            new_parent = home[parent_id]
            home[parent_id] = new_parent.model_copy(
                update={"child_ids": [*new_parent.child_ids, node_id]}
            )
            home[node_id] = node.model_copy(update={"parent_id": parent_id})
            moved = [node_id]
            for current in moved:
                moved.extend(home[current].child_ids)
            for current in moved:
                below = home[current]
                chain = self._chain(home, below.parent_id)
                home[current] = below.model_copy(
                    update={
                        "escalations": [
                            self._rerouted(escalation, chain)
                            for escalation in below.escalations
                        ]
                    }
                )
            if home is self._nodes:
                # The node's own rollup first, then both lines of teams
                # above it.
                for current in reversed(moved):
                    self._refresh_ancestors(current, always_announce_first=False)
                for current in moved:
                    below = home[current]
                    self._emit(NodeUpsertEvent(node=below), below.id, below.parent_id)
                self._refresh_ancestors(former, always_announce_first=True)
                self._refresh_ancestors(parent_id, always_announce_first=True)
            else:
                self._announce_sessions()
            self._announce_escalations()
            self._changed()
            return home[node_id]

    def _home_of(self, node_id: str) -> dict[str, OrgNode]:
        """Where ``node_id`` is kept: with the org tree, or with the teams
        of the sessions."""
        if node_id in self._nodes:
            return self._nodes
        return cast(dict[str, OrgNode], self._seats)

    @staticmethod
    def _rerouted(escalation: Escalation, chain: list[str]) -> Escalation:
        """An escalation as it stands when the teams above it are ``chain``."""
        kept: list[EscalationHandoff] = []
        for handoff, above in zip(escalation.handoffs, chain, strict=False):
            if handoff.node_id != above:
                break
            kept.append(handoff)
        holder = chain[len(kept)] if len(kept) < len(chain) else None
        return escalation.model_copy(update={"handoffs": kept, "held_by": holder})

    def remove(self, node_id: str) -> bool:
        """Remove a node and everything below it.

        Returns:
            Whether the node existed.
        """
        with self._lock:
            node = self.get(node_id)
            if node is None:
                return False
            if isinstance(node, SessionTeam):
                return self._remove_seat(node)

            doomed: list[OrgNode] = []
            stack = [node]
            while stack:
                current = stack.pop()
                doomed.append(current)
                stack.extend(self._nodes[child_id] for child_id in current.child_ids)
            # Leaves first, so a subscriber never holds a child of a missing
            # parent.
            doomed.reverse()

            # Decide who hears about each removal while the doomed nodes can
            # still be resolved, so a subscriber focused inside the removed
            # subtree is still matched.
            for gone in doomed:
                self._emit(NodeRemovedEvent(id=gone.id), gone.id, gone.parent_id)
            for gone in doomed:
                self._index_agents(gone, None)
                del self._nodes[gone.id]

            if node.parent_id is None:
                self._root_id = None
            else:
                parent = self._nodes[node.parent_id]
                self._nodes[parent.id] = parent.model_copy(
                    update={"child_ids": [c for c in parent.child_ids if c != node_id]}
                )
            self._refresh_ancestors(node.parent_id, always_announce_first=True)
            self._announce_escalations()
            self._changed()
            return True

    def _remove_seat(self, seat: SessionTeam) -> bool:
        """Remove what a session has at a team and at the teams below it.
        The place its work is reviewed from, and the team it was opened at,
        go only with the session."""
        if seat.parent_id is None or reviews_only(self._seats.get(seat.parent_id)):
            raise OrgStoreError(f"{seat.id} is removed by closing its session")
        doomed = [seat]
        for current in doomed:
            doomed.extend(self._seats[child_id] for child_id in current.child_ids)
        for gone in doomed:
            self._index_agents(gone, None)
            del self._seats[gone.id]
        parent = self._seats[seat.parent_id]
        self._seats[parent.id] = parent.model_copy(
            update={"child_ids": [c for c in parent.child_ids if c != seat.id]}
        )
        self._announce_sessions()
        self._announce_escalations()
        self._changed()
        return True

    def replace_all(self, inputs: Iterable[OrgNodeInput]) -> None:
        """Replace the whole tree without emitting events.

        Used when loading persisted state or a fixture. The store is left
        untouched if the input is not one well-formed tree. The sessions
        that are open stay, but for one that has lost a team.

        Raises:
            OrgStoreError: The input is not one well-formed tree.
        """
        nodes: dict[str, OrgNode] = {}
        for node_input in inputs:
            if node_input.id in nodes:
                raise OrgStoreError(f"Duplicate node id {node_input.id}")
            nodes[node_input.id] = OrgNode(**node_input.model_dump())

        root_id: str | None = None
        for node in nodes.values():
            if node.parent_id is None:
                if root_id is not None:
                    raise OrgStoreError(
                        f"More than one root node ({root_id}, {node.id})"
                    )
                root_id = node.id
                continue
            parent = nodes.get(node.parent_id)
            if parent is None:
                raise OrgStoreError(
                    f"Parent {node.parent_id} of node {node.id} does not exist"
                )
            parent.child_ids.append(node.id)
            self._check_escalations(node, self._chain(nodes, node.parent_id))
        if nodes and root_id is None:
            raise OrgStoreError("The tree has no root node")

        ordered = self._ordered(nodes, root_id)
        if len(ordered) != len(nodes):
            raise OrgStoreError("The tree contains a cycle")
        for node in reversed(ordered):
            node.descendant_counts, node.escalation_count = self._rollup(nodes, node)

        with self._lock:
            kept = [
                (session, list[SessionTeamInput](self.session(session.id)))
                for session in self._sessions.values()
            ]
            self._nodes = nodes
            self._root_id = root_id
            self._sessions = {}
            self._seats = {}
            self._agents = {}
            self._hand_overs = {}
            for node in nodes.values():
                self._index_agents(None, node)
            self.restore_sessions(kept)

    def restore_sessions(
        self, sessions: Iterable[tuple[SessionInput, list[SessionTeamInput]]]
    ) -> None:
        """Take sessions back as they were kept, each with its teams, the
        one it was opened at first, without emitting events.

        One is left out only when its teams do not hang together. A team
        of it that the org no longer has stays the session's: what becomes
        of it is decided when its work is reviewed.
        """
        with self._lock:
            for session, teams in sessions:
                if session.id in self._sessions:
                    continue
                self._sessions[session.id] = session
                try:
                    # The place a session is reviewed from is above the rest.
                    for team in sorted(teams, key=lambda team: not team.review_only):
                        self._restore_seat(team)
                    _ = self._seats[session_team_id(session.team_id, session.id)]
                except (OrgStoreError, KeyError):
                    for seat in self.session(session.id):
                        self._index_agents(seat, None)
                        del self._seats[seat.id]
                    del self._sessions[session.id]

    def _restore_seat(self, team: SessionTeamInput) -> None:
        """Put back what a session has at one team, without telling anyone."""
        above = self._seats.get(team.parent_id or "")
        if above is None and not team.review_only:
            raise OrgStoreError(f"Nothing is above {team.id}")
        self._check_escalations(team, self._chain(self._units(), team.parent_id))
        data = team.model_dump(exclude=_DERIVED_FIELDS)
        seat = SessionTeam(**data)
        self._seats[seat.id] = seat
        self._index_agents(None, seat)
        if isinstance(above, SessionTeam):
            above.child_ids.append(seat.id)

    def raise_escalation(
        self,
        node_id: str,
        escalation_id: str,
        *,
        kind: EscalationKind,
        title: str,
        detail: str = "",
        at: str | None = None,
        raised_by: AgentRole = "ic",
    ) -> Escalation:
        """Record something an agent of ``node_id`` cannot settle itself.

        It goes to the place above, never straight to the human, unless
        nobody works above ``node_id``: it is the place a session was opened
        at. One an IC raised
        blocks its team until it is settled.

        Raises:
            NotFound: The node does not exist.
            OrgStoreError: The node already has an escalation with this id.
        """
        with self._lock:
            node = self._require(node_id)
            if any(e.id == escalation_id for e in node.escalations):
                raise OrgStoreError(
                    f"Node {node_id} already has an escalation {escalation_id}"
                )
            escalation = Escalation(
                id=escalation_id,
                kind=kind,
                raised_by=raised_by,
                title=title,
                detail=detail,
                raised_at=at or _now(),
                held_by=next(iter(self._chain(self._units(), node.parent_id)), None),
            )
            self._set_escalations(node, [*node.escalations, escalation])
            return escalation

    def pass_up_escalation(
        self, node_id: str, escalation_id: str, *, reason: str, at: str | None = None
    ) -> Escalation:
        """The node holding an escalation could not settle it: send it one up.

        This is the only way an escalation moves, so it reaches the human
        only after every node above the one that raised it has said why it
        could not settle it. Design decisions in particular should almost
        always be settled on the way.

        Args:
            node_id: The node that raised the escalation, not the one holding it.
            reason: Why the node holding it could not settle it.

        Raises:
            NotFound: The node or the escalation does not exist.
            OrgStoreError: The escalation already waits on the human, or the
                reason is blank.
        """
        with self._lock:
            node = self._require(node_id)
            escalation = self._require_escalation(node, escalation_id)
            if escalation.held_by is None:
                raise OrgStoreError(
                    f"Escalation {escalation_id} of node {node_id} already waits "
                    "on the human"
                )
            if not reason.strip():
                raise OrgStoreError("Passing an escalation up needs a reason")
            holder = self._units()[escalation.held_by]
            passed = escalation.model_copy(
                update={
                    "held_by": next(
                        iter(self._chain(self._units(), holder.parent_id)), None
                    ),
                    "handoffs": [
                        *escalation.handoffs,
                        EscalationHandoff(
                            node_id=holder.id, reason=reason.strip(), at=at or _now()
                        ),
                    ],
                }
            )
            self._set_escalations(
                node,
                [passed if e.id == escalation_id else e for e in node.escalations],
            )
            return passed

    def ask_escalation(
        self,
        node_id: str,
        escalation_id: str,
        team_id: str | None,
        at: str | None = None,
    ) -> Escalation:
        """Give an escalation to a team that is not above the one that
        raised it, to settle directly; None gives it back to the node that
        holds it by the chain.

        Raises:
            NotFound: The node or the escalation does not exist.
        """
        with self._lock:
            node = self._require(node_id)
            escalation = self._require_escalation(node, escalation_id)
            asked = escalation.model_copy(
                update={
                    "asked_of": team_id,
                    "asked_at": (at or _now()) if team_id else None,
                }
            )
            self._set_escalations(
                node,
                [asked if e.id == escalation_id else e for e in node.escalations],
            )
            return asked

    def settle_escalation(self, node_id: str, escalation_id: str) -> None:
        """Close an escalation, whoever settled it.

        Raises:
            NotFound: The node or the escalation does not exist.
        """
        with self._lock:
            node = self._require(node_id)
            self._require_escalation(node, escalation_id)
            self._set_escalations(
                node, [e for e in node.escalations if e.id != escalation_id]
            )

    def escalations(self, focus_id: str) -> list[OpenEscalation]:
        """Every escalation at or below ``focus_id`` that waits on the human.

        Critical issues come first, then the longest waiting.
        """
        with self._lock:
            found: list[OpenEscalation] = []
            focus = self._nodes.get(focus_id)
            stack = [] if focus is None else [focus]
            while stack:
                node = stack.pop()
                # The rollup lets whole quiet subtrees be skipped.
                if node.escalation_count == 0:
                    continue
                waiting = [e for e in node.escalations if e.held_by is None]
                if waiting:
                    path = [
                        EscalationPathStep(id=step.id, title=step.title)
                        for step in (*self.ancestors(node.id), node)
                    ]
                    found.extend(
                        OpenEscalation(node_id=node.id, escalation=e, path=path)
                        for e in waiting
                    )
                stack.extend(self._nodes[c] for c in reversed(node.child_ids))
            if focus_id == self._root_id:
                # A session is under no node: what its teams wait on the
                # human for is told with the whole org's.
                for seat in self._seats.values():
                    # A team a session set up may not be the org's yet:
                    # the way to it is the one inside the session.
                    path = [
                        EscalationPathStep(id=team_id_of(step), title=step.title)
                        for step in (*self.ancestors(seat.id), seat)
                    ]
                    found.extend(
                        OpenEscalation(
                            node_id=seat.team_id,
                            session_id=seat.session,
                            session=self._sessions[seat.session].title,
                            escalation=e,
                            path=path,
                        )
                        for e in seat.escalations
                        if e.held_by is None
                    )
            found.sort(
                key=lambda item: (
                    item.escalation.kind != "critical_issue",
                    item.escalation.raised_at,
                )
            )
            return found

    def snapshot(
        self, focus_id: str, expanded: Collection[str] = (), *, full: bool = False
    ) -> SnapshotEvent | None:
        """The focused node, its ancestors, and the direct reports of the
        focused node and of every node in ``expanded``.

        With ``full``, every node below the focused one instead. Unknown ids
        in ``expanded`` are ignored.
        """
        with self._lock:
            focus = self._nodes.get(focus_id)
            if focus is None:
                return None
            nodes = {
                node.id: node
                for node in (
                    *self.ancestors(focus_id),
                    focus,
                    *self.children(focus_id),
                )
            }
            for node_id in expanded:
                if node_id in self._nodes:
                    for child in self.children(node_id):
                        nodes[child.id] = child
            if full:
                for node in self._ordered(self._nodes, focus_id):
                    nodes[node.id] = node
            return SnapshotEvent(
                focus_id=focus_id,
                nodes=list(nodes.values()),
                escalations=self._waiting_anywhere(),
                sessions=self.sessions(),
            )

    def subscribe(
        self,
        focus_id: str,
        listener: OrgListener,
        expanded: Collection[str] = (),
        *,
        full: bool = False,
    ) -> tuple[SnapshotEvent | None, Callable[[], None]]:
        """Receive changes to what a snapshot of ``focus_id`` contains.

        The snapshot and the subscription are taken together, so no change
        falls between them.

        Returns:
            The current snapshot (None when ``focus_id`` is unknown, in which
            case nothing is subscribed) and a function that unsubscribes.
        """
        with self._lock:
            snapshot = self.snapshot(focus_id, expanded, full=full)
            if snapshot is not None:
                self._listeners[listener] = _Subscription(
                    focus_id,
                    frozenset(expanded),
                    full,
                    frozenset(
                        self._chain(self._nodes, self._nodes[focus_id].parent_id)
                    ),
                    snapshot.escalations,
                )

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.pop(listener, None)

        return snapshot, unsubscribe

    @staticmethod
    def _ordered(nodes: dict[str, OrgNode], root_id: str | None) -> list[OrgNode]:
        ordered: list[OrgNode] = []
        queue = [] if root_id is None else [root_id]
        for node_id in queue:
            node = nodes[node_id]
            ordered.append(node)
            queue.extend(node.child_ids)
        return ordered

    @staticmethod
    def _rollup(nodes: dict[str, OrgNode], node: OrgNode) -> tuple[StatusCounts, int]:
        """The node's ``descendant_counts`` and ``escalation_count``."""
        counts = dict.fromkeys(NODE_STATUSES, 0)
        escalation_count = OrgStore._waiting_on_human(node)
        for child_id in node.child_ids:
            child = nodes[child_id]
            counts[child.status] += 1
            for status in NODE_STATUSES:
                counts[status] += getattr(child.descendant_counts, status)
            escalation_count += child.escalation_count
        return StatusCounts(**counts), escalation_count

    def _refresh_ancestors(
        self, start_id: str | None, *, always_announce_first: bool
    ) -> None:
        """Recompute rollups from ``start_id`` up to the root.

        Announces every node whose rollup changed.
        """
        current_id = start_id
        first = True
        while current_id is not None:
            current = self._nodes[current_id]
            counts, escalation_count = self._rollup(self._nodes, current)
            changed = (
                counts != current.descendant_counts
                or escalation_count != current.escalation_count
            )
            if changed:
                current = current.model_copy(
                    update={
                        "descendant_counts": counts,
                        "escalation_count": escalation_count,
                    }
                )
                self._nodes[current_id] = current
            if changed or (first and always_announce_first):
                self._emit(NodeUpsertEvent(node=current), current.id, current.parent_id)
            first = False
            current_id = current.parent_id

    def _emit(self, event: OrgChange, node_id: str, parent_id: str | None) -> None:
        if not self._listeners:
            return
        above = self._chain(self._nodes, parent_id)
        for listener, subscription in list(self._listeners.items()):
            focus_id = subscription.focus_id
            if (
                node_id == focus_id
                or parent_id == focus_id
                or parent_id in subscription.expanded
                or (node_id in subscription.above and focus_id in self._nodes)
                or (subscription.full and focus_id in above)
            ):
                listener(event)

    def _waiting_anywhere(self) -> list[OpenEscalation]:
        """What waits on the human in the whole tree.

        Subscribers get this whatever node they are focused on: it is the
        one thing the human must not miss by looking somewhere else.
        """
        return [] if self._root_id is None else self.escalations(self._root_id)

    @staticmethod
    def _waiting_on_human(node: OrgNodeInput) -> int:
        return sum(1 for e in node.escalations if e.held_by is None)

    def _units(self) -> Mapping[str, OrgNode]:
        """The teams of the org and of the sessions, by id."""
        return _Units(self._nodes, self._seats)

    def _session_view(self, session: SessionInput) -> Session:
        teams = self.session(session.id)
        top = self._seats[session_team_id(session.team_id, session.id)]
        return Session(
            **session.model_dump(),
            team=top.title,
            status=top.status,
            review=top.review.state if top.review else None,
            conversation_id=next(
                (agent.conversation_id for agent in top.agents if agent.role == "ic"),
                None,
            ),
            teams=teams,
        )

    def _announce_sessions(self) -> None:
        """Tell every subscriber what the sessions are now."""
        if not self._listeners:
            return
        event = SessionsEvent(sessions=self.sessions())
        for listener in list(self._listeners):
            listener(event)

    @staticmethod
    def _chain(nodes: Mapping[str, OrgNode], parent_id: str | None) -> list[str]:
        """Ids of the nodes above a node, nearest first.

        Stops at a missing parent or a cycle; the caller reports those.
        """
        chain: list[str] = []
        current = parent_id
        while current is not None and current in nodes and current not in chain:
            # Nobody stands at the place a session's work is reviewed from:
            # what goes up from the session goes to the human.
            if reviews_only(nodes[current]):
                break
            chain.append(current)
            current = nodes[current].parent_id
        return chain

    @staticmethod
    def _check_escalations(node: OrgNodeInput, chain: list[str]) -> None:
        """Refuse an escalation that skipped a node on its way up.

        Raises:
            OrgStoreError: An escalation's handoffs are not the unbroken chain
                from the node's parent, or it is held by the wrong node.
        """
        for escalation in node.escalations:
            passed = [handoff.node_id for handoff in escalation.handoffs]
            holder = chain[len(passed)] if len(passed) < len(chain) else None
            if passed != chain[: len(passed)] or escalation.held_by != holder:
                raise OrgStoreError(
                    f"Escalation {escalation.id} of node {node.id} must be passed "
                    f"up through {chain or 'nobody'} in order; it was passed up "
                    f"by {passed} and is held by {escalation.held_by}"
                )

    def _require(self, node_id: str) -> OrgNode:
        node = self.get(node_id)
        if node is None:
            raise NotFound(f"Unknown node: {node_id}")
        return node

    @staticmethod
    def _require_escalation(node: OrgNode, escalation_id: str) -> Escalation:
        for escalation in node.escalations:
            if escalation.id == escalation_id:
                return escalation
        raise NotFound(f"Node {node.id} has no escalation {escalation_id}")

    def _set_escalations(self, node: OrgNode, escalations: list[Escalation]) -> None:
        status = status_of(node.model_copy(update={"escalations": escalations}))
        self.update_node(node.id, escalations=escalations, status=status)

    def _index_agents(self, old: OrgNode | None, new: OrgNode | None) -> None:
        if old is not None:
            for agent in old.agents:
                if agent.conversation_id is not None:
                    self._agents.pop(agent.conversation_id, None)
                if agent.hand_over is not None:
                    self._hand_overs.pop(agent.hand_over.conversation_id, None)
        if new is not None:
            for agent in new.agents:
                if agent.conversation_id is not None:
                    self._agents[agent.conversation_id] = (new.id, agent.role)
                if agent.hand_over is not None:
                    self._hand_overs[agent.hand_over.conversation_id] = new.id

    def _changed(self) -> None:
        for listener in list(self._change_listeners):
            listener()

    def _announce_escalations(self) -> None:
        """Tell each subscriber when what waits on the human has changed."""
        current = self._waiting_anywhere()
        for listener, subscription in list(self._listeners.items()):
            if current != subscription.escalations:
                subscription.escalations = current
                listener(EscalationsEvent(escalations=current))
