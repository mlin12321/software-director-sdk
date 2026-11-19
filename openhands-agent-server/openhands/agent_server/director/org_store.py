"""In-memory org tree.

The store owns ``child_ids`` on every team and is the only thing that emits
org events, so every subscriber sees one consistent tree.
"""

import threading
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

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
    Stopped,
    Usage,
)
from openhands.sdk.errors import Invalid, NotFound


# An agent whose conversation failed, and one that has none on this server.
AGENT_BLOCKED = "blocked"
AGENT_LOST = "lost"


def status_of(node: SessionTeamInput) -> NodeStatus:
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
    if AGENT_BLOCKED in agents or any(
        escalation.raised_by == "ic" for escalation in node.escalations
    ):
        return "blocked"
    if "working" in agents:
        return "working"
    if node.review is not None and node.review.state != "changes_requested":
        # Submitted: it rests until its reviewer or something else wakes it.
        return "done"
    return "blocked" if AGENT_LOST in agents else "idle"


type OrgChange = (
    SnapshotEvent
    | NodeUpsertEvent
    | NodeRemovedEvent
    | EscalationsEvent
    | SessionsEvent
)
type OrgListener = Callable[[OrgChange], None]

_DERIVED_FIELDS = {"child_ids"}


def _of_all(top: NodeStatus, places: list[NodeStatus]) -> NodeStatus:
    """What a session is, from what the team it was opened at is and what
    every place of it is, its reviewer's included."""
    if top != "blocked" and "working" in places:
        return "working"
    return "blocked" if "blocked" in places else top


def session_team_id(team_id: str, session_id: str) -> str:
    """The id of what a session has at the team ``team_id``."""
    return f"{team_id}@{session_id}"


def division_lead_id(team_id: str, session_id: str) -> str:
    """The id of a session's Division Lead place, at the team
    ``team_id``."""
    return f"{team_id}@{session_id}@lead"


def is_division_lead(node: OrgNodeInput | None) -> bool:
    """Whether ``node`` is a session's Division Lead place."""
    return isinstance(node, SessionTeamInput) and node.division_lead


def session_id_of(node: OrgNodeInput) -> str:
    """The session ``node`` belongs to; '' for a team of the org."""
    return node.session if isinstance(node, SessionTeamInput) else ""


def team_id_of(node: OrgNodeInput) -> str:
    """The team of the org ``node`` is, or is the work of in a session."""
    return node.team_id if isinstance(node, SessionTeamInput) else node.id


@dataclass
class _Subscription:
    focus_id: str
    # Teams whose direct reports the subscriber also follows.
    expanded: frozenset[str]
    # Whether the subscriber follows every team below the focused one.
    full: bool
    # The teams above the focused one.
    above: frozenset[str]
    # What the subscriber was last told, to announce only real changes.
    escalations: list[OpenEscalation] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class OrgStoreError(Invalid):
    """A change would leave the org tree malformed."""


class OrgStore:
    """A tree of :class:`OrgNode`, safe to use from several threads.

    The tree is the org's structure: what a team is, and nothing of its
    work. Beside it the store holds the open sessions: each is a record of
    its own with, for the team it was opened at and every team below that
    one, a place (:class:`SessionTeam`). Places are in no team's
    ``child_ids`` and in no snapshot's ``nodes``. All work is at places:
    the agents, the review, the escalations.

    Listeners are called while the store is locked, in the order changes
    happened. They must return quickly and must not call back into the store.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: dict[str, OrgNode] = {}
        self._sessions: dict[str, SessionInput] = {}
        self._places: dict[str, SessionTeam] = {}
        self._root_id: str | None = None
        self._listeners: dict[OrgListener, _Subscription] = {}
        self._change_listeners: list[Callable[[], None]] = []
        self._agents: dict[str, tuple[str, AgentRole]] = {}
        # The hand-over conversation of each IC that has one -> its team.
        self._hand_overs: dict[str, str] = {}

    @property
    def root_id(self) -> str | None:
        return self._root_id

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> OrgNode | None:
        """A team of the org, or a place of a session, by its id."""
        return self._nodes.get(node_id) or self._places.get(node_id)

    def team(self, team_id: str) -> OrgNode | None:
        """A team of the org tree; None for anything else."""
        return self._nodes.get(team_id)

    def place(self, place_id: str) -> SessionTeam | None:
        """A place of a session; None for anything else."""
        return self._places.get(place_id)

    def sub_teams(self, team_id: str) -> list[OrgNode]:
        """The teams of the org directly below the team ``team_id``."""
        with self._lock:
            node = self._nodes.get(team_id)
            return [self._nodes[child] for child in node.child_ids] if node else []

    def below(self, place_id: str) -> list[SessionTeam]:
        """The places directly below the place ``place_id`` in its session."""
        with self._lock:
            place = self._places.get(place_id)
            return [self._places[child] for child in place.child_ids] if place else []

    def opens_session(self, place: SessionTeam) -> bool:
        """Whether ``node`` is the place a session was opened at."""
        with self._lock:
            session = self._sessions.get(place.session)
            return (
                session is not None
                and not place.division_lead
                and session.team_id == place.team_id
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
            places = [
                place for place in self._places.values() if place.session == session_id
            ]
            opened = self._sessions.get(session_id)
            top = session_team_id(opened.team_id, session_id) if opened else ""
            return sorted(
                places, key=lambda place: (place.division_lead, place.id != top)
            )

    def sessions(self) -> list[Session]:
        """The open sessions as a client reads them, oldest first."""
        with self._lock:
            # One that is being opened has its places only one by one.
            return [
                self._session_view(session)
                for session in self._sessions.values()
                if session_team_id(session.team_id, session.id) in self._places
            ]

    def instance(self, team_id: str, session_id: str) -> SessionTeam | None:
        """The place a session has at the team ``team_id``."""
        return self._places.get(session_team_id(team_id, session_id))

    def teams_below(self, place_id: str) -> int:
        """How many places are below the place ``place_id``, at any depth."""
        with self._lock:
            found = [place_id]
            for current in found:
                found.extend(team.id for team in self.below(current))
            return len(found) - 1

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
                return self._places[session_team_id(session.team_id, session.id)]
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
            before = self._places[session_team_id(was.team_id, was.id)]
            reviewed_from = self._places[before.parent_id or ""]
            if reviewed_from.agents or before.review is not None:
                raise OrgStoreError(f"The work of session {session.id} is in review")
            if (
                top.id != session_team_id(session.team_id, session.id)
                or top.id in self._places
                or top.parent_id != review.id
                or review.id != division_lead_id(session.above_id, session.id)
                or not review.division_lead
            ):
                raise OrgStoreError(f"Session {session.id} cannot go to {top.id}")
            self._sessions[session.id] = session
            del self._places[reviewed_from.id]
            self._places[review.id] = SessionTeam(
                **review.model_dump(), child_ids=[top.id]
            )
            self._places[top.id] = SessionTeam(
                **top.model_dump(), child_ids=[before.id]
            )
            self._places[before.id] = before.model_copy(update={"parent_id": top.id})
            moved = [before.id]
            for current in moved:
                moved.extend(self._places[current].child_ids)
            for current in moved:
                below = self._places[current]
                chain = self._chain(below.parent_id)
                self._places[current] = below.model_copy(
                    update={
                        "escalations": [
                            self._rerouted(escalation, chain)
                            for escalation in below.escalations
                        ]
                    }
                )
            for other in others:
                self._upsert_place(other)
            self._announce_sessions()
            self._announce_escalations()
            self._changed()
            return self._places[top.id]

    def set_shape(
        self, session_id: str, shape: list[tuple[str, str, str, str]]
    ) -> None:
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
                self._announce_escalations()

    def set_session_use(
        self, session_id: str, usage: Usage, cost_allowance: float
    ) -> Session | None:
        """Keep what a session has used and how much it may use.

        Returns:
            The session as it is now; None when it is not open.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            changed = session.model_copy(
                update={"usage": usage, "cost_allowance": cost_allowance}
            )
            if changed != session:
                self._sessions[session_id] = changed
                self._announce_sessions()
                self._changed()
            return self._session_view(changed)

    def close_session(self, session_id: str) -> bool:
        """Forget a session and everything it has at its teams.

        Returns:
            Whether the session was open.
        """
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                return False
            for place in self.session(session_id):
                self._index_agents(place, None)
                del self._places[place.id]
            self._announce_sessions()
            self._announce_escalations()
            self._changed()
            return True

    def teams_above(self, team_id: str) -> list[OrgNode]:
        """The teams of the org above the team ``team_id``, the top first."""
        with self._lock:
            chain: list[OrgNode] = []
            node = self._nodes.get(team_id)
            while node is not None and node.parent_id is not None:
                node = self._nodes.get(node.parent_id)
                if node is not None:
                    chain.append(node)
            chain.reverse()
            return chain

    def above(self, place_id: str) -> list[SessionTeam]:
        """The places above the place ``place_id`` in its session, the one
        the session was opened at first. Nobody stands at a session's
        Division Lead place, so it is not one of them."""
        with self._lock:
            return (
                [self._places[above] for above in reversed(self._chain(place_id))][:-1]
                if place_id in self._places
                else []
            )

    def places(self) -> list[SessionTeam]:
        """Every place of every session."""
        with self._lock:
            return list(self._places.values())

    def find_by_conversation(
        self, conversation_id: str
    ) -> tuple[SessionTeam, AgentRole] | None:
        """The place an agent's conversation belongs to, and the agent's role."""
        with self._lock:
            found = self._agents.get(conversation_id)
            if found is None:
                return None
            place_id, role = found
            return self._places[place_id], role

    def find_hand_over(self, conversation_id: str) -> SessionTeam | None:
        """The place whose IC has ``conversation_id`` as its hand-over
        conversation."""
        with self._lock:
            place_id = self._hand_overs.get(conversation_id)
            return self._places[place_id] if place_id is not None else None

    def set_hand_over(self, place_id: str, mode: HandOverMode | None) -> SessionTeam:
        """Put a place's IC in hand-over mode, change how that stands, or
        end it. The place's status does not follow: a hand-over says nothing
        of how the team's work stands.

        Raises:
            NotFound: The place does not exist.
        """
        with self._lock:
            node = self._require_place(place_id)
            agents = [
                agent.model_copy(update={"hand_over": mode})
                if agent.role == "ic"
                else agent
                for agent in node.agents
            ]
            if agents == node.agents:
                return node
            return self.update_place(place_id, agents=agents)

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

    def update_team(self, team_id: str, **changes: object) -> OrgNode:
        """Change fields of a team of the org.

        Raises:
            NotFound: The org has no such team.
            OrgStoreError: The changed team would not fit into the tree.
        """
        with self._lock:
            node = self._nodes.get(team_id)
            if node is None:
                raise NotFound(f"Unknown team: {team_id}")
            current = node.model_dump(exclude=_DERIVED_FIELDS)
            return self.upsert(OrgNodeInput.model_validate({**current, **changes}))

    def update_place(self, place_id: str, **changes: object) -> SessionTeam:
        """Change fields of a place of a session.

        Raises:
            NotFound: No session has such a place.
            OrgStoreError: The changed place would not fit into its session.
        """
        with self._lock:
            place = self._require_place(place_id)
            current = place.model_dump(exclude=_DERIVED_FIELDS)
            return self._upsert_place(
                SessionTeamInput.model_validate({**current, **changes})
            )

    def set_agent(
        self,
        place_id: str,
        role: AgentRole,
        *,
        conversation_id: str | None = None,
        status: str | None = None,
        kind: AgentKind | None = None,
        acp_server: str | None = None,
        model: str | None = None,
        reason: str | None = None,
    ) -> SessionTeam:
        """Change the conversation, the status or the kind of one agent of a
        place.

        The place's status follows (see :func:`status_of`). ``reason`` says
        why the agent is blocked or lost, and is forgotten once it is
        neither.

        Raises:
            NotFound: The place does not exist or has no agent in that role.
        """
        with self._lock:
            node = self._require_place(place_id)
            if all(agent.role != role for agent in node.agents):
                raise NotFound(f"Team {place_id} has no {role} agent")
            agents = [
                agent
                if agent.role != role
                else agent.model_copy(
                    update={
                        "conversation_id": conversation_id or agent.conversation_id,
                        "status": status or agent.status,
                        "reason": (reason or agent.reason)
                        if (status or agent.status) in (AGENT_BLOCKED, AGENT_LOST)
                        else None,
                        "kind": kind or agent.kind,
                        "acp_server": acp_server or agent.acp_server,
                        "model": model or agent.model,
                    }
                )
                for agent in node.agents
            ]
            if agents == node.agents:
                return node
            status = status_of(node.model_copy(update={"agents": agents}))
            return self.update_place(place_id, agents=agents, status=status)

    def set_review(
        self, place_id: str, review: Review | None, **changes: object
    ) -> SessionTeam:
        """Change where the review of a place's work stands, and with it the
        place's status. ``changes`` are other fields to change with it.

        Raises:
            NotFound: The place does not exist.
        """
        with self._lock:
            node = self._require_place(place_id)
            status = status_of(node.model_copy(update={"review": review}))
            return self.update_place(place_id, review=review, status=status, **changes)

    def set_stopped(self, place_id: str, stopped: Stopped | None) -> SessionTeam:
        """Record that a place was stopped, or that it goes on, and with it
        the place's status.

        Raises:
            NotFound: The place does not exist.
        """
        with self._lock:
            node = self._require_place(place_id)
            status = status_of(node.model_copy(update={"stopped": stopped}))
            return self.update_place(place_id, stopped=stopped, status=status)

    def upsert(self, node_input: OrgNodeInput) -> OrgNode:
        """Insert a team or update an existing one.

        A team cannot move to a different parent; remove it and insert it
        again instead.

        Raises:
            OrgStoreError: The team would not fit into the tree.
        """
        with self._lock:
            if isinstance(node_input, SessionTeamInput):
                return self._upsert_place(node_input)
            existing = self._nodes.get(node_input.id)
            parent_id = node_input.parent_id
            if existing is not None:
                if existing.parent_id != parent_id:
                    raise OrgStoreError(
                        f"Team {node_input.id} cannot change parent from "
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
                    f"Parent {parent_id} of team {node_input.id} does not exist"
                )
            node = OrgNode(
                **node_input.model_dump(),
                child_ids=existing.child_ids if existing else [],
            )
            self._nodes[node.id] = node
            if existing is None:
                if parent_id is None:
                    self._root_id = node.id
                else:
                    parent = self._nodes[parent_id]
                    self._nodes[parent_id] = parent.model_copy(
                        update={"child_ids": [*parent.child_ids, node.id]}
                    )

            self._emit(NodeUpsertEvent(node=node), node.id, parent_id)
            if existing is None and parent_id is not None:
                # Its ``child_ids`` changed.
                parent = self._nodes[parent_id]
                self._emit(NodeUpsertEvent(node=parent), parent.id, parent.parent_id)
            self._changed()
            return node

    def _upsert_place(self, place_input: SessionTeamInput) -> SessionTeam:
        """Insert or change what a session has at one team."""
        existing = self._places.get(place_input.id)
        parent_id = place_input.parent_id
        session = self._sessions.get(place_input.session)
        if session is None:
            raise OrgStoreError(f"Session {place_input.session} is not open")
        lead_id = division_lead_id(session.above_id, session.id)
        above = self._places.get(parent_id or "")
        if place_input.division_lead:
            fits = place_input.id == lead_id and parent_id is None
        elif place_input.id != session_team_id(place_input.team_id, session.id):
            raise OrgStoreError(f"{place_input.id} is no team of session {session.id}")
        elif place_input.team_id == session.team_id:
            fits = parent_id == lead_id and above is not None
        else:
            fits = (
                above is not None
                and above.session == session.id
                and not above.division_lead
            )
        if not fits:
            raise OrgStoreError(
                f"{place_input.id} cannot be below {parent_id} in session {session.id}"
            )
        if existing is not None and existing.parent_id != parent_id:
            raise OrgStoreError(f"{place_input.id} cannot change what is above it")
        self._check_escalations(place_input, self._chain(parent_id))
        place = SessionTeam(
            **place_input.model_dump(),
            child_ids=existing.child_ids if existing else [],
        )
        self._places[place.id] = place
        self._index_agents(existing, place)
        if existing is None and isinstance(above, SessionTeam):
            self._places[above.id] = above.model_copy(
                update={"child_ids": [*above.child_ids, place.id]}
            )
        self._announce_sessions()
        if (
            existing is None
            or existing.escalations != place.escalations
            or existing.title != place.title
        ):
            self._announce_escalations()
        self._changed()
        return place

    def move(self, team_id: str, parent_id: str) -> OrgNode:
        """Put a team of the org, with everything below it, under another
        team of the org.

        Raises:
            NotFound: The team or the new parent does not exist.
            OrgStoreError: The team is the root, or the new parent is the
                team itself or below it.
        """
        with self._lock:
            node = self._nodes.get(team_id)
            if node is None or parent_id not in self._nodes:
                raise NotFound(f"Unknown team: {team_id} or {parent_id}")
            former = node.parent_id
            above = [team.id for team in self.teams_above(parent_id)]
            if former is None or team_id in [parent_id, *above]:
                raise OrgStoreError(f"Team {team_id} cannot move under {parent_id}")
            if former == parent_id:
                return node
            self._unlink(self._nodes, former, team_id)
            new_parent = self._nodes[parent_id]
            self._nodes[parent_id] = new_parent.model_copy(
                update={"child_ids": [*new_parent.child_ids, team_id]}
            )
            moved = node.model_copy(update={"parent_id": parent_id})
            self._nodes[team_id] = moved
            self.show_again()
            self._changed()
            return moved

    def move_place(self, place_id: str, parent_id: str) -> SessionTeam:
        """Put a place, with everything below it, under another place of
        its session.

        An escalation of the place or of a place below it keeps the handoffs
        of the places that are still above it in the same order, and is held
        by the next place above those.

        Raises:
            NotFound: The place or the new parent does not exist.
            OrgStoreError: It is the place its session was opened at or is
                reviewed from, or the new parent is of another session, the
                place itself or below it.
        """
        with self._lock:
            place = self._require_place(place_id)
            above = self._require_place(parent_id)
            former = self._places.get(place.parent_id or "")
            if (
                former is None
                or former.division_lead
                or above.division_lead
                or above.session != place.session
                or place_id in [parent_id, *self._chain(parent_id)]
            ):
                raise OrgStoreError(f"Team {place_id} cannot move under {parent_id}")
            if former.id == parent_id:
                return place
            self._unlink(self._places, former.id, place_id)
            above = self._places[parent_id]
            self._places[parent_id] = above.model_copy(
                update={"child_ids": [*above.child_ids, place_id]}
            )
            self._places[place_id] = place.model_copy(update={"parent_id": parent_id})
            moved = [place_id]
            for current in moved:
                moved.extend(self._places[current].child_ids)
            for current in moved:
                below = self._places[current]
                chain = self._chain(below.parent_id)
                self._places[current] = below.model_copy(
                    update={
                        "escalations": [
                            self._rerouted(escalation, chain)
                            for escalation in below.escalations
                        ]
                    }
                )
            self._announce_sessions()
            self._announce_escalations()
            self._changed()
            return self._places[place_id]

    @staticmethod
    def _unlink[T: OrgNode](nodes: dict[str, T], parent_id: str, child_id: str) -> None:
        parent = nodes[parent_id]
        nodes[parent_id] = parent.model_copy(
            update={"child_ids": [c for c in parent.child_ids if c != child_id]}
        )

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
        """Remove a team and everything below it.

        Returns:
            Whether the team existed.
        """
        with self._lock:
            node = self.get(node_id)
            if node is None:
                return False
            if isinstance(node, SessionTeam):
                return self._remove_place(node)

            doomed: list[OrgNode] = []
            stack = [node]
            while stack:
                current = stack.pop()
                doomed.append(current)
                stack.extend(self._nodes[child_id] for child_id in current.child_ids)
            # Leaves first, so a subscriber never holds a child of a missing
            # parent.
            doomed.reverse()

            # Decide who hears about each removal while the doomed teams can
            # still be resolved, so a subscriber focused inside the removed
            # subtree is still matched.
            for gone in doomed:
                self._emit(NodeRemovedEvent(id=gone.id), gone.id, gone.parent_id)
            for gone in doomed:
                del self._nodes[gone.id]

            if node.parent_id is None:
                self._root_id = None
            else:
                self._unlink(self._nodes, node.parent_id, node_id)
                parent = self._nodes[node.parent_id]
                self._emit(NodeUpsertEvent(node=parent), parent.id, parent.parent_id)
            # What waits on the human names the teams on the way to it.
            self._announce_escalations()
            self._changed()
            return True

    def _remove_place(self, place: SessionTeam) -> bool:
        """Remove what a session has at a team and at the teams below it.
        Its Division Lead place, and the team it was opened at,
        go only with the session."""
        if place.parent_id is None or is_division_lead(
            self._places.get(place.parent_id)
        ):
            raise OrgStoreError(f"{place.id} is removed by closing its session")
        doomed = [place]
        for current in doomed:
            doomed.extend(self._places[child_id] for child_id in current.child_ids)
        for gone in doomed:
            self._index_agents(gone, None)
            del self._places[gone.id]
        self._unlink(self._places, place.parent_id, place.id)
        self._announce_sessions()
        self._announce_escalations()
        self._changed()
        return True

    def replace_all(self, inputs: Iterable[OrgNodeInput]) -> None:
        """Replace the whole tree; subscribers are shown theirs again.

        Used when loading persisted state. The store is left
        untouched if the input is not one well-formed tree. The sessions
        that are open stay, but for one that has lost a team.

        Raises:
            OrgStoreError: The input is not one well-formed tree.
        """
        nodes: dict[str, OrgNode] = {}
        for node_input in inputs:
            if node_input.id in nodes:
                raise OrgStoreError(f"Duplicate team id {node_input.id}")
            nodes[node_input.id] = OrgNode(**node_input.model_dump())

        root_id: str | None = None
        for node in nodes.values():
            if node.parent_id is None:
                if root_id is not None:
                    raise OrgStoreError(
                        f"More than one root team ({root_id}, {node.id})"
                    )
                root_id = node.id
                continue
            parent = nodes.get(node.parent_id)
            if parent is None:
                raise OrgStoreError(
                    f"Parent {node.parent_id} of team {node.id} does not exist"
                )
            parent.child_ids.append(node.id)
        if nodes and root_id is None:
            raise OrgStoreError("The tree has no root team")

        ordered = self._ordered(nodes, root_id)
        if len(ordered) != len(nodes):
            raise OrgStoreError("The tree contains a cycle")

        with self._lock:
            kept = [
                (session, list[SessionTeamInput](self.session(session.id)))
                for session in self._sessions.values()
            ]
            self._nodes = nodes
            self._root_id = root_id
            self._sessions = {}
            self._places = {}
            self._agents = {}
            self._hand_overs = {}
            self.restore_sessions(kept)
            self.show_again()

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
                    # Each place after the one above it, in whatever order
                    # they were kept.
                    waiting = list(teams)
                    while waiting:
                        ready = [
                            team
                            for team in waiting
                            if team.division_lead or team.parent_id in self._places
                        ]
                        if not ready:
                            raise OrgStoreError(f"Nothing is above {waiting[0].id}")
                        for team in ready:
                            self._restore_place(team)
                            waiting.remove(team)
                    _ = self._places[session_team_id(session.team_id, session.id)]
                except (OrgStoreError, KeyError):
                    for place in self.session(session.id):
                        self._index_agents(place, None)
                        del self._places[place.id]
                    del self._sessions[session.id]

    def _restore_place(self, team: SessionTeamInput) -> None:
        """Put back what a session has at one team, without telling anyone."""
        above = self._places.get(team.parent_id or "")
        if above is None and not team.division_lead:
            raise OrgStoreError(f"Nothing is above {team.id}")
        self._check_escalations(team, self._chain(team.parent_id))
        data = team.model_dump(exclude=_DERIVED_FIELDS)
        place = SessionTeam(**data)
        self._places[place.id] = place
        self._index_agents(None, place)
        if isinstance(above, SessionTeam):
            above.child_ids.append(place.id)

    def raise_escalation(
        self,
        place_id: str,
        escalation_id: str,
        *,
        kind: EscalationKind,
        title: str,
        detail: str = "",
        at: str | None = None,
        raised_by: AgentRole = "ic",
    ) -> Escalation:
        """Record something an agent of ``place_id`` cannot settle itself.

        It goes to the place above, never straight to the human, unless
        nobody works above ``place_id``: it is the place a session was opened
        at. One an IC raised
        blocks its team until it is settled.

        Raises:
            NotFound: The team does not exist.
            OrgStoreError: The team already has an escalation with this id.
        """
        with self._lock:
            node = self._require_place(place_id)
            if any(e.id == escalation_id for e in node.escalations):
                raise OrgStoreError(
                    f"Team {place_id} already has an escalation {escalation_id}"
                )
            escalation = Escalation(
                id=escalation_id,
                kind=kind,
                raised_by=raised_by,
                title=title,
                detail=detail,
                raised_at=at or utc_now(),
                held_by=next(iter(self._chain(node.parent_id)), None),
            )
            self._set_escalations(node, [*node.escalations, escalation])
            return escalation

    def pass_up_escalation(
        self, place_id: str, escalation_id: str, *, reason: str, at: str | None = None
    ) -> Escalation:
        """The team holding an escalation could not settle it: send it one up.

        This is the only way an escalation moves, so it reaches the human
        only after every team above the one that raised it has said why it
        could not settle it. Design decisions in particular should almost
        always be settled on the way.

        Args:
            node_id: The team that raised the escalation, not the one holding it.
            reason: Why the team holding it could not settle it.

        Raises:
            NotFound: The team or the escalation does not exist.
            OrgStoreError: The escalation already waits on the human, or the
                reason is blank.
        """
        with self._lock:
            node = self._require_place(place_id)
            escalation = self._require_escalation(node, escalation_id)
            if escalation.held_by is None:
                raise OrgStoreError(
                    f"Escalation {escalation_id} of team {place_id} already waits "
                    "on the human"
                )
            if not reason.strip():
                raise OrgStoreError("Passing an escalation up needs a reason")
            holder = self._places[escalation.held_by]
            passed = escalation.model_copy(
                update={
                    "held_by": next(iter(self._chain(holder.parent_id)), None),
                    "handoffs": [
                        *escalation.handoffs,
                        EscalationHandoff(
                            node_id=holder.id, reason=reason.strip(), at=at or utc_now()
                        ),
                    ],
                }
            )
            self._set_escalations(
                node,
                [passed if e.id == escalation_id else e for e in node.escalations],
            )
            return passed

    def settle_escalation(self, place_id: str, escalation_id: str) -> None:
        """Close an escalation, whoever settled it.

        Raises:
            NotFound: The team or the escalation does not exist.
        """
        with self._lock:
            node = self._require_place(place_id)
            self._require_escalation(node, escalation_id)
            self._set_escalations(
                node, [e for e in node.escalations if e.id != escalation_id]
            )

    def escalations(self) -> list[OpenEscalation]:
        """Every escalation that waits on the human.

        Critical issues come first, then the longest waiting.
        """
        with self._lock:
            found: list[OpenEscalation] = []
            for place in self._places.values():
                # A team a session set up may not be the org's yet: the way
                # to it is the one inside the session.
                path = [
                    EscalationPathStep(id=step.team_id, title=step.title)
                    for step in (*self.above(place.id), place)
                ]
                found.extend(
                    OpenEscalation(
                        node_id=place.team_id,
                        session_id=place.session,
                        session=self._sessions[place.session].title,
                        escalation=e,
                        path=path,
                    )
                    for e in place.escalations
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
        """The focused team, its ancestors, and the direct reports of the
        focused team and of every team in ``expanded``.

        With ``full``, every team below the focused one instead. Unknown ids
        in ``expanded`` are ignored.
        """
        with self._lock:
            focus = self._nodes.get(focus_id)
            if focus is None:
                return None
            nodes = {
                node.id: node
                for node in (
                    *self.teams_above(focus_id),
                    focus,
                    *self.sub_teams(focus_id),
                )
            }
            for team_id in expanded:
                if team_id in self._nodes:
                    for child in self.sub_teams(team_id):
                        nodes[child.id] = child
            if full:
                for node in self._ordered(self._nodes, focus_id):
                    nodes[node.id] = node
            return SnapshotEvent(
                focus_id=focus_id,
                nodes=list(nodes.values()),
                escalations=self.escalations(),
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
                    frozenset(team.id for team in self.teams_above(focus_id)),
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
        for team_id in queue:
            node = nodes[team_id]
            ordered.append(node)
            queue.extend(node.child_ids)
        return ordered

    def show_again(self) -> None:
        """Send every subscriber its snapshot anew: teams changed place, and
        what each follows is no longer what it was told one team at a time."""
        for listener, subscription in list(self._listeners.items()):
            focus_id = subscription.focus_id
            snapshot = self.snapshot(
                focus_id, subscription.expanded, full=subscription.full
            )
            if snapshot is None:
                continue
            subscription.above = frozenset(
                team.id for team in self.teams_above(focus_id)
            )
            subscription.escalations = snapshot.escalations
            listener(snapshot)

    def _emit(self, event: OrgChange, team_id: str, parent_id: str | None) -> None:
        if not self._listeners:
            return
        above = {team.id for team in self.teams_above(team_id)}
        for listener, subscription in list(self._listeners.items()):
            focus_id = subscription.focus_id
            if (
                team_id == focus_id
                or parent_id == focus_id
                or parent_id in subscription.expanded
                or (team_id in subscription.above and focus_id in self._nodes)
                or (subscription.full and focus_id in above)
            ):
                listener(event)

    def _session_view(self, session: SessionInput) -> Session:
        teams = self.session(session.id)
        top = self._places[session_team_id(session.team_id, session.id)]
        return Session(
            **session.model_dump(),
            team=top.title,
            status=_of_all(top.status, [team.status for team in teams]),
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

    def _chain(self, parent_id: str | None) -> list[str]:
        """Ids of the places above a place, nearest first.

        Stops at a missing parent or a cycle; the caller reports those.
        """
        chain: list[str] = []
        current = parent_id
        while current is not None and current in self._places and current not in chain:
            # Nobody stands at a session's Division Lead place:
            # what goes up from the session goes to the human.
            if self._places[current].division_lead:
                break
            chain.append(current)
            current = self._places[current].parent_id
        return chain

    @staticmethod
    def _check_escalations(node: SessionTeamInput, chain: list[str]) -> None:
        """Refuse an escalation that skipped a team on its way up.

        Raises:
            OrgStoreError: An escalation's handoffs are not the unbroken chain
                from the team's parent, or it is held by the wrong team.
        """
        for escalation in node.escalations:
            passed = [handoff.node_id for handoff in escalation.handoffs]
            holder = chain[len(passed)] if len(passed) < len(chain) else None
            if passed != chain[: len(passed)] or escalation.held_by != holder:
                raise OrgStoreError(
                    f"Escalation {escalation.id} of team {node.id} must be passed "
                    f"up through {chain or 'nobody'} in order; it was passed up "
                    f"by {passed} and is held by {escalation.held_by}"
                )

    def _require_place(self, place_id: str) -> SessionTeam:
        place = self._places.get(place_id)
        if place is None:
            raise NotFound(f"Unknown place: {place_id}")
        return place

    @staticmethod
    def _require_escalation(place: SessionTeam, escalation_id: str) -> Escalation:
        for escalation in place.escalations:
            if escalation.id == escalation_id:
                return escalation
        raise NotFound(f"Team {place.id} has no escalation {escalation_id}")

    def _set_escalations(
        self, place: SessionTeam, escalations: list[Escalation]
    ) -> None:
        status = status_of(place.model_copy(update={"escalations": escalations}))
        self.update_place(place.id, escalations=escalations, status=status)

    def _index_agents(self, old: SessionTeam | None, new: SessionTeam | None) -> None:
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
        current = self.escalations()
        for listener, subscription in list(self._listeners.items()):
            if current != subscription.escalations:
                subscription.escalations = current
                listener(EscalationsEvent(escalations=current))
