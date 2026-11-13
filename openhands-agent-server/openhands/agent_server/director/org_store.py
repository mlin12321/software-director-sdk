"""In-memory org tree.

The store owns the derived fields on every node (``child_ids``,
``descendant_counts`` and ``escalation_count``) and is the only thing that
emits org events, so every subscriber sees one consistent tree.
"""

import threading
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import get_args

from openhands.agent_server.director.models import (
    AgentRef,
    AgentRole,
    Escalation,
    EscalationHandoff,
    EscalationKind,
    EscalationPathStep,
    EscalationsEvent,
    NodeRemovedEvent,
    NodeStatus,
    NodeUpsertEvent,
    OpenEscalation,
    OrgNode,
    OrgNodeInput,
    SnapshotEvent,
    StatusCounts,
)
from openhands.sdk.errors import Invalid, NotFound


NODE_STATUSES: tuple[NodeStatus, ...] = get_args(NodeStatus)

type OrgChange = NodeUpsertEvent | NodeRemovedEvent | EscalationsEvent
type OrgListener = Callable[[OrgChange], None]

_DERIVED_FIELDS = {"child_ids", "descendant_counts", "escalation_count"}


@dataclass
class _Subscription:
    focus_id: str
    # Nodes whose direct reports the subscriber also follows.
    expanded: frozenset[str]
    # Whether the subscriber follows every node below the focused one.
    full: bool
    # What the subscriber was last told, to announce only real changes.
    escalations: list[OpenEscalation] = field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class OrgStoreError(Invalid):
    """A change would leave the org tree malformed."""


class OrgStore:
    """A tree of :class:`OrgNode`, safe to use from several threads.

    Listeners are called while the store is locked, in the order changes
    happened. They must return quickly and must not call back into the store.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: dict[str, OrgNode] = {}
        self._root_id: str | None = None
        self._listeners: dict[OrgListener, _Subscription] = {}
        self._change_listeners: list[Callable[[], None]] = []
        self._agents: dict[str, tuple[str, AgentRole]] = {}

    @property
    def root_id(self) -> str | None:
        return self._root_id

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> OrgNode | None:
        return self._nodes.get(node_id)

    def children(self, node_id: str) -> list[OrgNode]:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return []
            return [self._nodes[child_id] for child_id in node.child_ids]

    def ancestors(self, node_id: str) -> list[OrgNode]:
        """The nodes above ``node_id``, root first, excluding the node itself."""
        with self._lock:
            chain: list[OrgNode] = []
            node = self._nodes.get(node_id)
            while node is not None and node.parent_id is not None:
                node = self._nodes.get(node.parent_id)
                if node is not None:
                    chain.append(node)
            chain.reverse()
            return chain

    def all(self) -> list[OrgNode]:
        """Every node, parents before children."""
        with self._lock:
            return self._ordered(self._nodes, self._root_id)

    def find_by_conversation(
        self, conversation_id: str
    ) -> tuple[OrgNode, AgentRole] | None:
        """The node an agent's conversation belongs to, and the agent's role."""
        with self._lock:
            found = self._agents.get(conversation_id)
            if found is None:
                return None
            node_id, role = found
            return self._nodes[node_id], role

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
            return self.upsert(OrgNodeInput.model_validate({**current, **changes}))

    def set_agent(
        self,
        node_id: str,
        role: AgentRole,
        *,
        conversation_id: str | None = None,
        status: str | None = None,
    ) -> OrgNode:
        """Change the conversation or the status of one agent of a node.

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
                else AgentRef(
                    role=role,
                    conversation_id=conversation_id or agent.conversation_id,
                    status=status or agent.status,
                )
                for agent in node.agents
            ]
            if agents == node.agents:
                return node
            return self.update_node(node_id, agents=agents)

    def upsert(self, node_input: OrgNodeInput) -> OrgNode:
        """Insert a node or update an existing one.

        A node cannot move to a different parent; remove it and insert it
        again instead.

        Raises:
            OrgStoreError: The node would not fit into the tree.
        """
        with self._lock:
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
            self._announce_escalations()
            self._changed()
            return node

    def remove(self, node_id: str) -> bool:
        """Remove a node and everything below it.

        Returns:
            Whether the node existed.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return False

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

    def replace_all(self, inputs: Iterable[OrgNodeInput]) -> None:
        """Replace the whole tree without emitting events.

        Used when loading persisted state or a fixture. The store is left
        untouched if the input is not one well-formed tree.

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
            self._nodes = nodes
            self._root_id = root_id
            self._agents = {}
            for node in nodes.values():
                self._index_agents(None, node)

    def raise_escalation(
        self,
        node_id: str,
        escalation_id: str,
        *,
        kind: EscalationKind,
        title: str,
        detail: str = "",
        at: str | None = None,
    ) -> Escalation:
        """Record something ``node_id`` cannot settle itself.

        It goes to the node above, never straight to the human, unless
        ``node_id`` is the root and has nobody above it.

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
                title=title,
                detail=detail,
                raised_at=at or _now(),
                held_by=node.parent_id,
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
            holder = self._nodes[escalation.held_by]
            passed = escalation.model_copy(
                update={
                    "held_by": holder.parent_id,
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
                for child in self.children(node_id):
                    nodes[child.id] = child
            if full:
                for node in self._ordered(self._nodes, focus_id):
                    nodes[node.id] = node
            return SnapshotEvent(
                focus_id=focus_id,
                nodes=list(nodes.values()),
                escalations=self._waiting_anywhere(),
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
                    focus_id, frozenset(expanded), full, snapshot.escalations
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
        for listener, subscription in list(self._listeners.items()):
            if self._visible_to(subscription, node_id, parent_id):
                listener(event)

    def _visible_to(
        self, subscription: _Subscription, node_id: str, parent_id: str | None
    ) -> bool:
        focus_id = subscription.focus_id
        if node_id == focus_id or parent_id == focus_id:
            return True
        if parent_id in subscription.expanded:
            return True
        if subscription.full and any(
            ancestor.id == focus_id for ancestor in self.ancestors(node_id)
        ):
            return True
        return any(ancestor.id == node_id for ancestor in self.ancestors(focus_id))

    def _waiting_anywhere(self) -> list[OpenEscalation]:
        """What waits on the human in the whole tree.

        Subscribers get this whatever node they are focused on: it is the
        one thing the human must not miss by looking somewhere else.
        """
        return [] if self._root_id is None else self.escalations(self._root_id)

    @staticmethod
    def _waiting_on_human(node: OrgNodeInput) -> int:
        return sum(1 for e in node.escalations if e.held_by is None)

    @staticmethod
    def _chain(nodes: dict[str, OrgNode], parent_id: str | None) -> list[str]:
        """Ids of the nodes above a node, nearest first.

        Stops at a missing parent or a cycle; the caller reports those.
        """
        chain: list[str] = []
        current = parent_id
        while current is not None and current in nodes and current not in chain:
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
        node = self._nodes.get(node_id)
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
        self.update_node(node.id, escalations=escalations)

    def _index_agents(self, old: OrgNode | None, new: OrgNode | None) -> None:
        if old is not None:
            for agent in old.agents:
                if agent.conversation_id is not None:
                    self._agents.pop(agent.conversation_id, None)
        if new is not None:
            for agent in new.agents:
                if agent.conversation_id is not None:
                    self._agents[agent.conversation_id] = (new.id, agent.role)

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
