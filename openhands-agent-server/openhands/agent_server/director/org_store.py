"""In-memory org tree.

The store owns the two derived fields on every node (``child_ids`` and
``descendant_counts``) and is the only thing that emits org events, so every
subscriber sees one consistent tree.
"""

import threading
from collections.abc import Callable, Iterable
from typing import get_args

from openhands.agent_server.director.models import (
    HealthCounts,
    NodeHealth,
    NodeRemovedEvent,
    NodeUpsertEvent,
    OrgNode,
    OrgNodeInput,
    SnapshotEvent,
)
from openhands.sdk.errors import Invalid


NODE_HEALTHS: tuple[NodeHealth, ...] = get_args(NodeHealth)

type OrgChange = NodeUpsertEvent | NodeRemovedEvent
type OrgListener = Callable[[OrgChange], None]


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
        # Listener -> the node its subscription is focused on.
        self._listeners: dict[OrgListener, str] = {}

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

            node = OrgNode(
                **node_input.model_dump(),
                child_ids=existing.child_ids if existing else [],
                descendant_counts=(
                    existing.descendant_counts if existing else HealthCounts()
                ),
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
            # A new child changes its parent's ``child_ids``, so the parent is
            # always re-announced on insert; above that, only changed rollups.
            self._refresh_ancestors(parent_id, always_announce_first=existing is None)
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
                del self._nodes[gone.id]

            if node.parent_id is None:
                self._root_id = None
            else:
                parent = self._nodes[node.parent_id]
                self._nodes[parent.id] = parent.model_copy(
                    update={"child_ids": [c for c in parent.child_ids if c != node_id]}
                )
            self._refresh_ancestors(node.parent_id, always_announce_first=True)
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
        if nodes and root_id is None:
            raise OrgStoreError("The tree has no root node")

        ordered = self._ordered(nodes, root_id)
        if len(ordered) != len(nodes):
            raise OrgStoreError("The tree contains a cycle")
        for node in reversed(ordered):
            node.descendant_counts = self._rollup(nodes, node)

        with self._lock:
            self._nodes = nodes
            self._root_id = root_id

    def snapshot(self, focus_id: str) -> SnapshotEvent | None:
        """The focused node, its ancestors and its direct reports."""
        with self._lock:
            focus = self._nodes.get(focus_id)
            if focus is None:
                return None
            return SnapshotEvent(
                focus_id=focus_id,
                nodes=[*self.ancestors(focus_id), focus, *self.children(focus_id)],
            )

    def subscribe(
        self, focus_id: str, listener: OrgListener
    ) -> tuple[SnapshotEvent | None, Callable[[], None]]:
        """Receive changes to the nodes a snapshot of ``focus_id`` contains.

        The snapshot and the subscription are taken together, so no change
        falls between them.

        Returns:
            The current snapshot (None when ``focus_id`` is unknown, in which
            case nothing is subscribed) and a function that unsubscribes.
        """
        with self._lock:
            snapshot = self.snapshot(focus_id)
            if snapshot is not None:
                self._listeners[listener] = focus_id

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
    def _rollup(nodes: dict[str, OrgNode], node: OrgNode) -> HealthCounts:
        counts = dict.fromkeys(NODE_HEALTHS, 0)
        for child_id in node.child_ids:
            child = nodes[child_id]
            counts[child.health] += 1
            for health in NODE_HEALTHS:
                counts[health] += getattr(child.descendant_counts, health)
        return HealthCounts(**counts)

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
            counts = self._rollup(self._nodes, current)
            changed = counts != current.descendant_counts
            if changed:
                current = current.model_copy(update={"descendant_counts": counts})
                self._nodes[current_id] = current
            if changed or (first and always_announce_first):
                self._emit(NodeUpsertEvent(node=current), current.id, current.parent_id)
            first = False
            current_id = current.parent_id

    def _emit(self, event: OrgChange, node_id: str, parent_id: str | None) -> None:
        for listener, focus_id in list(self._listeners.items()):
            if self._visible_from(focus_id, node_id, parent_id):
                listener(event)

    def _visible_from(self, focus_id: str, node_id: str, parent_id: str | None) -> bool:
        if node_id == focus_id or parent_id == focus_id:
            return True
        return any(ancestor.id == node_id for ancestor in self.ancestors(focus_id))
