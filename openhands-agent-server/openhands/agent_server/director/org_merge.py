"""Merges two versions of an org tree that come from a common one.

A team changed on one side only is taken from that side. A team changed on
both is merged field by field, and a field changed on one side only is taken
from that side. What is left for the human is a conflict: the same field
changed on both sides, or a team removed on one side and changed, or given
sub-teams, on the other.

Where a team's agents run and where its checkout is belong to the machine
that runs them, so those fields are never taken from the other side for a
team that is here.
"""

import json
from collections.abc import Mapping
from typing import Any, NamedTuple

from openhands.agent_server.director.models import OrgConflict, Side


type Nodes = dict[str, dict[str, Any]]

# What only the machine that runs a team knows.
LOCAL_FIELDS = ("worktree_path", "agents")
_REMOVED = "The team is removed."
_KEPT = "The team is kept, with what was changed in it or below it."


class MergedOrg(NamedTuple):
    """The outcome of merging two trees."""

    nodes: list[dict[str, Any]]
    """The merged tree, parents before their sub-teams. Where a conflict is
    open it has this side's version."""
    conflicts: list[OrgConflict]
    """Every conflict, with the choice made for it, if any."""


def by_id(nodes: list[dict[str, Any]]) -> Nodes:
    return {node["id"]: node for node in nodes}


def merge_orgs(
    base: Nodes, mine: Nodes, theirs: Nodes, choices: Mapping[str, Side]
) -> MergedOrg:
    """Merge ``theirs`` into ``mine``, both descended from ``base``.

    Args:
        choices: The side chosen for a conflict, by the conflict's id.
    """
    conflicts: list[OrgConflict] = []
    merged: Nodes = {}
    settled: set[str] = set()

    for gone_from, kept_in, remover in (
        (mine, theirs, "mine"),
        (theirs, mine, "theirs"),
    ):
        for node_id in base:
            if node_id in gone_from or node_id in settled or node_id not in kept_in:
                continue
            parent = base[node_id]["parent_id"]
            if parent is not None and parent not in gone_from:
                continue  # settled with the removed team above it
            below = _subtree(kept_in, node_id)
            settled.update(below)
            if all(base.get(each) == kept_in[each] for each in below):
                continue  # removed on one side, untouched on the other
            conflict = OrgConflict(
                id=f"team:{node_id}",
                kind="team",
                node_id=node_id,
                title=kept_in[node_id]["title"],
                field="",
                mine=None if remover == "mine" else _KEPT,
                theirs=_KEPT if remover == "mine" else None,
                choice=choices.get(f"team:{node_id}"),
            )
            conflicts.append(conflict)
            keeper: Side = "theirs" if remover == "mine" else "mine"
            if (conflict.choice or "mine") == keeper:
                merged.update({each: kept_in[each] for each in below})

    for node_id in [*mine, *(each for each in theirs if each not in mine)]:
        if node_id in settled:
            continue
        ours, other = mine.get(node_id), theirs.get(node_id)
        if ours is None or other is None:
            present = ours if ours is not None else other
            if present is not None and node_id not in base:
                merged[node_id] = present
            continue
        merged[node_id] = _merge_node(
            base.get(node_id, {}), ours, other, choices, conflicts
        )

    return MergedOrg(_parents_first(merged), conflicts)


def _merge_node(
    base: dict[str, Any],
    mine: dict[str, Any],
    theirs: dict[str, Any],
    choices: Mapping[str, Side],
    conflicts: list[OrgConflict],
) -> dict[str, Any]:
    node = dict(mine)
    for field in [*mine, *(each for each in theirs if each not in mine)]:
        ours, other, before = mine.get(field), theirs.get(field), base.get(field)
        if field in LOCAL_FIELDS or ours == other or other == before:
            continue
        if ours == before:
            node[field] = other
            continue
        conflict = OrgConflict(
            id=f"field:{mine['id']}:{field}",
            kind="field",
            node_id=mine["id"],
            title=mine["title"],
            field=field,
            mine=_shown(ours),
            theirs=_shown(other),
            choice=choices.get(f"field:{mine['id']}:{field}"),
        )
        conflicts.append(conflict)
        if conflict.choice == "theirs":
            node[field] = other
    return node


def _shown(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2)


def _subtree(nodes: Nodes, top: str) -> list[str]:
    """``top`` and every node below it, parents first."""
    found = [top]
    for node_id in found:
        found.extend(
            each for each, node in nodes.items() if node["parent_id"] == node_id
        )
    return found


def _parents_first(nodes: Nodes) -> list[dict[str, Any]]:
    """The nodes in an order that names no parent after its sub-team. A node
    whose parent is not among them is left out, with everything below it."""
    ordered: list[dict[str, Any]] = []
    placed: set[str | None] = {None}
    waiting = list(nodes.values())
    while waiting:
        ready = [node for node in waiting if node["parent_id"] in placed]
        if not ready:
            break
        ordered.extend(ready)
        placed.update(node["id"] for node in ready)
        waiting = [node for node in waiting if node["id"] not in placed]
    return ordered


def rehomed(
    nodes: list[dict[str, Any]], here: Nodes, there_root: str | None, here_root: str
) -> list[dict[str, Any]]:
    """Point the teams that came from the other side at this machine.

    A team that works in the repository's own checkout is at another path
    here. A team with a checkout of its own keeps its path and gets the
    checkout made from its branch.

    Args:
        here: The teams this machine had before the merge.
        there_root: Where the repository is on the other side.
        here_root: Where it is here.
    """
    if not there_root or there_root == here_root:
        return nodes
    moved: list[dict[str, Any]] = []
    for node in nodes:
        path = node["worktree_path"]
        inside = path == there_root or path.startswith(f"{there_root}/")
        if node["id"] not in here and inside:
            node = {**node, "worktree_path": here_root + path[len(there_root) :]}
        moved.append(node)
    return moved
