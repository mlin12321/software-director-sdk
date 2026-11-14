"""Persists the org tree as one JSON file under its own hidden ref.

The server therefore needs no database for the tree and rebuilds it from git
on start.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from openhands.agent_server.director.git_store import (
    GitStore,
    StoreChange,
    StoreReader,
)
from openhands.agent_server.director.models import OrgNodeInput
from openhands.agent_server.director.org_store import OrgStore


ORG_REF = "refs/agents/org"
_ORG_FILE = "org.json"
_VERSION = 1


class _PersistedOrg(BaseModel):
    version: int
    nodes: list[OrgNodeInput]


def create_org_board(repo_dir: Path) -> GitStore:
    """The store that holds the org tree of ``repo_dir``."""
    return GitStore(repo_dir, ref=ORG_REF)


def _persisted(node: OrgNodeInput) -> dict[str, Any]:
    """A node without what its agents' conversations determine.

    Whether an agent is running changes many times a minute and is read back
    from the conversations on load, so it is not worth a commit each time.
    """
    data = node.model_dump(
        exclude={"child_ids", "descendant_counts", "escalation_count"}
    )
    for agent in data["agents"]:
        agent["status"] = "idle"
    if data["status"] == "working":
        data["status"] = "idle"
    data["waiting_to_start"] = False
    return data


def org_file(nodes: list[dict[str, Any]]) -> StoreChange:
    """The file that holds a tree of saved nodes."""
    content = json.dumps({"version": _VERSION, "nodes": nodes}, indent=2) + "\n"
    return StoreChange(_ORG_FILE, content)


def saved_nodes(board: GitStore, commit: str | None) -> list[dict[str, Any]]:
    """The nodes of the tree saved at ``commit``, as they are stored.

    Raises:
        ValueError: The saved tree has a format this version does not read.
    """
    raw = board.read_at(commit, _ORG_FILE)
    if raw is None:
        return []
    persisted = _PersistedOrg.model_validate_json(raw)
    if persisted.version != _VERSION:
        raise ValueError(
            f"Unrecognized org tree version {persisted.version} in {board.ref}"
        )
    return [_persisted(node) for node in persisted.nodes]


def stored_nodes(store: OrgStore) -> list[dict[str, Any]]:
    """The nodes of ``store`` as they are saved."""
    return [_persisted(node) for node in store.all()]


def save_org(board: GitStore, store: OrgStore) -> None:
    """Write the tree to ``board`` unless it is already stored unchanged."""
    content = org_file(stored_nodes(store)).content

    def mutate(read: StoreReader) -> list[StoreChange]:
        return [] if read(_ORG_FILE) == content else [StoreChange(_ORG_FILE, content)]

    board.commit(mutate, "org: save tree")


def load_org(board: GitStore, store: OrgStore) -> bool:
    """Load the saved tree into ``store``.

    Returns:
        Whether a saved tree was found.

    Raises:
        ValueError: The saved tree has a format this version does not read.
    """
    raw = board.read(_ORG_FILE)
    if raw is None:
        return False
    persisted = _PersistedOrg.model_validate_json(raw)
    if persisted.version != _VERSION:
        raise ValueError(
            f"Unrecognized org tree version {persisted.version} in {board.ref}"
        )
    store.replace_all(persisted.nodes)
    return True
