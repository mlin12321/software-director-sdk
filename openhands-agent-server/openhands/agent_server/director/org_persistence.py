"""Persists the org tree as one JSON file under its own hidden ref.

The server therefore needs no database for the tree and rebuilds it from git
on start.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from openhands.agent_server.director.git_blackboard import (
    BoardChange,
    BoardReader,
    GitBlackboard,
)
from openhands.agent_server.director.models import OrgNodeInput
from openhands.agent_server.director.org_store import OrgStore


ORG_REF = "refs/agents/org"
_ORG_FILE = "org.json"
_VERSION = 1


class _PersistedOrg(BaseModel):
    version: int
    nodes: list[OrgNodeInput]


def create_org_board(repo_dir: Path) -> GitBlackboard:
    """The blackboard that holds the org tree of ``repo_dir``."""
    return GitBlackboard(repo_dir, ref=ORG_REF)


def save_org(board: GitBlackboard, store: OrgStore) -> None:
    """Write the tree to ``board`` unless it is already stored unchanged."""
    nodes = [
        node.model_dump(exclude={"child_ids", "descendant_counts", "escalation_count"})
        for node in store.all()
    ]
    content = json.dumps({"version": _VERSION, "nodes": nodes}, indent=2) + "\n"

    def mutate(read: BoardReader) -> list[BoardChange]:
        return [] if read(_ORG_FILE) == content else [BoardChange(_ORG_FILE, content)]

    board.commit(mutate, "org: save tree")


def load_org(board: GitBlackboard, store: OrgStore) -> bool:
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
