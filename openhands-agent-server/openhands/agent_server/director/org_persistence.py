"""Persists the org tree as one JSON file under its own hidden ref.

The server therefore needs no database for the tree and rebuilds it from git
on start.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from openhands.agent_server.director.git_store import (
    GitStore,
    StoreChange,
    StoreReader,
)
from openhands.agent_server.director.models import OrgNodeInput, OrgSettings
from openhands.agent_server.director.org_store import OrgStore


ORG_REF = "refs/agents/org"
_ORG_FILE = "org.json"
# What the org sets for itself, beside its tree in the same ref.
_SETTINGS_FILE = "settings.json"
_VERSION = 1
_SETTINGS_VERSION = 1
_SETTINGS_VERSION_KEY = "schema_version"


class _PersistedOrg(BaseModel):
    version: int
    nodes: list[OrgNodeInput]


def create_org_board(repo_dir: Path) -> GitStore:
    """The store that holds the org tree of ``repo_dir``."""
    return GitStore(repo_dir, ref=ORG_REF)


def own_settings(board: GitStore) -> OrgSettings:
    """What the org of ``board`` sets for itself; nothing when it kept none
    or what it kept cannot be read.

    Raises:
        ValueError: The settings have a format this version does not read.
    """
    raw = board.read(_SETTINGS_FILE)
    if raw is None:
        return OrgSettings()
    try:
        kept = json.loads(raw)
        # A file kept before the version was written is of the first.
        version = kept.pop(_SETTINGS_VERSION_KEY, _SETTINGS_VERSION)
    except (ValueError, AttributeError):
        return OrgSettings()
    if version != _SETTINGS_VERSION:
        # Read as nothing, a newer director's settings would be saved over.
        raise ValueError(f"Unrecognized org settings version {version} in {board.ref}")
    try:
        return OrgSettings.model_validate(kept)
    except ValueError:
        return OrgSettings()


def save_own_settings(board: GitStore, settings: OrgSettings) -> None:
    """Keep what the org sets for itself beside its tree."""
    kept = {
        _SETTINGS_VERSION_KEY: _SETTINGS_VERSION,
        **settings.model_dump(mode="json", exclude_none=True),
    }
    content = json.dumps(kept, indent=2) + "\n"
    board.commit(
        lambda read: (
            []
            if read(_SETTINGS_FILE) == content
            else [StoreChange(_SETTINGS_FILE, content)]
        ),
        "org: save settings",
    )


def reference(node: OrgNodeInput) -> dict[str, Any]:
    """A team of the org as it is kept: what a new agent is briefed from.
    Nothing of the work, which is done in sessions and kept with them."""
    return node.model_dump(include=set(OrgNodeInput.model_fields))


def org_file(nodes: list[dict[str, Any]]) -> StoreChange:
    """The file that holds a tree of saved teams."""
    content = json.dumps({"version": _VERSION, "nodes": nodes}, indent=2) + "\n"
    return StoreChange(_ORG_FILE, content)


def saved_nodes(board: GitStore, commit: str | None) -> list[dict[str, Any]]:
    """The teams of the tree saved at ``commit``, as they are stored.

    Raises:
        ValueError: The saved tree has a format this version does not read.
    """
    raw = board.read_at(commit, _ORG_FILE)
    if raw is None:
        return []
    saved = _PersistedOrg.model_validate_json(raw)
    if saved.version != _VERSION:
        raise ValueError(
            f"Unrecognized org tree version {saved.version} in {board.ref}"
        )
    return [reference(node) for node in saved.nodes]


def stored_nodes(store: OrgStore) -> list[dict[str, Any]]:
    """The teams of ``store`` as they are saved."""
    return [reference(node) for node in store.teams()]


def saving(store: OrgStore) -> Callable[[StoreReader], list[StoreChange]]:
    """What saving the tree changes: nothing when it is stored unchanged."""
    content = org_file(stored_nodes(store)).content

    def mutate(read: StoreReader) -> list[StoreChange]:
        return [] if read(_ORG_FILE) == content else [StoreChange(_ORG_FILE, content)]

    return mutate


def save_org(board: GitStore, store: OrgStore) -> None:
    """Write the tree to ``board`` unless it is already stored unchanged."""
    board.commit(saving(store), "org: save tree")


def load_org(board: GitStore, store: OrgStore) -> bool:
    """Load the saved tree into ``store``.

    Returns:
        Whether a saved tree was found.

    Raises:
        ValueError: The saved tree or the org's settings have a format this
            version does not read.
    """
    raw = board.read(_ORG_FILE)
    if raw is None:
        return False
    saved = _PersistedOrg.model_validate_json(raw)
    if saved.version != _VERSION:
        raise ValueError(
            f"Unrecognized org tree version {saved.version} in {board.ref}"
        )
    # Refused with the tree, before anything of the org is used.
    own_settings(board)
    store.replace_all(saved.nodes)
    return True
