"""Exchanging an org's two stores between repositories kept in memory."""

from pathlib import Path

from openhands.agent_server.director.environment.memory.memory_store import (
    ORG_TREE,
    TEAM_MEMORY,
    MemoryRecords,
)
from openhands.agent_server.director.environment.remote import Remote
from openhands.sdk.errors import NotFound, UpstreamFailed


def _fetched(remote: str, store: str) -> str:
    """What a store fetched from a remote is called here."""
    return f"{remote}/{store}"


class MemoryRemote(Remote):
    """A remote of a repository is another repository of the same records."""

    def __init__(self, records: MemoryRecords) -> None:
        self.records = records

    def _other(self, repository: Path, remote: str) -> Path:
        other = self.records.remotes.get((repository, remote))
        if other is None:
            raise NotFound(f"{repository} has no remote {remote!r}")
        return other

    def fetch(self, repository: Path, remote: str) -> None:
        here = self.records.of(repository)
        there = self.records.of(self._other(repository, remote))
        for store in (ORG_TREE, TEAM_MEMORY):
            version = there.store(store).head()
            with here.lock:
                here.take(there, version)
                if version is None:
                    here.heads.pop(_fetched(remote, store), None)
                else:
                    here.heads[_fetched(remote, store)] = version

    def behind(self, repository: Path, remote: str) -> bool:
        here = self.records.of(repository)
        return any(
            (version := here.store(_fetched(remote, store)).head()) is not None
            and not here.store(store).contains(version)
            for store in (ORG_TREE, TEAM_MEMORY)
        )

    def push(self, repository: Path, remote: str) -> None:
        here = self.records.of(repository)
        there = self.records.of(self._other(repository, remote))
        with here.lock, there.lock:
            moves = []
            for store in (ORG_TREE, TEAM_MEMORY):
                ours, theirs = here.store(store).head(), there.store(store).head()
                if ours is None:
                    continue
                if theirs is not None and theirs not in here.history(ours):
                    raise UpstreamFailed(
                        f"{remote} has changes to {store} that are not here; "
                        "fetch and merge them first"
                    )
                moves.append((store, ours, theirs))
            for store, ours, theirs in moves:
                there.take(here, ours)
                there.swap(store, ours, theirs)
                here.heads[_fetched(remote, store)] = ours

    def org_tree_of(self, repository: Path, remote: str) -> str | None:
        return self.records.of(repository).store(_fetched(remote, ORG_TREE)).head()

    def team_memory_of(self, repository: Path, remote: str) -> str | None:
        return self.records.of(repository).store(_fetched(remote, TEAM_MEMORY)).head()

    def team_memory_name(self, remote: str) -> str:
        return _fetched(remote, TEAM_MEMORY)
