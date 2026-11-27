"""Stores of records kept in memory.

A version is every file of a store as it was, with the versions it was
made from. A repository has several stores (the org tree, the team
memory, and those fetched from a remote), each a name that points at a
version.
"""

import itertools
import threading
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import NamedTuple

from openhands.agent_server.director.environment.records import (
    Clash,
    Merged,
    RecordStore,
    Staged,
    StoreChange,
    StoreReader,
    merge_files,
)
from openhands.sdk.errors import NotFound


TEAM_MEMORY = "team-memory"
ORG_TREE = "org"

# Versions are named apart across repositories, so that one can be given
# to another as it is.
_numbers = itertools.count(1)


class Version(NamedTuple):
    """Every file of a store as it was at one time."""

    files: dict[str, str]
    parents: tuple[str, ...]
    message: str


class MemoryRepository:
    """The stores of one repository, and every version any of them had."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.versions: dict[str, Version] = {}
        self.heads: dict[str, str] = {}

    def store(self, name: str) -> "MemoryRecordStore":
        """The store called ``name``; empty until it is first written."""
        return MemoryRecordStore(self, name)

    def add(self, files: dict[str, str], parents: Sequence[str], message: str) -> str:
        """Keep a version that no store points at yet."""
        with self.lock:
            version = f"v{next(_numbers)}"
            self.versions[version] = Version(files, tuple(parents), message)
            return version

    def history(self, version: str | None) -> set[str]:
        """A version and every version it was made from."""
        seen: set[str] = set()
        todo = [version] if version else []
        with self.lock:
            while todo:
                at = todo.pop()
                if at not in seen:
                    seen.add(at)
                    todo.extend(self.versions[at].parents)
        return seen

    def resolve(self, name_or_version: str) -> str:
        """The version a store of this name points at, or the version
        itself.

        Raises:
            NotFound: It names neither.
        """
        with self.lock:
            if name_or_version in self.heads:
                return self.heads[name_or_version]
            if name_or_version in self.versions:
                return name_or_version
        raise NotFound(f"There is no store or version {name_or_version!r}")

    def take(self, other: "MemoryRepository", version: str | None) -> None:
        """Keep a version of another repository here, with its history."""
        with self.lock, other.lock:
            for kept in other.history(version):
                self.versions.setdefault(kept, other.versions[kept])

    def swap(self, name: str, version: str, expected: str | None) -> bool:
        """Point a store at ``version`` if it still points at ``expected``."""
        with self.lock:
            if self.heads.get(name) != expected:
                return False
            self.heads[name] = version
            return True

    def land_together(self, staged: Sequence[Staged]) -> bool:
        """Point every store at the version staged for it, all or none.

        Returns:
            False when one of them was changed by someone else meanwhile.
        """
        with self.lock:
            if any(self.heads.get(move.ref) != move.parent for move in staged):
                return False
            for move in staged:
                self.heads[move.ref] = move.commit
            return True


class MemoryRecords:
    """The records of every repository, by where the repository is, and
    which repository each of its remotes is."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._repositories: dict[Path, MemoryRepository] = {}
        self.remotes: dict[tuple[Path, str], Path] = {}

    def of(self, repository: Path) -> MemoryRepository:
        """The records of the repository at a path, empty at first."""
        with self._lock:
            return self._repositories.setdefault(repository, MemoryRepository())

    def add_remote(self, repository: Path, remote: str, other: Path) -> None:
        """Make another repository a remote of one, under a name."""
        self.remotes[(repository, remote)] = other


def _applied(files: dict[str, str], changes: Sequence[StoreChange]) -> dict[str, str]:
    made = dict(files)
    for path, content in changes:
        if content is None:
            made.pop(path, None)
        else:
            made[path] = content
    return made


class MemoryRecordStore(RecordStore):
    """One store of a repository's records, in memory.

    Unlike git, it merges files whole: a file both sides changed is
    settled as one (`records.merge_files`), where git first merges the
    lines only one side changed.
    """

    def __init__(self, repository: MemoryRepository, name: str) -> None:
        self.repository = repository
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def head(self) -> str | None:
        with self.repository.lock:
            return self.repository.heads.get(self._name)

    def _files(self, version: str | None) -> dict[str, str]:
        with self.repository.lock:
            return self.repository.versions[version].files if version else {}

    def read_at(self, commit: str | None, path: str) -> str | None:
        return self._files(commit).get(path)

    def files(self, directory: str) -> list[str]:
        inside = f"{directory.strip('/')}/"
        return sorted(
            path for path in self._files(self.head()) if path.startswith(inside)
        )

    def stage(
        self, mutate: Callable[[StoreReader], list[StoreChange]], message: str
    ) -> Staged | None:
        parent = self.head()
        changes = mutate(lambda path: self.read_at(parent, path))
        if not changes:
            return None
        version = self.repository.add(
            _applied(self._files(parent), changes), [parent] if parent else [], message
        )
        return Staged(self._name, version, parent)

    def commit(
        self, mutate: Callable[[StoreReader], list[StoreChange]], message: str
    ) -> str | None:
        while True:
            staged = self.stage(mutate, message)
            if staged is None:
                return self.head()
            if self.repository.swap(self._name, staged.commit, staged.parent):
                return staged.commit

    def land(
        self, changes: list[StoreChange], message: str, *, merging: str | None = None
    ) -> bool:
        parent = self.head()
        parents = [version for version in (parent, merging) if version]
        version = self.repository.add(
            _applied(self._files(parent), changes), parents, message
        )
        return self.repository.swap(self._name, version, parent)

    def _merged(
        self, ours: str, theirs: str, take_theirs: Collection[str]
    ) -> tuple[dict[str, str], list[str]]:
        changes, kept = merge_files(
            self._files(self.base_with(theirs)),
            self._files(ours),
            self._files(theirs),
            take_theirs,
        )
        return _applied(self._files(ours), changes), kept

    def clashes(self, theirs_ref: str) -> list[Clash]:
        theirs = self.repository.resolve(theirs_ref)
        ours = self.head()
        if (
            ours is None
            or theirs in self.repository.history(ours)
            or ours in self.repository.history(theirs)
        ):
            return []
        _, kept = self._merged(ours, theirs, ())
        return [
            Clash(path, self.read_at(ours, path), self.read_at(theirs, path))
            for path in kept
        ]

    def merge(self, theirs_ref: str, take_theirs: Collection[str] = ()) -> Merged:
        theirs = self.repository.resolve(theirs_ref)
        while True:
            ours = self.head()
            if ours is None:
                if self.repository.swap(self._name, theirs, None):
                    return Merged(theirs, [])
                continue
            if theirs in self.repository.history(ours):
                return Merged(ours, [])
            if ours in self.repository.history(theirs):
                if self.repository.swap(self._name, theirs, ours):
                    return Merged(theirs, [])
                continue
            files, kept = self._merged(ours, theirs, take_theirs)
            version = self.repository.add(
                files, [ours, theirs], f"memory: merge {theirs_ref}"
            )
            if self.repository.swap(self._name, version, ours):
                return Merged(version, kept)

    def move_to(self, commit: str) -> bool:
        parent = self.head()
        if parent is not None and parent not in self.repository.history(commit):
            return False
        return self.repository.swap(self._name, commit, parent)

    def contains(self, other: str) -> bool:
        return other in self.repository.history(self.head())

    def base_with(self, other: str) -> str | None:
        ours = self.repository.history(self.head())
        # The nearest version of the other's history that is in ours:
        # looked for level by level from the other.
        level = [self.repository.resolve(other)]
        seen: set[str] = set()
        while level:
            for version in level:
                if version in ours:
                    return version
            seen.update(level)
            with self.repository.lock:
                level = [
                    parent
                    for version in level
                    for parent in self.repository.versions[version].parents
                    if parent not in seen
                ]
        return None
