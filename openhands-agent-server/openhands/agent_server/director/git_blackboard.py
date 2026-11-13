"""Git-native blackboard: a virtual directory tree stored under a hidden ref.

Nothing here touches the working tree or the user's index. Every mutation
builds a tree in a throwaway index file and lands with a compare-and-swap
``git update-ref``, so concurrent writers never lose each other's changes.
"""

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from pydantic import TypeAdapter

from openhands.agent_server.director.git import GitError, git, git_version
from openhands.agent_server.director.models import (
    Autopsy,
    BoundaryMemory,
    InheritedSpec,
)
from openhands.sdk.errors import Conflict, Invalid, NotSupported, OperationFailed


BLACKBOARD_REF = "refs/agents/blackboard"
SPEC_FILE = "spec.md"
MEMORY_FILE = "memory.json"
AUTOPSIES_FILE = "autopsies.json"

_ZERO_OID = "0" * 40
_MAX_COMMIT_ATTEMPTS = 8
_MERGE_MIN_GIT = (2, 38)
_IDENTITY = {
    "GIT_AUTHOR_NAME": "Agency Director",
    "GIT_AUTHOR_EMAIL": "director@agency.invalid",
    "GIT_COMMITTER_NAME": "Agency Director",
    "GIT_COMMITTER_EMAIL": "director@agency.invalid",
}
_AUTOPSIES = TypeAdapter(list[Autopsy])


class BlackboardScopeError(Invalid):
    """A path escapes the blackboard or the scope it must stay inside."""


class BlackboardMergeConflictError(Conflict):
    """Two blackboards changed the same file in incompatible ways."""


class BoardChange(NamedTuple):
    """One file to write. ``content=None`` deletes the file."""

    path: str
    content: str | None


type BoardReader = Callable[[str], str | None]


def normalize_board_path(path: str) -> str:
    """Normalize a repo-relative path. ``""``, ``"."`` and ``"/"`` are the root.

    Raises:
        BlackboardScopeError: The path could escape the virtual tree.
    """
    segments = [
        segment
        for segment in path.replace("\\", "/").split("/")
        if segment not in ("", ".")
    ]
    if any(segment in ("..", ".git") for segment in segments):
        raise BlackboardScopeError(f"Path escapes the blackboard: {path}")
    return "/".join(segments)


def _join(directory: str, file: str) -> str:
    return f"{directory}/{file}" if directory else file


def _scoped(scope_path: str, file: str) -> str:
    relative = normalize_board_path(file)
    if not relative:
        raise BlackboardScopeError("A blackboard write needs a file name")
    return _join(normalize_board_path(scope_path), relative)


def _parse_memory(raw: str | None) -> BoundaryMemory:
    return BoundaryMemory.model_validate_json(raw) if raw else BoundaryMemory()


def _parse_autopsies(raw: str | None) -> list[Autopsy]:
    return _AUTOPSIES.validate_json(raw) if raw else []


def _serialize(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


class GitBlackboard:
    """Reads and writes one blackboard ref of one repository."""

    def __init__(self, repo_dir: Path, ref: str = BLACKBOARD_REF) -> None:
        self.repo_dir = repo_dir
        self.ref = ref

    def head(self) -> str | None:
        """The commit the ref points at, or None before the first write."""
        try:
            out = git(
                self.repo_dir,
                ["rev-parse", "--verify", "--quiet", f"{self.ref}^{{commit}}"],
            )
        except GitError:
            return None
        return out.strip()

    def read(self, path: str) -> str | None:
        """The content of ``path`` at the ref head, or None if it is absent."""
        return self._read_at(self.head(), path)

    def inherited(self, boundary_path: str) -> list[InheritedSpec]:
        """Specs from the repository root down to ``boundary_path``.

        Ancestor specs come back read-only; only the boundary's own spec is
        writable by its agents.
        """
        target = normalize_board_path(boundary_path)
        segments = target.split("/") if target else []
        directories = ["", *("/".join(segments[: i + 1]) for i in range(len(segments)))]
        commit = self.head()
        specs = []
        for directory in directories:
            spec = self._read_at(commit, _join(directory, SPEC_FILE))
            if spec is not None:
                specs.append(
                    InheritedSpec(
                        path=directory, spec=spec, readonly=directory != target
                    )
                )
        return specs

    def write(self, scope_path: str, file: str, content: str) -> str | None:
        """Write one file inside ``scope_path`` and return the new commit.

        A file that resolves outside the scope is rejected, which is what
        confines an agent to its own subtree.
        """
        path = _scoped(scope_path, file)
        return self.commit(
            lambda _read: [BoardChange(path, content)], f"blackboard: write {path}"
        )

    def memory(self, scope_path: str) -> BoundaryMemory:
        """The permanent memory of the boundary at ``scope_path``."""
        return _parse_memory(self.read(_scoped(scope_path, MEMORY_FILE)))

    def inherited_rules(self, boundary_path: str) -> list[str]:
        """The rules of every boundary from the repository root down to
        ``boundary_path``, outermost first, each once."""
        target = normalize_board_path(boundary_path)
        segments = target.split("/") if target else []
        commit = self.head()
        rules: list[str] = []
        for depth in range(len(segments) + 1):
            directory = "/".join(segments[:depth])
            memory = _parse_memory(self._read_at(commit, _join(directory, MEMORY_FILE)))
            rules.extend(rule for rule in memory.rules if rule not in rules)
        return rules

    def autopsies(self, scope_path: str) -> list[Autopsy]:
        """The autopsies recorded for the boundary at ``scope_path``."""
        return _parse_autopsies(self.read(_scoped(scope_path, AUTOPSIES_FILE)))

    def teach(self, scope_path: str, rule: str) -> str | None:
        """Append a rule to a boundary's permanent memory, once."""
        path = _scoped(scope_path, MEMORY_FILE)
        rule = rule.strip()
        if not rule:
            raise Invalid("A rule cannot be empty")

        def mutate(read: BoardReader) -> list[BoardChange]:
            memory = _parse_memory(read(path))
            if rule in memory.rules:
                return []
            memory.rules.append(rule)
            return [BoardChange(path, _serialize(memory.model_dump()))]

        return self.commit(mutate, f"blackboard: teach {path}")

    def append_autopsy(self, scope_path: str, autopsy: Autopsy) -> str | None:
        """Record why an attempt in a boundary was pruned."""
        path = _scoped(scope_path, AUTOPSIES_FILE)

        def mutate(read: BoardReader) -> list[BoardChange]:
            autopsies = [*_parse_autopsies(read(path)), autopsy]
            return [BoardChange(path, _serialize(_AUTOPSIES.dump_python(autopsies)))]

        return self.commit(mutate, f"blackboard: autopsy {path}")

    def commit(
        self, mutate: Callable[[BoardReader], list[BoardChange]], message: str
    ) -> str | None:
        """Read-modify-write against the ref.

        ``mutate`` reads the tree it is building on and is run again from
        scratch if another writer moved the ref first.

        Returns:
            The new commit, or the unchanged head when ``mutate`` produced no
            changes.
        """
        for _ in range(_MAX_COMMIT_ATTEMPTS):
            parent = self.head()
            changes = mutate(lambda path, parent=parent: self._read_at(parent, path))
            if not changes:
                return parent
            tree = self._build_tree(parent, changes)
            commit = self._commit_tree(tree, [parent] if parent else [], message)
            if self._swap(commit, parent):
                return commit
        raise OperationFailed(
            f"Gave up updating {self.ref} after {_MAX_COMMIT_ATTEMPTS} "
            "contended attempts"
        )

    def merge(self, theirs_ref: str) -> str:
        """Three-way merge another blackboard ref into this one.

        The merge happens entirely in the object store and needs git 2.38 or
        newer (``git merge-tree --write-tree``).

        Returns:
            The resulting head commit.

        Raises:
            NotSupported: The installed git is too old.
            BlackboardMergeConflictError: Both sides changed the same content.
        """
        version = git_version(self.repo_dir)
        if version < _MERGE_MIN_GIT:
            raise NotSupported(
                "Blackboard merge needs git 2.38 or newer "
                f"(found {version[0]}.{version[1]})"
            )
        theirs = git(
            self.repo_dir, ["rev-parse", "--verify", f"{theirs_ref}^{{commit}}"]
        ).strip()

        for _ in range(_MAX_COMMIT_ATTEMPTS):
            ours = self.head()
            if ours is None:
                if self._swap(theirs, None):
                    return theirs
                continue
            if ours == theirs:
                return ours
            try:
                tree = git(
                    self.repo_dir,
                    [
                        "merge-tree",
                        "--write-tree",
                        # Two blackboards that each started empty share no
                        # ancestor; merge them against an empty base.
                        "--allow-unrelated-histories",
                        ours,
                        theirs,
                    ],
                ).strip()
            except GitError as error:
                if error.code == 1:
                    raise BlackboardMergeConflictError(
                        f"Blackboard merge has conflicts:\n{error.stdout}"
                    ) from error
                raise
            commit = self._commit_tree(
                tree, [ours, theirs], f"blackboard: merge {theirs_ref}"
            )
            if self._swap(commit, ours):
                return commit
        raise OperationFailed(
            f"Gave up merging into {self.ref} after {_MAX_COMMIT_ATTEMPTS} "
            "contended attempts"
        )

    def _read_at(self, commit: str | None, path: str) -> str | None:
        if commit is None:
            return None
        try:
            return git(
                self.repo_dir,
                ["cat-file", "blob", f"{commit}:{normalize_board_path(path)}"],
            )
        except GitError:
            return None

    def _build_tree(self, parent: str | None, changes: list[BoardChange]) -> str:
        with tempfile.TemporaryDirectory(prefix="director-blackboard-") as scratch:
            env = {"GIT_INDEX_FILE": str(Path(scratch) / "index")}
            if parent:
                git(self.repo_dir, ["read-tree", parent], env=env)
            for change in changes:
                path = normalize_board_path(change.path)
                if change.content is None:
                    git(
                        self.repo_dir,
                        ["update-index", "--force-remove", path],
                        env=env,
                    )
                    continue
                blob = git(
                    self.repo_dir,
                    ["hash-object", "-w", "--stdin"],
                    stdin=change.content,
                ).strip()
                git(
                    self.repo_dir,
                    ["update-index", "--add", "--cacheinfo", f"100644,{blob},{path}"],
                    env=env,
                )
            return git(self.repo_dir, ["write-tree"], env=env).strip()

    def _commit_tree(self, tree: str, parents: list[str], message: str) -> str:
        parent_args = [arg for parent in parents for arg in ("-p", parent)]
        return git(
            self.repo_dir,
            ["commit-tree", tree, *parent_args, "-m", message],
            env=_IDENTITY,
        ).strip()

    def _swap(self, commit: str, expected: str | None) -> bool:
        """Move the ref from ``expected`` to ``commit`` unless someone moved it."""
        try:
            git(self.repo_dir, ["update-ref", self.ref, commit, expected or _ZERO_OID])
        except GitError:
            return False
        return True
