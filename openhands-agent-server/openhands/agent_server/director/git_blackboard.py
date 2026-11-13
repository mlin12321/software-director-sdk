"""Git-native blackboard: a virtual directory tree stored under a hidden ref.

Nothing here touches the working tree or the user's index. Every mutation
builds a tree in a throwaway index file and lands with a compare-and-swap
``git update-ref``, so concurrent writers never lose each other's changes.
"""

import json
import tempfile
from collections.abc import Callable, Collection
from pathlib import Path
from typing import NamedTuple

from pydantic import TypeAdapter, ValidationError

from openhands.agent_server.director.git import (
    DIRECTOR_IDENTITY,
    GitError,
    git,
    git_version,
)
from openhands.agent_server.director.models import (
    Autopsy,
    BoundaryContracts,
    BoundaryMemory,
    InheritedSpec,
)
from openhands.sdk.errors import Invalid, NotSupported, OperationFailed


BLACKBOARD_REF = "refs/agents/blackboard"
SPEC_FILE = "spec.md"
CONTRACTS_FILE = "contracts.md"
MEMORY_FILE = "memory.json"
AUTOPSIES_FILE = "autopsies.json"
# What the rules of a boundary looked like when they were last consolidated.
CONSOLIDATED_FILE = "consolidated.json"

_ZERO_OID = "0" * 40
_MAX_COMMIT_ATTEMPTS = 8
_MERGE_MIN_GIT = (2, 38)
_AUTOPSIES = TypeAdapter(list[Autopsy])


class BlackboardScopeError(Invalid):
    """A path escapes the blackboard or the scope it must stay inside."""


class BoardChange(NamedTuple):
    """One file to write. ``content=None`` deletes the file."""

    path: str
    content: str | None


class Merged(NamedTuple):
    """The outcome of merging another blackboard into this one."""

    head: str
    kept_local: list[str]
    """Files both sides changed where this side's text was kept."""


class Clash(NamedTuple):
    """A file both blackboards changed the same lines of."""

    path: str
    ours: str | None
    theirs: str | None


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


def _union[T](ours: list[T], theirs: list[T]) -> list[T]:
    return [*ours, *(item for item in theirs if item not in ours)]


def _settle(path: str, ours: str | None, theirs: str | None) -> tuple[str | None, bool]:
    """What a file both blackboards changed becomes.

    Returns:
        The content, and whether the other side's text was left out of it.
    """
    if ours is None or theirs is None:
        return (theirs if ours is None else ours), False
    name = path.rsplit("/", 1)[-1]
    try:
        if name == MEMORY_FILE:
            rules = _union(_parse_memory(ours).rules, _parse_memory(theirs).rules)
            return _serialize(BoundaryMemory(rules=rules).model_dump()), False
        if name == AUTOPSIES_FILE:
            autopsies = _union(_parse_autopsies(ours), _parse_autopsies(theirs))
            return _serialize(_AUTOPSIES.dump_python(autopsies)), False
    except ValidationError:
        return ours, True
    return ours, name != CONSOLIDATED_FILE


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

    def applies_to(
        self, boundary_path: str, above: str | None
    ) -> tuple[list[InheritedSpec], list[BoundaryContracts]]:
        """The spec and the contracts a team works to.

        That is the spec of its own boundary, the contracts between its own
        sub-teams, and the contracts the team above set between this team
        and the ones beside it. The specs of the teams above are theirs
        alone.

        Args:
            above: The boundary of the team above; None for the root team.
        """
        own = normalize_board_path(boundary_path)
        commit = self.head()
        spec = self._read_at(commit, _join(own, SPEC_FILE))
        contracts = [
            BoundaryContracts(path=path, contracts=text, own=path == own)
            for path in ([] if above is None else [normalize_board_path(above)]) + [own]
            if (text := self._read_at(commit, _join(path, CONTRACTS_FILE)))
        ]
        specs = (
            [] if spec is None else [InheritedSpec(path=own, spec=spec, readonly=False)]
        )
        return specs, contracts

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

    def directories_with(self, file: str) -> list[str]:
        """The directories of the blackboard that hold a file named ``file``."""
        commit = self.head()
        if commit is None:
            return []
        listed = git(self.repo_dir, ["ls-tree", "-r", "-z", "--name-only", commit])
        return sorted(
            path[: -len(file)].rstrip("/")
            for path in listed.split("\0")
            if path == file or path.endswith(f"/{file}")
        )

    def clashes(self, theirs_ref: str) -> list[Clash]:
        """The files a merge of ``theirs_ref`` would keep this side's text of,
        because both sides changed the same lines.

        Raises:
            NotSupported: The installed git is too old.
        """
        theirs = self._other(theirs_ref)
        ours = self.head()
        if ours is None or self._contains(ours, theirs) or self._contains(theirs, ours):
            return []
        _, kept = self._merged_tree(ours, theirs, ())
        return [
            Clash(path, self._read_at(ours, path), self._read_at(theirs, path))
            for path in kept
        ]

    def merge(self, theirs_ref: str, take_theirs: Collection[str] = ()) -> Merged:
        """Three-way merge another blackboard ref into this one.

        The merge happens entirely in the object store and needs git 2.38 or
        newer (``git merge-tree --write-tree``). Where both sides changed the
        same lines of a file, rules and autopsies become those of both; of
        anything else this side's text is kept and the other's stays in its
        history.

        Args:
            take_theirs: The files to take the other side's text of instead.

        Raises:
            NotSupported: The installed git is too old.
        """
        theirs = self._other(theirs_ref)

        for _ in range(_MAX_COMMIT_ATTEMPTS):
            ours = self.head()
            if ours is None:
                if self._swap(theirs, None):
                    return Merged(theirs, [])
                continue
            if ours == theirs or self._contains(ours, theirs):
                return Merged(ours, [])
            if self._contains(theirs, ours):
                if self._swap(theirs, ours):
                    return Merged(theirs, [])
                continue
            tree, kept = self._merged_tree(ours, theirs, take_theirs)
            commit = self._commit_tree(
                tree, [ours, theirs], f"blackboard: merge {theirs_ref}"
            )
            if self._swap(commit, ours):
                return Merged(commit, kept)
        raise OperationFailed(
            f"Gave up merging into {self.ref} after {_MAX_COMMIT_ATTEMPTS} "
            "contended attempts"
        )

    def land(
        self, changes: list[BoardChange], message: str, *, merging: str | None = None
    ) -> bool:
        """Commit ``changes`` on the head, once.

        Args:
            merging: A commit to record as the second parent.

        Returns:
            False when another writer moved the ref first; nothing was written.
        """
        parent = self.head()
        tree = self._build_tree(parent, changes)
        parents = [commit for commit in (parent, merging) if commit]
        return self._swap(self._commit_tree(tree, parents, message), parent)

    def move_to(self, commit: str) -> bool:
        """Point the ref at ``commit``, which has its head in its history.

        Returns:
            False when it has not, or another writer moved the ref first.
        """
        parent = self.head()
        if parent is not None and not self._contains(commit, parent):
            return False
        return self._swap(commit, parent)

    def contains(self, other: str) -> bool:
        """Whether the head has ``other`` in its history."""
        head = self.head()
        return head is not None and (head == other or self._contains(head, other))

    def base_with(self, other: str) -> str | None:
        """The commit this ref and ``other`` both come from, if any."""
        head = self.head()
        if head is None:
            return None
        try:
            return git(self.repo_dir, ["merge-base", head, other]).strip() or None
        except GitError:
            return None

    def read_at(self, commit: str | None, path: str) -> str | None:
        """The content of ``path`` at ``commit``, or None if it is absent."""
        return self._read_at(commit, path)

    def _other(self, theirs_ref: str) -> str:
        version = git_version(self.repo_dir)
        if version < _MERGE_MIN_GIT:
            raise NotSupported(
                "Blackboard merge needs git 2.38 or newer "
                f"(found {version[0]}.{version[1]})"
            )
        return git(
            self.repo_dir, ["rev-parse", "--verify", f"{theirs_ref}^{{commit}}"]
        ).strip()

    def _contains(self, commit: str, other: str) -> bool:
        try:
            git(self.repo_dir, ["merge-base", "--is-ancestor", other, commit])
        except GitError:
            return False
        return True

    def _merged_tree(
        self, ours: str, theirs: str, take_theirs: Collection[str]
    ) -> tuple[str, list[str]]:
        command = ["merge-tree", "--write-tree", "--name-only", "-z", "--no-messages"]
        # Two blackboards that each started empty share no ancestor; merge
        # them against an empty base.
        command += ["--allow-unrelated-histories", ours, theirs]
        try:
            return git(self.repo_dir, command).split("\0")[0].strip(), []
        except GitError as error:
            if error.code != 1:
                raise
            # The tree git wrote comes first, then the files it could not merge.
            tree, *conflicted = error.stdout.split("\0")
        changes: list[BoardChange] = []
        kept: list[str] = []
        for path in filter(None, conflicted):
            content, ours_kept = _settle(
                path, self._read_at(ours, path), self._read_at(theirs, path)
            )
            if ours_kept and path in take_theirs:
                content, ours_kept = self._read_at(theirs, path), False
            changes.append(BoardChange(path, content))
            if ours_kept:
                kept.append(path)
        return self._build_tree(tree.strip(), changes), kept

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
            env=DIRECTOR_IDENTITY,
        ).strip()

    def _swap(self, commit: str, expected: str | None) -> bool:
        """Move the ref from ``expected`` to ``commit`` unless someone moved it."""
        try:
            git(self.repo_dir, ["update-ref", self.ref, commit, expected or _ZERO_OID])
        except GitError:
            return False
        return True
