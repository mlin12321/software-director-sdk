"""A virtual directory tree stored under a hidden git ref.

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
    BoundaryContracts,
    InheritedSpec,
    Report,
    ThreadMessage,
)
from openhands.sdk.errors import Invalid, NotSupported, OperationFailed


MEMORY_REF = "refs/agents/memory"
SPEC_FILE = "spec.md"
CONTRACTS_FILE = "contracts.md"
# One file of reports for each agent, named after its conversation.
REPORTS_DIR = ".reports"
# One file for each Manager and IC that have written to each other.
THREADS_DIR = ".threads"
# What team memory was called before; a repository that still has it is moved.
_FORMER_MEMORY_REF = "refs/agents/blackboard"

_ZERO_OID = "0" * 40
_MAX_COMMIT_ATTEMPTS = 8
_MERGE_MIN_GIT = (2, 38)
_REPORTS = TypeAdapter(list[Report])
_MESSAGES = TypeAdapter(list[ThreadMessage])


class StoreScopeError(Invalid):
    """A path escapes the store or the scope it must stay inside."""


class StoreChange(NamedTuple):
    """One file to write. ``content=None`` deletes the file."""

    path: str
    content: str | None


class Merged(NamedTuple):
    """The outcome of merging another store into this one."""

    head: str
    kept_local: list[str]
    """Files both sides changed where this side's text was kept."""


class Clash(NamedTuple):
    """A file both stores changed the same lines of."""

    path: str
    ours: str | None
    theirs: str | None


type StoreReader = Callable[[str], str | None]


def normalize_store_path(path: str) -> str:
    """Normalize a repo-relative path. ``""``, ``"."`` and ``"/"`` are the root.

    Raises:
        StoreScopeError: The path could escape the virtual tree.
    """
    segments = [
        segment
        for segment in path.replace("\\", "/").split("/")
        if segment not in ("", ".")
    ]
    if any(segment in ("..", ".git") for segment in segments):
        raise StoreScopeError(f"Path escapes the store: {path}")
    return "/".join(segments)


def _join(directory: str, file: str) -> str:
    return f"{directory}/{file}" if directory else file


def _scoped(scope_path: str, file: str) -> str:
    relative = normalize_store_path(file)
    if not relative:
        raise StoreScopeError("A store write needs a file name")
    return _join(normalize_store_path(scope_path), relative)


def _parse_reports(raw: str | None) -> list[Report]:
    return _REPORTS.validate_json(raw) if raw else []


def _reports_file(conversation_id: str) -> str:
    return f"{REPORTS_DIR}/{conversation_id}.json"


def _parse_messages(raw: str | None) -> list[ThreadMessage]:
    return _MESSAGES.validate_json(raw) if raw else []


def _thread_file(manager_node_id: str, ic_node_id: str) -> str:
    return f"{THREADS_DIR}/{manager_node_id}/{ic_node_id}.json"


def _serialize(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


def _union[T](ours: list[T], theirs: list[T]) -> list[T]:
    return [*ours, *(item for item in theirs if item not in ours)]


def _settle(path: str, ours: str | None, theirs: str | None) -> tuple[str | None, bool]:
    """What a file both stores changed becomes.

    Returns:
        The content, and whether the other side's text was left out of it.
    """
    if ours is None or theirs is None:
        return (theirs if ours is None else ours), False
    try:
        if path.startswith(f"{REPORTS_DIR}/"):
            reports = _union(_parse_reports(ours), _parse_reports(theirs))
            reports.sort(key=lambda report: report.at)
            return _serialize(_REPORTS.dump_python(reports)), False
        if path.startswith(f"{THREADS_DIR}/"):
            messages = _union(_parse_messages(ours), _parse_messages(theirs))
            messages.sort(key=lambda message: message.at)
            return _serialize(_MESSAGES.dump_python(messages)), False
    except ValidationError:
        pass
    return ours, True


def adopt_former_memory(repo_dir: Path) -> None:
    """Move team memory kept under its former name to ``MEMORY_REF``."""
    former = GitStore(repo_dir, _FORMER_MEMORY_REF)
    commit = former.head()
    if commit is None:
        return
    if GitStore(repo_dir).head() is None:
        git(repo_dir, ["update-ref", MEMORY_REF, commit, _ZERO_OID])
    git(repo_dir, ["update-ref", "-d", _FORMER_MEMORY_REF, commit])


class GitStore:
    """Reads and writes one store ref of one repository."""

    def __init__(self, repo_dir: Path, ref: str = MEMORY_REF) -> None:
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
        own = normalize_store_path(boundary_path)
        commit = self.head()
        spec = self._read_at(commit, _join(own, SPEC_FILE))
        contracts = [
            BoundaryContracts(path=path, contracts=text, own=path == own)
            for path in ([] if above is None else [normalize_store_path(above)]) + [own]
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
            lambda _read: [StoreChange(path, content)], f"memory: write {path}"
        )

    def reports(self, conversation_id: str) -> list[Report]:
        """What the agent of one conversation recorded, oldest first."""
        return _parse_reports(self.read(_reports_file(conversation_id)))

    def append_report(self, conversation_id: str, report: Report) -> list[Report]:
        """Add to what an agent recorded and return all of it."""
        path = _reports_file(conversation_id)
        reports: list[Report] = []

        def mutate(read: StoreReader) -> list[StoreChange]:
            nonlocal reports
            reports = [*_parse_reports(read(path)), report]
            return [StoreChange(path, _serialize(_REPORTS.dump_python(reports)))]

        self.commit(mutate, f"memory: report {path}")
        return reports

    def thread(self, manager_node_id: str, ic_node_id: str) -> list[ThreadMessage]:
        """What a Manager and an IC have said to each other, oldest first."""
        return _parse_messages(self.read(_thread_file(manager_node_id, ic_node_id)))

    def append_message(
        self, manager_node_id: str, ic_node_id: str, message: ThreadMessage
    ) -> None:
        """Add to what a Manager and an IC have said to each other."""
        path = _thread_file(manager_node_id, ic_node_id)

        def mutate(read: StoreReader) -> list[StoreChange]:
            messages = [*_parse_messages(read(path)), message]
            return [StoreChange(path, _serialize(_MESSAGES.dump_python(messages)))]

        self.commit(mutate, f"memory: message {path}")

    def replace_reports(
        self, conversation_id: str, read_as: list[Report], reports: list[Report]
    ) -> bool:
        """Put ``reports`` in place of an agent's, unless they changed since
        they were read as ``read_as``.

        Returns:
            Whether they were replaced.
        """
        path = _reports_file(conversation_id)
        replaced = False

        def mutate(read: StoreReader) -> list[StoreChange]:
            nonlocal replaced
            replaced = _parse_reports(read(path)) == read_as
            if not replaced:
                return []
            return [StoreChange(path, _serialize(_REPORTS.dump_python(reports)))]

        self.commit(mutate, f"memory: compact {path}")
        return replaced

    def commit(
        self, mutate: Callable[[StoreReader], list[StoreChange]], message: str
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
        """Three-way merge another store ref into this one.

        The merge happens entirely in the object store and needs git 2.38 or
        newer (``git merge-tree --write-tree``). Where both sides changed the
        same lines of a file, reports and messages become those of both; of
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
                tree, [ours, theirs], f"memory: merge {theirs_ref}"
            )
            if self._swap(commit, ours):
                return Merged(commit, kept)
        raise OperationFailed(
            f"Gave up merging into {self.ref} after {_MAX_COMMIT_ATTEMPTS} "
            "contended attempts"
        )

    def land(
        self, changes: list[StoreChange], message: str, *, merging: str | None = None
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
                f"A merge needs git 2.38 or newer (found {version[0]}.{version[1]})"
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
        # Two stores that each started empty share no ancestor; merge
        # them against an empty base.
        command += ["--allow-unrelated-histories", ours, theirs]
        try:
            return git(self.repo_dir, command).split("\0")[0].strip(), []
        except GitError as error:
            if error.code != 1:
                raise
            # The tree git wrote comes first, then the files it could not merge.
            tree, *conflicted = error.stdout.split("\0")
        changes: list[StoreChange] = []
        kept: list[str] = []
        for path in filter(None, conflicted):
            content, ours_kept = _settle(
                path, self._read_at(ours, path), self._read_at(theirs, path)
            )
            if ours_kept and path in take_theirs:
                content, ours_kept = self._read_at(theirs, path), False
            changes.append(StoreChange(path, content))
            if ours_kept:
                kept.append(path)
        return self._build_tree(tree.strip(), changes), kept

    def _read_at(self, commit: str | None, path: str) -> str | None:
        if commit is None:
            return None
        try:
            return git(
                self.repo_dir,
                ["cat-file", "blob", f"{commit}:{normalize_store_path(path)}"],
            )
        except GitError:
            return None

    def _build_tree(self, parent: str | None, changes: list[StoreChange]) -> str:
        with tempfile.TemporaryDirectory(prefix="director-store-") as scratch:
            env = {"GIT_INDEX_FILE": str(Path(scratch) / "index")}
            if parent:
                git(self.repo_dir, ["read-tree", parent], env=env)
            for change in changes:
                path = normalize_store_path(change.path)
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
