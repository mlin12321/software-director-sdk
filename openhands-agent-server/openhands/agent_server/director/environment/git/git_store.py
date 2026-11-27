"""A virtual directory tree stored under a hidden git ref.

Nothing here touches the working tree or the user's index. Every mutation
builds a tree in a throwaway index file and lands with a compare-and-swap
``git update-ref``, so concurrent writers never lose each other's changes.
"""

import tempfile
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

from openhands.agent_server.director.environment.git.cli import (
    DIRECTOR_IDENTITY,
    GitError,
    git,
    git_version,
)
from openhands.agent_server.director.environment.records import (
    Clash,
    Merged,
    RecordStore,
    Staged,
    StoreChange,
    StoreReader,
    normalize_store_path,
    settle,
)
from openhands.sdk.errors import NotSupported, OperationFailed


MEMORY_REF = "refs/agents/memory"
_ZERO_OID = "0" * 40
_MAX_COMMIT_ATTEMPTS = 8
_MERGE_MIN_GIT = (2, 38)
# Where an org's tree is kept.
ORG_REF = "refs/agents/org"


def swap_together(repo_dir: Path, staged: Sequence[Staged]) -> bool:
    """Move every ref to the commit staged for it, all of them or none.

    Returns:
        False when one of them was moved by someone else meanwhile.
    """
    moves = "".join(
        f"update {move.ref} {move.commit} {move.parent or _ZERO_OID}\n"
        for move in staged
    )
    try:
        git(repo_dir, ["update-ref", "--stdin"], stdin=moves)
    except GitError:
        return False
    return True


class GitStore(RecordStore):
    """Reads and writes one store ref of one repository."""

    def __init__(self, repo_dir: Path, ref: str = MEMORY_REF) -> None:
        self.repo_dir = repo_dir
        self.ref = ref

    @property
    def name(self) -> str:
        return self.ref

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

    def stage(
        self, mutate: Callable[[StoreReader], list[StoreChange]], message: str
    ) -> Staged | None:
        """Make the commit ``mutate`` asks for without moving the ref to it.

        Returns:
            None when ``mutate`` produced no changes.
        """
        parent = self.head()
        changes = mutate(lambda path: self._read_at(parent, path))
        if not changes:
            return None
        tree = self._build_tree(parent, changes)
        commit = self._commit_tree(tree, [parent] if parent else [], message)
        return Staged(self.ref, commit, parent)

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
            content, ours_kept = settle(
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

    def files(self, directory: str) -> list[str]:
        """Every file under ``directory`` at the ref head."""
        commit = self.head()
        if commit is None:
            return []
        try:
            listed = git(
                self.repo_dir,
                ["ls-tree", "-r", "--name-only", "-z", commit, "--", directory],
            )
        except GitError:
            return []
        return [file for file in listed.split("\0") if file]

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
                        ["update-index", "--force-remove", "--", path],
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
