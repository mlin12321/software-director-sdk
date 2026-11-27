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
    EVENTS,
    FORMER_DIR,
    REPORTS,
    ROUTED,
    ROUTED_FILE,
    Clash,
    Landing,
    Merged,
    RecordStore,
    Staged,
    StoreChange,
    StoreReader,
    history_file,
    in_order,
    normalize_store_path,
    parse_reports,
    reports_file,
    scoped,
    serialize,
    settle,
    taking_in,
    team_key,
)
from openhands.agent_server.director.models import (
    FormerTeam,
    Report,
    RoutedRequest,
    TeamEvent,
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

    def read(self, path: str) -> str | None:
        """The content of ``path`` at the ref head, or None if it is absent."""
        return self._read_at(self.head(), path)

    def write(self, scope_path: str, file: str, content: str) -> str | None:
        """Write one file inside ``scope_path`` and return the new commit.

        A file that resolves outside the scope is rejected.
        """
        path = scoped(scope_path, file)
        return self.commit(
            lambda _read: [StoreChange(path, content)], f"memory: write {path}"
        )

    def reports(self, folder: str, role: str | None = None) -> list[Report]:
        """What the members of the team of ``folder`` recorded, oldest
        first: all of it, or what the member ``role`` did. It belongs to
        the team's member, whichever conversation it has."""
        return [
            report
            for report in parse_reports(self.read(reports_file(folder)))
            if role is None or report.role == role
        ]

    def former(self, node_id: str) -> list[FormerTeam]:
        """The records of the teams that were removed from a team id,
        oldest first."""
        commit = self.head()
        return [
            FormerTeam.model_validate_json(raw)
            for file in sorted(self._files(f"{FORMER_DIR}/{team_key(node_id)}"))
            if (raw := self._read_at(commit, file))
        ]

    def former_below(self, above: str) -> list[FormerTeam]:
        """The records of the teams that were removed from directly below
        a team, oldest first."""
        commit = self.head()
        records = [
            FormerTeam.model_validate_json(raw)
            for file in self._files(FORMER_DIR)
            if (raw := self._read_at(commit, file))
        ]
        return sorted(
            (record for record in records if record.above == team_key(above)),
            key=lambda record: record.at,
        )

    def take_in(self, landing: Landing, message: str) -> dict[str, str | None]:
        """Make of team memory what a session's accepted work makes of it,
        all of it or none, whoever else writes meanwhile.

        A file is written only where the text it was written over still
        stands, or the text a reviewer was shown; where the file already
        has the session's text there is nothing to write. A report, an
        event and a routed request are added once, by their ids. A record
        of a former team is written once.

        Returns:
            The files that were not written, with the text each has now.
        """
        mutate, left = taking_in(landing)
        self.commit(mutate, message)
        return left

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

    def history(self, node_id: str) -> list[TeamEvent]:
        """What became of the teams directly below a team, oldest first."""
        raw = self.read(history_file(node_id))
        return EVENTS.validate_json(raw) if raw else []

    def routed(self) -> list[RoutedRequest]:
        """What the human said that went straight to a team, oldest first."""
        raw = self.read(ROUTED_FILE)
        return ROUTED.validate_json(raw) if raw else []

    def replace_reports(
        self,
        folder: str,
        role: str,
        read_as: list[Report],
        reports: list[Report],
    ) -> bool:
        """Put ``reports`` in place of what one member of a team recorded,
        unless that changed since it was read as ``read_as``. What the
        team's other members recorded stays.

        Returns:
            Whether they were replaced.
        """
        path = reports_file(folder)
        replaced = False

        def mutate(read: StoreReader) -> list[StoreChange]:
            nonlocal replaced
            kept = parse_reports(read(path))
            replaced = [report for report in kept if report.role == role] == read_as
            if not replaced:
                return []
            others = [report for report in kept if report.role != role]
            return [
                StoreChange(
                    path,
                    serialize(REPORTS.dump_python(in_order([*others, *reports]))),
                )
            ]

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

    def _files(self, directory: str) -> list[str]:
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
