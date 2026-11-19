"""A virtual directory tree stored under a hidden git ref.

Nothing here touches the working tree or the user's index. Every mutation
builds a tree in a throwaway index file and lands with a compare-and-swap
``git update-ref``, so concurrent writers never lose each other's changes.
"""

import json
import tempfile
from collections.abc import Callable, Collection, Sequence
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
    AGENT_ROLES,
    AgentRole,
    FormerTeam,
    Report,
    RoutedRequest,
    TeamEvent,
)
from openhands.sdk.errors import Invalid, NotSupported, OperationFailed


MEMORY_REF = "refs/agents/memory"
SPEC_FILE = "spec.md"
CONTRACTS_FILE = "contracts.md"
# One file of reports for each agent of a team: `<team id>/<role>.json`.
REPORTS_DIR = ".reports"
# One record for each team that was replaced or pruned:
# `<team id>/<time>.json`.
FORMER_DIR = ".former"
HISTORY_DIR = ".history"
# What the human said that went straight to a team, for the whole org.
ROUTED_FILE = ".routed.json"

_ZERO_OID = "0" * 40
_MAX_COMMIT_ATTEMPTS = 8
_MERGE_MIN_GIT = (2, 38)
_REPORTS = TypeAdapter(list[Report])
_EVENTS = TypeAdapter(list[TeamEvent])
# How many events are kept of the teams below one team; the oldest go first.
HISTORY_KEPT = 300
_ROUTED = TypeAdapter(list[RoutedRequest])
ROUTED_KEPT = 200


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


class Landing(NamedTuple):
    """What a session's accepted work makes of team memory, to be taken in
    as one commit."""

    # Each file the session wrote, with its text (None to remove it) and
    # the text it was written over.
    files: dict[str, tuple[str | None, str | None]]
    # The text a reviewer was shown of a file and decided to put the
    # session's in place of.
    seen: dict[str, str | None]
    # What its agents recorded, by team and in the order it was written.
    reports: dict[str, list[Report]]
    # The records of the teams it removed or that the org did not take.
    former: list[FormerTeam]
    # The teams whose own records go: another team has their id now, or
    # none.
    forget: list[str]
    # What became of the teams below a team, by that team.
    events: dict[str, list[TeamEvent]]
    # Its requests that went straight to a team.
    routed: list[RoutedRequest]


class Staged(NamedTuple):
    """A commit made for a ref that the ref does not point at yet."""

    ref: str
    commit: str
    parent: str | None


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


def _reports_file(node_id: str, role: AgentRole) -> str:
    return f"{REPORTS_DIR}/{node_id}/{role}.json"


def _former_file(node_id: str, at: str) -> str:
    return f"{FORMER_DIR}/{node_id}/{at}.json"


def team_key(node_id: str) -> str:
    """The team what is kept for ``node_id`` is kept under: the team of the
    org itself, whichever session's place the id names."""
    return node_id.split("@")[0]


def _history_file(node_id: str) -> str:
    return f"{HISTORY_DIR}/{team_key(node_id)}.json"


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
        # What this side removed stays removed, with their text left out.
        return ours, ours is None
    try:
        if path.startswith(f"{REPORTS_DIR}/"):
            reports = _union(_parse_reports(ours), _parse_reports(theirs))
            reports.sort(key=lambda report: report.at)
            return _serialize(_REPORTS.dump_python(reports)), False
        if path.startswith(f"{HISTORY_DIR}/"):
            events = _union(_EVENTS.validate_json(ours), _EVENTS.validate_json(theirs))
            events.sort(key=lambda event: event.at)
            return _serialize(_EVENTS.dump_python(events[-HISTORY_KEPT:])), False
        if path == ROUTED_FILE:
            routed = _union(_ROUTED.validate_json(ours), _ROUTED.validate_json(theirs))
            routed.sort(key=lambda request: request.at)
            return _serialize(_ROUTED.dump_python(routed[-ROUTED_KEPT:])), False
    except ValidationError:
        pass
    return ours, True


def taking_in(
    landing: Landing,
) -> tuple[Callable[[StoreReader], list[StoreChange]], dict[str, str | None]]:
    """What taking a landing in changes of team memory.

    Returns:
        The mutation, and the files it did not write with the text each
        has, filled in whenever the mutation is run.
    """
    left: dict[str, str | None] = {}

    def mutate(read: StoreReader) -> list[StoreChange]:
        left.clear()
        out: dict[str, str | None] = {}

        def now(path: str) -> str | None:
            return out[path] if path in out else read(path)

        for team in landing.forget:
            for file in (
                *(_reports_file(team, role) for role in AGENT_ROLES),
                _history_file(team),
            ):
                if read(file) is not None:
                    out[file] = None
        for record in landing.former:
            path = _former_file(record.id, record.at)
            if read(path) is None:
                out[path] = record.model_dump_json(indent=2) + "\n"
        for team, reports in landing.reports.items():
            for role in AGENT_ROLES:
                path = _reports_file(team, role)
                kept = _parse_reports(now(path))
                known = {report.id for report in kept if report.id}
                added = [
                    report
                    for report in reports
                    if report.role == role and report.id not in known
                ]
                if added:
                    out[path] = _serialize(_REPORTS.dump_python([*kept, *added]))
        for team, events in landing.events.items():
            path = _history_file(team)
            raw = now(path)
            had = _EVENTS.validate_json(raw) if raw else []
            known = {event.id for event in had if event.id}
            more = [event for event in events if event.id not in known]
            if more:
                kept = [*had, *more][-HISTORY_KEPT:]
                out[path] = _serialize(_EVENTS.dump_python(kept))
        if landing.routed:
            raw = read(ROUTED_FILE)
            asked = _ROUTED.validate_json(raw) if raw else []
            known = {request.id for request in asked if request.id}
            new = [r for r in landing.routed if r.id not in known]
            if new:
                asked = sorted([*asked, *new], key=lambda r: r.at)[-ROUTED_KEPT:]
                out[ROUTED_FILE] = _serialize(_ROUTED.dump_python(asked))
        for path, (text, over) in landing.files.items():
            has = read(path)
            if has == text:
                continue
            if has == over or (path in landing.seen and landing.seen[path] == has):
                out[path] = text
            else:
                left[path] = has
        return [StoreChange(path, content) for path, content in out.items()]

    return mutate, left


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

    def write(self, scope_path: str, file: str, content: str) -> str | None:
        """Write one file inside ``scope_path`` and return the new commit.

        A file that resolves outside the scope is rejected.
        """
        path = _scoped(scope_path, file)
        return self.commit(
            lambda _read: [StoreChange(path, content)], f"memory: write {path}"
        )

    def reports(self, node_id: str, role: AgentRole) -> list[Report]:
        """What one agent of a team recorded, oldest first. They belong to
        the team's agent, whichever conversation it has."""
        return _parse_reports(self.read(_reports_file(node_id, role)))

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
        raw = self.read(_history_file(node_id))
        return _EVENTS.validate_json(raw) if raw else []

    def routed(self) -> list[RoutedRequest]:
        """What the human said that went straight to a team, oldest first."""
        raw = self.read(ROUTED_FILE)
        return _ROUTED.validate_json(raw) if raw else []

    def replace_reports(
        self,
        node_id: str,
        role: AgentRole,
        read_as: list[Report],
        reports: list[Report],
    ) -> bool:
        """Put ``reports`` in place of an agent's, unless they changed since
        they were read as ``read_as``.

        Returns:
            Whether they were replaced.
        """
        path = _reports_file(node_id, role)
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
