"""What must outlive the server and travel between machines: files kept
apart from any checkout, in a store that takes each change whole or not
at all.

The director keeps two such stores for an org: its team memory and its
tree. `RecordStore` is what one does; the helpers here say how team
memory is laid out in one, whatever keeps it.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from typing import NamedTuple

from pydantic import TypeAdapter, ValidationError

from openhands.agent_server.director.models import (
    FormerTeam,
    Report,
    RoutedRequest,
    TeamEvent,
)
from openhands.sdk.errors import Invalid


# What the members of a team recorded, in the team's folder beside its
# other documents: one file for all of them.
REPORTS_FILE = "report.json"
# One record for each team that was replaced or pruned:
# `<team id>/<time>.json`.
FORMER_DIR = ".former"
HISTORY_DIR = ".history"
# What the human said that went straight to a team, for the whole org.
ROUTED_FILE = ".routed.json"

REPORTS = TypeAdapter(list[Report])
EVENTS = TypeAdapter(list[TeamEvent])
# How many events are kept of the teams below one team; the oldest go first.
HISTORY_KEPT = 300
ROUTED = TypeAdapter(list[RoutedRequest])
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
    # What its agents recorded, by the folder of the team and in the order
    # it was written.
    reports: dict[str, list[Report]]
    # The records of the teams it removed or that the org did not take.
    former: list[FormerTeam]
    # The teams whose own records go: another team has their id now, or
    # none.
    forget: list[str]
    # The folders of those teams: what was recorded there goes with them.
    forget_reports: list[str]
    # What became of the teams below a team, by that team.
    events: dict[str, list[TeamEvent]]
    # Its requests that went straight to a team.
    routed: list[RoutedRequest]


class Staged(NamedTuple):
    """A commit made for a ref that the ref does not point at yet."""

    ref: str
    commit: str
    parent: str | None


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


def scoped(scope_path: str, file: str) -> str:
    relative = normalize_store_path(file)
    if not relative:
        raise StoreScopeError("A store write needs a file name")
    return _join(normalize_store_path(scope_path), relative)


def parse_reports(raw: str | None) -> list[Report]:
    return REPORTS.validate_json(raw) if raw else []


def reports_file(folder: str) -> str:
    """The file of what the members of the team of ``folder`` recorded."""
    return _join(normalize_store_path(folder), REPORTS_FILE)


def _is_reports(path: str) -> bool:
    return path == REPORTS_FILE or path.endswith(f"/{REPORTS_FILE}")


def in_order(reports: list[Report]) -> list[Report]:
    return sorted(reports, key=lambda report: report.at)


def _former_file(node_id: str, at: str) -> str:
    return f"{FORMER_DIR}/{node_id}/{at}.json"


def team_key(node_id: str) -> str:
    """The team what is kept for ``node_id`` is kept under: the team of the
    org itself, whichever session's place the id names."""
    return node_id.split("@")[0]


def history_file(node_id: str) -> str:
    return f"{HISTORY_DIR}/{team_key(node_id)}.json"


def serialize(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


def _union[T](ours: list[T], theirs: list[T]) -> list[T]:
    return [*ours, *(item for item in theirs if item not in ours)]


def settle(path: str, ours: str | None, theirs: str | None) -> tuple[str | None, bool]:
    """What a file both stores changed becomes.

    Returns:
        The content, and whether the other side's text was left out of it.
    """
    if ours is None or theirs is None:
        # This side's stays, removed or changed, and what the other side
        # did is left out.
        return ours, True
    try:
        if _is_reports(path):
            reports = in_order(_union(parse_reports(ours), parse_reports(theirs)))
            return serialize(REPORTS.dump_python(reports)), False
        if path.startswith(f"{HISTORY_DIR}/"):
            events = _union(EVENTS.validate_json(ours), EVENTS.validate_json(theirs))
            events.sort(key=lambda event: event.at)
            return serialize(EVENTS.dump_python(events[-HISTORY_KEPT:])), False
        if path == ROUTED_FILE:
            routed = _union(ROUTED.validate_json(ours), ROUTED.validate_json(theirs))
            routed.sort(key=lambda request: request.at)
            return serialize(ROUTED.dump_python(routed[-ROUTED_KEPT:])), False
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

        for file in (
            *(reports_file(folder) for folder in landing.forget_reports),
            *(history_file(team) for team in landing.forget),
        ):
            if read(file) is not None:
                out[file] = None
        for record in landing.former:
            path = _former_file(record.id, record.at)
            if read(path) is None:
                out[path] = record.model_dump_json(indent=2) + "\n"
        for folder, reports in landing.reports.items():
            path = reports_file(folder)
            kept = parse_reports(now(path))
            known = {report.id for report in kept if report.id}
            added = [report for report in reports if report.id not in known]
            if added:
                out[path] = serialize(REPORTS.dump_python(in_order([*kept, *added])))
        for team, events in landing.events.items():
            path = history_file(team)
            raw = now(path)
            had = EVENTS.validate_json(raw) if raw else []
            known = {event.id for event in had if event.id}
            more = [event for event in events if event.id not in known]
            if more:
                kept = [*had, *more][-HISTORY_KEPT:]
                out[path] = serialize(EVENTS.dump_python(kept))
        if landing.routed:
            raw = read(ROUTED_FILE)
            asked = ROUTED.validate_json(raw) if raw else []
            known = {request.id for request in asked if request.id}
            new = [r for r in landing.routed if r.id not in known]
            if new:
                asked = sorted([*asked, *new], key=lambda r: r.at)[-ROUTED_KEPT:]
                out[ROUTED_FILE] = serialize(ROUTED.dump_python(asked))
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


class RecordStore(ABC):
    """One store of an org's records.

    A version of the store is named by a string; ``None`` names the store
    before its first write.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """What the store is called where it is kept, for a message."""

    @abstractmethod
    def head(self) -> str | None:
        """The commit the ref points at, or None before the first write."""

    @abstractmethod
    def read(self, path: str) -> str | None:
        """The content of ``path`` at the ref head, or None if it is absent."""

    @abstractmethod
    def write(self, scope_path: str, file: str, content: str) -> str | None:
        """Write one file inside ``scope_path`` and return the new commit.

        A file that resolves outside the scope is rejected.
        """

    @abstractmethod
    def reports(self, folder: str, role: str | None = None) -> list[Report]:
        """What the members of the team of ``folder`` recorded, oldest
        first: all of it, or what the member ``role`` did. It belongs to
        the team's member, whichever conversation it has."""

    @abstractmethod
    def former(self, node_id: str) -> list[FormerTeam]:
        """The records of the teams that were removed from a team id,
        oldest first."""

    @abstractmethod
    def former_below(self, above: str) -> list[FormerTeam]:
        """The records of the teams that were removed from directly below
        a team, oldest first."""

    @abstractmethod
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

    @abstractmethod
    def stage(
        self, mutate: Callable[[StoreReader], list[StoreChange]], message: str
    ) -> Staged | None:
        """Make the commit ``mutate`` asks for without moving the ref to it.

        Returns:
            None when ``mutate`` produced no changes.
        """

    @abstractmethod
    def history(self, node_id: str) -> list[TeamEvent]:
        """What became of the teams directly below a team, oldest first."""

    @abstractmethod
    def routed(self) -> list[RoutedRequest]:
        """What the human said that went straight to a team, oldest first."""

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def clashes(self, theirs_ref: str) -> list[Clash]:
        """The files a merge of ``theirs_ref`` would keep this side's text of,
        because both sides changed the same lines.

        Raises:
            NotSupported: The installed git is too old.
        """

    @abstractmethod
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

    @abstractmethod
    def land(
        self, changes: list[StoreChange], message: str, *, merging: str | None = None
    ) -> bool:
        """Commit ``changes`` on the head, once.

        Args:
            merging: A commit to record as the second parent.

        Returns:
            False when another writer moved the ref first; nothing was written.
        """

    @abstractmethod
    def move_to(self, commit: str) -> bool:
        """Point the ref at ``commit``, which has its head in its history.

        Returns:
            False when it has not, or another writer moved the ref first.
        """

    @abstractmethod
    def contains(self, other: str) -> bool:
        """Whether the head has ``other`` in its history."""

    @abstractmethod
    def base_with(self, other: str) -> str | None:
        """The commit this ref and ``other`` both come from, if any."""

    @abstractmethod
    def read_at(self, commit: str | None, path: str) -> str | None:
        """The content of ``path`` at ``commit``, or None if it is absent."""
