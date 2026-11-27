"""Where the documents of teams are kept.

A document is kept for the org once the work of the session that wrote it
is accepted. Until then what a session writes is its own: it reads its own
text, and every other session reads the org's.

The store keeps and gives back. It does not decide who may read or write a
document, nor what one means.
"""

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from openhands.agent_server.director.environment.records import RecordStore
from openhands.agent_server.director.models import Report as LogEntry
from openhands.agent_server.director.run_state import FILES, RunState
from openhands.sdk.errors import NotSupported


# The record every member of a team keeps of what it learned. It is the
# director's own, as messages are, and not a document an org invents: the
# one log the store keeps.
REPORT = "report"


class DocumentRef(NamedTuple):
    """One document: its name, in the folder of the team that keeps it."""

    folder: str
    name: str

    @property
    def path(self) -> str:
        """Where it is kept, from the top of the store."""
        file = f"{self.name}.md"
        return f"{self.folder}/{file}" if self.folder else file

    @classmethod
    def at(cls, path: str) -> "DocumentRef":
        """The document kept at ``path``."""
        folder, _, file = path.rpartition("/")
        return cls(folder, file.removesuffix(".md"))


@dataclass(frozen=True)
class TeamRef:
    """The team a document is told of: the work of one team in a session.

    Attributes:
        id: The id of the session's place of the team.
        session: The session the work is of.
        folder: The team's folder, where its documents are kept.
    """

    id: str
    session: str
    folder: str

    def keeps(self, name: str) -> DocumentRef:
        """The team's document of that name."""
        return DocumentRef(self.folder, name)


class DocumentDraft(NamedTuple):
    """What a session made of a document, which is its own until its work
    is accepted."""

    text: str | None
    """None where the session removed the document."""
    over: str | None
    """What the org had when the session first wrote it; None for nothing."""


class DocumentStore(ABC):
    """Keeps the documents of teams. Safe from any thread."""

    @abstractmethod
    def read(self, session: str, ref: DocumentRef) -> str | None:
        """A document as ``session`` has it: what it wrote, else the
        org's. '' for the session reads the org's. None where there is
        none."""

    @abstractmethod
    def write(self, session: str, ref: DocumentRef, text: str) -> None:
        """Put ``text`` in the place of a document, for ``session`` alone
        until its work is accepted."""

    @abstractmethod
    def remove(self, session: str, ref: DocumentRef) -> None:
        """Take a document away, for ``session`` alone until its work is
        accepted."""

    @abstractmethod
    def discard(self, session: str, ref: DocumentRef) -> None:
        """Forget what ``session`` made of a document: the org's holds
        for it again."""

    @abstractmethod
    def drafts(self, session: str) -> dict[DocumentRef, DocumentDraft]:
        """Every document ``session`` wrote or removed and the org has not
        taken yet."""

    @abstractmethod
    def entries(self, team: TeamRef, name: str) -> list[LogEntry]:
        """What was added to the log named ``name`` of a team, oldest
        first: what the org has of it and what the team's session added."""

    @abstractmethod
    def append(self, team: TeamRef, name: str, entry: LogEntry) -> None:
        """Add an entry to the log named ``name`` of a team, for its
        session alone until its work is accepted. The store says when it
        was added and in which session."""


class RecordDocumentStore(DocumentStore):
    """Documents in the org's team memory, a hidden ref of its git
    repository; what sessions made of them is kept beside the director's
    unfinished work until it is accepted."""

    def __init__(
        self,
        memory: Callable[[], RecordStore],
        runs: Callable[[], RunState],
        changed: Callable[[str], None] = lambda _session: None,
        logs: "Logs | None" = None,
    ) -> None:
        """
        Args:
            memory: The org's team memory, as it is when asked.
            runs: Where the director keeps its unfinished work.
            changed: Told the session whenever one writes or removes a
                document.
            logs: Where what teams add to their logs is kept; None for a
                store that keeps none.
        """
        self._memory = memory
        self._runs = runs
        self._changed = changed
        self._logs = logs
        self._lock = threading.RLock()

    def read(self, session: str, ref: DocumentRef) -> str | None:
        draft = self.drafts(session).get(ref)
        return draft.text if draft else self._memory().read(ref.path)

    def write(self, session: str, ref: DocumentRef, text: str) -> None:
        self._keep(session, ref, text)

    def remove(self, session: str, ref: DocumentRef) -> None:
        self._keep(session, ref, None)

    def discard(self, session: str, ref: DocumentRef) -> None:
        self._runs().unmark(FILES, _key(session, ref))

    def drafts(self, session: str) -> dict[DocumentRef, DocumentDraft]:
        if not session:
            return {}
        prefix = f"{session}\n"
        found = {}
        for key, raw in self._runs().marked(FILES).items():
            if key.startswith(prefix):
                draft = json.loads(raw)
                found[DocumentRef.at(key.removeprefix(prefix))] = DocumentDraft(
                    draft["text"], draft["over"]
                )
        return found

    def _keep(self, session: str, ref: DocumentRef, text: str | None) -> None:
        # What it is written over is what the session first wrote over, so
        # that what the org made of it since is not lost sight of.
        with self._lock:
            earlier = self.drafts(session).get(ref)
            over = earlier.over if earlier else self._memory().read(ref.path)
            self._runs().mark(
                FILES, _key(session, ref), json.dumps({"text": text, "over": over})
            )
        self._changed(session)

    def entries(self, team: TeamRef, name: str) -> list[LogEntry]:
        return self._logs.entries(team, name) if self._logs else []

    def append(self, team: TeamRef, name: str, entry: LogEntry) -> None:
        if self._logs is None:
            raise NotSupported("This store keeps no logs")
        self._logs.append(team, name, entry)


def _key(session: str, ref: DocumentRef) -> str:
    return f"{session}\n{ref.path}"


class Logs(Protocol):
    """Where what teams add to their logs is kept."""

    def entries(self, team: TeamRef, name: str) -> list[LogEntry]: ...

    def append(self, team: TeamRef, name: str, entry: LogEntry) -> None: ...
