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
from typing import NamedTuple

from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.run_state import FILES, RunState


class DocRef(NamedTuple):
    """One document: its name, in the folder of the team that keeps it."""

    folder: str
    name: str

    @property
    def path(self) -> str:
        """Where it is kept, from the top of the store."""
        file = f"{self.name}.md"
        return f"{self.folder}/{file}" if self.folder else file

    @classmethod
    def at(cls, path: str) -> "DocRef":
        """The document kept at ``path``."""
        folder, _, file = path.rpartition("/")
        return cls(folder, file.removesuffix(".md"))


class Draft(NamedTuple):
    """What a session made of a document, which is its own until its work
    is accepted."""

    text: str | None
    """None where the session removed the document."""
    over: str | None
    """What the org had when the session first wrote it; None for nothing."""


class DocumentStore(ABC):
    """Keeps the documents of teams. Safe from any thread."""

    @abstractmethod
    def read(self, session: str, ref: DocRef) -> str | None:
        """A document as ``session`` has it: what it wrote, else the
        org's. '' for the session reads the org's. None where there is
        none."""

    @abstractmethod
    def write(self, session: str, ref: DocRef, text: str) -> None:
        """Put ``text`` in the place of a document, for ``session`` alone
        until its work is accepted."""

    @abstractmethod
    def remove(self, session: str, ref: DocRef) -> None:
        """Take a document away, for ``session`` alone until its work is
        accepted."""

    @abstractmethod
    def discard(self, session: str, ref: DocRef) -> None:
        """Forget what ``session`` made of a document: the org's holds
        for it again."""

    @abstractmethod
    def drafts(self, session: str) -> dict[DocRef, Draft]:
        """Every document ``session`` wrote or removed and the org has not
        taken yet."""


class GitDocumentStore(DocumentStore):
    """Documents in the org's team memory, a hidden ref of its git
    repository; what sessions made of them is kept beside the director's
    unfinished work until it is accepted."""

    def __init__(
        self, memory: Callable[[], GitStore], runs: Callable[[], RunState]
    ) -> None:
        """
        Args:
            memory: The org's team memory, as it is when asked.
            runs: Where the director keeps its unfinished work.
        """
        self._memory = memory
        self._runs = runs
        self._lock = threading.RLock()

    def read(self, session: str, ref: DocRef) -> str | None:
        draft = self.drafts(session).get(ref)
        return draft.text if draft else self._memory().read(ref.path)

    def write(self, session: str, ref: DocRef, text: str) -> None:
        self._keep(session, ref, text)

    def remove(self, session: str, ref: DocRef) -> None:
        self._keep(session, ref, None)

    def discard(self, session: str, ref: DocRef) -> None:
        self._runs().unmark(FILES, _key(session, ref))

    def drafts(self, session: str) -> dict[DocRef, Draft]:
        if not session:
            return {}
        prefix = f"{session}\n"
        found = {}
        for key, raw in self._runs().marked(FILES).items():
            if key.startswith(prefix):
                draft = json.loads(raw)
                found[DocRef.at(key.removeprefix(prefix))] = Draft(
                    draft["text"], draft["over"]
                )
        return found

    def _keep(self, session: str, ref: DocRef, text: str | None) -> None:
        # What it is written over is what the session first wrote over, so
        # that what the org made of it since is not lost sight of.
        with self._lock:
            earlier = self.drafts(session).get(ref)
            over = earlier.over if earlier else self._memory().read(ref.path)
            self._runs().mark(
                FILES, _key(session, ref), json.dumps({"text": text, "over": over})
            )


def _key(session: str, ref: DocRef) -> str:
    return f"{session}\n{ref.path}"
