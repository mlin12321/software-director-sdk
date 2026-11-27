"""What a session keeps to itself until its work is accepted: what became
of its teams, what its agents recorded and wrote to each other, the teams
it removed, and what was said and passed to its teams."""

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter

from openhands.agent_server.director.models import (
    FormerTeam,
    FormerThread,
    Report,
    RoutedRequest,
    TeamEvent,
    ThreadMessage,
)
from openhands.agent_server.director.run_state import (
    EVENTS,
    FORMER,
    HANDED,
    REPORTS,
    ROUTED,
    SENT,
    THREADS,
    RunState,
)


_REPORTS = TypeAdapter(list[Report])
_EVENTS = TypeAdapter(list[TeamEvent])
_MESSAGES = TypeAdapter(list[ThreadMessage])


def thread_name(writer_node_id: str, ic_node_id: str, ic_to_ic: bool) -> str:
    """What names the messages of one agent to another."""
    kind = ".threads-ic" if ic_to_ic else ".threads"
    return f"{kind}/{writer_node_id}/{ic_node_id}.json"


@dataclass
class SessionDraftService:
    """Keeps what the sessions open on this machine have not had accepted
    yet: their drafts of what the org will have on record.

    None of it is the org's, and none of it is a lasting log: the director
    makes it the org's when a session's work is accepted, and drops it
    when a session is closed. It decides neither; it reads, adds and
    forgets. Safe from any thread.
    """

    run_state: Callable[[], RunState]
    """Where the drafts are kept."""
    # Held while one record is read and added to.
    _keeping: threading.RLock = field(default_factory=threading.RLock, init=False)

    @contextmanager
    def unchanged(self) -> Iterator[None]:
        """Nothing is added to a record meanwhile: for whoever reads the
        records, does something with them and forgets them."""
        with self._keeping:
            yield

    def events(self, session: str, team: str) -> list[TeamEvent]:
        """What a session kept of what became of the teams below a team,
        oldest first."""
        kept = self.run_state().marked(EVENTS).get(f"{session}\n{team}")
        return _EVENTS.validate_json(kept) if kept else []

    def events_by_team(
        self, session: str | None = None
    ) -> dict[tuple[str, str], list[TeamEvent]]:
        """The events kept, by the session and the team they were kept
        for; with ``session``, that session's alone."""
        found = {}
        for key, raw in self.run_state().marked(EVENTS).items():
            of, _, team = key.partition("\n")
            if session is None or of == session:
                found[(of, team)] = _EVENTS.validate_json(raw)
        return found

    def keep_event(self, session: str, team: str, event: TeamEvent) -> None:
        with self._keeping:
            events = [*self.events(session, team), event]
            self.run_state().mark(
                EVENTS, f"{session}\n{team}", _EVENTS.dump_json(events).decode()
            )

    def forget_events(self, session: str, team: str) -> None:
        self.run_state().unmark(EVENTS, f"{session}\n{team}")

    def routed(self, session: str | None = None) -> list[RoutedRequest]:
        """The requests that went straight to a team; with ``session``,
        that session's alone."""
        return [
            RoutedRequest.model_validate_json(raw)
            for key, raw in self.run_state().marked(ROUTED).items()
            if session is None or key.startswith(f"{session}\n")
        ]

    def keep_routed(self, session: str, request: RoutedRequest) -> None:
        self.run_state().mark(
            ROUTED, f"{session}\n{request.id}", request.model_dump_json()
        )

    def forget_routed(self, session: str, request: RoutedRequest) -> None:
        self.run_state().unmark(ROUTED, f"{session}\n{request.id}")

    def reports(self, place_id: str) -> list[Report]:
        """What the agents of a session's place recorded, oldest first."""
        kept = self.run_state().marked(REPORTS).get(place_id)
        return _REPORTS.validate_json(kept) if kept else []

    def keep_report(self, place_id: str, report: Report) -> None:
        """Add to what the agents of a place recorded; a report that has
        no id is given one."""
        named = report.model_copy(update={"id": report.id or uuid4().hex})
        with self._keeping:
            reports = [*self.reports(place_id), named]
            self.run_state().mark(
                REPORTS, place_id, _REPORTS.dump_json(reports).decode()
            )

    def forget_reports(self, place_id: str) -> None:
        self.run_state().unmark(REPORTS, place_id)

    def former(self, session: str) -> list[FormerTeam]:
        """The records of the teams a session removed."""
        if not session:
            return []
        return [
            FormerTeam.model_validate_json(raw)
            for key, raw in self.run_state().marked(FORMER).items()
            if key.startswith(f"{session}\n")
        ]

    def keep_former(self, session: str, record: FormerTeam) -> None:
        self.run_state().mark(
            FORMER, f"{session}\n{record.id}\n{record.at}", record.model_dump_json()
        )

    def forget_former(self, session: str, record: FormerTeam) -> None:
        self.run_state().unmark(FORMER, f"{session}\n{record.id}\n{record.at}")

    def thread(self, name: str) -> list[ThreadMessage]:
        """What was written in the thread kept by ``name``
        (:func:`thread_name`), oldest first."""
        kept = self.run_state().marked(THREADS).get(name)
        return _MESSAGES.validate_json(kept) if kept else []

    def keep_message(self, name: str, message: ThreadMessage) -> None:
        with self._keeping:
            messages = [*self.thread(name), message]
            self.run_state().mark(THREADS, name, _MESSAGES.dump_json(messages).decode())

    def take_threads(self, place_id: str) -> list[FormerThread]:
        """Take out what the agents at a place wrote and were written, so
        that a later team at the same place does not find it."""
        run = self.run_state()
        taken = [
            FormerThread(file=name, messages=_MESSAGES.validate_json(raw))
            for name, raw in run.marked(THREADS).items()
            if place_id in (Path(name).parent.name, Path(name).stem)
        ]
        for thread in taken:
            run.unmark(THREADS, thread.file)
        return taken

    def passed_to(self, place_id: str) -> list[tuple[str, str]]:
        """The documents other teams passed to a place to read: the
        folder and the name of each."""
        kept = self.run_state().marked(SENT).get(place_id)
        return [(folder, name) for folder, name in json.loads(kept)] if kept else []

    def keep_passed(self, place_id: str, folder: str, name: str) -> None:
        """Keep that a document was passed to a place, once."""
        with self._keeping:
            passed = self.passed_to(place_id)
            if (folder, name) not in passed:
                self.run_state().mark(
                    SENT, place_id, json.dumps([*passed, (folder, name)])
                )

    def forget_passed(self, place_id: str) -> None:
        self.run_state().unmark(SENT, place_id)

    def handed_over_to(self, place_id: str) -> list[dict[str, str]]:
        """What the team above said to a place of the teams that were
        there before it: who said it (``by``), when (``at``) and the
        ``text``."""
        kept = self.run_state().marked(HANDED).get(place_id)
        return json.loads(kept) if kept else []

    def keep_handed_over(self, place_id: str, entry: dict[str, str]) -> None:
        with self._keeping:
            said = self.handed_over_to(place_id)
            self.run_state().mark(HANDED, place_id, json.dumps([*said, entry]))

    def forget_handed_over(self, place_id: str) -> None:
        self.run_state().unmark(HANDED, place_id)

    def forget_session(self, session: str) -> None:
        """Forget what was kept by a session and not by one of its
        places: its events, its routed requests and the teams it removed."""
        run = self.run_state()
        for kind in (EVENTS, ROUTED, FORMER):
            for key in run.marked(kind):
                if key.startswith(f"{session}\n"):
                    run.unmark(kind, key)
