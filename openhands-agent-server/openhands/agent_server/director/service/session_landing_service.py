"""Landing: what a session kept to itself becomes the org's once its
work is accepted, as one commit of team memory."""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import NamedTuple

from openhands.agent_server.director.documents.store import DocumentRef, DocumentStore
from openhands.agent_server.director.environment.records import (
    Landing,
    RecordStore,
    normalize_store_path,
)
from openhands.agent_server.director.models import (
    FormerTeam,
    LeftoverDocument,
    Report,
    SessionTeam,
    TeamEvent,
    TeamEventKind,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    is_division_lead,
    utc_now,
)
from openhands.agent_server.director.run_state import TASK, RunState
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.split import home_of_lost


def _directory_of(file: str) -> str:
    """The folder a document's file of team memory is of."""
    return file.rpartition("/")[0]


class Taken(NamedTuple):
    """What became of a session's records when its work was accepted."""

    grown: list[tuple[str, str]]
    # The files that were not written: someone else changed them meanwhile.
    left: list[str]


@dataclass
class SessionLandingService:
    """Makes what a session kept to itself the org's.

    It reads what the session drafted (documents, reports, events, the
    records of teams it removed, routed requests), works out where each
    goes in the org as it is now, has it committed with or without the
    tree, and forgets what landed. It decides nothing about whether work
    is accepted or which teams wait: it is told.
    """

    org: Callable[[], OrgStore]
    """The org a session's work lands in."""
    run_state: Callable[[], RunState]
    """Where the task each team was given is kept."""
    team_memory: Callable[[], RecordStore]
    """The team memory that is landed in, for the text a file has now."""
    documents: DocumentStore
    """Where the documents a session wrote wait."""
    team_types: TeamTypeService
    """What each document does: whether it stays when its team is gone."""
    session_drafts: SessionDraftService
    """What the session recorded and has not had accepted."""
    org_save: OrgSaveService
    """What takes a landing into team memory, with the tree or without."""
    event: Callable[[TeamEventKind, SessionTeam, str], TeamEvent]
    """Makes the event that is recorded of a team."""

    def home_of(self, top: SessionTeam, path: str) -> str:
        """The team of the org that has what a session's team the org does
        not have left: the nearest one that holds its directory."""
        store = self.org()
        session = store.session_of(top)
        return home_of_lost(
            path, store.teams(), session.above_id if session else top.team_id
        )

    def _drafts(self, session: str) -> dict[str, tuple[str | None, str | None]]:
        """The files a session wrote: the text of each (None where it
        removed the file) and the text it was written over."""
        return {
            ref.path: (draft.text, draft.over)
            for ref, draft in self.documents.drafts(session).items()
        }

    def _stays_when_team_is_removed(self, name: str) -> bool:
        protocol = self.team_types.protocols.get(name)
        return protocol is not None and protocol.stays_when_team_is_removed

    def land(
        self,
        top: SessionTeam,
        seen: dict[str, str | None] | None = None,
        chosen: dict[str, str | None] | None = None,
        *,
        waits: Collection[str] = (),
        held: Collection[str] = (),
        tree: bool = False,
    ) -> Taken:
        """Make what a session kept to itself the org's, now that its work
        is accepted: the documents it wrote, what its agents
        recorded and the records of the teams it removed. All of it is one
        commit of team memory.

        What is of a team that waits stays the session's until then. What
        is of a team the org does not have, or has another team in place
        of, is kept as that team's record with the nearest team of the org
        that holds its directory. A document is put in place of one that
        changed only where its reviewer was shown that one.

        Args:
            seen: The documents the reviewer was shown, by file.
            chosen: The documents that waited with the teams and are
                settled, by file: the text that holds.
            waits: The teams whose records stay the session's for now.
            held: The files that stay the session's for now.
            tree: Whether the org tree as it is now lands with it.

        Returns:
            The records that grew and the files that were not written.
        """
        # Nothing is written to the session's records between reading
        # them and forgetting them: it would be lost.
        with self.session_drafts.unchanged():
            return self._land(top, seen, chosen, waits=waits, held=held, tree=tree)

    def _land(
        self,
        top: SessionTeam,
        seen: dict[str, str | None] | None,
        chosen: dict[str, str | None] | None,
        *,
        waits: Collection[str],
        held: Collection[str],
        tree: bool,
    ) -> Taken:
        store = self.org()
        run = self.run_state()
        session = top.session
        review = store.place(top.parent_id or "")
        held = set(held) - set(chosen or {})
        places = [
            place for place in store.session(session) if not is_division_lead(place)
        ]
        by_path = {normalize_store_path(p.boundary.path): p for p in places}
        stamp = utc_now()
        left = {
            place.id: place
            for place in places
            if place.team_id not in waits and not store.is_orgs(place)
        }
        files: dict[str, tuple[str | None, str | None]] = {}
        passed_on: dict[str, SessionTeam] = {}
        of_left: dict[str, dict[str, str]] = {}
        drafts = self._drafts(session)
        for file, text in (chosen or {}).items():
            if file not in drafts:
                # What came with a team: there is no draft of it.
                files[file] = (text, self.team_memory().read(file))
        for file, (text, over) in drafts.items():
            owner = by_path.get(_directory_of(file))
            if file in held or (owner is not None and owner.team_id in waits):
                continue
            if owner is not None and owner.id in left:
                kind = DocumentRef.at(file).name
                kept_of_it = of_left.setdefault(owner.id, {})
                if text:
                    kept_of_it[kind] = text
                if (
                    self._stays_when_team_is_removed(kind)
                    and text
                    and store.team(owner.team_id) is None
                    and all(
                        normalize_store_path(team.boundary.path) != _directory_of(file)
                        for team in store.teams()
                    )
                ):
                    # No team has the directory: the document stays there
                    # as an old one of the team that holds it now.
                    files[file] = (text, over)
                    passed_on[file] = owner
                continue
            if chosen is not None and file in chosen:
                files[file] = (chosen[file], self.team_memory().read(file))
            else:
                files[file] = (text, over)
        # By the folder of each team, where what it recorded is kept.
        reports: dict[str, list[Report]] = {}
        for place in places:
            if place.team_id not in waits and place.id not in left:
                at = normalize_store_path(place.boundary.path)
                reports[at] = [
                    *reports.get(at, []),
                    *self.session_drafts.reports(place.id),
                ]
        if review is not None:
            # What its reviewers recorded is the team's they review from.
            home = (
                review
                if store.is_orgs(review)
                else store.team(self.home_of(top, review.boundary.path))
            )
            at = normalize_store_path((home or review).boundary.path)
            reports[at] = sorted(
                [*reports.get(at, []), *self.session_drafts.reports(review.id)],
                key=lambda report: report.at,
            )
        not_taken = [
            FormerTeam(
                id=place.team_id,
                title=place.title,
                above=self.home_of(top, place.boundary.path),
                path=place.boundary.path,
                purpose=run.marked(TASK).get(place.id, place.summary),
                ended="not_taken",
                at=stamp,
                review=place.review.state if place.review else None,
                reports=self.session_drafts.reports(place.id),
                compacted_report="\n\n".join(
                    r.text for r in self.session_drafts.reports(place.id)
                ),
                documents=of_left.get(place.id, {}),
                session=store.session_title(place),
                since=place.since,
            )
            for place in left.values()
            if self.session_drafts.reports(place.id) or place.id in of_left
        ]
        removed = [
            record
            for record in self.session_drafts.former(session)
            if record.id not in waits
        ]
        former, forget, forget_reports = [], [], []
        for record in removed:
            team = store.team(record.id)
            if team is not None and team.since == record.since:
                # The org kept the team the session removed.
                former.append(record.model_copy(update={"ended": "not_taken"}))
            else:
                former.append(record)
                ours = next((p for p in places if p.team_id == record.id), None)
                if team is None or (ours is not None and ours.since == team.since):
                    forget.append(record.id)
                    forget_reports.append(record.path)
        of_team = {place.team_id: place for place in places}
        events: dict[str, list[TeamEvent]] = {}
        kept_events: list[str] = []
        for (_, kept_for), kept in self.session_drafts.events_by_team(session).items():
            team_id = kept_for
            place = of_team.get(team_id)
            if team_id in waits:
                continue
            if place is not None and place.id in left:
                team_id = self.home_of(top, place.boundary.path)
            elif store.team(team_id) is None:
                team_id = self.home_of(top, top.boundary.path)
            events.setdefault(team_id, []).extend(
                event.model_copy(update={"unaccepted": False}) for event in kept
            )
            kept_events.append(kept_for)
        for record in not_taken:
            place = next(p for p in left.values() if p.team_id == record.id)
            events.setdefault(record.above, []).append(
                self.event(
                    "not_taken",
                    place,
                    "what its session wrote for it is kept in its record",
                ).model_copy(update={"unaccepted": False})
            )
        routed = self.session_drafts.routed(session)
        conflicts = self.org_save.commit_landing(
            Landing(
                files=files,
                seen=seen or {},
                reports={team: kept for team, kept in reports.items() if kept},
                former=[*former, *not_taken],
                forget=forget,
                forget_reports=forget_reports,
                events=events,
                routed=routed,
            ),
            f"memory: the accepted work of session {session}",
            tree,
        )
        for kept_for in kept_events:
            self.session_drafts.forget_events(session, kept_for)
        for request in routed:
            self.session_drafts.forget_routed(session, request)
        if review is not None:
            self.session_drafts.forget_reports(review.id)
            self.session_drafts.forget_passed(review.id)
        for file in files:
            if file in conflicts:
                continue
            self.documents.discard(session, DocumentRef.at(file))
            if file in passed_on:
                lost = passed_on[file]
                home = store.team(self.home_of(top, lost.boundary.path))
                name = DocumentRef.at(file).name
                rewritten = (
                    DocumentRef(normalize_store_path(home.boundary.path), name).path
                    if home
                    else ""
                )
                directory = _directory_of(file)
                if (
                    home is not None
                    and not (
                        files.get(rewritten, (None,))[0] and rewritten not in conflicts
                    )
                    and all(
                        (old.name, old.path) != (name, directory)
                        for old in home.leftover_documents
                    )
                ):
                    store.update_team(
                        home.id,
                        leftover_documents=[
                            *home.leftover_documents,
                            LeftoverDocument(
                                name=name, path=directory, team=lost.title
                            ),
                        ],
                    )
                continue
            owner = by_path.get(_directory_of(file))
            name = DocumentRef.at(file).name
            if (
                owner is not None
                and self._stays_when_team_is_removed(name)
                and files[file][0] is not None
            ):
                # The team has one of its own now: what teams before it
                # left of that document is no longer its guide.
                team = store.team(owner.team_id)
                store.update_team(
                    owner.team_id,
                    summary=owner.summary,
                    leftover_documents=[
                        old
                        for old in (team.leftover_documents if team else [])
                        if old.name != name
                    ],
                )
        for place in left.values():
            for file in self._drafts(session):
                # What waits for the org's part of the landing stays a draft.
                if by_path.get(_directory_of(file)) is place and file not in held:
                    self.documents.discard(session, DocumentRef.at(file))
        for place in places:
            if place.team_id not in waits:
                self.session_drafts.forget_reports(place.id)
                self.session_drafts.forget_passed(place.id)
        for record in removed:
            self.session_drafts.forget_former(session, record)
        grown: list[tuple[str, str]] = [
            (team, role)
            for team, kept in reports.items()
            for role in dict.fromkeys(report.role for report in kept)
        ]
        return Taken(grown, sorted(conflicts))
