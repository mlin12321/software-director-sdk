"""What a session wrote of team memory, which is its own until its work
is accepted, and what it leaves when it is closed."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from openhands.agent_server.director.documents.store import DocumentRef, DocumentStore
from openhands.agent_server.director.documents.team_documents import (
    RemovedTeam,
    TeamDocuments,
)
from openhands.agent_server.director.environment.records import (
    RecordStore,
    normalize_store_path,
)
from openhands.agent_server.director.models import (
    LeftoverDocument,
    OrgNode,
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    is_division_lead,
    session_of,
    team_ref,
    utc_now,
)
from openhands.agent_server.director.run_state import (
    USE,
    RunState,
)
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)


class Inherited(NamedTuple):
    """A document a session wrote for a team the org no longer has."""

    name: str
    # The folder of the team of the org that holds the lost team's now.
    home: str
    # That team's own document of the name, as the org has it.
    now: str
    # What the lost team was called.
    lost: str
    written: str


@dataclass
class SessionDocumentService:
    """Reads and keeps the documents of a session, and drops what a
    session that is closed kept to itself."""

    org: Callable[[], OrgStore]
    """The org the sessions work in."""
    team_memory: Callable[[], RecordStore]
    """The org's team memory, as it is when asked."""
    require_team_memory: Callable[[], RecordStore]
    """The same, raising ``Unavailable`` where the org has no team memory."""
    run_state: Callable[[], RunState]
    """Where what a session used is counted."""
    documents: DocumentStore
    """Where the documents are read and written."""
    team_documents: Callable[[], TeamDocuments]
    """The documents of the org as each team reads them."""
    team_types: TeamTypeService
    """What the members of a team are called."""
    session_drafts: SessionDraftService
    """What a session keeps until its work is accepted."""
    home_of: Callable[[SessionTeam, str], str]
    """The team of the org that holds a folder once a session lands
    (``SessionLandingService.home_of``)."""
    drop_merge_checkout: Callable[[str], None]
    """Discards the merge of a team's work that its reviewer finishes."""

    def drafts(self, session: str) -> dict[str, tuple[str | None, str | None]]:
        """The files of team memory a session wrote, which are its own
        until its work is accepted.

        Returns:
            For each file, its text (None where the session removed it)
            and the text it was written over (None where there was none).
        """
        return {
            ref.path: (draft.text, draft.over)
            for ref, draft in self.documents.drafts(session).items()
        }

    def draft(self, session: str, file: str, text: str | None) -> None:
        """Keep, for a session alone, what it makes of a file of team
        memory."""
        self.require_team_memory()  # refuses where the org has no team memory
        ref = DocumentRef.at(file)
        if text is None:
            self.documents.remove(session, ref)
        else:
            self.documents.write(session, ref, text)

    def undraft(self, session: str, file: str) -> None:
        self.documents.discard(session, DocumentRef.at(file))

    def read_file(self, node: OrgNode, file: str) -> str | None:
        """A file of team memory as the agents of ``node`` have it: what
        their session wrote there, else the org's."""
        return self.documents.read(session_of(node), DocumentRef.at(file))

    def documents_left_in(
        self, place: SessionTeam, path: str, removed: list[SessionTeam]
    ) -> list[LeftoverDocument]:
        """The documents a team that is given ``path`` in the session of
        ``place`` finds there, with the ``removed`` teams gone."""
        return self.team_documents().left_in(
            place.session,
            path,
            [
                RemovedTeam(team.boundary.path, team.title, team.leftover_documents)
                for team in removed
            ],
        )

    def documents_inherited(self, top: SessionTeam) -> list[Inherited]:
        """The documents a session wrote for teams the org had when it
        began and has no more, of the kind that stays when a team is
        removed: each is left with the team that holds the folder now."""
        store = self.org()
        session = store.session_of(top)
        if session is None:
            return []
        began = {team for team, *_ in session.shape}
        memory = self.team_memory()
        found = []
        for place in store.session(top.session):
            if (
                is_division_lead(place)
                or place.team_id not in began
                or store.team(place.team_id) is not None
            ):
                continue
            home = store.team(self.home_of(top, place.boundary.path))
            if home is None:
                continue
            path = normalize_store_path(home.boundary.path)
            documents = self.team_documents()
            for name, protocol in documents.protocols.items():
                draft = documents.draft(team_ref(place), name)
                if draft is None or not protocol.stays_when_team_is_removed:
                    continue
                now = memory.read(DocumentRef(path, name).path) or ""
                found.append(Inherited(name, path, now, place.title, draft[0]))
        return found

    def leave(self, top: SessionTeam) -> str:
        """Drop what a session that is being closed kept to itself: what
        it wrote for work that was never accepted goes with it.

        Returns:
            What went with the session, in words; '' when nothing did.
        """
        store = self.org()
        run = self.run_state()
        session = top.session
        places = store.session(session)
        reports = sum(len(self.session_drafts.reports(place.id)) for place in places)
        files = sorted(self.drafts(session))
        records = self.session_drafts.former(session)
        for place in places:
            self.session_drafts.forget_reports(place.id)
            self.session_drafts.forget_passed(place.id)
            self.session_drafts.forget_handed_over(place.id)
            self.drop_merge_checkout(place.id)
        for file in files:
            self.undraft(session, file)
        self.session_drafts.forget_session(session)
        for place in places:
            self.session_drafts.take_threads(place.id)
        run.unmark(USE, session)
        return ", ".join(
            said
            for said in (
                f"{reports} reports" if reports else "",
                ", ".join(files),
                f"the records of {len(records)} teams it removed" if records else "",
            )
            if said
        )

    def removed_team_handed_over(
        self, place: SessionTeam, above: SessionTeam, member: str, text: str
    ) -> None:
        """Keep what a member of the team above says to ``place`` of the
        teams that were there before it. The team is told with its next
        message, or in its briefing, and no document is written by this.
        Safe from any thread."""
        self.session_drafts.keep_handed_over(
            place.id,
            {
                "by": (
                    f"the {self.team_types.team(above).title(member)} of {above.title}"
                ),
                "at": utc_now(),
                "text": text,
            },
        )
