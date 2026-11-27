"""What the service reads and keeps of the teams: their context, their
records and what they needed of each other."""

from functools import cached_property
from typing import NamedTuple

from openhands.agent_server.director.documents.store import REPORT, DocumentRef, TeamRef
from openhands.agent_server.director.documents.team_documents import RemovedTeam
from openhands.agent_server.director.git_store import (
    normalize_store_path,
)
from openhands.agent_server.director.models import (
    LeftoverDocument,
    OrgNode,
    Report,
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    is_division_lead,
    session_of,
    team_ref,
    utc_now,
)
from openhands.agent_server.director.recovery import STEP_NAMES
from openhands.agent_server.director.run_state import (
    USE,
    Intent,
)
from openhands.agent_server.director.service.agent_start_service import REASON_KEPT
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
)
from openhands.agent_server.director.service.session_landing_service import (
    SessionLandingService,
)
from openhands.agent_server.director.service.state import (
    ServiceState,
    _instant,
)
from openhands.agent_server.director.service.team_context_service import (
    TeamContextService,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.sdk.errors import NotFound, NotSupported
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


def _directory_of(file: str) -> str:
    """The folder a document's file of team memory is of."""
    return file.rpartition("/")[0]


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


class Records(ServiceState):
    """Reads and keeps what is known of the teams."""

    @cached_property
    def session_landing(self) -> SessionLandingService:
        """What makes a session's records the org's once its work is
        accepted."""
        return SessionLandingService(
            org=lambda: self.org,
            run_state=self.org_state.run_state,
            team_memory=self._memory,
            documents=self.documents,
            team_types=self.team_types,
            session_drafts=self.session_drafts,
            org_save=self.org_save,
            event=self.team_history.event,
        )

    @cached_property
    def team_history(self) -> TeamHistoryService:
        """The history of the org's teams."""
        return TeamHistoryService(
            org=lambda: self.org,
            team_memory=self._memory,
            require_team_memory=lambda: self.team_memory,
            session_drafts=self.session_drafts,
            org_changed=self._note_change,
        )

    @cached_property
    def team_context(self) -> TeamContextService:
        """What each agent reads of its place."""
        return TeamContextService(
            org=lambda: self.org,
            org_state=self.org_state,
            org_settings=self.org_settings,
            team_types=self.team_types,
            team_history=self.team_history,
            session_drafts=self.session_drafts,
            documents=self.documents,
            team_documents=lambda: self.team_documents,
        )

    @cached_property
    def check_counts(self) -> CheckCountService:
        """What a Manager's checks count."""
        return CheckCountService(
            org=lambda: self.org, run_state=self.org_state.run_state
        )

    def _keep_failure(self, intent: Intent, error: Exception) -> None:
        """Keep a step that failed where the team above and the human read
        what became of a team."""
        team = self.org.place(intent.node_id)
        if team is None:
            return
        what = STEP_NAMES.get(intent.kind, intent.kind)
        self.team_history.record_event(
            team.parent_id or team.id,
            "failed",
            team,
            f"{what}: not done. {type(error).__name__}: {error}"[:REASON_KEPT],
        )

    def _note_change(self, session: str) -> None:
        """Remember that a session whose work is accepted has changed
        something since: the org takes it only when it is submitted again."""
        top = next(iter(self.org.session(session)), None)
        if top is not None and top.review is not None:
            if top.review.state == "accepted" and self.org.opens_session(top):
                self._changed_since_accept.setdefault(session, False)

    def reports_of(self, node: OrgNode) -> list[Report]:
        """What the two agents of a team recorded, oldest first. Reports
        are the team's: an agent started again reads what it wrote before."""
        team_memory = self._memory()
        reports = (
            team_memory.reports(node.boundary.path) if self.org.is_orgs(node) else []
        )
        own = self.session_drafts.reports(node.id)
        return sorted([*reports, *own], key=lambda report: report.at)

    def entries(self, team: TeamRef, name: str) -> list[Report]:
        """The one log the director keeps for a team, as the store of
        documents is asked for it: what its members recorded."""
        place = self.org.place(team.id)
        return self.reports_of(place) if place and name == REPORT else []

    def append(self, team: TeamRef, name: str, entry: Report) -> None:
        """Add to what the members of a team recorded, as the store of
        documents is asked to.

        Raises:
            NotSupported: It is another log than the one the director keeps.
            NotFound: The team is no place of a session.
        """
        if name != REPORT:
            raise NotSupported(f"The director keeps no log named {name!r}")
        place = self.org.place(team.id)
        if place is None:
            raise NotFound(f"There is no team {team.id}")
        said_where = {
            "session": self.org.session_title(place),
            "session_id": session_of(place),
        }
        # Its session's alone until the session's work is accepted.
        self.session_drafts.keep_report(
            place.id,
            entry.model_copy(update={"at": entry.at or _instant(), **said_where}),
        )

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

    def _draft(self, session: str, file: str, text: str | None) -> None:
        """Keep, for a session alone, what it makes of a file of team
        memory."""
        _ = self.team_memory  # refuses where the org has no team memory
        ref = DocumentRef.at(file)
        if text is None:
            self.documents.remove(session, ref)
        else:
            self.documents.write(session, ref, text)

    def _document_changed(self, session: str) -> None:
        self._note_change(session)

    def _undraft(self, session: str, file: str) -> None:
        self.documents.discard(session, DocumentRef.at(file))

    def read_file(self, node: OrgNode, file: str) -> str | None:
        """A file of team memory as the agents of ``node`` have it: what
        their session wrote there, else the org's."""
        return self.documents.read(session_of(node), DocumentRef.at(file))

    def _documents_left_in(
        self, place: SessionTeam, path: str, removed: list[SessionTeam]
    ) -> list[LeftoverDocument]:
        """The documents a team that is given ``path`` in the session of
        ``place`` finds there, with the ``removed`` teams gone."""
        return self.team_documents.left_in(
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
        store = self.org
        session = store.session_of(top)
        if session is None:
            return []
        began = {team for team, *_ in session.shape}
        memory = self._memory()
        found = []
        for place in store.session(top.session):
            if (
                is_division_lead(place)
                or place.team_id not in began
                or store.team(place.team_id) is not None
            ):
                continue
            home = store.team(self.session_landing.home_of(top, place.boundary.path))
            if home is None:
                continue
            path = normalize_store_path(home.boundary.path)
            documents = self.team_documents
            for name, protocol in documents.protocols.items():
                draft = documents.draft(team_ref(place), name)
                if draft is None or not protocol.stays_when_team_is_removed:
                    continue
                now = memory.read(DocumentRef(path, name).path) or ""
                found.append(Inherited(name, path, now, place.title, draft[0]))
        return found

    def _leave(self, top: SessionTeam) -> str:
        """Drop what a session that is being closed kept to itself: what
        it wrote for work that was never accepted goes with it.

        Returns:
            What went with the session, in words; '' when nothing did.
        """
        store = self.org
        run = self.org_state.run_state()
        session = top.session
        places = store.session(session)
        reports = sum(len(self.session_drafts.reports(place.id)) for place in places)
        files = sorted(self.drafts(session))
        records = self.session_drafts.former(session)
        for place in places:
            self.session_drafts.forget_reports(place.id)
            self.session_drafts.forget_passed(place.id)
            self.session_drafts.forget_handed_over(place.id)
            self._drop_merge_checkout(place.id)
        for file in files:
            self._undraft(session, file)
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
