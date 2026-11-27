"""What the service reads and keeps of the teams: their context, their
records and what they needed of each other."""

from functools import cached_property

from openhands.agent_server.director.recovery import STEP_NAMES
from openhands.agent_server.director.run_state import (
    Intent,
)
from openhands.agent_server.director.service.agent_start_service import REASON_KEPT
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
)
from openhands.agent_server.director.service.session_document_service import (
    SessionDocumentService,
)
from openhands.agent_server.director.service.session_landing_service import (
    SessionLandingService,
)
from openhands.agent_server.director.service.state import (
    ServiceState,
)
from openhands.agent_server.director.service.team_context_service import (
    TeamContextService,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


def _directory_of(file: str) -> str:
    """The folder a document's file of team memory is of."""
    return file.rpartition("/")[0]


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

    @cached_property
    def session_documents(self) -> SessionDocumentService:
        """What a session wrote of team memory and what it leaves."""
        return SessionDocumentService(
            org=lambda: self.org,
            team_memory=self._memory,
            require_team_memory=lambda: self.team_memory,
            run_state=self.org_state.run_state,
            documents=self.documents,
            team_documents=lambda: self.team_documents,
            team_types=self.team_types,
            session_drafts=self.session_drafts,
            home_of=self.session_landing.home_of,
            drop_merge_checkout=self._drop_merge_checkout,
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
