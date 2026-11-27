"""Setting up, replacing and pruning teams."""

from functools import cached_property

from openhands.agent_server.director.service.team_split_service import TeamSplitService
from openhands.agent_server.director.service.watch import Watch
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class Splits(Watch):
    """Builds what sets up, replaces and prunes teams."""

    @cached_property
    def team_splits(self) -> TeamSplitService:
        """What carries out what was approved or decided about teams."""
        return TeamSplitService(
            org=lambda: self.org,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.org_settings,
            team_types=self.team_types,
            team_history=self.team_history,
            team_context=self.team_context,
            team_reports=self.team_reports,
            team_summaries=self.team_summaries,
            team_messaging=self.team_messaging,
            message_delivery=self.message_delivery,
            agent_start=self.agent_start,
            check_counts=self.check_counts,
            confinement=self.confinement,
            session_documents=self.session_documents,
            session_drafts=self.session_drafts,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            team_documents=lambda: self.team_documents,
            launcher=self._require_launcher,
            is_running=lambda: self._loop is not None,
            review_instructions=self.review_instructions,
            retry=self._run_limit_retry,
        )
