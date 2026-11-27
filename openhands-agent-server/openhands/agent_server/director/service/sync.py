"""An org that is paused, comes back, is pulled, pushed or merged."""

from functools import cached_property

from openhands.agent_server.director.service.org_run_service import OrgRunService
from openhands.agent_server.director.service.org_sync_service import OrgSyncService
from openhands.agent_server.director.service.sessions import Sessions
from openhands.agent_server.director.service.state import (
    lane_for,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class Sync(Sessions):
    """Builds what runs the org and what moves it between machines."""

    @cached_property
    def org_run(self) -> OrgRunService:
        """What pauses and resumes the org and follows its agents."""
        return OrgRunService(
            checkouts=self.checkouts,
            org=lambda: self.org,
            find_org=lambda: self._store,
            repo_dir=lambda: self._repo_dir,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.org_settings,
            start_line=self.start_line,
            message_delivery=self.message_delivery,
            team_messaging=self.team_messaging,
            team_types=self.team_types,
            agent_start=self.agent_start,
            confinement=self.confinement,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            team_splits=self.team_splits,
            launcher=self._require_launcher,
            find_launcher=lambda: self._launcher,
            loop=lambda: self._loop,
            home=self._home,
            instance=self._instance,
            working=self._working,
            lane_of=lane_for,
            task_of=self._task_of,
            reviews=self.reviews,
            changed_since_accept=self._changed_since_accept,
        )

    @cached_property
    def org_sync(self) -> OrgSyncService:
        """What pushes, pulls and merges the org."""
        return OrgSyncService(
            environment=self.environment,
            org=lambda: self.org,
            repo_dir=lambda: self._repo_dir,
            repository=self._repository,
            org_state=self.org_state,
            org_save=self.org_save,
            work_queue=self.work_queue,
            org_run=self.org_run,
            launcher=self._require_launcher,
            join=self._join,
            working=self._working,
        )

    async def _came_back(self) -> None:
        await self.org_run.came_back()
