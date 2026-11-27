"""Sessions: the lines of work the human opens at a team. All work is in one."""

from functools import cached_property

from openhands.agent_server.director.service.request_routing_service import (
    RequestRoutingService,
)
from openhands.agent_server.director.service.session_lifecycle_service import (
    SessionLifecycleService,
)
from openhands.agent_server.director.service.splits import Splits
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class Sessions(Splits):
    """Builds what opens and closes the human's sessions."""

    @cached_property
    def request_routing(self) -> RequestRoutingService:
        """Chooses the team a request goes straight to. Built here, where
        everything it is given is defined."""
        return RequestRoutingService(
            checkouts=self.checkouts,
            org=lambda: self.org,
            model_calls=self.model_calls,
            checkout=lambda: self._repo_dir,
            purpose=self.team_messaging.purpose,
            routed_requests=self.team_history.routed_requests,
            keep_routed=self.team_history.keep_routed,
            hand_on=lambda *args, **kwargs: self.session_lifecycle.hand_on(
                *args, **kwargs
            ),
            paused=lambda: self.org_state.paused is not None,
        )

    @cached_property
    def session_lifecycle(self) -> SessionLifecycleService:
        """What opens and closes the human's sessions."""
        return SessionLifecycleService(
            checkouts=self.checkouts,
            org=lambda: self.org,
            repo_dir=lambda: self._repo_dir,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.org_settings,
            team_types=self.team_types,
            team_history=self.team_history,
            team_summaries=self.team_summaries,
            team_messaging=self.team_messaging,
            message_delivery=self.message_delivery,
            agent_start=self.agent_start,
            check_counts=self.check_counts,
            confinement=self.confinement,
            session_documents=self.session_documents,
            session_drafts=self.session_drafts,
            team_splits=self.team_splits,
            model_calls=self.model_calls,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            request_routing=lambda: self.request_routing,
            launcher=self._require_launcher,
            checkout_for_session=self._checkout_for_session,
            reviews=self.reviews,
            session_merges=self.session_merges,
        )
