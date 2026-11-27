"""How the service reaches agents: messages, the steps that wait for an
agent to be quiet, and the questions put to a model."""

from functools import cached_property

from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.records import Records
from openhands.agent_server.director.service.resumable_work_service import (
    ResumableWorkService,
)
from openhands.agent_server.director.service.state import (
    SAVED_BEFORE_FORGOTTEN,
    lane_for,
)
from openhands.agent_server.director.service.team_summary_service import (
    TeamSummaryService,
)
from openhands.agent_server.director.service.work_queue_service import (
    ORG_LANE,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class Delivery(Records):
    """Sends agents what the director has for them."""

    @cached_property
    def resumable_work(self) -> ResumableWorkService:
        """What the director begins for a team and finishes after a pause
        or a restart. The service registers the step of each kind."""
        return ResumableWorkService(
            run_state=self.org_state.run_state,
            work_queue=self.work_queue,
            lane_of=lane_for,
            is_paused=lambda: self.org_state.paused is not None,
            save=self.org_save.save,
            save_soon=self.org_save.save_soon,
            let_go=self.message_delivery.let_go,
            keep_failure=self._keep_failure,
            left=self.start_line.leave,
            saved_before_forgotten=SAVED_BEFORE_FORGOTTEN,
        )

    @cached_property
    def team_summaries(self) -> TeamSummaryService:
        """What a model writes of the teams: headlines, names, reports
        rewritten shorter."""
        return TeamSummaryService(
            org=lambda: self.org,
            team_memory=self._memory,
            org_state=self.org_state,
            org_settings=self.org_settings,
            team_types=self.team_types,
            model_calls=self.model_calls,
            work_queue=self.work_queue,
            launcher=self._require_launcher,
        )

    @cached_property
    def message_delivery(self) -> MessageDeliveryService:
        """What sends agents the director's messages."""
        return MessageDeliveryService(
            org=lambda: self.org,
            org_state=self.org_state,
            org_save=self.org_save,
            work_queue=self.work_queue,
            team_types=self.team_types,
            launcher=self._require_launcher,
            is_running=lambda: self._loop is not None,
            context_of=self.team_context.context_of,
            is_stopped=lambda conversation_id: self.org.is_stopped(conversation_id),
            lane_of=lane_for,
            take_up=lambda intent_id, lane: self.resumable_work.take_up(
                intent_id, lane
            ),
            default_lane=ORG_LANE,
            retry=self._run_limit_retry,
        )
