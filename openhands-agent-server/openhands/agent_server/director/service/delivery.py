"""How the service reaches agents: messages, the steps that wait for an
agent to be quiet, and the questions put to a model."""

import asyncio
from contextlib import suppress
from functools import cached_property
from uuid import UUID

from openhands.agent_server.director import compaction
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    Report,
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    division_lead_id,
    instant,
)
from openhands.agent_server.director.run_state import (
    JOINED,
)
from openhands.agent_server.director.service.agent_start_service import agent_title
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.records import Records
from openhands.agent_server.director.service.resumable_work_service import (
    ResumableWorkService,
)
from openhands.agent_server.director.service.state import (
    MODELS_LANE,
    ORG_LANE,
    SAVED_BEFORE_FORGOTTEN,
    lane_for,
)
from openhands.sdk.errors import (
    NotFound,
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

    def refresh_headline(self, place_id: str, report: str) -> None:
        """Have a team's headline rewritten after it reported. Any thread."""
        place = self.org.place(place_id)
        if place is not None:
            task = f"{place.summary}\n\nLatest report of the team: {report}"
            self.work_queue.enqueue(
                lambda: self._summarize(place_id, task), f"headline:{place_id}"
            )

    def reports_grew(self, folder: str, role: str) -> None:
        """Compact what one member of the team of ``folder`` recorded, if
        it has grown past what the org set. Safe from any thread."""
        reports = self._memory().reports(folder, role)
        if compaction.due(reports, self.org_settings.settings.report_compaction_chars):
            self.work_queue.enqueue(lambda: self._compact(folder, role), MODELS_LANE)

    def joined(self, conversation_id: str) -> None:
        """Note that the human has written in an agent's conversation, so
        that it is listed with the conversations the human started. Safe
        from any thread, and to call again.

        Raises:
            NotFound: No agent of the org has this conversation.
        """
        found = self.org.find_by_conversation(conversation_id)
        if found is None:
            raise NotFound(f"No agent of the org has conversation {conversation_id}")
        run = self.org_state.run_state()
        if conversation_id in run.marked(JOINED):
            return
        run.mark(JOINED, conversation_id)
        # After what starts the conversation, when it is not there yet.
        self.work_queue.enqueue(lambda: self._mark_joined(conversation_id), found[0].id)

    async def _mark_joined(self, conversation_id: str) -> None:
        try:
            await self._require_launcher().mark_joined(UUID(conversation_id))
        except NotFound:
            # Not started yet, or gone: asked for again when it is written to.
            self.org_state.run_state().unmark(JOINED, conversation_id)

    async def _compact(self, folder: str, role: str) -> None:
        """Have a model rewrite one agent's reports shorter, if they are
        still too long."""
        asker = self.model_calls.asker()
        if asker is None or self.org_state.paused is not None:
            return
        team_memory = self._memory()
        reports = await asyncio.to_thread(team_memory.reports, folder, role)
        if not compaction.due(
            reports, self.org_settings.settings.report_compaction_chars
        ):
            return
        answer = await self.model_calls.ask(
            asker, *compaction.prompt(reports), compaction.USAGE_ID
        )
        shorter = compaction.compacted(answer or "", reports, instant())
        if shorter is None:
            logger.warning("No usable rewrite of the reports of %s %s", folder, role)
            return
        # Rewriting them changes what the org shares.
        await self._base(
            lambda: asyncio.to_thread(
                team_memory.replace_reports, folder, role, reports, shorter
            )
        )

    async def _as_one_report(self, reports: list[Report]) -> str:
        """``reports`` as one report: rewritten shorter by a model when
        there are several, and in full when it gives no usable answer."""
        full = "\n\n".join(report.text for report in reports)
        asker = self.model_calls.asker()
        if len(reports) < 2 or asker is None:
            return full
        answer = await self.model_calls.ask(
            asker, *compaction.prompt(reports), compaction.USAGE_ID
        )
        shorter = compaction.compacted(answer or "", reports, instant())
        return shorter[0].text if shorter else full

    async def _name_org(self, top_id: str) -> None:
        """Give the top team the name a model made of the first task. The
        Division Lead of that session is called what its team is."""
        store = self.org
        top = store.place(top_id)
        if top is None or store.root_id is None:
            return
        store.update_team(store.root_id, title=top.title, headline=top.headline)
        lead = store.place(division_lead_id(store.root_id, top.session))
        if lead is not None:
            store.update_place(lead.id, title=top.title)

    async def _summarize(
        self,
        place_id: str,
        task: str,
        *,
        rename: bool = False,
        first: bool = False,
    ) -> None:
        """Have a model write a team's headline and, with ``rename``, its title.

        Args:
            first: This is the summary of the task the team was made for. It
                is dropped once the team has reported, which may happen
                before the model was asked.
        """
        store = self.org

        def reported() -> bool:
            current = store.place(place_id)
            return (
                first
                and current is not None
                and current.headline != provisional_summary(task).headline
            )

        place = store.place(place_id)
        conversation_id = self.team_types.writer_conversation(place) if place else None
        if place is None or conversation_id is None or reported():
            return
        if self.org_state.paused is not None:
            return
        launcher = self._require_launcher()
        summary = await self.model_calls.summarize(conversation_id, task)
        if summary is None or reported():
            return
        changes: dict[str, object] = {"headline": summary.headline}
        if rename:
            changes["title"] = summary.title
        try:
            place = store.update_place(place_id, **changes)
        except NotFound:
            return
        if not rename:
            return
        if store.opens_session(place):
            # A session the human gave no name is called what it is for.
            store.rename_session(place.session, summary.title)
        await self._rename_agents(place, launcher)

    async def _name_session(self, place_id: str, request: str) -> None:
        """Call a session the human gave no name by what a model makes of
        its request. The team it was opened at keeps its own name."""
        store = self.org
        place = store.place(place_id)
        conversation_id = self.team_types.writer_conversation(place) if place else None
        if (
            place is None
            or conversation_id is None
            or self.org_state.paused is not None
        ):
            return
        launcher = self._require_launcher()
        summary = await self.model_calls.summarize(conversation_id, request)
        place = store.place(place_id)
        if summary is None or place is None:
            return
        store.rename_session(place.session, summary.title)
        await self._rename_agents(place, launcher)

    async def _rename_agents(self, place: SessionTeam, launcher: TeamLauncher) -> None:
        store = self.org
        for agent in place.agents:
            if agent.conversation_id is not None:
                with suppress(NotFound):
                    await launcher.rename(
                        UUID(agent.conversation_id),
                        agent_title(
                            place.title,
                            self.team_types.team(place).title(agent.role),
                            store.session_title(place),
                        ),
                    )
