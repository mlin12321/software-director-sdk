"""What a model writes for the org: the headline and the summary of a
team, the names of the org and of a session, and the reports of a team
rewritten shorter once they have grown long."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from uuid import UUID

from openhands.agent_server.director import compaction
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    Report,
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    division_lead_id,
    instant,
)
from openhands.agent_server.director.service.agent_start_service import agent_title
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import (
    BASE_LANE,
    MODELS_LANE,
    WorkQueueService,
)
from openhands.sdk.errors import (
    NotFound,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


@dataclass
class TeamSummaryService:
    """Has a model name and summarize what the org shows of its teams.

    Each is asked for in a lane of its own, so that no team waits for a
    model, and what a model does not answer leaves what was there.
    """

    org: Callable[[], OrgStore]
    """The org whose teams are named and summarized."""
    team_memory: Callable[[], GitStore]
    """Where the reports of the teams are."""
    org_state: OrgStateService
    """The run state, and whether the org is paused."""
    org_settings: OrgSettingsService
    """How long reports may grow before they are rewritten."""
    team_types: TeamTypeService
    """Who the members of a team are, for what their conversations are
    called."""
    model_calls: ModelCallService
    """Puts the questions to a model."""
    work_queue: WorkQueueService
    """What carries the questions out."""
    launcher: Callable[[], TeamLauncher]
    """What renames the conversations; raises where there is none."""

    def refresh_headline(self, place_id: str, report: str) -> None:
        """Have a team's headline rewritten after it reported. Any thread."""
        place = self.org().place(place_id)
        if place is not None:
            task = f"{place.summary}\n\nLatest report of the team: {report}"
            self.work_queue.enqueue(
                lambda: self.summarize(place_id, task), f"headline:{place_id}"
            )

    def reports_grew(self, folder: str, role: str) -> None:
        """Compact what one member of the team of ``folder`` recorded, if
        it has grown past what the org set. Safe from any thread."""
        reports = self.team_memory().reports(folder, role)
        if compaction.due(reports, self.org_settings.settings.report_compaction_chars):
            self.work_queue.enqueue(lambda: self.compact(folder, role), MODELS_LANE)

    async def compact(self, folder: str, role: str) -> None:
        """Have a model rewrite one agent's reports shorter, if they are
        still too long."""
        asker = self.model_calls.asker()
        if asker is None or self.org_state.paused is not None:
            return
        team_memory = self.team_memory()
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
        await self.work_queue.result_in(
            BASE_LANE,
            lambda: asyncio.to_thread(
                team_memory.replace_reports, folder, role, reports, shorter
            ),
        )

    async def as_one_report(self, reports: list[Report]) -> str:
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

    async def name_org(self, top_id: str) -> None:
        """Give the top team the name a model made of the first task. The
        Division Lead of that session is called what its team is."""
        store = self.org()
        top = store.place(top_id)
        if top is None or store.root_id is None:
            return
        store.update_team(store.root_id, title=top.title, headline=top.headline)
        lead = store.place(division_lead_id(store.root_id, top.session))
        if lead is not None:
            store.update_place(lead.id, title=top.title)

    async def summarize(
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
        store = self.org()

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
        launcher = self.launcher()
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

    async def name_session(self, place_id: str, request: str) -> None:
        """Call a session the human gave no name by what a model makes of
        its request. The team it was opened at keeps its own name."""
        store = self.org()
        place = store.place(place_id)
        conversation_id = self.team_types.writer_conversation(place) if place else None
        if (
            place is None
            or conversation_id is None
            or self.org_state.paused is not None
        ):
            return
        launcher = self.launcher()
        summary = await self.model_calls.summarize(conversation_id, request)
        place = store.place(place_id)
        if summary is None or place is None:
            return
        store.rename_session(place.session, summary.title)
        await self._rename_agents(place, launcher)

    async def _rename_agents(self, place: SessionTeam, launcher: TeamLauncher) -> None:
        store = self.org()
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
