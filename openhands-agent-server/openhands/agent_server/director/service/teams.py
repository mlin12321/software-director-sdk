"""Starting a team's agents and what is sent to them."""

import asyncio
from contextlib import suppress
from functools import cached_property
from pathlib import Path

from openhands.agent_server.director.git import (
    GitError,
    bring_over,
    uncommitted,
)
from openhands.agent_server.director.intents import Brief, LaunchMember, StartTeam
from openhands.agent_server.director.models import (
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    is_division_lead,
)
from openhands.agent_server.director.prompts import (
    left_for_team,
)
from openhands.agent_server.director.run_state import (
    TASK,
    Intent,
)
from openhands.agent_server.director.service.agent_start_service import (
    AgentStartService,
)
from openhands.agent_server.director.service.delivery import Delivery
from openhands.agent_server.director.service.state import (
    LEFT_REFS,
    checkout_root,
)
from openhands.agent_server.director.service.team_messaging_service import (
    TeamMessagingService,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class Teams(Delivery):
    """Starts agents and carries what is said between teams."""

    @cached_property
    def agent_start(self) -> AgentStartService:
        """What starts the agents of a team."""
        return AgentStartService(
            org=lambda: self.org,
            repo_dir=lambda: self._repo_dir,
            org_state=self.org_state,
            start_line=self.start_line,
            message_delivery=self.message_delivery,
            confinement=self.confinement,
            team_types=self.team_types,
            team_documents=lambda: self.team_documents,
            launcher=self._require_launcher,
            context_of=self.team_context.context_of,
            tell_of_failure=lambda place_id, role, reason: (
                self.team_messaging.tell_of_failure(place_id, role, reason)
            ),
        )

    @cached_property
    def team_messaging(self) -> TeamMessagingService:
        """What carries what is said between teams."""
        return TeamMessagingService(
            org=lambda: self.org,
            org_state=self.org_state,
            org_settings=self.org_settings,
            org_save=self.org_save,
            team_types=self.team_types,
            team_history=self.team_history,
            session_drafts=self.session_drafts,
            documents=self.documents,
            check_counts=self.check_counts,
            message_delivery=self.message_delivery,
            agent_start=self.agent_start,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            launcher=self._require_launcher,
            joined=self.joined,
            clock=self._clock,
        )

    def _task_of(self, place: SessionTeam) -> str:
        """What a team that is started again is to do, from what the org
        kept of it. What its documents say comes with its briefing."""
        task = place.summary
        if place.headline and place.headline not in task:
            task = f"{task}\n\nWhere the team last was: {place.headline}"
        return (
            f"{task}\n\nThis team was worked on before. "
            "What it committed is in your checkout: look at it first and "
            "carry on from there."
        )

    async def _do_brief(self, intent: Intent) -> None:
        brief = intent.of(Brief)
        place = self.org.place(intent.node_id)
        if place and brief.role == self.team_types.team(place).writer:
            self.org_state.run_state().mark(TASK, intent.node_id, brief.task)
        await self.agent_start.brief(intent.node_id, brief.role, brief.task)

    async def _do_launch_member(self, intent: Intent) -> None:
        launch = intent.of(LaunchMember)
        await self.agent_start.launch_member(intent.node_id, launch.task, launch.role)

    async def _do_start_team(self, intent: Intent) -> None:
        start = intent.of(StartTeam)
        place_id, task = intent.node_id, start.task
        above = self.org.place(start.parent or "")
        parent_ic = (
            await self.agent_start.started_ic(above)
            or await self.agent_start.template()
        )
        checkout = start.checkout
        self.org_state.run_state().mark(TASK, place_id, task)
        self.team_messaging.progressed(place_id, by_above=True)
        started = self.org.place(place_id)
        left: list[str] = []
        # A session starts beside the team's own work: what is uncommitted
        # in the team's directory is not the session's.
        opens = started is not None and (
            self.org.opens_session(started) or is_division_lead(started)
        )
        if started is not None:
            # A team of the same name before it left nothing to count on.
            self.check_counts.work_accepted(started)
            if above is not None and not opens:
                with suppress(GitError, OSError):
                    left = await asyncio.to_thread(
                        uncommitted,
                        Path(above.worktree_path),
                        started.boundary.path,
                    )
        if started is None:
            return
        await self.agent_start.start_agent(
            place_id,
            self.team_types.team(started).writer,
            parent_ic,
            f"{task}\n\n{left_for_team(left, names=self.team_types.names_at(place_id))}"
            if left
            else task,
            checkout,
        )
        if above is not None and not opens:
            await asyncio.to_thread(self._pass_down, above, place_id)
        if opens and not is_division_lead(started):
            self._checkout_for_session(place_id)
        purpose = start.purpose or task
        self.work_queue.enqueue(
            lambda: self._summarize(place_id, purpose, first=True), place_id
        )

    def _checkout_for_session(self, place_id: str) -> None:
        """Give the teams of a session that have not started the checkout
        of the team the session was opened at, as a team that is set up
        has the checkout of the team above."""
        store = self.org
        top = store.place(place_id)
        if top is None or not top.agents:
            return
        teams = store.below(place_id)
        for team in teams:
            if not team.agents:
                store.update_place(
                    team.id, worktree_path=top.worktree_path, branch=top.branch
                )
                teams.extend(store.below(team.id))

    def _pass_down(self, above: SessionTeam, place_id: str) -> None:
        """Close a started team's directory to the IC above, and bring what
        that IC left uncommitted there into the team's own checkout."""
        team = self.org.place(place_id)
        if team is None or not team.worktree_path:
            return
        with suppress(GitError, OSError):
            # The shell the IC above starts next cannot write there.
            self.confinement.confine(above.id, self.team_types.team(above).writer)
        source, target = Path(above.worktree_path), Path(team.worktree_path)
        if checkout_root(str(source)) == checkout_root(str(target)):
            # In one checkout the team finds the files where they are.
            return
        try:
            bring_over(source, target, team.boundary.path, f"{LEFT_REFS}/{team.id}")
        except (GitError, OSError):
            logger.exception("Could not pass what was left to team %s", team.id)
