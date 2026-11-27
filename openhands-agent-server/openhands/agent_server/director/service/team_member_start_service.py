"""Starting agents: the conversation of a member of a team is derived
from one that runs already, briefed with its team's context, confined to
its team's folder and started when the server has room."""

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from openhands.agent_server.director.context import briefing_block, steady
from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.environment import EnvironmentFailed
from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.intents import Brief, LaunchMember, StartTeam
from openhands.agent_server.director.launch import StartedAgent, TeamLauncher
from openhands.agent_server.director.models import SessionTeam
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    OrgStore,
    is_division_lead,
)
from openhands.agent_server.director.prompts import (
    brief as first_message,
    hand_over_brief,
    left_for_team,
    role_text,
)
from openhands.agent_server.director.run_state import TASK, TOLD, Intent
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import Held
from openhands.agent_server.director.service.start_line_service import StartLineService
from openhands.agent_server.director.service.team_history_service import REASON_KEPT
from openhands.agent_server.director.service.team_summary_service import (
    TeamSummaryService,
    agent_title,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.errors import Conflict, NotFound
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.spec import Tool


logger = get_logger(__name__)

# Where what an IC left uncommitted in the directory of a sub-team with a
# checkout of its own is kept, as it went to the team when it started.
LEFT_REFS = "refs/agents/left"


@dataclass
class TeamMemberStartService:
    """Starts the agents of a team.

    An agent is derived from a conversation that has started already (the
    team above, or the one a session started with), or from the profile
    its team names. Its first message is its briefing; what the briefing
    told is kept, so that later messages carry only what changed.
    """

    org: Callable[[], OrgStore]
    """The org whose agents are started."""
    checkouts: Checkouts
    """Where each team's work is and how it moves."""
    repo_dir: Callable[[], Path]
    """The repository the org is in."""
    org_state: OrgStateService
    """Whether the org is paused, and where what an agent was told is
    kept."""
    start_line: StartLineService
    """Has a start wait while the server is at its run limit."""
    message_delivery: MessageDeliveryService
    """Sends a briefing to an agent that runs already, and is told which
    agents run."""
    confinement: AgentConfinementService
    """Holds an agent to its team's folder."""
    team_types: TeamTypeService
    """Who the members of a team are and what each is offered and told."""
    team_documents: Callable[[], TeamDocuments]
    """The documents of the org as each team reads them."""
    launcher: Callable[[], TeamLauncher]
    """What starts the conversations; raises where there is none."""
    context_of: Callable[[SessionTeam, str], dict[str, Any]]
    """What a member of a team reads of its place."""
    tell_of_failure: Callable[[str, str, str], None]
    """Tells the team above that a member of a team could not be started:
    the place, the member and the reason."""
    check_counts: CheckCountService
    """Forgets what a team of the same name before counted."""
    team_summaries: TeamSummaryService
    """Writes the first headline of a team that starts."""
    work_queue: WorkQueueService

    async def template(self) -> str | None:
        """A conversation of the org to derive an agent from when nothing
        above it has started: the IC a session started with, oldest first."""
        store, launcher = self.org(), self.launcher()
        for session in store.sessions():
            ic = session.conversation_id
            if ic is not None and await launcher.started(UUID(ic)) is not None:
                return ic
        return None

    def briefing(self, place: SessionTeam, role: str, task: str) -> tuple[str, str]:
        """An agent's first message, and the context it tells, as kept to
        find later what changed."""
        context = self.context_of(place, role)
        text = first_message(
            place,
            role,
            task,
            self.team_documents().briefed(self.team_types.reader_of(place)),
            briefing_block(
                context, self.team_documents().told(self.team_types.reader_of(place))
            ),
            self.team_types.type_of(place).text_for(role),
            works=role == self.team_types.team(place).writer,
            titles=self.team_types.type_of(place).titles,
            role_said=self.team_types.role_text(place, role),
            names=self.team_types.team(place).names,
        )
        return text, json.dumps(steady(context))

    def _hand_over_briefing(self, place: SessionTeam, job: str) -> str:
        """The first message of an IC's hand-over conversation."""
        return hand_over_brief(
            place,
            job,
            self.team_documents().briefed(self.team_types.reader_of(place)),
            briefing_block(
                self.context_of(place, self.team_types.team(place).writer),
                self.team_documents().told(self.team_types.reader_of(place)),
            ),
            role=role_text(
                self.team_types.team(place).names,
                works=True,
                hand_over=True,
                rules=self.team_types.document_rules(place),
            ),
            names=self.team_types.team(place).names,
        )

    async def start_hand_over(
        self, place: SessionTeam, conversation_id: str, job: str
    ) -> bool:
        """Start the hand-over conversation of a team's IC, derived from its
        main one.

        Returns:
            Whether it runs.
        """
        names = self.team_types.team(place).names
        launcher = self.launcher()
        try:
            if await launcher.started(UUID(conversation_id)) is not None:
                return True
            parent = await self.started_ic(place)
            if parent is None:
                raise NotFound(f"Team {place.id} has no {names.writer} to hand over")
            text = await asyncio.to_thread(self._hand_over_briefing, place, job)
            tool_params = await asyncio.to_thread(
                self.confinement.confine,
                place.id,
                self.team_types.team(place).writer,
                conversation_id,
            )
            adapt = await asyncio.to_thread(
                self.confinement.adapter,
                conversation_id,
                self.team_types.hand_over_tools,
            )

            async def start() -> StartedAgent:
                return await launcher.start_derived(
                    UUID(parent),
                    UUID(conversation_id),
                    title=f"{place.title} ({names.writer}, hand-over)",
                    task=text,
                    isolation="shared",
                    tool_params=tool_params,
                    adapt_agent=adapt,
                )

            await self.start_line.when_there_is_room(place.id, start)
            if self.org_state.paused is not None:
                self.message_delivery.note_running(conversation_id, place.id, True)
                await launcher.pause(UUID(conversation_id))
        except Held:
            raise
        except Exception:
            logger.exception("Could not start the hand-over of team %s", place.id)
            return False
        return True

    async def brief(self, place_id: str, role: str, task: str) -> None:
        place = self.org().place(place_id)
        if place is None:
            return
        conversation_id = self.team_types.team(place).conversation(role)
        if conversation_id is not None:
            text, told = await asyncio.to_thread(self.briefing, place, role, task)
            self.org_state.run_state().mark(TOLD, conversation_id, told)
            await self.message_delivery.send(conversation_id, text)

    async def start_agent(
        self,
        place_id: str,
        role: str,
        parent_conversation: str | None,
        task: str,
        checkout: str | None = None,
    ) -> bool:
        """Start one agent of a team from ``parent_conversation``.

        Args:
            checkout: The team's checkout when it has one already and the
                agent is to work on in it.

        Returns:
            Whether it runs. A failure is recorded on the team.
        """
        store = self.org()
        place = store.place(place_id)
        conversation_id = (
            self.team_types.team(place).conversation(role) if place else None
        )
        if place is None or conversation_id is None:
            return False
        launcher = self.launcher()
        try:
            started = await launcher.started(UUID(conversation_id))
            if started is None:
                started = await self._start_new(
                    place, role, conversation_id, parent_conversation, task, checkout
                )
                if self.org_state.paused is not None:
                    self.message_delivery.note_running(conversation_id, place_id, True)
                    await launcher.pause(UUID(conversation_id))
        except Held:
            raise
        except Exception as error:
            logger.exception("Could not start the %s of team %s", role, place_id)
            store.set_agent(place_id, role, status=AGENT_LOST)
            self.tell_of_failure(
                place_id, role, f"It could not be started: {error}"[:REASON_KEPT]
            )
            return False
        store.set_agent(
            place_id,
            role,
            kind=started.kind,
            acp_server=started.acp_server,
            model=started.model,
        )
        if role == self.team_types.team(place).writer:
            store.update_place(
                place_id,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(
                    self.checkouts.branch, started.workspace
                ),
                base_commit=place.base_commit
                or await asyncio.to_thread(self.checkouts.version, started.workspace)
                or "",
                boundary_enforcement=self.confinement.enforcement(started.kind),
            )
        return True

    async def _start_new(
        self,
        place: SessionTeam,
        role: str,
        conversation_id: str,
        parent_conversation: str | None,
        task: str,
        checkout: str | None = None,
    ) -> StartedAgent:
        """Start the conversation of an agent that has none yet.

        Raises:
            NotFound: There is nobody to derive the agent from.
            Conflict: A step of the director is replacing or removing the team.
            Held: The org was paused while the team waited for a run.
        """
        place_id = place.id
        launcher = self.launcher()
        if self.message_delivery.is_settling(place_id):
            # The step fixed which conversations it stops before this one.
            raise Conflict(
                f"Team {place.title!r} is being replaced or removed, so its "
                f"{role} was not started"
            )
        text, told = await asyncio.to_thread(self.briefing, place, role, task)
        tool_params = await asyncio.to_thread(self.confinement.confine, place_id, role)
        if parent_conversation is None:
            raise NotFound(f"Team {place_id} has nobody to derive its {role} from")
        parent_id = UUID(parent_conversation)
        # A team given a profile of its own starts its IC from it; its
        # Manager, like every Manager, is derived from the IC.
        works = role == self.team_types.team(place).writer
        profile = place.agent_profile if works else None
        kind = await (
            launcher.profile_kind(profile)
            if profile
            else launcher.agent_kind(parent_id)
        )
        isolation = place.isolation if works else "shared"
        leads = is_division_lead(place)
        if kind == "acp" and works and not self.confinement.is_sandboxed:
            # Nothing holds its process to the team's directory, so its
            # work is kept apart and checked when it is merged.
            isolation = "worktree"
        tools = self.team_types.tools_for(place, role)
        adapt = await asyncio.to_thread(
            self.confinement.adapter, conversation_id, tools
        )
        # A session starts from where the repository's checkout is.
        base = (
            await asyncio.to_thread(self.checkouts.version, self.repo_dir())
            if works
            and not checkout
            and isolation == "worktree"
            and self.org().opens_session(place)
            else None
        )
        title = agent_title(
            place.title,
            self.team_types.team(place).title(role),
            self.org().session_title(place),
            division_lead=leads,
        )

        async def start() -> StartedAgent:
            if profile:
                return await launcher.start_from_profile(
                    profile,
                    parent_id,
                    UUID(conversation_id),
                    [Tool(name=name) for name in tools],
                    title=title,
                    task=text,
                    isolation=isolation,
                    tool_params=tool_params,
                    adapt_agent=adapt,
                    checkout=checkout,
                    base=base,
                )
            return await launcher.start_derived(
                parent_id,
                UUID(conversation_id),
                title=title,
                task=text,
                isolation=isolation,
                tool_params=tool_params,
                adapt_agent=adapt,
                checkout=checkout,
                base=base,
            )

        started = await self.start_line.when_there_is_room(place_id, start)
        self.org_state.run_state().mark(TOLD, conversation_id, told)
        return started

    async def started_ic(self, place: SessionTeam | None) -> str | None:
        """The IC of ``node`` or, while that team has not started, of the
        nearest team above that has: the conversation a new agent is
        derived from."""
        store, launcher = self.org(), self.launcher()
        while place is not None:
            ic = self.team_types.writer_conversation(place)
            # Never a Division Lead IC, whose role of reviewer the new agent
            # would inherit.
            if (
                ic is not None
                and not is_division_lead(place)
                and await launcher.started(UUID(ic)) is not None
            ):
                return ic
            place = store.place(place.parent_id) if place.parent_id else None
        return None

    async def launch_member(self, place_id: str, task: str, member: str) -> None:
        place = self.org().place(place_id)
        if place is not None:
            # A Division Lead Manager is derived from the session's IC,
            # never from the Division Lead IC, whose role it would inherit.
            derived_from = await self.started_ic(
                self.org().place(self.org().derived_from(place) or "")
                if is_division_lead(place)
                else place
            )
            # A member that reviews a session's work without being the one
            # that works there still does it in the place's own checkout.
            reviews_here = (
                is_division_lead(place)
                and member == self.team_types.team(place).reviewer
            )
            await self.start_agent(
                place_id,
                member,
                derived_from,
                task,
                place.worktree_path if reviews_here else None,
            )

    def task_of(self, place: SessionTeam) -> str:
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

    async def do_brief(self, intent: Intent) -> None:
        brief = intent.of(Brief)
        place = self.org().place(intent.node_id)
        if place and brief.role == self.team_types.team(place).writer:
            self.org_state.run_state().mark(TASK, intent.node_id, brief.task)
        await self.brief(intent.node_id, brief.role, brief.task)

    async def do_launch_member(self, intent: Intent) -> None:
        launch = intent.of(LaunchMember)
        await self.launch_member(intent.node_id, launch.task, launch.role)

    async def do_start_team(self, intent: Intent) -> None:
        start = intent.of(StartTeam)
        place_id, task = intent.node_id, start.task
        above = self.org().place(start.parent or "")
        parent_ic = await self.started_ic(above) or await self.template()
        checkout = start.checkout
        self.org_state.run_state().mark(TASK, place_id, task)
        self.org_state.progressed(place_id, by_above=True)
        started = self.org().place(place_id)
        left: list[str] = []
        # A session starts beside the team's own work: what is uncommitted
        # in the team's directory is not the session's.
        opens = started is not None and (
            self.org().opens_session(started) or is_division_lead(started)
        )
        if started is not None:
            # A team of the same name before it left nothing to count on.
            self.check_counts.work_accepted(started)
            if above is not None and not opens:
                with suppress(EnvironmentFailed, OSError):
                    left = await asyncio.to_thread(
                        self.checkouts.uncommitted,
                        Path(above.worktree_path),
                        started.boundary.path,
                    )
        if started is None:
            return
        await self.start_agent(
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
            self.checkout_for_session(place_id)
        purpose = start.purpose or task
        self.work_queue.enqueue(
            lambda: self.team_summaries.summarize(place_id, purpose, first=True),
            place_id,
        )

    def checkout_for_session(self, place_id: str) -> None:
        """Give the teams of a session that have not started the checkout
        of the team the session was opened at, as a team that is set up
        has the checkout of the team above."""
        store = self.org()
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
        team = self.org().place(place_id)
        if team is None or not team.worktree_path:
            return
        with suppress(EnvironmentFailed, OSError):
            # The shell the IC above starts next cannot write there.
            self.confinement.confine(above.id, self.team_types.team(above).writer)
        source, target = Path(above.worktree_path), Path(team.worktree_path)
        if self.checkouts.root_of(str(source)) == self.checkouts.root_of(str(target)):
            # In one checkout the team finds the files where they are.
            return
        try:
            self.checkouts.bring_over(
                source, target, team.boundary.path, f"{LEFT_REFS}/{team.id}"
            )
        except (EnvironmentFailed, OSError):
            logger.exception("Could not pass what was left to team %s", team.id)
