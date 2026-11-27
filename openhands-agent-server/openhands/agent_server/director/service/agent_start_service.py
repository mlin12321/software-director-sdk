"""Starting agents: the conversation of a member of a team is derived
from one that runs already, briefed with its team's context, confined to
its team's folder and started when the server has room."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from openhands.agent_server.director.context import briefing_block, steady
from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.launch import StartedAgent, TeamLauncher
from openhands.agent_server.director.models import SessionTeam
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    OrgStore,
    is_division_lead,
    session_team_id,
)
from openhands.agent_server.director.prompts import (
    brief as first_message,
    hand_over_brief,
    role_text,
)
from openhands.agent_server.director.run_state import TOLD
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import Held
from openhands.agent_server.director.service.start_line_service import StartLineService
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.sdk.errors import Conflict, NotFound
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.spec import Tool


logger = get_logger(__name__)

# How much of an error is kept as the reason an agent failed.
REASON_KEPT = 600


def agent_title(
    team: str, name: str, session: str = "", *, division_lead: bool = False
) -> str:
    """What an agent's conversation is called. The session is named unless
    it is called what its team is, and the agents a session's work is
    reviewed by are called its Division Leads.

    Args:
        name: What the agent's member is called.
    """
    where = f"{team} · {session}" if session and session != team else team
    return f"{where} ({'Division Lead ' + name if division_lead else name})"


@dataclass
class AgentStartService:
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

    def in_place(self, place: SessionTeam) -> bool:
        """Whether a team that starts works in the checkout it has: a
        session's Division Lead place, and the team a session is opened
        at when sessions get no worktree or the session has one already."""
        return is_division_lead(place) or (
            self.org().opens_session(place)
            and (
                place.isolation != "worktree"
                or self.checkouts.root_of(place.worktree_path)
                != self.checkouts.root_of(str(self.repo_dir()))
            )
        )

    def derived_from(self, place: SessionTeam) -> str | None:
        """The place whose IC the IC of ``node`` is derived from when it
        starts: the team above, or for the team a session is opened at and
        its reviewer, who of the session works already."""
        store = self.org()
        opened = store.session_of(place)
        if is_division_lead(place) and opened:
            return session_team_id(opened.team_id, opened.id)
        if store.opens_session(place):
            # A session that went a team higher has an IC below already.
            return next(
                (team.id for team in store.below(place.id) if team.agents),
                place.parent_id,
            )
        return place.parent_id

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
                self.org().place(self.derived_from(place) or "")
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
