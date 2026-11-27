"""Confinement: what each agent of the org may write, the shell and the
tools it is started with to hold it to that, and the tokens its tool
calls are known by."""

import os
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import UUID

from openhands.agent_server.conversation_service import ToolParams
from openhands.agent_server.director.acp import conversation_of
from openhands.agent_server.director.boundary import scope_within, worded
from openhands.agent_server.director.environment import (
    CodingEnvironment,
    Confinement,
    EnvironmentFailed,
)
from openhands.agent_server.director.environment.checkouts import (
    MergeCheckout,
)
from openhands.agent_server.director.models import (
    AgentKind,
    SessionTeam,
    TeamEnforcement,
)
from openhands.agent_server.director.org_store import OrgStore, is_division_lead
from openhands.agent_server.director.prompts import NEED_WRITE
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.run_state import MERGING, RunState
from openhands.agent_server.director.scope import (
    Denied,
    MergeArea,
    Scope,
    named_paths,
    searched_folder,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.sdk.agent.base import AgentBase
from openhands.sdk.errors import NotFound
from openhands.sdk.event import ActionEvent
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Tools that start agents nothing holds to the team's folder. An org's
# agent hands work on by proposing sub-teams.
_SUB_AGENT_TOOLS = (
    "task_tool_set",
    "task",
    "delegate",
    "workflow_tool_set",
    "workflow",
)
# In the repository's git directory: the merges reviewers finish.
_MERGING_DIR = "director-merging"
# Set by the server's entry point to the address it is bound to.
_INTERNAL_SERVER_URL_ENV = "OH_INTERNAL_SERVER_URL"
# In the remembered file: what the tokens of the org's agents are signed with.
_TOOL_KEY = "tool_key"


@dataclass
class AgentConfinementService:
    """Decides what an agent of a team may write and holds it to that.

    :meth:`scope_of` is the one place the org decides what its agents are
    held to. The rest puts a scope in force: the shell an agent's terminal
    starts, the tools its agent is given, the check of a file tool's
    action, and the token a tool call of an agent with its own tools is
    known by.
    """

    repo_dir: Callable[[], Path]
    """The repository the org is in."""
    worktree_root: Path | None
    find_org: Callable[[], OrgStore | None]
    """The org if it is loaded; asking does not load it."""
    record_need: Callable[[SessionTeam, str, str], object]
    """Keeps that a team needed to write outside its folder
    (:meth:`TeamHistoryService.record_need`)."""
    """Where the server makes the git worktrees of conversations."""
    find_sandbox: Callable[[], str | None]
    """Looks for the bubblewrap binary to confine shells with."""
    remembered: RememberedFile
    """Where the key of the tool tokens is kept: no agent may open it."""
    server_url: str | None
    """Where this server answers; None for the address it is bound to."""
    org: Callable[[], OrgStore]
    """The org whose agents are held."""
    environment: CodingEnvironment
    """Where the agents work, and what holds them to their scopes there."""
    team_types: TeamTypeService
    """What a team's type says of who writes and who reviews."""
    run_state: Callable[[], RunState]
    """Where the merges reviewers finish are kept."""
    _bwrap: str | None = field(default=None, init=False)
    _tool_key: bytes | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def look_for_sandbox(self) -> None:
        """Find what confines the shells on this server. Without it the
        director checks file tools, and nothing holds the process of an
        agent that runs its own tools."""
        self._bwrap = self.find_sandbox()
        if self._bwrap is None:
            logger.warning(
                "Director cannot confine shells here (no bubblewrap): it checks "
                "file tools, and nothing holds the process of an ACP agent"
            )

    @property
    def is_sandboxed(self) -> bool:
        """Whether the agents' shells are confined on this server."""
        return self._bwrap is not None

    def tool_key(self) -> bytes:
        """What the tokens of the org's agents are signed with. It is kept
        beside the server's conversations, never in the repository."""
        with self._lock:
            if self._tool_key is None:
                kept = self.remembered.read().get(_TOOL_KEY)
                if not isinstance(kept, str):
                    kept = secrets.token_hex(32)
                    self.remembered.keep(**{_TOOL_KEY: kept})
                self._tool_key = kept.encode()
            return self._tool_key

    def agent_of(self, token: str) -> UUID:
        """The conversation of the agent a tool call comes from.

        Raises:
            NotFound: The token was not given out by this director.
        """
        conversation_id = conversation_of(self.tool_key(), token)
        if conversation_id is None:
            raise NotFound("That is not the token of an agent of this org")
        return UUID(conversation_id)

    def _checkouts(self) -> list[Path]:
        """Where the org's checkouts are: the repository and the worktrees."""
        roots = [self.repo_dir(), *([self.worktree_root] if self.worktree_root else [])]
        return [Path(os.path.realpath(root)) for root in roots]

    def _confinement(self) -> Confinement:
        """What holds the org's agents to their scopes, as the server is
        now."""
        return self.environment.confinement(
            self.repo_dir(),
            checkouts=tuple(self._checkouts()),
            sandbox=self._bwrap,
            secret=self.remembered.path,
            # Where an agent that runs its own tools reaches this server.
            server_url=self.server_url or os.environ.get(_INTERNAL_SERVER_URL_ENV),
            key=self.tool_key,
        )

    def enforcement(self, kind: AgentKind) -> TeamEnforcement:
        """What holds an agent of that kind here."""
        return self._confinement().enforcement(kind == "acp")

    def merges_for(self, place_id: str) -> Path:
        """Where the merges the reviewer of a team finishes are laid out:
        one folder for each reviewer, which its shell alone may write."""
        root = self.environment.state_dir(self.repo_dir()) / _MERGING_DIR
        return root / place_id.replace("@", "-")

    def merge_checkout(self, place_id: str) -> MergeCheckout | None:
        """The merge of a team's work that its reviewer finishes."""
        kept = self.run_state().marked(MERGING).get(place_id)
        return MergeCheckout.model_validate_json(kept) if kept else None

    def keep_merge_checkout(self, place_id: str, merge: MergeCheckout) -> None:
        """Remember the merge that was laid out for a team's work, so that
        it is found again after a restart."""
        self.run_state().mark(MERGING, place_id, merge.model_dump_json())

    def drop_merge_checkout(self, place_id: str) -> None:
        """Remove the merge laid out for a team's work, if there is one."""
        merge = self.merge_checkout(place_id)
        if merge is not None:
            self.environment.checkouts.drop_merge(self.repo_dir(), Path(merge.where))
            self.run_state().unmark(MERGING, place_id)

    def scope_of(
        self, place: SessionTeam, role: str, hand_over: str | None = None
    ) -> Scope:
        """What an agent of ``place`` may write.

        Args:
            hand_over: The IC's hand-over conversation, when the scope is
                that conversation's and not the IC's own.
        """
        store = self.org()
        children = store.below(place.id)
        kind = self.team_types.type_of(place)
        reviewer = self.team_types.team(place).reviewer
        # A team that has not started holds no one out of its directory:
        # the IC above may finish what it has there.
        started = [child for child in children if child.agents]
        merges = []
        for team_id, kept in self.run_state().marked(MERGING).items():
            merge = MergeCheckout.model_validate_json(kept)
            team = store.place(team_id)
            merges.append(
                MergeArea(
                    where=merge.where,
                    files=tuple(merge.files),
                    own=role == reviewer
                    and team is not None
                    and team.parent_id == place.id,
                )
            )
        within = scope_within(
            place,
            started,
            # A hand-over conversation is no agent of the place.
            checkout="" if hand_over else place.worktree_path,
            edits=role == kind.writer,
        )
        repo_checkout = self.environment.checkouts.root_of(str(self.repo_dir()))
        return replace(
            within,
            # It writes in its boundary from its main conversation alone;
            # at a Division Lead place it only reviews, in the human's own
            # checkout.
            writes=role == kind.writer
            and hand_over is None
            and not is_division_lead(place),
            aside=hand_over is not None,
            # The repository's own checkout is the human's, on whatever
            # branch they have it.
            branch=place.branch or ""
            if place.worktree_path
            and self.environment.checkouts.root_of(place.worktree_path) != repo_checkout
            else "",
            # Its git commits nothing in a sub-team's directory; it may still
            # write in that of a team that has not started.
            hidden=tuple(child.boundary.path for child in children),
            merges=tuple(merges),
            merge_root=str(self.merges_for(place.id))
            if role == reviewer and hand_over is None
            else "",
        )

    def check(
        self, scope: Scope | None, tool_name: str, action: Mapping[str, Any]
    ) -> Denied | None:
        """Why an action of a file tool may not run, if it may not.

        Args:
            scope: The scope of the agent; None for a conversation that is
                held to none, which is still kept from the director's own
                file.
        """
        return self._confinement().check(scope, tool_name, action)

    def confine(
        self, place_id: str, role: str | None = None, hand_over: str | None = None
    ) -> ToolParams | None:
        """Write the shell an agent's terminal starts.

        Args:
            role: The agent's member; None for the one that works in the
                team's folder.

        Returns:
            What to start the agent's terminal with, or None when it needs
            no shell of its own.
        """
        place = self.org().place(place_id)
        if place is None:
            return None
        team = self.team_types.team(place)
        role = role or team.writer
        conversation_id = hand_over or team.conversation(role)
        if conversation_id is None:
            return None
        return self._confinement().confine(
            conversation_id, self.scope_of(place, role, hand_over)
        )

    def confine_all(self) -> None:
        """Write the shell of every agent of the org again. One that
        cannot be written is logged and the rest are written."""
        for node in self.org().places():
            for agent in node.agents:
                try:
                    self.confine(node.id, agent.role)
                except (EnvironmentFailed, OSError):
                    logger.exception("Could not write the shell of team %s", node.id)

    def forget_shells(self, team: SessionTeam) -> None:
        """Remove what the shells of a team's agents were made of."""
        self._confinement().forget(
            [agent.conversation_id for agent in team.agents if agent.conversation_id],
            Path(team.worktree_path),
        )

    def adapter(
        self, conversation_id: str, offered: Sequence[str]
    ) -> Callable[[AgentBase], AgentBase]:
        """What is changed on an agent as its conversation starts: of the
        director's tools it gets the ones in ``offered``, and the tools
        that start agents nothing holds are taken from it."""
        return self._confinement().adapter(
            conversation_id,
            offered,
            known=self.team_types.tool_names,
            never=_SUB_AGENT_TOOLS,
        )

    def with_tools_again(self, conversation_id: UUID, agent: AgentBase) -> AgentBase:
        """The saved agent of a conversation that is loaded again, able to
        reach the director's tools. Safe from any thread."""
        return self._confinement().with_tools_again(str(conversation_id), agent)

    def guard_action(self, conversation_id: UUID, event: ActionEvent) -> str | None:
        """Why an agent of the org may not carry out an action, if it may not.

        A file tool is refused a write outside the boundary of the agent's
        team. Called on the agent's own thread, before the action runs.
        """
        store = self.find_org()
        if store is None or event.action is None:
            return None
        action = event.action.model_dump()
        if (
            not named_paths(event.tool_name, action)
            and searched_folder(event.tool_name, action) is None
        ):
            # Nothing a scope is asked about: most actions.
            return None
        found = store.find_by_conversation(str(conversation_id))
        if found is not None:
            scope = self.scope_of(*found)
        else:
            aside = store.find_hand_over(str(conversation_id))
            scope = (
                self.scope_of(
                    aside, self.team_types.team(aside).writer, str(conversation_id)
                )
                if aside
                else None
            )
        denied = self.check(scope, event.tool_name, action)
        if denied is None:
            return None
        works = found is not None and found[1] == self.team_types.team(found[0]).writer
        if found is not None and works and denied.relative is not None:
            # The refusal stands whether or not it could be kept.
            with suppress(Exception):
                self.record_need(found[0], denied.relative or ".", NEED_WRITE)
        return worded(
            denied,
            scope or Scope(checkout="", directory=""),
            names=self.team_types.names_at(found[0].id if found else ""),
        )
