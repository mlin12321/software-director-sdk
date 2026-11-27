"""Holds an agent to its scope.

Three things hold it: the shell its terminal starts (a sandbox where the
server has one, and a ``git`` that keeps to the agent's folder), a check of
every action of a file tool, and, for an agent that runs its own tools, the
script its whole process is started by. None of them knows what a team is.
"""

import os
import shutil
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openhands.agent_server.conversation_service import ToolParams
from openhands.agent_server.director.acp import (
    MCP_SERVER_NAME,
    agent_token,
    as_org_agent,
    tools_server,
)
from openhands.agent_server.director.environment.git.cli import (
    GitError,
    checkout_root,
    common_dir,
    git,
)
from openhands.agent_server.director.environment.git.git_index import (
    excludes_file,
    git_script,
    rcfile,
)
from openhands.agent_server.director.sandbox import (
    command_script,
    shell_script,
    write_shell,
)
from openhands.agent_server.director.scope import (
    Denied,
    Scope,
    named_paths,
    searched_folder,
    written_paths,
)
from openhands.sdk.agent import ACPAgent
from openhands.sdk.agent.base import AgentBase
from openhands.sdk.errors import Unavailable
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.spec import Tool
from openhands.sdk.utils.files import atomic_write_text


logger = get_logger(__name__)

# Where the shells of the agents are kept, in the repository's git directory.
SHELLS_DIR = "director-shells"
# What no agent's shell may write in the git directory, besides its shells.
PRIVATE = ("director", "hooks")
# The script an agent that runs its own tools is started by.
ACP_LAUNCHER = ".acp.sh"

Enforcement = Literal["sandbox", "checks", "merge"]


def is_in(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


@dataclass(frozen=True)
class Confinement:
    """What holds the agents of one repository to their scopes.

    Attributes:
        repo_dir: The repository.
        checkouts: Where its checkouts are: the repository and the
            worktrees.
        bwrap: The bubblewrap binary; None leaves the shells unconfined.
        secret: A file of the director's own that no agent may open.
        server_url: Where an agent that runs its own tools reaches the
            server; None when that is not known.
        key: Gives what the tokens of such agents are signed with.
    """

    repo_dir: Path
    checkouts: tuple[Path, ...]
    bwrap: str | None
    secret: Path | None
    server_url: str | None
    key: Callable[[], bytes]

    @property
    def shells(self) -> Path:
        return common_dir(self.repo_dir) / SHELLS_DIR

    def enforcement(self, own_tools: bool) -> Enforcement:
        """What holds an agent here.

        Args:
            own_tools: Whether the agent runs its own tools.
        """
        if self.bwrap is not None:
            return "sandbox"
        return "merge" if own_tools else "checks"

    def confine(self, conversation_id: str, scope: Scope) -> ToolParams | None:
        """Write the shell an agent's terminal starts.

        Where the server can, the shell holds the agent to its scope. The
        shell of an agent that writes also finds a ``git`` that keeps the
        agent's index apart from those of the agents it shares a checkout
        with.

        Returns:
            What to start the agent's terminal with, or None when it needs
            no shell of its own.
        """
        bash, real_git = shutil.which("bash"), shutil.which("git")
        own_git = scope.writes and os.name != "nt" and bash and real_git
        if self.bwrap is None and not own_git:
            return None
        shells = self.shells
        git_dir = str(shells.parent)
        hidden = list(scope.hidden)
        closed = [folder.path for folder in scope.closed]
        rc: Path | None = None
        bin_dir: Path | None = None
        if own_git and real_git:
            rc = shells / f"{conversation_id}.rc"
            bin_dir = shells / f"{conversation_id}.bin"
            excludes = shells / f"{conversation_id}.exclude"
            write_shell(
                bin_dir / "git",
                git_script(
                    real_git,
                    common_dir=git_dir,
                    conversation_id=conversation_id,
                    directory=scope.directory,
                    sub_team_directories=hidden,
                    excludes=excludes,
                    flock=shutil.which("flock"),
                    readonly=list(scope.readonly),
                    writable=list(scope.writable),
                    checkout=str(checkout_root(scope.checkout)),
                    branch=scope.branch,
                ),
            )
            atomic_write_text(
                excludes, excludes_file(self.repo_dir, scope.directory, hidden)
            )
            atomic_write_text(rc, rcfile(bin_dir))
        # An agent's commits need the git directory, but not what the director
        # keeps there, nor the hooks git runs for everyone.
        private = [shells, *(shells.parent / name for name in PRIVATE)]
        # Made now: one missing would be left for an agent to make.
        for path in private:
            path.mkdir(exist_ok=True)
        script = shells / f"{conversation_id}.sh"
        checkouts = list(self.checkouts)
        merges = Path(scope.merge_root) if scope.merge_root else None
        if merges is not None:
            merges.mkdir(parents=True, exist_ok=True)
        directory = scope.directory if scope.writes else None
        write_shell(
            script,
            shell_script(
                self.bwrap,
                bash or "/bin/sh",
                checkouts=checkouts,
                private=private,
                secret=self.secret,
                directory=directory,
                sub_team_directories=closed,
                merges=merges,
                rcfile=rc,
            ),
        )
        # An agent that runs its own tools has no terminal of the server's;
        # its whole process is started by this instead.
        write_shell(
            shells / f"{conversation_id}{ACP_LAUNCHER}",
            command_script(
                self.bwrap,
                checkouts=checkouts,
                private=private,
                secret=self.secret,
                directory=directory,
                sub_team_directories=closed,
                merges=merges,
                first_on_path=bin_dir,
            ),
        )
        # Only the subprocess terminal starts the shell it is given.
        return {"terminal": {"terminal_type": "subprocess", "shell_path": str(script)}}

    def adapter(
        self,
        conversation_id: str,
        offered: Sequence[str],
        *,
        known: Collection[str],
        never: Collection[str] = (),
    ) -> Callable[[AgentBase], AgentBase]:
        """What is changed on an agent as its conversation starts.

        An agent that runs its own tools gets the director's through an MCP
        server and is started by the script that holds its process to its
        scope. Any other agent keeps the tools it has and, of the
        director's, gets the ones in ``offered``: an agent derived from
        another carries that agent's.

        Args:
            known: Every tool of the director's, offered or not.
            never: Tools no agent here keeps, whoever gave them.
        """
        launchers = self.shells
        launcher = launchers / f"{conversation_id}{ACP_LAUNCHER}"

        def adapt(agent: AgentBase) -> AgentBase:
            if not isinstance(agent, ACPAgent):
                kept = [
                    tool
                    for tool in agent.tools
                    if tool.name not in never
                    and (tool.name not in known or tool.name in offered)
                ]
                has = {tool.name for tool in kept}
                added = [Tool(name=name) for name in offered if name not in has]
                return agent.model_copy(update={"tools": [*kept, *added]})
            if self.server_url is None:
                raise Unavailable(
                    "The director does not know the address of its own "
                    "server, which an ACP agent needs to reach its tools"
                )
            if self.bwrap is not None and not launcher.exists():
                logger.warning(
                    "The ACP agent of conversation %s starts with nothing that "
                    "holds its process to its team's folder: %s is missing",
                    conversation_id,
                    launcher,
                )
            return as_org_agent(
                agent,
                tools=tools_server(
                    self.server_url, agent_token(self.key(), conversation_id)
                ),
                launcher=launcher if launcher.exists() else None,
                launchers=launchers,
            )

        return adapt

    def with_tools_again(self, conversation_id: str, agent: AgentBase) -> AgentBase:
        """The saved agent of a conversation that is loaded again, able to
        reach the director's tools.

        An agent that runs its own tools reaches them through an MCP server
        that is given its token in its environment. A server with no key to
        keep secrets under does not keep that between runs, so it is made
        again here: the token is worked out from the director's own key,
        which is kept. Any other agent is returned as it is.
        """
        if not isinstance(agent, ACPAgent) or MCP_SERVER_NAME not in agent.mcp_config:
            return agent
        if self.server_url is None:
            return agent
        tools = tools_server(self.server_url, agent_token(self.key(), conversation_id))
        return agent.model_copy(
            update={"mcp_config": {**agent.mcp_config, MCP_SERVER_NAME: tools}}
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
        secret = self.secret and os.path.realpath(self.secret)
        workspace = scope.checkout if scope else ""
        for named in named_paths(tool_name, action):
            # Read, the key in it lets an agent act as any other.
            if os.path.realpath(os.path.join(workspace, named)) == secret:
                return Denied("private", named)
        searched = searched_folder(tool_name, action)
        if secret and searched is not None:
            folder = os.path.realpath(os.path.join(workspace, searched))
            # What a search finds in it tells of it, a character at a time.
            if secret.startswith(folder.rstrip(os.sep) + os.sep):
                return Denied("holds_private", searched)
        paths = written_paths(tool_name, action)
        if not paths or scope is None:
            return None
        if scope.aside:
            return Denied("aside", paths[0])
        top = checkout_root(scope.checkout)
        for named in paths:
            target = Path(os.path.realpath(Path(scope.checkout) / named))
            merging = next(
                (
                    merge
                    for merge in scope.merges
                    if is_in(target, Path(os.path.realpath(merge.where)))
                ),
                None,
            )
            if merging is not None:
                # Only what does not merge is its reviewer's to write.
                if not merging.own:
                    return Denied("merge_of_another", named)
                relative = target.relative_to(os.path.realpath(merging.where))
                if relative.as_posix() not in merging.files:
                    return Denied("merged_by_itself", named, files=merging.files)
            elif is_in(target, top):
                relative = target.relative_to(top).as_posix()
                denied = scope.refuse("" if relative == "." else relative, named)
                if denied is not None:
                    return denied
            elif any(is_in(target, checkout) for checkout in self.checkouts):
                return Denied("other_checkout", named)
            elif is_in(target, Path(os.path.realpath(common_dir(self.repo_dir)))):
                # A git directory that is not inside its checkout.
                return Denied("git_directory", named)
        return None

    def forget(self, conversation_ids: Iterable[str], checkout: Path) -> None:
        """Remove what the shells of agents that worked in ``checkout`` were
        made of."""
        try:
            common, own = (
                Path(
                    git(
                        directory, ["rev-parse", "--path-format=absolute", kind]
                    ).strip()
                )
                for directory, kind in (
                    (self.repo_dir, "--git-common-dir"),
                    (checkout, "--git-dir"),
                )
            )
        except (GitError, OSError):
            return
        for name in conversation_ids:
            for left in (
                *(common / SHELLS_DIR).glob(f"{name}.*"),
                *own.glob(f"agent-index-{name}*"),
            ):
                if left.is_dir():
                    shutil.rmtree(left, ignore_errors=True)
                else:
                    left.unlink(missing_ok=True)
