"""Holding an agent to its scope: what it may write of the checkout it
works in, and what it may not reach at all."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from openhands.agent_server.conversation_service import ToolParams
from openhands.agent_server.director.scope import Denied, Scope
from openhands.sdk.agent.base import AgentBase


Enforcement = Literal["sandbox", "checks", "merge"]


class Confinement(ABC):
    """What holds the agents of one repository to their scopes.

    Three things hold an agent: the shell its terminal starts, a check of
    every action of a file tool, and, for an agent that runs its own
    tools, the script its whole process is started by. None of them knows
    what a team is.
    """

    @abstractmethod
    def enforcement(self, own_tools: bool) -> Enforcement:
        """What holds an agent here.

        Args:
            own_tools: Whether the agent runs its own tools.
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def with_tools_again(self, conversation_id: str, agent: AgentBase) -> AgentBase:
        """The saved agent of a conversation that is loaded again, able to
        reach the director's tools.

        An agent that runs its own tools reaches them through an MCP server
        that is given its token in its environment. A server with no key to
        keep secrets under does not keep that between runs, so it is made
        again here: the token is worked out from the director's own key,
        which is kept. Any other agent is returned as it is.
        """

    @abstractmethod
    def check(
        self, scope: Scope | None, tool_name: str, action: Mapping[str, Any]
    ) -> Denied | None:
        """Why an action of a file tool may not run, if it may not.

        Args:
            scope: The scope of the agent; None for a conversation that is
                held to none, which is still kept from the director's own
                file.
        """

    @abstractmethod
    def forget(self, conversation_ids: Iterable[str], checkout: Path) -> None:
        """Remove what the shells of agents that worked in ``checkout`` were
        made of."""
