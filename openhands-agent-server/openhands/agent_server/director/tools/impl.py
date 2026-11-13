"""Runs a director tool for the agent that called it."""

from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import UUID

from openhands.agent_server.director.git import GitError
from openhands.agent_server.director.pod_actions import Caller, caller_of
from openhands.agent_server.director.runtime import current_director
from openhands.agent_server.director.service import DirectorService
from openhands.sdk.errors import DomainError
from openhands.sdk.tool.tool import Action, Observation, ToolExecutor


if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation


class DirectorObservation(Observation):
    """What the director answered."""


class DirectorExecutor[ActionT: Action](ToolExecutor[ActionT, DirectorObservation]):
    """Resolves the calling agent, then lets ``act`` change the org.

    What the director refuses comes back as an error the agent can read and
    act on, not as a failed tool call.
    """

    def __init__(
        self,
        conversation_id: UUID,
        act: Callable[[DirectorService, Caller, ActionT], str],
    ) -> None:
        self.conversation_id = conversation_id
        self._act = act

    def __call__(
        self,
        action: ActionT,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> DirectorObservation:
        try:
            service = current_director()
            caller = caller_of(service, self.conversation_id)
            return DirectorObservation.from_text(self._act(service, caller, action))
        except (DomainError, GitError) as error:
            return DirectorObservation.from_text(str(error), is_error=True)
