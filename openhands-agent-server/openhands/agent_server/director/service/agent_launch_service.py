"""What reaches the agents' conversations while the director runs agents."""

import asyncio
from dataclasses import dataclass, field

from openhands.agent_server.director.launch import TeamLauncher
from openhands.sdk.errors import Unavailable


@dataclass
class AgentLaunchService:
    """Holds the launcher the director runs agents through and the loop it
    works on, from :meth:`begin` to :meth:`end`.

    Before and after, the director runs no agents on this server: the tree
    can still be read and taught.
    """

    launcher: TeamLauncher | None = field(default=None, init=False)
    """What starts and reaches conversations; None while no agents run."""
    loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)
    """The loop the director works on; None while no agents run."""

    @property
    def is_running(self) -> bool:
        """Whether the director runs agents on this server now."""
        return self.launcher is not None

    def require(self) -> TeamLauncher:
        """The launcher.

        Raises:
            Unavailable: The director runs no agents on this server.
        """
        if self.launcher is None:
            raise Unavailable("The director is not running agents on this server")
        return self.launcher

    def begin(self, launcher: TeamLauncher) -> None:
        """Run agents through ``launcher``, on the running loop."""
        self.loop = asyncio.get_running_loop()
        self.launcher = launcher

    def end(self) -> None:
        self.launcher = None
        self.loop = None
