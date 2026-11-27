"""What the members of a team recorded: its reports, the one log the
director keeps for a team."""

from collections.abc import Callable
from dataclasses import dataclass

from openhands.agent_server.director.documents.store import REPORT, TeamRef
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.models import OrgNode, Report
from openhands.agent_server.director.org_store import OrgStore, instant, session_of
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.sdk.errors import NotFound, NotSupported


@dataclass
class TeamReportService:
    """Reads and adds to the reports of a team.

    What a session adds is its own until its work is accepted; reading
    puts it after what the org has. The store of documents is given this
    service as the keeper of its logs (``entries`` and ``append``).
    """

    org: Callable[[], OrgStore]
    """The org whose teams report."""
    team_memory: Callable[[], GitStore]
    """Where the reports the org has are."""
    session_drafts: SessionDraftService
    """Where what a session reported waits for its work to be accepted."""

    def reports_of(self, node: OrgNode) -> list[Report]:
        """What the two agents of a team recorded, oldest first. Reports
        are the team's: an agent started again reads what it wrote before."""
        team_memory = self.team_memory()
        reports = (
            team_memory.reports(node.boundary.path) if self.org().is_orgs(node) else []
        )
        own = self.session_drafts.reports(node.id)
        return sorted([*reports, *own], key=lambda report: report.at)

    def entries(self, team: TeamRef, name: str) -> list[Report]:
        """The one log the director keeps for a team, as the store of
        documents is asked for it: what its members recorded."""
        place = self.org().place(team.id)
        return self.reports_of(place) if place and name == REPORT else []

    def append(self, team: TeamRef, name: str, entry: Report) -> None:
        """Add to what the members of a team recorded, as the store of
        documents is asked to.

        Raises:
            NotSupported: It is another log than the one the director keeps.
            NotFound: The team is no place of a session.
        """
        if name != REPORT:
            raise NotSupported(f"The director keeps no log named {name!r}")
        place = self.org().place(team.id)
        if place is None:
            raise NotFound(f"There is no team {team.id}")
        said_where = {
            "session": self.org().session_title(place),
            "session_id": session_of(place),
        }
        # Its session's alone until the session's work is accepted.
        self.session_drafts.keep_report(
            place.id,
            entry.model_copy(update={"at": entry.at or instant(), **said_where}),
        )
