"""A team as the director reads it: who is in it and whom it turns to.

A view over one place of the org and its type. It keeps nothing: the place
is the org store's, the members are the type's.
"""

from dataclasses import dataclass

from openhands.agent_server.director.agent_types import Names, TeamTemplate
from openhands.agent_server.director.models import AgentRef, SessionTeam
from openhands.agent_server.director.org_store import OrgStore, is_division_lead


@dataclass(frozen=True)
class Team:
    """One team of the org.

    Attributes:
        place: The team's place in the org.
    """

    org: OrgStore
    place: SessionTeam

    @property
    def kind(self) -> TeamTemplate:
        return self.org.type_of(self.place)

    @property
    def is_lead(self) -> bool:
        """Whether it only reviews the work of a session."""
        return is_division_lead(self.place)

    @property
    def members(self) -> tuple[str, ...]:
        """The members its type names, started or not."""
        return tuple(self.kind.agents)

    @property
    def writer(self) -> str:
        """The member that works in the team's folder: the one a task, a
        message from above and a person's words go to."""
        return self.kind.writer

    def member_with(self, tool: str) -> str | None:
        """The member that has ``tool`` here."""
        return self.kind.member_with(tool, lead=self.is_lead)

    @property
    def reviewer(self) -> str:
        """The member that reviews the work of the teams below, and that a
        team below turns to."""
        kind = self.kind
        return (
            self.member_with("review_team")
            or kind.member_with("review_team")
            or kind.writer
        )

    @property
    def manager(self) -> str:
        """The member that settles what the team's sub-teams are: the one
        that approves them, and where the team only reviews the work of a
        session, the one that settles that session's."""
        kind = self.kind
        tool = "settle_teams" if self.is_lead else "approve_teams"
        return (
            kind.member_with(tool, lead=self.is_lead)
            or kind.member_with("approve_teams")
            or kind.writer
        )

    @property
    def names(self) -> Names:
        """What the members are called here, and what the team above calls
        the ones this team turns to."""
        names = self.kind.names
        above = self.org.place(self.place.parent_id) if self.place.parent_id else None
        return names if above is None else names.under(self.org.type_of(above).names)

    def title(self, member: str) -> str:
        """What ``member`` is called in what the director writes."""
        return self.kind.title_of(member)

    def answers(self, raised_by: str) -> str:
        """The member here that an escalation of a team below is for: the
        one of the same name as the member that raised it, and where this
        team has none, the one the teams below turn to."""
        return raised_by if raised_by in self.members else self.reviewer

    def agent(self, member: str) -> AgentRef | None:
        """The agent that is ``member``, once the team has one."""
        return next((a for a in self.place.agents if a.role == member), None)

    def conversation(self, member: str) -> str | None:
        """The conversation of ``member``, if it has one."""
        agent = self.agent(member)
        return agent.conversation_id if agent else None

    @property
    def above(self) -> "Team | None":
        """The team it answers to, along the privileged edge."""
        parent = self.place.parent_id
        place = self.org.place(parent) if parent else None
        return Team(self.org, place) if place else None

    @property
    def below(self) -> list["Team"]:
        """The teams that answer to it."""
        return [Team(self.org, place) for place in self.org.below(self.place.id)]
