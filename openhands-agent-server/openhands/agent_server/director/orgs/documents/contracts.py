"""What the sub-teams of a team promise each other."""

from collections.abc import Mapping

from openhands.agent_server.director.documents.events import (
    HandedOver,
    SubTeamsCreated,
    TeamRef,
    TeamRemoved,
    WorkAccepted,
)
from openhands.agent_server.director.documents.protocol import (
    DocumentProtocolBase,
    register_protocol,
)


class ContractsProtocol(DocumentProtocolBase):
    """Contracts are true of the teams they are between, so they become
    the org's when those teams do."""

    def waits_for_team_changes(self) -> bool:
        return True

    def when_sub_teams_are_created(self, created: SubTeamsCreated) -> None:
        # Contracts that named teams which are gone do not stay beside the
        # new ones.
        self._record(created.team, created.given, replace=created.replaced)

    def when_sub_team_work_is_accepted(self, accepted: WorkAccepted) -> None:
        self._record(accepted.team, accepted.given, replace=False)

    def when_removed_team_is_handed_over(self, handed: HandedOver) -> None:
        self._record(handed.team, handed.given, replace=True)

    def when_team_is_removed(self, removed: TeamRemoved) -> str | None:
        # The folder is left without the contracts of a team that is gone.
        ref = removed.team.keeps(self.document.name)
        contracts = self.store.read(removed.team.session, ref)
        if contracts:
            self.store.remove(removed.team.session, ref)
        return contracts or None

    def _record(
        self, team: TeamRef, given: Mapping[str, str], *, replace: bool
    ) -> None:
        """Add what was given to the team's contracts, or put it in their
        place."""
        contracts = given.get(self.document.name, "")
        if not contracts.strip():
            return
        ref = team.keeps(self.document.name)
        earlier = self.store.read(team.session, ref)
        text = f"{contracts.strip()}\n"
        if replace:
            if earlier != text:
                self.store.write(team.session, ref, text)
            return
        if earlier and text.strip() in earlier:
            return
        self.store.write(
            team.session,
            ref,
            f"{earlier.rstrip()}\n\n{text}" if earlier and earlier.strip() else text,
        )


register_protocol("contracts", ContractsProtocol)
