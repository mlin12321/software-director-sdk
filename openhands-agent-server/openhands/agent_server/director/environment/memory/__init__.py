"""An org's records kept in memory: for tests, and to hold the
interfaces of `director.environment` to what they say and not to git.

Records and the remote only. Where the teams' work is (`Checkouts`) and
what holds agents to their scopes (`Confinement`) are another
environment's, which `MemoryRecordsEnvironment` is given.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from openhands.agent_server.director.environment import CodingEnvironment
from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.environment.confinement import Confinement
from openhands.agent_server.director.environment.memory.memory_remote import (
    MemoryRemote,
)
from openhands.agent_server.director.environment.memory.memory_store import (
    ORG_TREE,
    TEAM_MEMORY,
    MemoryRecords,
    MemoryRecordStore,
    MemoryRepository,
)
from openhands.agent_server.director.environment.records import Staged


__all__ = [
    "ORG_TREE",
    "TEAM_MEMORY",
    "MemoryRecordStore",
    "MemoryRecords",
    "MemoryRecordsEnvironment",
    "MemoryRemote",
    "MemoryRepository",
]


@dataclass
class MemoryRecordsEnvironment(CodingEnvironment):
    """The teams' work where another environment has it, and the org's
    records in memory."""

    work: CodingEnvironment
    """Where the teams' work is, and what confines their agents."""
    records: MemoryRecords = field(default_factory=MemoryRecords)
    """The records of each repository."""

    @property
    def checkouts(self) -> Checkouts:
        return self.work.checkouts

    @property
    def remote(self) -> MemoryRemote:
        return MemoryRemote(self.records)

    def confinement(
        self,
        repository: Path,
        *,
        checkouts: tuple[Path, ...],
        sandbox: str | None,
        secret: Path | None,
        server_url: str | None,
        key: Callable[[], bytes],
    ) -> Confinement:
        return self.work.confinement(
            repository,
            checkouts=checkouts,
            sandbox=sandbox,
            secret=secret,
            server_url=server_url,
            key=key,
        )

    def team_memory(self, repository: Path) -> MemoryRecordStore:
        return self.records.of(repository).store(TEAM_MEMORY)

    def org_tree(self, repository: Path) -> MemoryRecordStore:
        return self.records.of(repository).store(ORG_TREE)

    def land_together(self, repository: Path, staged: Sequence[Staged]) -> bool:
        return self.records.of(repository).land_together(staged)

    def state_dir(self, repository: Path) -> Path:
        return self.work.state_dir(repository)
