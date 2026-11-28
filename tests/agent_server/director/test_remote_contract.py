"""What exchanging an org's records with a remote does, whatever keeps
them: the same tests against git and against memory."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from openhands.agent_server.director.environment import CodingEnvironment
from openhands.agent_server.director.environment.git import GitCodingEnvironment
from openhands.agent_server.director.environment.memory import (
    MemoryRecordsEnvironment,
)
from openhands.agent_server.director.environment.records import StoreChange
from openhands.sdk.errors import NotFound, UpstreamFailed
from tests.agent_server.director.conftest import run_git


@dataclass
class Exchange:
    """A repository and another one that is its remote ``origin``."""

    environment: CodingEnvironment
    here: Path
    there: Path

    def write(self, repository: Path, file: str, text: str) -> None:
        """Change both of a repository's stores."""
        for store in (
            self.environment.org_tree(repository),
            self.environment.team_memory(repository),
        ):
            store.commit(lambda read: [StoreChange(file, text)], f"write {file}")

    def fetch(self) -> None:
        self.environment.remote.fetch(self.here, "origin")

    def push(self) -> None:
        self.environment.remote.push(self.here, "origin")

    def behind(self) -> bool:
        return self.environment.remote.behind(self.here, "origin")

    def merge(self) -> None:
        """Take in the team memory that was fetched."""
        remote = self.environment.remote
        self.environment.team_memory(self.here).merge(remote.team_memory_name("origin"))
        fetched = remote.org_tree_of(self.here, "origin")
        assert fetched is not None
        self.environment.org_tree(self.here).merge(fetched)


def in_git(repo: Path, tmp_path: Path) -> Exchange:
    there = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--quiet", "--bare", str(there))
    run_git(repo, "remote", "add", "origin", str(there))
    return Exchange(GitCodingEnvironment(), repo, there)


def in_memory(repo: Path, tmp_path: Path) -> Exchange:
    environment = MemoryRecordsEnvironment(work=GitCodingEnvironment())
    there = tmp_path / "remote"
    environment.records.add_remote(repo, "origin", there)
    return Exchange(environment, repo, there)


@pytest.fixture(params=[in_git, in_memory], ids=["git", "memory"])
def exchange(request: pytest.FixtureRequest, repo: Path, tmp_path: Path) -> Exchange:
    return request.param(repo, tmp_path)


def test_a_remote_the_repository_does_not_have_cannot_be_fetched(
    exchange: Exchange,
):
    with pytest.raises(NotFound):
        exchange.environment.remote.fetch(exchange.here, "elsewhere")


def test_a_remote_without_records_has_nothing_to_fetch(exchange: Exchange):
    exchange.write(exchange.here, "a", "ours")

    exchange.fetch()

    remote = exchange.environment.remote
    assert remote.team_memory_of(exchange.here, "origin") is None
    assert remote.org_tree_of(exchange.here, "origin") is None
    assert not exchange.behind()


def test_a_push_gives_the_remote_both_stores(exchange: Exchange):
    exchange.write(exchange.here, "a", "ours")

    exchange.push()

    environment = exchange.environment
    assert environment.team_memory(exchange.there).read("a") == "ours"
    assert environment.org_tree(exchange.there).read("a") == "ours"
    exchange.fetch()
    assert not exchange.behind()


def test_what_is_fetched_stays_beside_ours_until_it_is_merged(exchange: Exchange):
    exchange.write(exchange.here, "a", "ours")
    exchange.push()
    exchange.write(exchange.there, "b", "theirs")

    exchange.fetch()

    environment = exchange.environment
    assert exchange.behind()
    assert environment.team_memory(exchange.here).read("b") is None
    assert (
        environment.remote.team_memory_of(exchange.here, "origin")
        == environment.team_memory(exchange.there).head()
    )

    exchange.merge()

    assert environment.team_memory(exchange.here).read("b") == "theirs"
    assert environment.org_tree(exchange.here).read("b") == "theirs"
    assert not exchange.behind()


def test_a_push_is_refused_while_the_remote_has_what_is_not_here(
    exchange: Exchange,
):
    exchange.write(exchange.here, "a", "ours")
    exchange.push()
    exchange.write(exchange.there, "b", "theirs")
    exchange.write(exchange.here, "c", "ours again")

    with pytest.raises(UpstreamFailed):
        exchange.push()

    assert exchange.environment.team_memory(exchange.there).read("c") is None
