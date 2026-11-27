"""What a model writes of the teams, by itself: a store of teams, a run
state in memory and a launcher that answers by hand."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from openhands.agent_server.director.environment.git import (
    GitCodingEnvironment,
)
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.headline import TeamSummary
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.team_summary_service import (
    TeamSummaryService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    FakeLauncher,
    open_session_in,
    small_tree,
)
from tests.agent_server.director.test_team_documents import (
    RootAboveApi,
    TextsInADict,
)


A = "a@s1"


class Summaries:
    def __init__(self, repo: Path) -> None:
        self.store = OrgStore(lambda node: team_types.type_of(node))
        tree = RootAboveApi()
        team_types = TeamTypeService(
            config_file=None,
            documents=TextsInADict(),
            tree=tree,
            org=lambda: self.store,
        )
        for node in small_tree():
            self.store.upsert(node)
        open_session_in(self.store, "root", "s1", checkout=str(repo))
        self.agent = uuid4()
        self.store.set_agent(A, "ic", conversation_id=str(self.agent))
        self.launcher = FakeLauncher()
        self.launcher.known[self.agent] = ConversationExecutionStatus.IDLE
        self.launcher.summary = TeamSummary(
            title="Parser", headline="Reads every sample file"
        )
        self.queue = WorkQueueService(default_lane="org")
        self.org_state = OrgStateService(
            repo_dir=lambda: repo,
            clock=lambda: datetime.now(UTC),
            environment=GitCodingEnvironment(),
        )
        self.service = TeamSummaryService(
            org=lambda: self.store,
            team_memory=lambda: GitStore(repo),
            org_state=self.org_state,
            org_settings=OrgSettingsService(
                config_file=None,
                remembered=RememberedFile(None),
                team_types=team_types,
                tree=tree,
                org=lambda: self.store,
                repo_dir=lambda: repo,
                max_concurrent_runs=None,
            ),
            team_types=team_types,
            model_calls=ModelCallService(
                launcher=lambda: self.launcher,
                sessions=lambda: self.store.sessions(),
                call_timeout=5,
                summary_timeout=5,
            ),
            work_queue=self.queue,
            launcher=lambda: self.launcher,
        )


@pytest.fixture
async def summaries(repo: Path) -> AsyncIterator[Summaries]:
    made = Summaries(repo)
    async with made.queue:
        yield made


async def test_a_model_writes_the_headline_of_a_team(summaries: Summaries):
    before = summaries.store.place(A)
    assert before is not None

    await summaries.service.summarize(A, "Build the parser for the new format")

    after = summaries.store.place(A)
    assert after is not None
    assert after.headline == "Reads every sample file"
    assert after.title == before.title


async def test_nothing_is_asked_of_a_model_while_the_org_is_paused(
    summaries: Summaries,
):
    before = summaries.store.place(A)
    assert before is not None
    summaries.org_state.hold("human")

    await summaries.service.summarize(A, "Build the parser", rename=True)

    assert summaries.store.place(A) == before


async def test_a_report_rewrites_the_headline_in_a_lane_of_its_own(
    summaries: Summaries,
):
    before = summaries.store.place(A)
    assert before is not None

    summaries.service.refresh_headline(A, "The parser reads every sample file.")
    await summaries.queue.settled()

    after = summaries.store.place(A)
    assert after is not None and after.headline == "Reads every sample file"
    [(asked, task)] = summaries.launcher.summarized
    assert asked == summaries.agent and "every sample file" in task


async def test_a_team_that_is_gone_is_left_alone(summaries: Summaries):
    await summaries.service.summarize("gone@s1", "Build the parser")
    summaries.service.refresh_headline("gone@s1", "Done.")
    await summaries.queue.settled()
