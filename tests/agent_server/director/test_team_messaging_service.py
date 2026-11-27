"""What is said between teams, by itself: the services it is given over
a small tree, and a launcher that records what it is sent."""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from openhands.agent_server.director.environment.git import GitCodingEnvironment
from openhands.agent_server.director.environment.git.git_store import GitStore
from openhands.agent_server.director.models import DirectorSettings, Stopped
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.run_state import CHECKS, PROGRESS
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
)
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.resumable_work_service import (
    ResumableWorkService,
)
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_messaging_service import (
    TeamMessagingService,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.test_team_documents import RootAboveApi
from tests.agent_server.director.test_team_member_start_service import ROOT, A, Starts


NOW = datetime(2026, 1, 2, tzinfo=UTC)


class Messaging(Starts):
    """The messaging service over the same tree, with both ICs running."""

    def __init__(self, repo: Path) -> None:
        super().__init__(repo)
        self.launcher.known[self.agent] = ConversationExecutionStatus.IDLE
        session_drafts = SessionDraftService(run_state=self.org_state.run_state)
        self.history = TeamHistoryService(
            org=lambda: self.store,
            team_memory=lambda: GitStore(repo),
            require_team_memory=lambda: GitStore(repo),
            session_drafts=session_drafts,
            org_changed=lambda session: None,
        )
        self.joined: list[str] = []
        self.messaging = TeamMessagingService(
            release_reviews=lambda place_id: None,
            org=lambda: self.store,
            org_state=self.org_state,
            org_settings=OrgSettingsService(
                environment=GitCodingEnvironment(),
                config_file=None,
                remembered=RememberedFile(None),
                team_types=self.team_types,
                tree=RootAboveApi(),
                documents=self.documents,
                org=lambda: self.store,
                repo_dir=lambda: repo,
                max_concurrent_runs=None,
            ),
            org_save=self.org_save,
            team_types=self.team_types,
            team_history=self.history,
            session_drafts=session_drafts,
            documents=self.documents,
            check_counts=CheckCountService(
                org=lambda: self.store, run_state=self.org_state.run_state
            ),
            message_delivery=self.message_delivery,
            member_start=self.service,
            resumable_work=ResumableWorkService(
                run_state=self.org_state.run_state,
                work_queue=self.queue,
                lane_of=lambda intent: intent.node_id,
                is_paused=lambda: self.org_state.paused is not None,
                save=self.org_save.save,
                save_soon=self.org_save.save_soon,
                let_go=self.message_delivery.let_go,
                keep_failure=lambda intent, error: None,
                left=self.start_line.leave,
                saved_before_forgotten=frozenset(),
            ),
            work_queue=self.queue,
            launcher=lambda: self.launcher,
            joined=self.joined.append,
            clock=lambda: NOW,
        )

    def sent(self) -> list[str]:
        return self.launcher.messages_to(str(self.agent))


@pytest.fixture
async def messaging(repo: Path) -> AsyncIterator[Messaging]:
    made = Messaging(repo)
    async with made.queue:
        yield made


async def test_what_the_director_tells_a_team_reaches_the_member_that_works(
    messaging: Messaging,
):
    told = messaging.messaging.tell_writer(A, "Mind the deadline.")
    await messaging.queue.settled()

    assert told == str(messaging.agent)
    [text] = messaging.sent()
    assert "Mind the deadline." in text
    assert messaging.joined == []


async def test_what_the_human_says_marks_the_conversation_as_joined(
    messaging: Messaging,
):
    messaging.messaging.tell_writer(A, "Mind the deadline.", human=True)
    await messaging.queue.settled()

    assert messaging.joined == [str(messaging.agent)]


async def test_a_message_between_teams_is_kept_in_their_thread(messaging: Messaging):
    sent = messaging.messaging.send_msg(ROOT, "ic", A, "ic", "Start with the lexer.")
    await messaging.queue.settled()

    # Nothing was stopped, so nothing was set going again.
    assert not sent
    [text] = messaging.sent()
    assert "Start with the lexer." in text


async def test_a_failure_is_recorded_above_and_waits_for_the_human(
    messaging: Messaging,
):
    messaging.messaging.tell_of_failure(A, "ic", "It could not be started.")
    await messaging.queue.settled()

    [event] = messaging.history.history_of(ROOT)
    assert (event.kind, event.team_id) == ("failed", "a")
    assert "It could not be started." in event.detail


def test_progress_is_counted_from_the_clock_it_is_given(messaging: Messaging):
    messaging.messaging.progressed(A)

    seen = json.loads(messaging.org_state.run_state().marked(PROGRESS)[A])
    assert seen["at"] == NOW.isoformat()
    messaging.messaging.progressed(None)


async def test_a_manager_is_woken_once_enough_reviews_were_accepted(
    messaging: Messaging,
):
    limit = DirectorSettings().manager_check_accepted_reviews
    for _ in range(limit - 1):
        messaging.messaging.count_accepted_review(ROOT)
    kept = json.loads(messaging.org_state.run_state().marked(CHECKS)[ROOT])
    assert kept["accepted"] == limit - 1

    messaging.messaging.count_accepted_review(ROOT)
    await messaging.queue.settled()

    kept = json.loads(messaging.org_state.run_state().marked(CHECKS)[ROOT])
    assert kept["accepted"] == 0


async def test_the_agents_of_a_stopped_team_that_run_are_paused(
    messaging: Messaging,
):
    messaging.org_state.running.add(str(messaging.agent))
    await messaging.messaging.stop(A)
    assert messaging.launcher.paused == []

    messaging.store.update_place(A, stopped=Stopped(at=NOW.isoformat(), reason="Wait."))
    await messaging.messaging.stop(A)

    assert messaging.launcher.paused == [messaging.agent]
