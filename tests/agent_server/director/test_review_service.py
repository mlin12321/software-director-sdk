"""The stages of a review by themselves: the two review services over a
small tree, built from what they are given, with no director."""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.environment.git import GitCheckouts
from openhands.agent_server.director.environment.git.git_store import GitStore
from openhands.agent_server.director.models import Review
from openhands.agent_server.director.run_state import TURN
from openhands.agent_server.director.service.review_service import ReviewService
from openhands.agent_server.director.service.session_merge_service import (
    SessionMergeService,
)
from openhands.agent_server.director.service.session_record_service import (
    SessionRecordService,
)
from tests.agent_server.director.test_team_documents import AnyOrg, RootAboveApi
from tests.agent_server.director.test_team_member_start_service import SESSION, A
from tests.agent_server.director.test_team_messaging_service import NOW, Messaging


B = f"b@{SESSION}"


class Reviews(Messaging):
    """Both review services over the same tree. What no stage tested here
    asks (a model, a checkout laid out for a merge, summaries) is absent."""

    def __init__(self, repo: Path) -> None:
        super().__init__(repo)
        absent = cast(Any, None)

        def team_documents() -> TeamDocuments:
            return TeamDocuments(
                self.documents, self.team_types.protocols, AnyOrg(RootAboveApi())
            )

        session_drafts = self.messaging.session_drafts
        self.records = SessionRecordService(
            org=lambda: self.store,
            team_memory=lambda: GitStore(repo),
            require_team_memory=lambda: GitStore(repo),
            run_state=self.org_state.run_state,
            documents=self.documents,
            team_documents=team_documents,
            team_types=self.team_types,
            session_drafts=session_drafts,
            org_save=self.org_save,
            event=self.history.event,
            confinement=absent,
        )
        self.merges = SessionMergeService(
            org=lambda: self.store,
            team_memory=lambda: GitStore(repo),
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.messaging.org_settings,
            team_types=self.team_types,
            team_history=self.history,
            team_summaries=absent,
            team_messaging=self.messaging,
            message_delivery=self.message_delivery,
            resumable_work=self.messaging.resumable_work,
            session_records=self.records,
            confinement=absent,
        )
        self.reviews = ReviewService(
            org=lambda: self.store,
            repo_dir=lambda: repo,
            checkouts=GitCheckouts(),
            launcher=lambda: self.launcher,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.messaging.org_settings,
            team_types=self.team_types,
            team_history=self.history,
            team_context=absent,
            team_messaging=self.messaging,
            message_delivery=self.message_delivery,
            resumable_work=self.messaging.resumable_work,
            work_queue=self.queue,
            check_counts=self.messaging.check_counts,
            confinement=cast(Any, SimpleNamespace(drop_merge_checkout=lambda id: None)),
            model_calls=absent,
            session_records=self.records,
            session_merges=self.merges,
        )

    def submit(self, place_id: str) -> None:
        """Put a team's work up for review, after whatever was submitted."""
        self.store.set_review(
            place_id,
            Review(
                state="pending",
                seq=self.merges.next_submission(),
                round=1,
                at=NOW.isoformat(),
            ),
        )

    def state_of(self, place_id: str) -> str | None:
        place = self.store.place(place_id)
        assert place is not None
        return place.review.state if place.review else None


@pytest.fixture
async def reviews(repo: Path) -> AsyncIterator[Reviews]:
    made = Reviews(repo)
    async with made.queue:
        yield made


async def test_work_that_was_not_submitted_is_not_accepted(reviews: Reviews):
    await reviews.reviews._accept(A, "")

    assert reviews.state_of(A) is None


async def test_work_is_accepted_in_the_order_it_was_submitted(reviews: Reviews):
    reviews.submit(A)
    reviews.submit(B)
    first, second = reviews.store.place(A), reviews.store.place(B)
    assert first is not None and second is not None

    assert reviews.reviews._is_its_turn(first, None)
    assert not reviews.reviews._is_its_turn(second, None)

    # Its reviewer is told when the earlier work is decided on.
    assert B in reviews.org_state.run_state().marked(TURN)
    assert A not in reviews.org_state.run_state().marked(TURN)


async def test_accepting_out_of_turn_leaves_the_work_waiting(reviews: Reviews):
    reviews.submit(A)
    reviews.submit(B)

    await reviews.reviews._accept(B, "")

    assert reviews.state_of(B) == "pending"


async def test_work_whose_acceptance_is_kept_is_accepted_and_recorded(
    reviews: Reviews,
):
    reviews.submit(A)
    place, above = reviews.store.place(A), reviews.store.place(f"root@{SESSION}")
    assert place is not None and above is not None

    reviews.reviews._record_accepted(place, above, "abc123", None, (), "")

    assert reviews.state_of(A) == "accepted"
    kinds = [event.kind for event in reviews.history.history_of(above.id)]
    assert kinds == ["accepted"]
