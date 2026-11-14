"""How work a team submits is reviewed by the IC above and taken in."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    CreateOrgRequest,
    ReviewFinding,
    SplitChild,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgManagerTool,
    RaiseEscalationTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    RaiseEscalationAction,
    ReviewTeamAction,
    SubmitReviewAction,
)
from openhands.agent_server.director.tools.impl import DirectorObservation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    FakeLauncher,
    root_conversation,
    run_git,
)
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    ic,
    manager,
    node,
    proposing,
    split_root,
    use,
)


INSTRUCTIONS = "Review the diff for bugs."
WRITE_UP = "pull request description"
ACCOUNT = "Adds the checkout endpoint in app.py."
FINDING = ReviewFinding(path="services/api/app.py", line=1, issue="Returns a string.")
DONE = SubmitReviewAction(summary="Endpoint shipped.")


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    service = DirectorService(
        repo,
        save_delay=0,
        run_limit_retry=0,
        find_review_instructions=lambda: INSTRUCTIONS,
    )
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout",
            task="Build the checkout flow",
            conversation=root_conversation(repo),
        )
    )
    await service.settled()
    yield service
    await service.aclose()


def commit(checkout: Path, path: str, content: str) -> None:
    file = checkout / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    run_git(checkout, "add", path)
    run_git(checkout, "commit", "--quiet", "-m", f"Write {path}")


async def api_submits(service: DirectorService, repo: Path) -> DirectorObservation:
    commit(repo, "services/api/app.py", "order_id = '1'\n")
    return await use(service, ic(service, "services.api"), SubmitReviewTool, DONE)


async def reviewer_wakes(
    service: DirectorService, launcher: FakeLauncher, node_id: str = "root"
) -> None:
    """The IC of a team starts a turn, as when anything is sent to it."""
    reviewer = ic(service, node_id)
    assert reviewer is not None
    launcher.report(reviewer, ConversationExecutionStatus.RUNNING)
    await service.settled()


def requests(service: DirectorService, launcher: FakeLauncher) -> list[str]:
    """The review requests the top team's IC got."""
    return [text for text in launcher.messages_to(ic(service)) if "Review mode" in text]


def reject(node_id: str = "services.api") -> ReviewTeamAction:
    return ReviewTeamAction(node_id=node_id, decision="reject", findings=[FINDING])


def accept(node_id: str = "services.api", needed_above: str = "") -> ReviewTeamAction:
    return ReviewTeamAction(
        node_id=node_id, decision="accept", needed_above=needed_above
    )


async def test_submitted_work_goes_to_the_ic_above_with_the_teams_own_account(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    launcher.agent_answers = {
        WRITE_UP: ACCOUNT,
        INSTRUCTIONS: json.dumps({"findings": [FINDING.model_dump()]}),
    }
    base = node(service, "services.api").base_commit
    to_manager = len(launcher.messages_to(manager(service)))

    await api_submits(service, repo)
    await reviewer_wakes(service, launcher)

    api = node(service, "services.api")
    assert api.status == "idle"
    assert api.review is not None
    assert (api.review.state, api.review.round) == ("pending", 1)
    assert api.review.automated == [FINDING]
    request = launcher.messages_to(ic(service))[-1]
    assert api.review.description == ACCOUNT
    assert "Review mode" in request and "Endpoint shipped." in request
    assert f"git diff {base} HEAD -- services/api" in request
    assert ACCOUNT in request and "Returns a string." in request
    assert len(launcher.messages_to(manager(service))) == to_manager
    (first, write_up), (second, review) = launcher.agents_asked
    assert str(first) == str(second) == ic(service, "services.api")
    assert WRITE_UP in write_up and "services/api/app.py" in write_up
    assert INSTRUCTIONS in review and "POST /checkout" in review
    assert "order_id = '1'" not in write_up + review
    assert launcher.asked == []


async def test_work_that_is_not_committed_cannot_be_submitted(
    service: DirectorService, repo: Path
):
    await split_root(service)
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "draft.py").write_text("x = 1\n")

    refused = await use(service, ic(service, "services.api"), SubmitReviewTool, DONE)

    assert refused.is_error and "services/api/draft.py" in refused.text
    assert node(service, "services.api").review is None


@pytest.mark.parametrize("answer", [None, "I could not tell."])
async def test_a_review_request_says_what_the_team_did_not_answer(
    service: DirectorService, launcher: FakeLauncher, repo: Path, answer: str | None
):
    await split_root(service)
    launcher.agent_answers = {INSTRUCTIONS: answer}

    await api_submits(service, repo)
    await reviewer_wakes(service, launcher)

    review = node(service, "services.api").review
    assert review is not None and not review.automated_ran
    assert review.description == ""
    request = launcher.messages_to(ic(service))[-1]
    assert "did not run" in request and "gave no account" in request


async def test_without_review_instructions_the_team_is_only_asked_for_its_account(
    repo: Path, launcher: FakeLauncher
):
    service = DirectorService(repo, save_delay=0, run_limit_retry=0)
    await service.start(launcher)
    try:
        await service.create_org(
            CreateOrgRequest(
                title="Checkout", task="Build", conversation=root_conversation(repo)
            )
        )
        await service.settled()
        await split_root(service)

        await api_submits(service, repo)
        await reviewer_wakes(service, launcher)

        asked = [WRITE_UP in question for _, question in launcher.agents_asked]
        assert asked == [True]
        assert "Review mode" in launcher.messages_to(ic(service))[-1]
    finally:
        await service.aclose()


async def test_rejected_work_goes_back_to_the_team_and_comes_again_as_a_new_round(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await api_submits(service, repo)

    await use(service, ic(service), ReviewTeamTool, reject())

    api = node(service, "services.api")
    assert api.review is not None and api.review.state == "changes_requested"
    told = launcher.messages_to(ic(service, "services.api"))[-1]
    assert "services/api/app.py:1" in told and "Returns a string." in told
    assert "submit_review" in told

    commit(repo, "services/api/app.py", "order_id = 1\n")
    await use(service, ic(service, "services.api"), SubmitReviewTool, DONE)

    await reviewer_wakes(service, launcher)

    again = node(service, "services.api").review
    assert again is not None and (again.state, again.round) == ("pending", 2)
    assert "review round 2" in launcher.messages_to(ic(service))[-1]


async def test_work_is_sent_back_as_often_as_its_reviewer_asks(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    api = "services.api"

    for attempt in range(5):
        commit(repo, "services/api/app.py", f"order_id = {attempt}\n")
        await use(service, ic(service, api), SubmitReviewTool, DONE)
        sent_back = await use(service, ic(service), ReviewTeamTool, reject())
        assert not sent_back.is_error

    review = node(service, api).review
    assert review is not None
    assert (review.state, review.round) == ("changes_requested", 5)
    assert node(service, api).status == "idle"
    assert launcher.stopped == []


async def test_an_ic_submits_once_and_says_what_its_team_delivered(
    service: DirectorService, repo: Path
):
    await split_root(service)
    api = ic(service, "services.api")

    nothing_said = await use(
        service, api, SubmitReviewTool, SubmitReviewAction(summary="  ")
    )
    by_manager = await use(
        service, manager(service, "services.api"), SubmitReviewTool, DONE
    )
    await api_submits(service, repo)
    twice = await use(service, api, SubmitReviewTool, DONE)

    assert nothing_said.is_error and by_manager.is_error
    assert twice.is_error and "already waits" in twice.text


async def test_reviews_wait_until_every_sub_team_has_submitted(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)

    submitted = await api_submits(service, repo)

    assert "Stop here" in submitted.text
    assert requests(service, launcher) == []

    commit(repo, "apps/web/page.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)

    [both] = requests(service, launcher)
    assert "Team 'API'" in both and "Team 'Web'" in both
    assert "review_team` for node services.api" in both
    assert "review_team` for node apps.web" in both


async def test_a_blocked_sub_team_is_not_waited_for(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    web = ic(service, "apps.web")
    assert web is not None
    await api_submits(service, repo)

    launcher.report(web, ConversationExecutionStatus.ERROR)
    await service.settled()

    assert node(service, "apps.web").status == "blocked"
    [request] = requests(service, launcher)
    assert "Team 'API'" in request and "Team 'Web'" not in request


async def test_an_ic_that_is_in_a_turn_gets_each_review_at_once(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await reviewer_wakes(service, launcher)

    await api_submits(service, repo)

    assert len(requests(service, launcher)) == 1


async def test_an_escalation_brings_the_reviews_that_wait_with_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await api_submits(service, repo)

    await use(
        service,
        ic(service, "apps.web"),
        RaiseEscalationTool,
        RaiseEscalationAction(category="design_decision", title="Which router?"),
    )

    escalation, review = launcher.messages_to(ic(service))[-2:]
    assert "Which router?" in escalation
    assert "Review mode" in review and "Team 'API'" in review


async def test_reviews_that_waited_through_a_pause_go_out_after_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await api_submits(service, repo)
    await service.pause()
    commit(repo, "apps/web/page.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    assert requests(service, launcher) == []

    await service.resume()
    await service.settled()

    [both] = requests(service, launcher)
    assert "Team 'API'" in both and "Team 'Web'" in both


async def test_only_the_ic_directly_above_reviews_and_only_work_that_waits(
    service: DirectorService, repo: Path
):
    await split_root(service)

    nothing_waits = await use(service, ic(service), ReviewTeamTool, accept())
    await api_submits(service, repo)
    by_manager = await use(service, manager(service), ReviewTeamTool, accept())
    by_a_peer = await use(service, ic(service, "apps.web"), ReviewTeamTool, accept())
    without_findings = await use(
        service,
        ic(service),
        ReviewTeamTool,
        ReviewTeamAction(node_id="services.api", decision="reject"),
    )

    assert nothing_waits.is_error and "no work waiting" in nothing_waits.text
    assert by_manager.is_error and "Only the IC" in by_manager.text
    assert by_a_peer.is_error and "not a team directly below" in by_a_peer.text
    assert without_findings.is_error
    review = node(service, "services.api").review
    assert review is not None and review.state == "pending"


async def split_api(service: DirectorService) -> None:
    await use(
        service,
        ic(service, "services.api"),
        MsgManagerTool,
        proposing(
            [
                SplitChild(
                    title="Schema", path="services/api/schema", task="Define orders"
                )
            ],
            "Order has an id.",
        ),
    )
    await use(service, manager(service, "services.api"), ApproveTeamsTool, APPROVE)


@pytest.mark.parametrize("needed_above", ["", "Web builds its forms from the schema."])
async def test_accepted_work_is_told_further_up_only_when_a_team_there_needs_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path, needed_above: str
):
    await split_root(service)
    await split_api(service)
    schema = "services.api.schema"
    commit(repo, "services/api/schema/order.py", "id: int\n")
    await use(service, ic(service, schema), SubmitReviewTool, DONE)
    to_root = len(launcher.messages_to(ic(service)))
    to_root_manager = len(launcher.messages_to(manager(service)))

    await use(
        service,
        ic(service, "services.api"),
        ReviewTeamTool,
        accept(schema, needed_above),
    )

    review = node(service, schema).review
    assert review is not None and review.state == "accepted"
    assert (
        node(service, schema).base_commit == run_git(repo, "rev-parse", "HEAD").strip()
    )
    assert "accepted" in launcher.messages_to(ic(service, schema))[-1]
    notes = launcher.messages_to(ic(service))[to_root:]
    assert len(notes) == (1 if needed_above else 0)
    if needed_above:
        assert needed_above in notes[0] and "Schema" in notes[0]
    assert len(launcher.messages_to(manager(service))) == to_root_manager


def in_own_worktree(
    service: DirectorService, tmp_path: Path, repo: Path, *teams: str
) -> Path:
    """Give teams (the Web team, when none is named) the one worktree a
    conversation of theirs would have made."""
    teams = teams or ("apps.web",)
    name = teams[0].rsplit(".", 1)[-1]
    worktree = tmp_path / name
    run_git(repo, "worktree", "add", "--quiet", "-b", name, str(worktree))
    for team in teams:
        service.store.update_node(team, worktree_path=str(worktree), branch=name)
    return worktree


async def test_accepted_work_of_a_worktree_team_is_merged_into_the_checkout_above(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    await split_root(service)
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, "apps/web/page.tsx", "export {}\n")
    commit(repo, "README.md", "changed above\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)

    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    assert (repo / "apps" / "web" / "page.tsx").read_text() == "export {}\n"
    assert (repo / "README.md").read_text() == "changed above\n"
    assert run_git(repo, "status", "--porcelain") == ""
    review = node(service, "apps.web").review
    assert review is not None and review.state == "accepted"
    assert "merged" in launcher.messages_to(ic(service))[-1]


@pytest.mark.parametrize(
    ("path", "said"),
    [("apps/web/page.tsx", "do not merge"), ("README.md", "outside apps/web")],
)
async def test_work_that_cannot_be_taken_in_goes_back_to_the_team(
    service: DirectorService,
    launcher: FakeLauncher,
    repo: Path,
    tmp_path: Path,
    path: str,
    said: str,
):
    await split_root(service)
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, path, "theirs\n")
    commit(repo, path, "ours\n")
    head = run_git(repo, "rev-parse", "HEAD")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)

    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    web = node(service, "apps.web")
    assert web.review is not None and web.review.state == "changes_requested"
    assert run_git(repo, "rev-parse", "HEAD") == head
    assert (repo / path).read_text() == "ours\n"
    assert said in launcher.messages_to(ic(service, "apps.web"))[-1]
    assert "sent back" in launcher.messages_to(ic(service))[-1]


async def test_a_merge_the_checkout_above_is_not_ready_for_waits_for_its_ic(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    await split_root(service)
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, "apps/web/page.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    (repo / "apps" / "web").mkdir(parents=True)
    (repo / "apps" / "web" / "page.tsx").write_text("in the way\n")

    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    review = node(service, "apps.web").review
    assert review is not None and review.state == "pending"
    assert (repo / "apps" / "web" / "page.tsx").read_text() == "in the way\n"
    assert "accept it again" in launcher.messages_to(ic(service))[-1]


async def test_the_top_team_is_done_once_it_submits_and_hears_its_own_findings(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    launcher.agent_answers = {
        INSTRUCTIONS: json.dumps({"findings": [FINDING.model_dump()]})
    }
    commit(repo, "app.py", "x = 1\n")

    done = await use(service, ic(service), SubmitReviewTool, DONE)

    root = node(service, "root")
    assert not done.is_error and root.status == "done"
    assert root.review is not None and root.review.state == "unreviewed"
    assert "Returns a string." in launcher.messages_to(ic(service))[-1]


SCHEMA = "services.api.schema"


async def schema_is_accepted(service: DirectorService, checkout: Path) -> None:
    commit(checkout, "services/api/schema/order.py", "id: int\n")
    await use(service, ic(service, SCHEMA), SubmitReviewTool, DONE)
    await use(service, ic(service, "services.api"), ReviewTeamTool, accept(SCHEMA))


async def test_accepted_work_goes_on_up_unreviewed_when_the_ic_changed_nothing(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await split_api(service)

    await schema_is_accepted(service, repo)

    api = node(service, "services.api")
    assert api.review is not None and api.review.state == "accepted"
    assert api.status == "done"
    assert api.base_commit == run_git(repo, "rev-parse", "HEAD").strip()
    assert "without another review" in launcher.messages_to(ic(service, api.id))[-1]
    assert requests(service, launcher) == []
    # The top team has another sub-team still at work.
    assert node(service, "root").review is None


async def test_the_top_team_is_told_when_all_below_is_accepted_and_submits_itself(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await split_api(service)
    commit(repo, "apps/web/page.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    await schema_is_accepted(service, repo)

    told = launcher.messages_to(ic(service))[-1]
    assert "all your sub-teams is accepted" in told and "submit_review" in told
    root = node(service, "root")
    assert root.review is None and root.status != "done"


@pytest.mark.parametrize("committed", [True, False])
async def test_an_ic_that_changed_files_itself_submits_them_with_the_accepted_work(
    service: DirectorService, launcher: FakeLauncher, repo: Path, committed: bool
):
    await split_root(service)
    await split_api(service)
    if committed:
        commit(repo, "services/api/app.py", "import schema\n")
    else:
        (repo / "services" / "api").mkdir(parents=True)
        (repo / "services" / "api" / "app.py").write_text("import schema\n")

    await schema_is_accepted(service, repo)

    assert node(service, "services.api").review is None
    # Nothing since it was asked to review its sub-team.
    assert "Review mode" in launcher.messages_to(ic(service, "services.api"))[-1]
    assert requests(service, launcher) == []

    if not committed:
        run_git(repo, "add", "-A")
        run_git(repo, "commit", "--quiet", "-m", "Use the schema")
    await use(service, ic(service, "services.api"), SubmitReviewTool, DONE)
    await reviewer_wakes(service, launcher)

    [request] = requests(service, launcher)
    assert "Team 'API'" in request


async def test_work_that_comes_up_on_its_own_tells_an_ic_with_changes_of_its_own(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await split_api(service)
    commit(repo, "README.md", "How to run it.\n")
    commit(repo, "apps/web/page.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    await schema_is_accepted(service, repo)

    told = launcher.messages_to(ic(service))[-1]
    assert "README.md" in told and "nobody has reviewed" in told


async def test_work_that_goes_up_unreviewed_is_merged_into_each_checkout_above(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    await split_root(service)
    await split_api(service)
    worktree = in_own_worktree(service, tmp_path, repo, "services.api", SCHEMA)

    await schema_is_accepted(service, worktree)

    assert (repo / "services/api/schema/order.py").read_text() == "id: int\n"
    api = node(service, "services.api")
    assert api.review is not None and api.review.state == "accepted"
    assert api.base_commit == run_git(worktree, "rev-parse", "HEAD").strip()


async def test_work_that_does_not_merge_on_the_way_up_stops_with_that_teams_ic(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    await split_root(service)
    await split_api(service)
    worktree = in_own_worktree(service, tmp_path, repo, "services.api", SCHEMA)
    commit(repo, "services/api/schema/order.py", "id: str\n")
    head = run_git(repo, "rev-parse", "HEAD")

    await schema_is_accepted(service, worktree)

    schema = node(service, SCHEMA).review
    assert schema is not None and schema.state == "accepted"
    assert node(service, "services.api").review is None
    assert run_git(repo, "rev-parse", "HEAD") == head
    told = launcher.messages_to(ic(service, "services.api"))[-1]
    assert "does not merge" in told and "services/api/schema/order.py" in told
    assert "submit_review" in told
    assert requests(service, launcher) == []
