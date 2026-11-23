"""How work a team submits is reviewed by the IC above and taken in."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

from openhands.agent_server.director.models import (
    CreateOrgRequest,
    ReviewFinding,
    SplitChild,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgTeamTool,
    ProposeTeamsTool,
    RaiseEscalationTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgTeamAction,
    RaiseEscalationAction,
    ReviewTeamAction,
    SubmitReviewAction,
)
from openhands.agent_server.director.tools.impl import DirectorObservation
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.event import ActionEvent
from openhands.tools.file_editor.definition import FileEditorAction
from tests.agent_server.director.conftest import (
    FakeLauncher,
    commit,
    root_conversation,
    run_git,
    tasked,
)
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    division_lead_ic,
    ic,
    manager,
    node,
    proposing,
    split_root,
    use,
    woken,
)


INSTRUCTIONS = "Review the diff for bugs."
WRITE_UP = "pull request description"
ACCOUNT = "Adds the checkout endpoint in app.py."
FINDING = ReviewFinding(path="services/api/app.py", line=1, issue="Returns a string.")
DONE = SubmitReviewAction(summary="Endpoint shipped.")


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
    return [
        text for text in launcher.messages_to(ic(service)) if "Review request" in text
    ]


def reject(node_id: str = "services.api") -> ReviewTeamAction:
    return ReviewTeamAction(team=node_id, decision="reject", findings=[FINDING])


def accept(node_id: str = "services.api", needed_above: str = "") -> ReviewTeamAction:
    return ReviewTeamAction(team=node_id, decision="accept", needed_above=needed_above)


async def test_submitted_work_goes_to_the_ic_above_with_the_teams_own_account(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    launcher.agent_answers = {WRITE_UP: ACCOUNT}
    base = node(service, "services.api").base_commit
    to_manager = len(launcher.messages_to(manager(service)))

    await api_submits(service, repo)
    await reviewer_wakes(service, launcher)

    api = node(service, "services.api")
    assert api.status == "done"
    assert api.review is not None
    assert (api.review.state, api.review.round) == ("pending", 1)
    request = launcher.messages_to(ic(service))[-1]
    assert api.review.description == ACCOUNT
    assert "Review request" in request and "Endpoint shipped." in request
    assert f"git diff {base} HEAD -- services/api" in request
    assert ACCOUNT in request and INSTRUCTIONS in request
    assert "Returns a string." not in request
    assert len(launcher.messages_to(manager(service))) == to_manager
    [(asked_ic, write_up)] = launcher.agents_asked
    assert str(asked_ic) == ic(service, "services.api")
    assert WRITE_UP in write_up and "services/api/app.py" in write_up
    assert INSTRUCTIONS not in write_up
    assert "order_id = '1'" not in write_up
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


async def test_a_review_request_says_when_the_team_gave_no_account(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)

    await api_submits(service, repo)
    await reviewer_wakes(service, launcher)

    review = node(service, "services.api").review
    assert review is not None and review.description == ""
    request = launcher.messages_to(ic(service))[-1]
    assert "gave no account" in request and INSTRUCTIONS in request


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
        request = launcher.messages_to(ic(service))[-1]
        assert "Review request" in request and INSTRUCTIONS not in request
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
        service, await woken(service, "services.api"), SubmitReviewTool, DONE
    )
    await api_submits(service, repo)
    twice = await use(service, api, SubmitReviewTool, DONE)

    assert nothing_said.is_error and by_manager.is_error
    assert twice.is_error and "already waits" in twice.text


async def test_an_ic_that_submitted_rests_and_its_team_shows_as_done(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    api = ic(service, "services.api")
    assert api is not None
    launcher.report(api, ConversationExecutionStatus.RUNNING)
    assert node(service, "services.api").status == "working"

    await api_submits(service, repo)

    # Its turn was ended for it; nothing is left to its own word.
    assert launcher.paused == [UUID(api)]
    assert node(service, "services.api").status == "done"

    await use(service, ic(service), ReviewTeamTool, reject())
    launcher.report(api, ConversationExecutionStatus.RUNNING)

    # Sent back, it is woken and no longer done.
    assert "asks for" in launcher.messages_to(api)[-1]
    assert node(service, "services.api").status == "working"


async def test_the_top_team_rests_after_it_submits_too(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await use(service, ic(service), SubmitReviewTool, DONE)

    assert launcher.paused == [UUID(str(ic(service)))]
    assert node(service, "root").status == "done"


async def test_a_review_is_asked_for_as_soon_as_its_team_submits(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)

    submitted = await api_submits(service, repo)

    # The team beside it still works: its review does not wait for that.
    assert "Your turn ends here" in submitted.text
    [first] = requests(service, launcher)
    assert "Team 'API'" in first and "Team 'Web'" not in first
    assert "review_team` for team services.api" in first
    assert "Still waiting for your decision" not in first

    # The reviewer ended its turn without deciding, to compare the two.
    commit(repo, "apps/web/page.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)

    [_, second] = requests(service, launcher)
    assert "Team 'Web'" in second.split("---")[0]
    assert "review_team` for team apps.web" in second
    # What it was asked before is named again, not sent again.
    assert "Still waiting for your decision" in second
    assert "'API'" in second.split("Still waiting for your decision")[1]
    assert "Team 'API' (id" not in second


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

    # The two come on their own lanes, in either order.
    review, escalation = sorted(
        launcher.messages_to(ic(service))[-2:], key=lambda text: "escalates" in text
    )
    assert "Which router?" in escalation
    assert "Review request" in review and "Team 'API'" in review


async def test_reviews_that_waited_through_a_pause_go_out_after_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await service.pause()
    await api_submits(service, repo)
    commit(repo, "apps/web/page.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    assert requests(service, launcher) == []

    await service.resume()
    await service.settled()

    asked = "\n".join(requests(service, launcher))
    assert asked.count("Team 'API' (id") == 1 and asked.count("Team 'Web' (id") == 1


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
        ReviewTeamAction(team="services.api", decision="reject"),
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
        ProposeTeamsTool,
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
    await tasked(service)


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
        service.org.update_place(
            node(service, team).id, worktree_path=str(worktree), branch=name
        )
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


async def test_work_outside_the_teams_directory_goes_back_to_the_team(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    await split_root(service)
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, "README.md", "theirs\n")
    commit(repo, "README.md", "ours\n")
    head = run_git(repo, "rev-parse", "HEAD")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)

    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    web = node(service, "apps.web")
    assert web.review is not None and web.review.state == "changes_requested"
    assert run_git(repo, "rev-parse", "HEAD") == head
    assert (repo / "README.md").read_text() == "ours\n"
    assert "outside apps/web" in launcher.messages_to(ic(service, "apps.web"))[-1]
    assert "sent back" in launcher.messages_to(ic(service))[-1]


PAGE = "apps/web/page.tsx"


async def test_work_that_does_not_merge_is_merged_by_its_reviewer(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    await split_root(service)
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, PAGE, "theirs\n")
    commit(worktree, "apps/web/cart.tsx", "cart\n")
    commit(repo, PAGE, "ours\n")
    head = run_git(repo, "rev-parse", "HEAD")
    web = node(service, "apps.web")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)

    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    # The team is not sent back: the merge is laid out for its reviewer.
    merge = service.merge_checkout(web.id)
    assert merge is not None and merge.files == [PAGE]
    review = node(service, "apps.web").review
    assert review is not None and review.state == "pending"
    assert run_git(repo, "rev-parse", "HEAD") == head
    told = launcher.messages_to(ic(service))[-1]
    assert merge.where in told and PAGE in told and "yours to finish" in told
    laid_out = Path(merge.where)
    assert "<<<<<<< " in (laid_out / PAGE).read_text()
    assert (laid_out / "apps/web/cart.tsx").read_text() == "cart\n"

    # Only what does not merge is the reviewer's to write there.
    def writing(conversation: str | None, path: Path) -> str | None:
        action = FileEditorAction(command="create", path=str(path), file_text="x")
        event = cast(
            ActionEvent, SimpleNamespace(tool_name="file_editor", action=action)
        )
        return service.guard_action(UUID(conversation or ""), event)

    assert writing(ic(service), laid_out / PAGE) is None
    refused = writing(ic(service), laid_out / "apps/web/cart.tsx")
    assert refused is not None and "merged by itself" in refused
    other = writing(ic(service, "apps.web"), laid_out / PAGE)
    assert other is not None and "another team's reviewer" in other

    # It is taken only once it is finished, and as git merged the rest.
    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))
    assert "still have conflict markers" in launcher.messages_to(ic(service))[-1]
    (laid_out / PAGE).write_text("ours and theirs\n")
    (laid_out / "apps/web/cart.tsx").write_text("changed\n")
    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))
    assert "not yours to change" in launcher.messages_to(ic(service))[-1]
    assert run_git(repo, "rev-parse", "HEAD") == head
    (laid_out / "apps/web/cart.tsx").write_text("cart\n")

    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    review = node(service, "apps.web").review
    assert review is not None and review.state == "accepted"
    assert (repo / PAGE).read_text() == "ours and theirs\n"
    assert (repo / "apps/web/cart.tsx").read_text() == "cart\n"
    assert run_git(repo, "status", "--porcelain") == ""
    parents = run_git(repo, "log", "-1", "--format=%P").split()
    assert parents == [head.strip(), run_git(worktree, "rev-parse", "HEAD").strip()]
    assert service.merge_checkout(web.id) is None and not laid_out.exists()
    said = [
        text
        for text in launcher.messages_to(ic(service, "apps.web"))
        if "finished the merge" in text
    ]
    assert len(said) == 1 and PAGE in said[0]


async def test_a_merge_laid_out_goes_with_work_that_is_sent_back(
    service: DirectorService, repo: Path, tmp_path: Path
):
    await split_root(service)
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, PAGE, "theirs\n")
    commit(repo, PAGE, "ours\n")
    web = node(service, "apps.web")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))
    merge = service.merge_checkout(web.id)
    assert merge is not None

    await use(
        service,
        ic(service),
        ReviewTeamTool,
        ReviewTeamAction(team="apps.web", decision="reject", findings=[FINDING]),
    )

    assert service.merge_checkout(web.id) is None
    assert not Path(merge.where).exists()
    assert merge.where not in run_git(repo, "worktree", "list")


async def test_what_the_ic_above_wrote_in_a_started_teams_directory_is_put_aside(
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
    assert review is not None and review.state == "accepted"
    assert (repo / "apps" / "web" / "page.tsx").read_text() == "export {}\n"
    [aside] = [
        text for text in launcher.messages_to(ic(service)) if "were taken out" in text
    ]
    assert "your sub-team 'Web': apps/web/page.tsx." in aside
    [kept] = run_git(
        repo, "for-each-ref", "--format=%(refname)", "refs/agents/aside"
    ).split()
    assert kept in aside
    assert run_git(repo, "show", f"{kept}:apps/web/page.tsx") == "in the way\n"


async def test_the_top_team_is_done_once_it_submits_and_its_reviewer_is_instructed(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    commit(repo, "app.py", "x = 1\n")

    done = await use(service, ic(service), SubmitReviewTool, DONE)

    root = node(service, "root")
    assert not done.is_error and root.status == "done"
    assert root.review is not None and root.review.state == "pending"
    start = launcher.derived[-1]
    assert str(start.conversation_id) == division_lead_ic(service)
    assert INSTRUCTIONS in start.task
    assert "Returns a string." not in start.task


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
    assert api.review.passed_up and api.status == "done"
    schema = node(service, SCHEMA).review
    assert schema is not None and not schema.passed_up
    assert api.base_commit == run_git(repo, "rev-parse", "HEAD").strip()
    assert "without another review" in launcher.messages_to(ic(service, api.id))[-1]
    assert requests(service, launcher) == []
    # The top team has another sub-team still at work.
    assert node(service, "root").review is None


async def test_a_top_team_that_changed_nothing_is_accepted_when_all_below_is(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await split_api(service)
    commit(repo, "apps/web/page.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    await schema_is_accepted(service, repo)

    root = node(service, "root")
    assert root.review is not None and root.review.state == "accepted"
    assert root.review.passed_up and root.status == "done"
    # Each change was reviewed once already: nobody is asked again.
    assert division_lead_ic(service) is None
    assert [team.id for team in service.org.teams()] == [
        "root",
        "services.api",
        "apps.web",
        SCHEMA,
    ]


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
    assert "Review request" in launcher.messages_to(ic(service, "services.api"))[-1]
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


async def test_a_team_given_more_work_after_an_accept_owes_it_a_review(
    service: DirectorService,
):
    await split_root(service)
    checkout = Path(node(service, "root").worktree_path)
    for team, file in (
        ("services.api", "services/api/app.py"),
        ("apps.web", "apps/web/page.tsx"),
    ):
        commit(checkout, file, "a = 1\n")
        await use(service, ic(service, team), SubmitReviewTool, DONE)
        await use(service, ic(service), ReviewTeamTool, accept(team))
    first = node(service, "root").review
    assert first is not None and first.passed_up

    for team in ("services.api", "apps.web"):
        await use(
            service, ic(service), MsgTeamTool, MsgTeamAction(team=team, text="Refunds.")
        )
    # API commits what it has so far and does not submit.
    commit(checkout, "services/api/refunds.py", "WIP = True\n")
    api = node(service, "services.api")
    assert api.review is None and service._may_stall(api)
    commit(checkout, "apps/web/refunds.tsx", "export {}\n")
    await use(service, ic(service, "apps.web"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    # Nothing went up: API's work is reviewed by nobody yet.
    assert node(service, "root").review == first


async def test_work_accepted_twice_before_it_is_carried_out_goes_up_once(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await split_api(service)
    commit(repo, "services/api/schema/order.py", "id: int\n")
    await use(service, ic(service, SCHEMA), SubmitReviewTool, DONE)
    reviewer = ic(service, "services.api")
    state = cast(ConversationState, SimpleNamespace(id=UUID(reviewer)))
    executor = ReviewTeamTool.create(state)[0].executor
    assert executor is not None

    await asyncio.to_thread(executor, accept(SCHEMA))
    again = await asyncio.to_thread(executor, accept(SCHEMA))
    await service.settled()

    assert again.is_error and "being accepted already" in again.text
    went_up = [
        text
        for text in launcher.messages_to(reviewer)
        if "changed nothing itself, so its work went up" in text
    ]
    assert len(went_up) == 1


async def test_work_goes_up_only_when_every_sub_team_below_is_finished(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    web = ic(service, "apps.web")
    assert web is not None
    launcher.report(web, ConversationExecutionStatus.RUNNING)
    await service.settled()

    # One sub-team still works.
    working = await use(service, ic(service), SubmitReviewTool, DONE)
    assert working.is_error and "'Web' is working" in working.text

    # One waits for this IC's review.
    launcher.report(web, ConversationExecutionStatus.FINISHED)
    await api_submits(service, repo)
    waiting = await use(service, ic(service), SubmitReviewTool, DONE)
    assert waiting.is_error and "'API' waits for your review" in waiting.text
    assert "review_team" in waiting.text and "prune_team" in waiting.text

    # One was sent back and has not submitted again.
    await use(service, ic(service), ReviewTeamTool, reject())
    launcher.report(
        str(ic(service, "services.api")), ConversationExecutionStatus.FINISHED
    )
    await service.settled()
    sent_back = await use(service, ic(service), SubmitReviewTool, DONE)
    assert sent_back.is_error and "was sent back" in sent_back.text

    # Accepted: nothing below is unfinished, and the work goes up.
    commit(repo, "services/api/app.py", "order_id = '2'\n")
    await use(service, ic(service, "services.api"), SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, accept())
    # The accepted team is in the turn that tells it so: it is finished.
    launcher.report(
        str(ic(service, "services.api")), ConversationExecutionStatus.RUNNING
    )
    await service.settled()
    done = await use(service, ic(service), SubmitReviewTool, DONE)
    assert not done.is_error, done.text


async def test_a_blocked_sub_team_does_not_keep_work_from_going_up(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    web = ic(service, "apps.web")
    assert web is not None
    launcher.report(web, ConversationExecutionStatus.ERROR)
    await service.settled()
    assert node(service, "apps.web").status == "blocked"

    done = await use(service, ic(service), SubmitReviewTool, DONE)

    assert not done.is_error, done.text
