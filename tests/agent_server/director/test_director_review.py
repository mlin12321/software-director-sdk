"""How work a team reports done is reviewed by the IC above and taken in."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.models import (
    CreateOrgRequest,
    ReviewFinding,
    SplitChild,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    DecideSplitTool,
    ProposeBoundarySplitTool,
    RecordRuleTool,
    ReviewTeamTool,
    SetStatusTool,
)
from openhands.agent_server.director.tools.definition import (
    DecideSplitAction,
    ProposeBoundarySplitAction,
    RecordRuleAction,
    ReviewTeamAction,
    SetStatusAction,
)
from tests.agent_server.director.conftest import (
    FakeLauncher,
    root_conversation,
    run_git,
)
from tests.agent_server.director.test_director_tools import (
    ic,
    manager,
    node,
    split_root,
    use,
)


INSTRUCTIONS = "Review the diff for bugs."
WRITE_UP = "pull request description"
ACCOUNT = "Adds the checkout endpoint in app.py."
FINDING = ReviewFinding(path="services/api/app.py", line=1, issue="Returns a string.")
DONE = SetStatusAction(status="done", summary="Endpoint shipped.")


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


async def api_reports_done(service: DirectorService, repo: Path) -> None:
    commit(repo, "services/api/app.py", "order_id = '1'\n")
    await use(service, ic(service, "services.api"), SetStatusTool, DONE)


def reject(node_id: str = "services.api") -> ReviewTeamAction:
    return ReviewTeamAction(node_id=node_id, decision="reject", findings=[FINDING])


def accept(node_id: str = "services.api", needed_above: str = "") -> ReviewTeamAction:
    return ReviewTeamAction(
        node_id=node_id, decision="accept", needed_above=needed_above
    )


async def test_done_work_goes_to_the_ic_above_with_the_teams_own_account(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    launcher.agent_answers = {
        WRITE_UP: ACCOUNT,
        INSTRUCTIONS: json.dumps({"findings": [FINDING.model_dump()]}),
    }
    base = node(service, "services.api").base_commit
    to_manager = len(launcher.messages_to(manager(service)))

    await api_reports_done(service, repo)

    api = node(service, "services.api")
    assert api.status == "done"
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


async def test_work_that_is_not_committed_cannot_be_reported_done(
    service: DirectorService, repo: Path
):
    await split_root(service)
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "draft.py").write_text("x = 1\n")

    refused = await use(service, ic(service, "services.api"), SetStatusTool, DONE)

    assert refused.is_error and "services/api/draft.py" in refused.text
    assert node(service, "services.api").status != "done"


@pytest.mark.parametrize("answer", [None, "I could not tell."])
async def test_a_review_request_says_what_the_team_did_not_answer(
    service: DirectorService, launcher: FakeLauncher, repo: Path, answer: str | None
):
    await split_root(service)
    launcher.agent_answers = {INSTRUCTIONS: answer}

    await api_reports_done(service, repo)

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

        await api_reports_done(service, repo)

        asked = [WRITE_UP in question for _, question in launcher.agents_asked]
        assert asked == [True]
        assert "Review mode" in launcher.messages_to(ic(service))[-1]
    finally:
        await service.aclose()


async def test_rejected_work_goes_back_to_the_team_and_comes_again_as_a_new_round(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    await api_reports_done(service, repo)

    await use(service, ic(service), ReviewTeamTool, reject())

    api = node(service, "services.api")
    assert api.status == "working"
    assert api.review is not None and api.review.state == "changes_requested"
    told = launcher.messages_to(ic(service, "services.api"))[-1]
    assert "services/api/app.py:1" in told and "Returns a string." in told

    commit(repo, "services/api/app.py", "order_id = 1\n")
    await use(service, ic(service, "services.api"), SetStatusTool, DONE)

    again = node(service, "services.api").review
    assert again is not None and (again.state, again.round) == ("pending", 2)
    assert "review round 2" in launcher.messages_to(ic(service))[-1]


async def test_a_third_rejection_stops_the_team_and_tells_nobody(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)

    async def report_done(attempt: int) -> None:
        commit(repo, "services/api/app.py", f"order_id = {attempt}\n")
        await use(service, ic(service, "services.api"), SetStatusTool, DONE)

    def heard() -> list[int]:
        return [
            len(launcher.messages_to(ic(service, "services.api"))),
            len(launcher.messages_to(manager(service))),
            len(launcher.messages_to(manager(service, "services.api"))),
        ]

    for attempt in range(2):
        await report_done(attempt)
        await use(service, ic(service), ReviewTeamTool, reject())
    await report_done(2)
    before = heard()

    last = await use(service, ic(service), ReviewTeamTool, reject())

    api = node(service, "services.api")
    assert api.status == "blocked"
    assert api.review is not None and api.review.findings == [FINDING]
    assert "stopped" in last.text
    assert {UUID(str(agent.conversation_id)) for agent in api.agents} <= set(
        launcher.stopped
    )
    assert heard() == before


async def test_only_the_ic_directly_above_reviews_and_only_work_that_waits(
    service: DirectorService, repo: Path
):
    await split_root(service)

    nothing_waits = await use(service, ic(service), ReviewTeamTool, accept())
    await api_reports_done(service, repo)
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
        ProposeBoundarySplitTool,
        ProposeBoundarySplitAction(
            children=[
                SplitChild(
                    title="Schema", path="services/api/schema", task="Define orders"
                )
            ],
            contracts="Order has an id.",
        ),
    )
    await use(
        service,
        manager(service, "services.api"),
        DecideSplitTool,
        DecideSplitAction(decision="approve"),
    )


@pytest.mark.parametrize("needed_above", ["", "Web builds its forms from the schema."])
async def test_accepted_work_is_told_further_up_only_when_a_team_there_needs_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path, needed_above: str
):
    await split_root(service)
    await split_api(service)
    schema = "services.api.schema"
    commit(repo, "services/api/schema/order.py", "id: int\n")
    await use(service, ic(service, schema), SetStatusTool, DONE)
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


def in_own_worktree(service: DirectorService, tmp_path: Path, repo: Path) -> Path:
    """Give the Web team the worktree its conversation would have made."""
    worktree = tmp_path / "web"
    run_git(repo, "worktree", "add", "--quiet", "-b", "web", str(worktree))
    service.store.update_node("apps.web", worktree_path=str(worktree), branch="web")
    return worktree


async def test_accepted_work_of_a_worktree_team_is_merged_into_the_checkout_above(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    await split_root(service)
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, "apps/web/page.tsx", "export {}\n")
    commit(repo, "README.md", "changed above\n")
    await use(service, ic(service, "apps.web"), SetStatusTool, DONE)

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
    await use(service, ic(service, "apps.web"), SetStatusTool, DONE)

    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    web = node(service, "apps.web")
    assert web.status == "working"
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
    await use(service, ic(service, "apps.web"), SetStatusTool, DONE)
    (repo / "apps" / "web").mkdir(parents=True)
    (repo / "apps" / "web" / "page.tsx").write_text("in the way\n")

    await use(service, ic(service), ReviewTeamTool, accept("apps.web"))

    review = node(service, "apps.web").review
    assert review is not None and review.state == "pending"
    assert (repo / "apps" / "web" / "page.tsx").read_text() == "in the way\n"
    assert "accept it again" in launcher.messages_to(ic(service))[-1]


async def test_the_root_team_is_done_without_a_review_and_hears_the_findings(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    launcher.agent_answers = {
        INSTRUCTIONS: json.dumps({"findings": [FINDING.model_dump()]})
    }
    commit(repo, "app.py", "x = 1\n")

    done = await use(service, ic(service), SetStatusTool, DONE)

    root = node(service, "root")
    assert not done.is_error and root.status == "done"
    assert root.review is not None and root.review.state == "unreviewed"
    assert "Returns a string." in launcher.messages_to(ic(service))[-1]


async def test_a_team_that_works_on_takes_its_work_back_from_review(
    service: DirectorService, repo: Path
):
    await split_root(service)
    await api_reports_done(service, repo)

    await use(
        service,
        ic(service, "services.api"),
        SetStatusTool,
        SetStatusAction(status="working"),
    )
    refused = await use(service, ic(service), ReviewTeamTool, accept())

    assert node(service, "services.api").review is None
    assert refused.is_error


async def test_an_agent_records_a_rule_for_its_own_boundary(service: DirectorService):
    await split_root(service)

    await use(
        service,
        ic(service, "services.api"),
        RecordRuleTool,
        RecordRuleAction(rule="Amounts are integer cents."),
    )
    blank = await use(
        service, manager(service), RecordRuleTool, RecordRuleAction(rule="  ")
    )

    assert service.blackboard.memory("services/api").rules == [
        "Amounts are integer cents."
    ]
    assert service.blackboard.memory("").rules == []
    assert blank.is_error


async def test_a_team_that_was_done_works_while_it_reviews_and_is_done_again(
    service: DirectorService, repo: Path
):
    await split_root(service)
    await split_api(service)
    schema = "services.api.schema"
    await api_reports_done(service, repo)
    assert node(service, "services.api").status == "done"

    commit(repo, "services/api/schema/order.py", "id: int\n")
    await use(service, ic(service, schema), SetStatusTool, DONE)

    assert node(service, "services.api").status == "working"

    await use(service, ic(service, "services.api"), ReviewTeamTool, reject(schema))

    assert node(service, "services.api").status == "done"
