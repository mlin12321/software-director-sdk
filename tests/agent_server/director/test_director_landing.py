"""Tests for what a session's accepted work makes the org's: its
sub-teams, its specs and what its agents recorded."""

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from openhands.agent_server.director.changes import accepted_changes
from openhands.agent_server.director.intents import Accept
from openhands.agent_server.director.models import (
    CreateOrgRequest,
    DirectorSettingsUpdate,
    DocumentChoice,
    OpenSessionRequest,
    Report,
    RouteMessageRequest,
    SessionTeam,
    TeamChoice,
)
from openhands.agent_server.director.org_persistence import create_org_board, load_org
from openhands.agent_server.director.org_store import (
    OrgStore,
    session_id_of,
    team_id_of,
)
from openhands.agent_server.director.prompts import (
    DIVISION_LEAD_IC_ROLE,
    DIVISION_LEAD_MANAGER_ROLE,
)
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.run_state import SESSION
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import (
    MERGE_WAITS,
    team_ref,
)
from openhands.agent_server.director.tools import (
    PruneTeamTool,
    RaiseEscalationTool,
    ReviewTeamTool,
    SettleTeamsTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    RaiseEscalationAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import NotFound, OperationFailed
from tests.agent_server.director.conftest import (
    DONE,
    FINDING,
    JOBS,
    PRUNE_API,
    FakeLauncher,
    accept,
    at_api,
    commit,
    division_lead_ic,
    division_lead_manager,
    division_lead_of,
    ic,
    landed,
    manager,
    node,
    org_gains,
    own_contracts,
    path_of,
    record_of,
    root_conversation,
    run_git,
    session_accepted,
    session_contracts,
    sets_up_jobs,
    shipped,
    shipped_type,
    split_root,
    started,
    take_in,
    use,
    waiting,
)
from tests.agent_server.director.document_tools import (
    ReportAction,
    ReportTool,
    ReviewTeamAction,
    SettleTeamsAction,
    WriteSpecAction,
    WriteSpecTool,
    report,
)


async def test_the_org_has_a_sessions_sub_teams_once_its_work_is_accepted(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)

    assert [team.id for team in service.org.teams()] == ["root"]
    assert division_lead_ic(service) is None

    await use(service, ic(service), SubmitReviewTool, DONE)

    # Its Division Lead IC is the top team's IC, started for this.
    asked = division_lead_ic(service)
    assert asked is not None
    start = launcher.derived[-1]
    assert str(start.conversation_id) == asked
    assert (
        "submits its work" in start.task
        and shipped(DIVISION_LEAD_IC_ROLE) in start.task
    )
    assert start.title == "Checkout (Division Lead IC)"
    assert start.checkout == node(service, division_lead_of(service)).worktree_path

    await use(service, asked, ReviewTeamTool, accept(node(service, "root").id))

    assert [team.id for team in service.org.teams()] == [
        "root",
        "services.api",
        "apps.web",
    ]
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"
    assert service.org.place(division_lead_of(service)) is not None


async def test_what_a_session_records_is_the_teams_once_its_work_is_accepted(
    service: DirectorService,
):
    await use(service, ic(service), ReportTool, ReportAction(text="Chose REST."))

    assert [r.text for r in service.reports_of(node(service, "root"))] == [
        "Chose REST."
    ]
    assert service.team_memory.reports("", "ic") == []

    await landed(service)

    assert [r.text for r in service.team_memory.reports("", "ic")] == ["Chose REST."]
    assert service.own_reports(node(service, "root")) == []


async def test_what_is_recorded_while_work_lands_is_not_lost(
    service: DirectorService, monkeypatch: pytest.MonkeyPatch
):
    await use(service, ic(service), ReportTool, ReportAction(text="Chose REST."))
    root, commit = node(service, "root"), service._commit_landing
    late: list[threading.Thread] = []

    def a_report_comes(*args: Any, **kwargs: Any):
        # The session's records are read by now and not yet cleared.
        writer = threading.Thread(
            target=report, args=(service, root, "ic", "Chose JSON.")
        )
        writer.start()
        writer.join(0.3)
        late.append(writer)
        return commit(*args, **kwargs)

    monkeypatch.setattr(service, "_commit_landing", a_report_comes)

    await landed(service)
    await service.settled()
    for writer in late:
        writer.join()

    assert late
    kept = service.team_memory.reports("", "ic") + service.own_reports(root)
    assert sorted(report.text for report in kept) == ["Chose JSON.", "Chose REST."]


async def test_what_became_of_a_sessions_teams_is_on_record_once_it_is_accepted(
    service: DirectorService,
):
    await split_root(service)
    top = node(service, "root")

    own = service.history_of(top.id)
    assert "set_up" in [event.kind for event in own]
    assert all(event.unaccepted for event in own)
    assert service.team_memory.history("root") == []

    await landed(service)

    kept = service.team_memory.history("root")
    assert {"set_up", "accepted"} <= {event.kind for event in kept}
    assert not any(event.unaccepted for event in kept)
    assert service.own_events(top.session, "root") == []


async def test_what_a_reviewer_records_is_the_teams_once_the_work_is_accepted(
    service: DirectorService,
):
    await use(service, ic(service), SubmitReviewTool, DONE)
    asked = division_lead_ic(service)
    assert asked is not None
    await use(service, asked, ReportTool, ReportAction(text="Read the diff twice."))

    assert service.team_memory.reports("", "ic") == []

    await use(service, asked, ReviewTeamTool, accept(node(service, "root").id))

    assert [r.text for r in service.team_memory.reports("", "ic")] == [
        "Read the diff twice."
    ]


async def test_what_agents_write_to_each_other_is_never_the_orgs(
    service: DirectorService, repo: Path
):
    top = node(service, "root")
    service.send_msg(top.id, "ic", top.id, "manager", "Split in two?")

    await landed(service)

    [thread] = service.threads_of(node(service, "root"))
    assert [message.text for message in thread.messages] == ["Split in two?"]
    kept = run_git(repo, "ls-tree", "-r", "--name-only", "refs/agents/memory")
    assert ".threads" not in kept


async def test_the_end_of_a_session_the_org_knows_is_on_its_record(
    service: DirectorService,
):
    await landed(service)

    service.close_session(node(service, "root").session)
    await service.settled()

    kinds = [event.kind for event in service.team_memory.history("root")]
    assert kinds.count("session_opened") == kinds.count("session_closed") == 1


def own_worktree(service: DirectorService, repo: Path, tmp_path: Path) -> Path:
    """Give the session opened first the worktree its conversation would
    have made."""
    worktree = tmp_path / "session"
    run_git(repo, "worktree", "add", "--quiet", "-b", "session", str(worktree))
    for place in service.org.session(session_id_of(node(service, "root"))):
        if not place.division_lead:
            service.org.update_place(
                place.id, worktree_path=str(worktree), branch="session"
            )
    return worktree


async def test_accepted_work_is_merged_into_the_repositorys_checkout(
    service: DirectorService, repo: Path, tmp_path: Path
):
    worktree = own_worktree(service, repo, tmp_path)
    commit(worktree, "app.py", "refunds = True\n")

    await landed(service)

    assert (repo / "app.py").read_text() == "refunds = True\n"
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"


async def test_an_accept_on_record_says_which_commits_it_took_the_work_between(
    service: DirectorService, repo: Path, tmp_path: Path
):
    started_at = run_git(repo, "rev-parse", "HEAD").strip()
    worktree = own_worktree(service, repo, tmp_path)
    commit(worktree, "app.py", "refunds = True\n")
    accepted_at = run_git(worktree, "rev-parse", "HEAD").strip()

    await landed(service)

    [session_accepted] = [
        event
        for event in service.team_memory.history("root")
        if event.kind == "accepted"
    ]
    assert (session_accepted.base, session_accepted.head) == (started_at, accepted_at)
    # The top team's folder is the whole repository.
    assert session_accepted.paths == [""]
    # The session can be closed: the repository has both commits.
    changes = accepted_changes(
        repo, session_accepted.base, session_accepted.head, session_accepted.paths[0]
    )
    assert [(file.path, file.kind) for file in changes.files] == [("app.py", "added")]


async def test_the_human_is_asked_when_their_uncommitted_work_is_in_the_way(
    service: DirectorService, repo: Path, tmp_path: Path
):
    commit(repo, "app.py", "refunds = False\n")
    worktree = own_worktree(service, repo, tmp_path)
    commit(worktree, "app.py", "refunds = True\n")
    (repo / "app.py").write_text("refunds = None\n")

    await landed(service)

    top = node(service, "root")
    assert top.review is not None and top.review.state == "pending"
    [asked] = top.escalations
    assert (asked.id, asked.held_by) == (MERGE_WAITS, None)
    assert (repo / "app.py").read_text() == "refunds = None\n"

    run_git(repo, "checkout", "--", "app.py")
    service.answer_escalation(top.id, MERGE_WAITS, "Cleared.")
    await service.settled()

    assert (repo / "app.py").read_text() == "refunds = True\n"
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"


async def test_work_is_not_merged_into_another_branch_than_the_session_started_from(
    service: DirectorService, repo: Path, tmp_path: Path
):
    worktree = own_worktree(service, repo, tmp_path)
    commit(worktree, "app.py", "refunds = True\n")
    run_git(repo, "checkout", "--quiet", "-b", "elsewhere")

    await landed(service)

    top = node(service, "root")
    assert top.review is not None and top.review.state == "pending"
    [asked] = top.escalations
    assert (asked.id, asked.held_by) == (MERGE_WAITS, None)
    assert "your checkout is now on 'elsewhere'" in asked.detail
    assert not (repo / "app.py").exists()

    run_git(repo, "checkout", "--quiet", "-")
    service.answer_escalation(top.id, MERGE_WAITS, "Switched back.")
    await service.settled()

    assert (repo / "app.py").read_text() == "refunds = True\n"
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"


async def test_what_a_team_needs_of_another_goes_to_the_ic_above_both(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    need = RaiseEscalationAction(
        category="design_decision",
        title="The cart must send the order id",
        detail="POST /checkout needs it of the cart page, which is Web's.",
    )

    await use(service, ic(service, "services.api"), RaiseEscalationTool, need)
    await service.settled()

    [raised] = node(service, "services.api").escalations
    assert raised.held_by == node(service, "root").id
    assert (
        "escalates 'The cart must send the order id'"
        in launcher.messages_to(ic(service))[-1]
    )
    # The team it concerns hears of it only from the IC above both.
    assert not any(
        "The cart must send" in told
        for told in launcher.messages_to(ic(service, "apps.web"))
    )


async def test_a_new_request_cannot_go_to_a_team_only_a_session_has(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    launcher.answers[USAGE_ID] = "Teams: apps.web\nReason: The cart page is Web's."

    # The org does not have Web yet; the session that set it up does.
    assert service.org.team("apps.web") is None
    sent = await service.route_message(
        RouteMessageRequest(text="Make the cart page faster.", auto=True)
    )
    assert team_id_of(node(service, sent.node_id)) == "root"


async def test_what_the_human_says_in_a_session_is_routed_among_its_own_teams(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    launcher.answers[USAGE_ID] = "Teams: apps.web\nReason: The cart page is Web's."
    [session] = service.org.sessions()

    said = await service.route_message(
        RouteMessageRequest(text="Make the cart page faster.", session_id=session.id)
    )
    await service.settled()

    # Web is a team only this session has. No session was opened for it.
    assert [open_now.id for open_now in service.org.sessions()] == [session.id]
    assert (said.node_id, said.chosen_by) == (node(service, "apps.web").id, "model")
    assert said.conversation_id == ic(service, "apps.web")
    assert launcher.messages_to(said.conversation_id)[-1] == (
        "Make the cart page faster."
    )
    # Its IC may say it is not Web's, which the IC above then has.
    handed = await use(
        service,
        said.conversation_id,
        RaiseEscalationTool,
        RaiseEscalationAction(category="misrouted", title="Not ours", detail="API's."),
    )
    assert "Handed to the IC of the team above" in handed.text
    assert "Make the cart page faster." in launcher.messages_to(ic(service))[-1]

    service.update_settings(DirectorSettingsUpdate(route_requests=False))
    kept = await service.route_message(
        RouteMessageRequest(text="Make the cart page lighter.", session_id=session.id)
    )
    assert (kept.node_id, kept.conversation_id) == (
        node(service, "root").id,
        ic(service),
    )
    with pytest.raises(NotFound):
        await service.route_message(
            RouteMessageRequest(text="Anyone?", session_id="s000000")
        )


async def test_work_is_accepted_and_a_team_the_org_lost_meanwhile_stays_gone(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    await use(
        service,
        conversation,
        ReportTool,
        ReportAction(text="Refunds are in."),
    )
    service.org.remove("services.api")

    await use(service, conversation, SubmitReviewTool, DONE)
    session_accepted = await use(
        service, division_lead_ic(service, top), ReviewTeamTool, accept(top)
    )

    # The session set up no teams of its own, so nothing is asked and the
    # team stays gone; its code is merged all the same.
    assert "Accepting the work of 'API'" in session_accepted.text
    assert service.org.team("services.api") is None
    place = service.org.place(top)
    assert place is not None and place.review is not None
    assert place.review.state == "accepted"
    assert service.own_reports(place) == []
    # What it recorded is not dropped with the team: it is that team's
    # record, with the nearest team that holds its directory.
    [left] = service.team_memory.former("services.api")
    assert (left.ended, left.above) == ("not_taken", "root")
    assert [report.text for report in left.reports] == ["Refunds are in."]
    assert service.team_memory.reports("services/api", "ic") == []
    assert "not_taken" in [e.kind for e in record_of(service, "root")]


async def test_a_stale_session_adds_its_team_and_keeps_what_the_org_gained(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, "services.api.cache", "services/api/cache")

    answer = await session_accepted(service, top, conversation)

    # Each team is settled on its own, and nobody is asked.
    assert "Accepting the work of 'API'" in answer
    assert path_of(service, JOBS) == "services/api/jobs"
    assert path_of(service, "services.api.cache") == "services/api/cache"
    assert waiting(service, top) == []
    assert division_lead_manager(service, top) is None

    # What the session did not change is not brought up again either.
    service.org.remove("services.api.cache")
    await use(service, conversation, SubmitReviewTool, DONE)
    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))
    await service.settled()
    assert path_of(service, JOBS) == "services/api/jobs"
    assert service.org.team("services.api.cache") is None


@pytest.mark.parametrize("org_holds", [True, False])
async def test_a_team_both_made_something_else_of_waits_for_the_manager_above(
    service: DirectorService, launcher: FakeLauncher, repo: Path, org_holds: bool
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    commit(repo, "services/api/refunds.py", "refunds = True\n")
    place = service.org.place(f"{JOBS}@{top.split('@')[1]}")
    assert place is not None
    service.documents.write(
        place.session, team_ref(place).keeps("spec"), "Run the refund jobs."
    )

    answer = await session_accepted(service, top, conversation)

    # The code is merged whatever becomes of the team; the org's stays.
    assert "Accepting the work of 'API'" in answer
    review = node(service, top).review
    assert review is not None and review.state == "accepted"
    assert path_of(service, JOBS) == "services/api/workers"
    assert waiting(service, top) == [JOBS]
    asked = division_lead_manager(service, top)
    assert asked is not None
    # It was asked when the work was submitted, beside the review.
    [start] = [d for d in launcher.derived if str(d.conversation_id) == asked]
    assert shipped(DIVISION_LEAD_MANAGER_ROLE) in start.task
    told = launcher.messages_to(asked)[-1]
    assert "the session: 'API' in services/api/jobs, under services.api" in told
    assert "the org now: 'API' in services/api/workers, under services.api" in told
    # What the session wrote for it waits with it.
    assert service.document_draft(place, "spec") is not None

    by_ic = await use(
        service, division_lead_ic(service, top), SettleTeamsTool, SettleTeamsAction()
    )
    unknown = await use(
        service,
        asked,
        SettleTeamsTool,
        SettleTeamsAction(teams=[TeamChoice(team="apps.web", use="org")]),
    )
    settled = await use(
        service,
        asked,
        SettleTeamsTool,
        SettleTeamsAction(
            teams=[TeamChoice(team=JOBS, use="org")] if org_holds else [],
            note="One queue.",
        ),
    )
    again = await use(service, asked, SettleTeamsTool, SettleTeamsAction())

    assert by_ic.is_error and unknown.is_error and again.is_error
    assert f"The teams that wait: {JOBS}" in unknown.text
    assert not settled.is_error
    assert waiting(service, top) == []
    assert path_of(service, JOBS) == (
        "services/api/workers" if org_holds else "services/api/jobs"
    )
    assert service.document_draft(place, "spec") is None
    said = launcher.messages_to(conversation)[-1]
    assert "One queue." in said
    assert ("The org's teams stay" in said) is org_holds
    # What the session wrote for the team is never dropped: it is the
    # team's spec where the org took the team, and that team's record with
    # the team above where it did not.
    written = service.team_memory.read("services/api/jobs/spec.md")
    assert written == (None if org_holds else "Run the refund jobs.")
    kept = service.team_memory.former(JOBS)
    assert [
        (left.ended, left.above, left.documents.get("spec", "")) for left in kept
    ] == ([("not_taken", "services.api", "Run the refund jobs.")] if org_holds else [])


async def test_a_team_set_up_under_one_the_org_lost_goes_under_the_nearest_team(
    service: DirectorService, launcher: FakeLauncher
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    session_contracts(service, api_place(service, top), IDEMPOTENT)
    service.org.remove("services.api")

    answer = await session_accepted(service, top, conversation)

    assert "Accepting the work of 'API'" in answer
    assert service.org.team("services.api") is None
    assert waiting(service, top) == [JOBS]
    asked = division_lead_manager(service, top)
    assert asked is not None
    told = launcher.messages_to(asked)[-1]
    assert "it goes under team root" in told and "is not put back" in told
    # The contracts the lost team set for it come with it, and nothing of
    # them is the org's before it is settled.
    assert f"it comes with team {JOBS}" in told and IDEMPOTENT in told
    before = service.team_memory.read("contracts.md")
    assert before is not None and IDEMPOTENT not in before

    await use(service, asked, SettleTeamsTool, SettleTeamsAction())

    # The team the org removed stays removed.
    assert service.org.team("services.api") is None
    jobs = service.org.team(JOBS)
    assert jobs is not None and jobs.parent_id == "root"
    assert path_of(service, JOBS) == "services/api/jobs"
    assert "moved" in [event.kind for event in record_of(service, "root")]
    # They are added to the contracts of the team it went under.
    assert service.team_memory.read("contracts.md") == (
        f"{before.rstrip()}\n\n{IDEMPOTENT}\n"
    )


API_CONTRACTS = "services/api/contracts.md"
API_SPEC = "services/api/spec.md"
IDEMPOTENT = "Jobs are idempotent."


def api_place(service: DirectorService, top: str) -> SessionTeam:
    place = service.org.place(top)
    assert isinstance(place, SessionTeam)
    return place


async def test_contracts_and_reports_a_session_writes_are_the_orgs_on_accept(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    place = api_place(service, top)

    report(service, place, "ic", "The old jobs team retried forever.")
    session_contracts(service, place, "Jobs are idempotent.")

    # Until its work is accepted they are the session's alone.
    assert service.team_memory.read(API_CONTRACTS) is None
    assert service.team_memory.reports("services/api", "ic") == []
    assert own_contracts(service, place) == "Jobs are idempotent.\n"
    assert [r.text for r in service.reports_of(place)] == [
        "The old jobs team retried forever."
    ]

    await session_accepted(service, top, conversation)

    assert service.team_memory.read(API_CONTRACTS) == "Jobs are idempotent.\n"
    [note] = service.team_memory.reports("services/api", "ic")
    assert note.text == "The old jobs team retried forever."
    assert waiting(service, top) == []


@pytest.mark.parametrize(
    ("which", "holds"),
    [
        ("org", "Orders and carts go through Jobs.\n"),
        ("session", "Refunds go through Jobs.\n"),
        ("replacement", "Orders, carts and refunds go through Jobs.\n"),
    ],
)
async def test_contracts_both_changed_wait_for_the_manager_above(
    service: DirectorService, launcher: FakeLauncher, which: str, holds: str
):
    top, conversation = await at_api(service)
    session_contracts(service, api_place(service, top), "Refunds go through Jobs.")
    # Other accepted work set the contracts of the same directory meanwhile.
    service.team_memory.write(
        "services/api", "contracts.md", "Orders go through Jobs.\n"
    )

    answer = await session_accepted(service, top, conversation)

    # The code is merged; of the contracts the org's stay until they are
    # settled, and nothing is overwritten.
    assert "Accepting the work of 'API'" in answer
    assert service.team_memory.read(API_CONTRACTS) == "Orders go through Jobs.\n"
    asked = division_lead_manager(service, top)
    assert asked is not None
    told = launcher.messages_to(asked)[-1]
    assert "Refunds go through Jobs." in told and "Orders go through Jobs." in told
    # It is derived from the session's IC, so it has one role, its own.
    [started] = [
        start for start in launcher.derived if str(start.conversation_id) == asked
    ]
    assert str(started.parent_id) == conversation
    assert "the contracts of services/api" in told
    merge = node(service, division_lead_of(service, top)).merge
    assert merge is not None
    assert [
        (item.kind, item.document, item.key)
        for item in merge.items
        if item.gate == "manager"
    ] == [("document", "contracts", "services/api")]
    # Its Division Lead IC judged the code by the contracts it was written to.
    request = [
        start.task for start in launcher.derived if "submits its work" in start.task
    ][-1]
    assert "What it wrote as the contracts of services/api" in request
    assert "Refunds go through Jobs." in request

    # What the Manager decided on is held to what it was shown.
    service.team_memory.write(
        "services/api", "contracts.md", "Orders and carts go through Jobs.\n"
    )
    choice = DocumentChoice.model_validate(
        {
            "document": "contracts",
            "folder": "services/api",
            "use": which,
            "replacement": "Orders, carts and refunds go through Jobs.",
        }
    )
    unknown = await use(
        service,
        asked,
        SettleTeamsTool,
        SettleTeamsAction(
            merge_choices=[
                DocumentChoice(document="contracts", folder="apps/web", use="org")
            ]
        ),
    )
    stale = await use(
        service, asked, SettleTeamsTool, SettleTeamsAction(merge_choices=[choice])
    )
    assert unknown.is_error and "services/api" in unknown.text
    assert stale.is_error and "changed since you were shown" in stale.text
    assert "Orders and carts go through Jobs." in stale.text
    assert service.team_memory.read(API_CONTRACTS) == (
        "Orders and carts go through Jobs.\n"
    )

    settled = await use(
        service, asked, SettleTeamsTool, SettleTeamsAction(merge_choices=[choice])
    )

    assert not settled.is_error
    assert service.team_memory.read(API_CONTRACTS) == holds
    place = service.org.place(division_lead_of(service, top))
    assert place is not None and place.merge is None
    # The session reads what holds now, not what it wrote.
    contracts = [
        shown
        for shown in service.documents_shown(api_place(service, top))
        if shown.name == "contracts"
    ]
    assert [shown.text for shown in contracts if shown.own] == [holds]
    # Its opening place also works to the contracts of the team above.
    assert [shown.folder for shown in contracts] == ["", "services/api"]
    # Where its own do not hold, its IC is told the ones that do, to follow
    # up on the code it delivered, and it is on the team's record.
    last = launcher.messages_to(conversation)[-1]
    on_record = [
        event
        for event in service.org_history(50).events
        if event.kind == "not_taken" and "the contracts it wrote" in event.detail
    ]
    if which == "session":
        assert "is not what you wrote" not in last and not on_record
    else:
        assert "is not what you wrote" in last and holds.strip() in last
        assert [event.detail for event in on_record] == [
            "the contracts it wrote for services/api: its merged code was written to it"
        ]


async def test_what_a_reviewer_reconciled_holds_only_for_what_it_was_shown(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    await use(
        service, conversation, WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )
    service.team_memory.write("services/api", "spec.md", "# Orders")
    await use(service, conversation, SubmitReviewTool, DONE)
    decided = ReviewTeamAction(team=top, decision="accept", reconciled="Refunds.")

    # Saying it is reconciled before being shown anything settles nothing.
    unseen = await use(service, division_lead_ic(service, top), ReviewTeamTool, decided)
    assert unseen.is_error and "# Orders" in unseen.text

    # The org's spec changes again before the reviewer answers.
    service.team_memory.write("services/api", "spec.md", "# Orders and carts")
    stale = await use(service, division_lead_ic(service, top), ReviewTeamTool, decided)

    assert stale.is_error
    assert "changed since you were shown it" in stale.text
    assert "# Orders and carts" in stale.text
    assert service.team_memory.read(API_SPEC) == "# Orders and carts"

    again = await use(service, division_lead_ic(service, top), ReviewTeamTool, decided)
    await service.settled()

    assert not again.is_error
    assert service.team_memory.read(API_SPEC) == "# Refunds"


async def test_what_a_session_wrote_for_a_team_that_was_replaced_is_not_the_new_teams(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    await use(
        service, conversation, WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )
    await use(service, conversation, ReportTool, ReportAction(text="Chose a ledger."))
    # Other accepted work put another team in the same directory, which has
    # the same id.
    service.org.update_team("services.api", since="set up by other work")

    await session_accepted(service, top, conversation)

    assert service.team_memory.read(API_SPEC) is None
    assert service.team_memory.reports("services/api", "ic") == []
    [left] = service.team_memory.former("services.api")
    # It is kept for the team that has its directory now.
    assert (left.ended, left.above) == ("not_taken", "services.api")
    assert left.documents == {"spec": "# Refunds"}
    assert [report.text for report in left.reports] == ["Chose a ledger."]
    # The session no longer reads the other team's records as its own.
    assert service.reports_of(api_place(service, top)) == []


async def test_a_team_only_the_session_renamed_is_renamed_in_the_org(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    service.org.update_place(top, title="Payments API")

    await session_accepted(service, top, conversation)

    team = service.org.team("services.api")
    assert team is not None and team.title == "Payments API"
    assert waiting(service, top) == []


async def test_closing_a_session_leaves_the_org_nothing_it_never_had_accepted(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    await use(
        service, conversation, WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )
    await use(service, conversation, ReportTool, ReportAction(text="Chose a ledger."))
    place = api_place(service, top)

    service.close_session(place.session)
    await service.settled()

    assert service.team_memory.read(API_SPEC) is None
    assert service.team_memory.reports("services/api", "ic") == []
    # Nor is it on the org's record that it was closed.
    assert "session_closed" not in [e.kind for e in record_of(service, "root")]
    assert service.drafts(place.session) == {}


async def test_closing_a_session_keeps_what_waited_of_its_accepted_work(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    jobs = service.org.place(f"{JOBS}@{api_place(service, top).session}")
    assert jobs is not None
    service.documents.write(
        jobs.session, team_ref(jobs).keeps("spec"), "Run the refund jobs."
    )
    await session_accepted(service, top, conversation)
    assert waiting(service, top) == [JOBS]

    service.close_session(jobs.session)
    await service.settled()

    # Nobody settled it, so the org's team stays and what the session wrote
    # for its own is that team's record.
    assert path_of(service, JOBS) == "services/api/workers"
    [left] = service.team_memory.former(JOBS)
    assert (left.ended, left.documents) == (
        "not_taken",
        {"spec": "Run the refund jobs."},
    )


async def test_a_team_a_session_prunes_is_the_orgs_until_its_work_is_accepted(
    service: DirectorService,
):
    await split_root(service)
    await landed(service)
    await service.settled()
    kept = Report(at="2026-11-01T00:00:00+00:00", role="ic", text="Chose REST.")
    take_in(service.team_memory, reports={"services/api": [kept]})

    await use(service, manager(service), PruneTeamTool, PRUNE_API)
    await service.settled()

    # Other sessions still have the team and all it recorded.
    assert service.org.team("services.api") is not None
    assert service.team_memory.reports("services/api", "ic") == [kept]
    assert service.team_memory.former("services.api") == []

    await landed(service)
    await service.settled()

    assert service.org.team("services.api") is None
    [record] = service.team_memory.former("services.api")
    assert (record.ended, record.reports) == ("pruned", [kept])
    assert service.team_memory.reports("services/api", "ic") == []


async def test_an_ic_is_told_once_that_the_org_lacks_what_changed_after_an_accept(
    service: DirectorService, launcher: FakeLauncher
):
    await split_root(service)
    await landed(service)
    await service.settled()
    root_ic, root_manager = ic(service), manager(service)
    assert root_ic is not None and root_manager is not None

    def reminders() -> list[str]:
        return [
            text
            for text in launcher.messages_to(root_ic)
            if "The org does not have these changes" in text
        ]

    # A turn that ends with nothing changed since the accept says nothing.
    launcher.report(root_manager, ConversationExecutionStatus.FINISHED)
    await service.settled()
    assert reminders() == []

    await use(service, root_manager, PruneTeamTool, PRUNE_API)
    await service.settled()
    for conversation, status in list(launcher.known.items()):
        if status == ConversationExecutionStatus.RUNNING:
            launcher.report(str(conversation), ConversationExecutionStatus.FINISHED)
    launcher.report(root_manager, ConversationExecutionStatus.FINISHED)
    await service.settled()
    assert len(reminders()) == 1

    # Not with every turn that ends.
    launcher.report(root_ic, ConversationExecutionStatus.FINISHED)
    await service.settled()
    assert len(reminders()) == 1

    # Accepted again, the org has it, and a later change is said again.
    await landed(service)
    await service.settled()
    launcher.report(root_ic, ConversationExecutionStatus.FINISHED)
    await service.settled()
    assert len(reminders()) == 1


def merge_of(service: DirectorService, top: str):
    place = service.org.place(division_lead_of(service, top))
    assert isinstance(place, SessionTeam)
    return place.merge


def refs(repo: Path) -> dict[str, str]:
    """What the org shares, as it is stored."""
    return {
        ref: run_git(repo, "rev-parse", ref).strip()
        for ref in ("refs/agents/org", "refs/agents/memory")
    }


WORKERS = "services.api.workers"


async def test_what_later_work_makes_of_the_org_waits_for_earlier_work(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    first, conversation = await at_api(service)
    opened = await service.open_session(
        OpenSessionRequest(text="Add workers.", node_id="services.api")
    )
    await service.settled()
    second, other = opened.session.teams[0].id, opened.conversation_id
    sets_up_jobs(service, first)
    org_gains(service, JOBS, "services/api/queue")
    sets_up_jobs(service, second, "services/api/workers", WORKERS)
    await use(service, conversation, SubmitReviewTool, DONE)
    await use(service, other, SubmitReviewTool, DONE)

    # Work is accepted in the order it was submitted.
    early = await use(
        service, division_lead_ic(service, second), ReviewTeamTool, accept(second)
    )
    assert early.is_error and "its Division Lead IC has not decided" in early.text
    await use(service, division_lead_ic(service, first), ReviewTeamTool, accept(first))
    # The reviewer that was made to wait is told when it is its turn.
    assert (
        "It is your turn" in launcher.messages_to(division_lead_ic(service, second))[-1]
    )
    await use(
        service, division_lead_ic(service, second), ReviewTeamTool, accept(second)
    )
    await service.settled()

    # The code of the later work is in although the Manager of the earlier
    # has not decided; the team it set up waits for that.
    for top in (first, second):
        review = node(service, top).review
        assert review is not None and review.state == "accepted"
    waits = merge_of(service, second)
    assert waits is not None and waits.code_landed and waits.org_queued
    assert service.org.team(WORKERS) is None
    assert division_lead_manager(service, second) is None
    late = await use(
        service, division_lead_ic(service, second), SettleTeamsTool, SettleTeamsAction()
    )
    assert late.is_error

    await use(
        service,
        division_lead_manager(service, first),
        SettleTeamsTool,
        SettleTeamsAction(),
    )

    # Then both are the org's, the earlier first.
    assert path_of(service, JOBS) == "services/api/jobs"
    assert path_of(service, WORKERS) == "services/api/workers"
    assert merge_of(service, first) is None and merge_of(service, second) is None
    log = run_git(repo, "log", "--format=%s", "refs/agents/memory").splitlines()
    sessions = [first.split("@")[1], second.split("@")[1]]
    landings = [
        name for line in reversed(log) for name in sessions if line.endswith(name)
    ]
    assert landings[-2:] == sessions


async def test_settled_before_the_code_is_accepted_it_all_lands_in_one_step(
    service: DirectorService, repo: Path
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    await use(service, conversation, ReportTool, ReportAction(text="Refunds are in."))
    await use(service, conversation, SubmitReviewTool, DONE)

    # The Manager is asked beside the review, and what it settles is not
    # the org's before the code is accepted.
    asked = division_lead_manager(service, top)
    settled = await use(service, asked, SettleTeamsTool, SettleTeamsAction())
    assert not settled.is_error and "when the IC" in settled.text
    assert path_of(service, JOBS) == "services/api/workers"
    assert service.team_memory.reports("services/api", "ic") == []
    before = refs(repo)

    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))
    await service.settled()

    assert path_of(service, JOBS) == "services/api/jobs"
    assert merge_of(service, top) is None
    after = refs(repo)
    # The tree and team memory changed together: one commit each, made of
    # the same landing.
    for ref in before:
        assert run_git(repo, "rev-parse", f"{after[ref]}~1").strip() == before[ref]
    assert f"session {top.split('@')[1]}" in run_git(
        repo, "log", "-1", "--format=%s", "refs/agents/org"
    )


async def test_a_save_that_comes_in_the_middle_of_a_landing_waits_for_it(
    service: DirectorService, repo: Path, monkeypatch: pytest.MonkeyPatch
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    await use(service, conversation, SubmitReviewTool, DONE)
    asked = division_lead_manager(service, top)
    await use(service, asked, SettleTeamsTool, SettleTeamsAction())
    before = refs(repo)
    store = service.org
    set_shape = store.set_shape
    saves: list[threading.Thread] = []

    def a_save_begins(session_id: str, shape: list[tuple[str, str, str, str]]) -> None:
        # The org is half changed here: a save of the tree is due now.
        save = threading.Thread(target=service._save_tree, args=(store,))
        save.start()
        save.join(0.3)
        saves.append(save)
        set_shape(session_id, shape)

    monkeypatch.setattr(store, "set_shape", a_save_begins)

    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))
    await service.settled()
    for save in saves:
        save.join()

    assert saves
    after = refs(repo)
    landing = f"session {top.split('@')[1]}"
    for ref in before:
        # The landing is the first commit after what was there, on both.
        first = run_git(
            repo, "rev-list", "--reverse", f"{before[ref]}..{after[ref]}"
        ).split()[0]
        assert landing in run_git(repo, "log", "-1", "--format=%s", first)


async def test_work_that_is_sent_back_waits_for_no_manager(
    service: DirectorService, launcher: FakeLauncher
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    await use(service, conversation, SubmitReviewTool, DONE)
    asked = division_lead_manager(service, top)
    assert waiting(service, top) == [JOBS]

    await use(
        service,
        division_lead_ic(service, top),
        ReviewTeamTool,
        ReviewTeamAction(team=top, decision="reject", findings=[FINDING]),
    )

    assert merge_of(service, top) is None
    assert "waits no longer" in launcher.messages_to(asked)[-1]
    gone = await use(service, asked, SettleTeamsTool, SettleTeamsAction())
    assert gone.is_error
    assert path_of(service, JOBS) == "services/api/workers"


async def test_work_that_is_sent_back_joins_the_end_and_keeps_what_was_decided(
    service: DirectorService, launcher: FakeLauncher
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    await use(service, conversation, SubmitReviewTool, DONE)
    asked = division_lead_manager(service, top)
    first = merge_of(service, top)
    assert first is not None
    await use(
        service,
        asked,
        SettleTeamsTool,
        SettleTeamsAction(teams=[TeamChoice(team=JOBS, use="org")]),
    )

    await use(
        service,
        division_lead_ic(service, top),
        ReviewTeamTool,
        ReviewTeamAction(team=top, decision="reject", findings=[FINDING]),
    )
    back = launcher.messages_to(conversation)[-1]
    heard = len(launcher.messages_to(asked))
    await use(service, conversation, SubmitReviewTool, DONE)
    await service.settled()

    assert "joins the end of the line again" in back
    again = merge_of(service, top)
    assert again is not None and again.seq > first.seq
    # The Manager decided this already and nothing of it changed.
    assert [(item.key, item.use) for item in again.items if item.kind == "team"] == [
        (JOBS, "org")
    ]
    assert len(launcher.messages_to(asked)) == heard


async def test_the_division_lead_ic_is_told_of_a_spec_that_could_not_be_written(
    service: DirectorService,
    launcher: FakeLauncher,
    monkeypatch: pytest.MonkeyPatch,
):
    top, conversation = await at_api(service)
    await use(
        service, conversation, WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )
    commit_landing = service._commit_landing

    def changed_meanwhile(landing, message, tree):
        return {**commit_landing(landing, message, tree), API_SPEC: "# Orders"}

    monkeypatch.setattr(service, "_commit_landing", changed_meanwhile)

    await session_accepted(service, top, conversation)

    told = launcher.messages_to(division_lead_ic(service, top))[-1]
    assert API_SPEC in told and "was not written" in told
    # The session keeps its text, to be shown with the org's next time.
    assert API_SPEC in service.drafts(node(service, top).session)


async def test_a_reviewer_writes_the_spec_that_holds(service: DirectorService):
    top, conversation = await at_api(service)
    await use(
        service, conversation, WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )
    service.team_memory.write("services/api", "spec.md", "# Orders")
    await use(service, conversation, SubmitReviewTool, DONE)
    shown = await use(
        service, division_lead_ic(service, top), ReviewTeamTool, accept(top)
    )
    assert shown.is_error and "say in `merge_choices`" in shown.text

    def decided(path: str, use: str, text: str = "", document: str = "spec") -> Any:
        return ReviewTeamAction(
            team=top,
            decision="accept",
            reconciled="Both.",
            merge_choices=[
                DocumentChoice.model_validate(
                    {
                        "document": document,
                        "folder": path,
                        "use": use,
                        "replacement": text,
                    }
                )
            ],
        )

    # Only the documents that land with the work are its reviewer's to
    # decide: one that waits with the teams is settled with them, and the
    # org keeps no other.
    reviewer = division_lead_ic(service, top)
    with_teams = await use(
        service,
        reviewer,
        ReviewTeamTool,
        decided("services/api", "org", "", "contracts"),
    )
    unknown = await use(
        service, reviewer, ReviewTeamTool, decided("services/api", "org", "", "minutes")
    )
    assert with_teams.is_error and "waits with the teams" in with_teams.text
    assert unknown.is_error and "keeps no document named 'minutes'" in unknown.text
    # And only the ones of the folders of the work.
    elsewhere = await use(
        service,
        division_lead_ic(service, top),
        ReviewTeamTool,
        decided("apps/web", "org"),
    )
    empty = await use(
        service,
        division_lead_ic(service, top),
        ReviewTeamTool,
        decided("services/api", "replacement"),
    )
    assert elsewhere.is_error and "not a folder of this work" in elsewhere.text
    assert empty.is_error and "in `replacement`" in empty.text

    written = await use(
        service,
        division_lead_ic(service, top),
        ReviewTeamTool,
        decided("services/api", "replacement", "# Orders and refunds"),
    )
    await service.settled()

    assert not written.is_error
    assert service.team_memory.read(API_SPEC) == "# Orders and refunds"
    assert service.document_draft(api_place(service, top), "spec") is None


async def test_a_reviewer_keeps_the_orgs_spec(service: DirectorService):
    top, conversation = await at_api(service)
    await use(
        service, conversation, WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )
    service.team_memory.write("services/api", "spec.md", "# Orders")
    await use(service, conversation, SubmitReviewTool, DONE)
    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))

    await use(
        service,
        division_lead_ic(service, top),
        ReviewTeamTool,
        ReviewTeamAction(
            team=top,
            decision="accept",
            reconciled="Theirs.",
            merge_choices=[
                DocumentChoice(document="spec", folder="services/api", use="org")
            ],
        ),
    )
    await service.settled()

    assert service.team_memory.read(API_SPEC) == "# Orders"
    assert service.document_draft(api_place(service, top), "spec") is None


async def test_a_manager_puts_a_team_it_takes_where_it_fits_the_org(
    service: DirectorService, launcher: FakeLauncher
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    org_gains(service, "services.api.queue", "services/api/queue")
    await use(service, conversation, SubmitReviewTool, DONE)
    asked = division_lead_manager(service, top)

    def placed(**choice: str) -> Any:
        return SettleTeamsAction(
            teams=[TeamChoice.model_validate({"team": JOBS, **choice})]
        )

    # A team goes only under one that holds its directory, and only a team
    # taken from the session is put or called otherwise.
    outside = await use(
        service, asked, SettleTeamsTool, placed(use="session", under="apps.web")
    )
    beside = await use(
        service,
        asked,
        SettleTeamsTool,
        placed(use="session", under="services.api.queue"),
    )
    kept = await use(service, asked, SettleTeamsTool, placed(use="org", title="Jobs"))
    # Beside the team that holds its folder, both would own it.
    above = await use(
        service, asked, SettleTeamsTool, placed(use="session", under="root")
    )
    assert outside.is_error and "is not inside" in outside.text
    assert beside.is_error and "is not inside" in beside.text
    assert kept.is_error and "stays as the org has it" in kept.text
    assert above.is_error and "goes under team services.api" in above.text

    settled = await use(
        service,
        asked,
        SettleTeamsTool,
        placed(use="session", under="services.api", title="Refund jobs"),
    )
    assert not settled.is_error
    # Nothing of it is the org's before the code is accepted.
    assert path_of(service, JOBS) == "services/api/workers"

    await use(service, division_lead_ic(service, top), ReviewTeamTool, accept(top))
    await service.settled()

    jobs = service.org.team(JOBS)
    assert jobs is not None
    assert (jobs.parent_id, jobs.title, jobs.boundary.path) == (
        "services.api",
        "Refund jobs",
        "services/api/jobs",
    )


@pytest.mark.parametrize("rewritten", [True, False])
async def test_a_spec_written_for_a_team_the_org_lost_is_left_to_the_team_that_holds_it(
    service: DirectorService, rewritten: bool
):
    top, conversation = await at_api(service)
    await use(
        service, conversation, WriteSpecTool, WriteSpecAction(file_text="# Refunds")
    )
    service.org.remove("services.api")
    await use(service, conversation, SubmitReviewTool, DONE)

    # Its reviewer is shown it with the spec of the team that holds the
    # directory now, and may write that one from both.
    shown = await use(
        service, division_lead_ic(service, top), ReviewTeamTool, accept(top)
    )
    assert shown.is_error and "a team the org no longer has" in shown.text
    assert "# Refunds" in shown.text
    await use(
        service,
        division_lead_ic(service, top),
        ReviewTeamTool,
        ReviewTeamAction(
            team=top,
            decision="accept",
            reconciled="Kept what holds.",
            merge_choices=[
                DocumentChoice(
                    document="spec",
                    folder="",
                    use="replacement",
                    replacement="# Checkout, with refunds",
                )
            ]
            if rewritten
            else [],
        ),
    )
    await service.settled()

    root = service.org.team("root")
    assert root is not None
    assert service.team_memory.read(API_SPEC) == "# Refunds"
    if rewritten:
        assert service.team_memory.read("spec.md") == "# Checkout, with refunds"
        assert root.leftover_documents == []
    else:
        # Left alone, it is an old spec the next session there is told of.
        assert [(old.path, old.team) for old in root.leftover_documents] == [
            ("services/api", "API")
        ]
    [left] = service.team_memory.former("services.api")
    assert (left.ended, left.documents) == ("not_taken", {"spec": "# Refunds"})


async def test_a_manager_that_does_not_answer_is_asked_again_and_then_the_human(
    repo: Path, launcher: FakeLauncher
):
    now = [datetime(2026, 11, 14, tzinfo=UTC)]
    service = await started(repo, launcher, clock=lambda: now[0], stall_poll=3600)
    service.update_settings(DirectorSettingsUpdate(soft_check_in_minutes=60))
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    org_gains(service, JOBS, "services/api/workers")
    await session_accepted(service, top, conversation)
    asked = division_lead_manager(service, top)
    told = len(launcher.messages_to(asked))

    async def later(minutes: int) -> None:
        now[0] += timedelta(minutes=minutes)
        await service._check_stalls()
        await service.settled()

    await later(30)
    assert len(launcher.messages_to(asked)) == told
    await later(31)
    assert "still waits for you" in launcher.messages_to(asked)[-1]
    assert node(service, top).escalations == []
    await later(30)
    assert node(service, top).escalations == []

    # After as long again the human is asked, once.
    await later(31)
    await later(61)
    [waits] = node(service, top).escalations
    assert waits.held_by is None and "has not settled" in waits.title
    assert path_of(service, JOBS) == "services/api/workers"

    # Their answer goes on without the Manager: the org's stay.
    service.answer_escalation(top, waits.id, "Go on.")
    await service.settled()

    assert merge_of(service, top) is None
    assert path_of(service, JOBS) == "services/api/workers"
    assert [left.ended for left in service.team_memory.former(JOBS)] == []
    assert node(service, top).escalations == []
    await service.aclose()


async def test_what_a_session_has_is_kept_at_once(repo: Path, launcher: FakeLauncher):
    service = DirectorService(repo, save_delay=3600)
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    await service.settled()
    opened = await service.open_session(
        OpenSessionRequest(text="Add refunds.", node_id="root")
    )
    await service.settled()

    # The tree waits to be saved; what the session has does not.
    kept = service._runs().marked(SESSION)
    assert opened.session.id in kept
    assert "Add refunds." in kept[opened.session.id]
    await service.aclose()


async def test_closing_a_session_leaves_the_humans_own_checkout_as_it_is(
    service: DirectorService, repo: Path, tmp_path: Path
):
    await landed(service)
    await service.settled()
    top = node(service, "root")
    # The session works apart; its Division Lead place is in the repository.
    worktree = tmp_path / "apart"
    run_git(repo, "worktree", "add", "--quiet", "-b", "apart", str(worktree))
    service.org.update_place(top.id, worktree_path=str(worktree), isolation="worktree")
    assert node(service, division_lead_of(service)).agents
    (repo / "notes.txt").write_text("the human's, not committed\n")

    service.close_session(top.session)
    await service.settled()

    assert (repo / "notes.txt").read_text() == "the human's, not committed\n"


async def test_work_that_goes_up_by_itself_is_not_merged_into_another_branch(
    service: DirectorService, repo: Path, tmp_path: Path
):
    await split_root(service)
    worktree = own_worktree(service, repo, tmp_path)
    commit(worktree, "services/api/app.py", "x = 1\n")
    run_git(repo, "checkout", "--quiet", "-b", "elsewhere")
    head = run_git(repo, "rev-parse", "HEAD").strip()
    for team in ("services.api", "apps.web"):
        await use(service, ic(service, team), SubmitReviewTool, DONE)

    for team in ("services.api", "apps.web"):
        await use(service, ic(service), ReviewTeamTool, accept(team))
    await service.settled()

    # The human's checkout is where they left it; the session's IC submits,
    # and the human is asked when that is accepted.
    assert run_git(repo, "rev-parse", "HEAD").strip() == head
    assert node(service, "root").review is None


async def test_a_division_lead_ic_is_not_told_to_submit_work(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_root(service)
    commit(repo, "README.md", "the team's own change\n")
    await landed(service)
    await service.settled()
    lead = division_lead_ic(service)
    assert lead is not None
    told = len(launcher.messages_to(lead))
    commit(repo, "services/api/app.py", "x = 1\n")
    for team in ("services.api", "apps.web"):
        await use(service, ic(service, team), SubmitReviewTool, DONE)

    for team in ("services.api", "apps.web"):
        await use(service, ic(service), ReviewTeamTool, accept(team))
    await service.settled()

    assert not any(
        "submit_review" in message for message in launcher.messages_to(lead)[told:]
    )


async def test_an_acceptance_cut_off_before_its_landing_lands_when_it_is_run_again(
    service: DirectorService, monkeypatch: pytest.MonkeyPatch
):
    await use(service, ic(service), WriteSpecTool, WriteSpecAction(file_text="# Spec"))
    await use(service, ic(service), SubmitReviewTool, DONE)
    top = node(service, "root")
    land = service._land_accepted

    def cut_off(*args: object) -> None:
        raise RuntimeError("the server ends here")

    monkeypatch.setattr(service, "_land_accepted", cut_off)
    await use(service, division_lead_ic(service), ReviewTeamTool, accept(top.id))
    await service.settled()
    monkeypatch.setattr(service, "_land_accepted", land)

    # What a restart does with the step that was still open.
    service._begin(top.id, Accept(needed_above="", base=True))
    await service.settled()

    assert service.team_memory.read("spec.md") == "# Spec"
    assert service._in_order() == []


async def test_work_that_waited_for_the_human_is_merged_as_its_reviewer_decided(
    service: DirectorService, repo: Path, tmp_path: Path
):
    commit(repo, "app.py", "refunds = False\n")
    worktree = own_worktree(service, repo, tmp_path)
    commit(worktree, "app.py", "refunds = True\n")
    (repo / "app.py").write_text("in the way\n")
    await use(
        service, ic(service), WriteSpecTool, WriteSpecAction(file_text="# Session")
    )
    await use(service, ic(service), SubmitReviewTool, DONE)
    top = node(service, "root")
    decided = DocumentChoice.model_validate(
        {
            "document": "spec",
            "folder": "",
            "use": "replacement",
            "replacement": "# Reviewer",
        }
    )
    await use(
        service,
        division_lead_ic(service),
        ReviewTeamTool,
        ReviewTeamAction(team=top.id, decision="accept", merge_choices=[decided]),
    )
    assert [e.id for e in node(service, "root").escalations] == [MERGE_WAITS]
    run_git(repo, "checkout", "--", "app.py")

    service.answer_escalation(top.id, MERGE_WAITS, "Cleared.")
    await service.settled()

    assert (repo / "app.py").read_text() == "refunds = True\n"
    assert service.team_memory.read("spec.md") == "# Reviewer"


async def test_contracts_the_org_did_not_take_are_on_record_when_the_session_closes(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    session_contracts(service, api_place(service, top), "Refunds go through Jobs.")
    service.team_memory.write(
        "services/api", "contracts.md", "Orders go through Jobs.\n"
    )
    await session_accepted(service, top, conversation)

    service.close_session(api_place(service, top).session)
    await service.settled()

    assert any(
        event.kind == "not_taken" and "the contracts it wrote" in event.detail
        for event in service.org_history(100).events
    )


async def test_contracts_for_a_team_the_org_lost_wait_with_the_rest_of_the_landing(
    service: DirectorService,
):
    await split_root(service)
    await landed(service)
    # Work of another session waits ahead of this one's.
    web = await service.open_session(
        OpenSessionRequest(text="Web work.", node_id="apps.web")
    )
    await service.settled()
    ahead = web.session.teams[0].id
    session_contracts(service, node(service, ahead), "The web session's contracts.")
    service.team_memory.write("apps/web", "contracts.md", "The org's.\n")
    await session_accepted(service, ahead, web.conversation_id)
    opened = await service.open_session(
        OpenSessionRequest(text="Add refunds.", node_id="services.api")
    )
    await service.settled()
    top = opened.session.teams[0].id
    place = api_place(service, top)
    session_contracts(service, place, "Refunds go through Jobs.")
    # Other accepted work set the team at that directory up anew meanwhile.
    service.org.update_team("services.api", since="set up anew")

    await session_accepted(service, top, opened.conversation_id)

    # Only its code has landed: what it wrote is not lost before the rest.
    assert [text for text, _ in service.drafts(place.session).values()] == [
        "Refunds go through Jobs.\n"
    ]


async def test_a_team_beside_one_inside_its_directory_waits_for_the_manager_above(
    service: DirectorService,
):
    top, conversation = await at_api(service)
    sets_up_jobs(service, top)
    # Other accepted work gave the org a team inside the same directory.
    org_gains(service, "services.api.nightly", "services/api/jobs/nightly")

    await session_accepted(service, top, conversation)

    # Side by side, one would hold the other's directory: not the org's yet.
    assert service.org.team(JOBS) is None
    assert waiting(service, top) == [JOBS]


async def test_work_that_went_up_without_a_review_counts_as_an_accept_for_reminders(
    service: DirectorService,
):
    await split_root(service)
    session = node(service, "root").session
    # The IC was reminded once of what changed since an earlier accept.
    service._changed_since_accept[session] = True

    for team in ("services.api", "apps.web"):
        await use(service, ic(service, team), SubmitReviewTool, DONE)
        await use(service, ic(service), ReviewTeamTool, accept(node(service, team).id))
    await service.settled()

    review = node(service, "root").review
    assert review is not None and review.passed_up
    assert session not in service._changed_since_accept


async def test_a_landing_that_cannot_be_committed_leaves_the_tree_as_it_was(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    real = service._commit_landing

    def fails(landing, message, tree):
        if tree:
            raise OperationFailed("another writer kept moving a ref first")
        return real(landing, message, tree)

    service._commit_landing = fails  # type: ignore[method-assign]
    await use(service, ic(service), SubmitReviewTool, DONE)
    top = node(service, "root").id
    await use(service, division_lead_ic(service), ReviewTeamTool, accept(top))
    await service.settled()
    service._commit_landing = real  # type: ignore[method-assign]
    await service._save()

    # Neither the tree nor what decides how it lands moved without the
    # team memory that goes with it.
    assert [team.id for team in service.org.teams()] == ["root"]
    saved = OrgStore(shipped_type)
    load_org(create_org_board(repo), saved)
    assert [team.id for team in saved.teams()] == ["root"]
    lead = service.org.place(division_lead_of(service))
    assert lead is not None and lead.merge is not None
    await service.aclose()


async def test_an_org_that_kept_reports_by_team_id_has_them_in_its_folders(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    kept = Report(at="2026-11-01T00:00:00+00:00", role="ic", text="Chose REST.")
    service.team_memory.write(".reports/root", "ic.json", f"[{kept.model_dump_json()}]")
    await service.aclose()

    again = DirectorService(
        repo, save_delay=0, remembered=repo.parent / "state" / "director.json"
    )

    assert again.org.team("root") is not None
    assert again.team_memory.reports("") == [kept]
    assert again.team_memory.read(".reports/root/ic.json") is None
