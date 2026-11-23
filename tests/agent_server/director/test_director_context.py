"""What an agent is told of its team's context without asking: all of it in
its briefing, and what changed with the next message it gets anyway."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from openhands.agent_server.director.context import (
    CUT,
    UPDATE_HEADING,
    changed,
    steady,
    update_block,
)
from openhands.agent_server.director.run_state import TOLD
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    MsgIcTool,
    PruneTeamTool,
    RaiseEscalationTool,
    ReportTool,
    ReviewTeamTool,
    SubmitReviewTool,
    WriteSpecTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgIcAction,
    PruneTeamAction,
    RaiseEscalationAction,
    ReportAction,
    WriteSpecAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    DONE,
    WRITE_UP,
    FakeLauncher,
    accept,
    again,
    commit,
    ic,
    manager,
    node,
    reject,
    reviewer_wakes,
    split_root,
    started,
    use,
    woken,
)


def context(**changes: Any) -> dict[str, Any]:
    return {
        "status": "idle",
        "sub_teams": [
            {"team_id": "api", "status": "working", "headline": "Writes the API."}
        ],
        "reports_of_your_team": [],
        "escalations_you_raised": [],
        "review_of_your_work": None,
        **changes,
    }


def update_of(message: str) -> dict[str, Any] | None:
    """What a message says changed in the context, if it says so."""
    _, heading, block = message.partition(UPDATE_HEADING)
    return json.loads(block[block.index("{") :]) if heading else None


async def tell(service: DirectorService, launcher: FakeLauncher, to: str) -> str:
    """Send the IC of a team a message from the human; what it got."""
    team = "apps.web" if to else "root"
    service.tell_writer(node(service, team).id, "Go on.", human=True)
    await service.settled()
    return launcher.messages_to(ic(service, team))[-1]


def test_a_team_that_starts_or_ends_a_turn_has_not_changed():
    working = context(status="working")
    idle = context(
        sub_teams=[{"team_id": "api", "status": "idle", "headline": "Waits."}]
    )

    assert steady(working) == steady(idle)
    assert "headline" not in steady(idle)["sub_teams"][0]
    blocked = context(sub_teams=[{"team_id": "api", "status": "blocked"}])
    assert changed(steady(idle), steady(blocked), "ic", works=True) == {
        "sub_teams": {"changed": [{"team_id": "api", "status": "blocked"}]}
    }


def test_of_the_sub_teams_only_those_that_changed_or_went_are_told():
    api = {"team_id": "api", "status": "at work", "review": None}
    web = {"team_id": "web", "status": "at work", "review": None}
    db = {"team_id": "db", "status": "at work", "review": None}
    before = context(sub_teams=[api, web])

    submitted = context(sub_teams=[api, {**web, "status": "done"}])
    replaced = context(sub_teams=[api, db])

    assert changed(before, submitted, "ic", works=True) == {
        "sub_teams": {"changed": [{**web, "status": "done"}]}
    }
    assert changed(before, replaced, "ic", works=True) == {
        "sub_teams": {"added": [db], "removed": ["web"]}
    }
    # The first sub-teams a team gets are told as they are.
    assert changed(context(sub_teams=[]), before, "ic", works=True) == {
        "sub_teams": {"added": [api, web]}
    }


def test_only_what_differs_is_told_and_of_a_list_that_grew_only_what_was_added():
    first = {"by": "manager", "text": "Schema first."}
    before = context(reports_of_your_team=[first])
    now = context(
        reports_of_your_team=[first, {"by": "manager", "text": "Then the API."}],
        review_of_your_work={"state": "accepted"},
    )

    assert changed(before, before, "ic", works=True) == {}
    assert changed(before, now, "ic", works=True) == {
        "reports_of_your_team": {"added": [{"by": "manager", "text": "Then the API."}]},
        "review_of_your_work": {"state": "accepted"},
    }
    # Rewritten, not added to: all of it.
    rewritten = context(reports_of_your_team=[{"by": "ic", "text": "In short."}])
    assert changed(before, rewritten, "manager", works=False) == {
        "reports_of_your_team": [{"by": "ic", "text": "In short."}]
    }


def test_an_agent_is_not_told_back_what_it_added_itself():
    before = context()
    now = context(
        reports_of_your_team=[
            {"by": "ic", "text": "Mine."},
            {"by": "manager", "text": "Theirs."},
        ],
        escalations_you_raised=[{"escalation_id": "jwt"}],
    )

    assert changed(before, now, "ic", works=True) == {
        "reports_of_your_team": {"added": [{"by": "manager", "text": "Theirs."}]}
    }


def test_of_a_list_known_by_id_only_what_was_added_changed_or_went_is_told():
    first = {"escalation_id": "e1", "title": "Which router?"}
    second = {"escalation_id": "e2", "title": "Which store?"}
    before = context(escalations_for_you_to_settle=[first, second])
    settled = context(escalations_for_you_to_settle=[second])
    passed = context(
        escalations_for_you_to_settle=[
            first,
            {**second, "passed_up": [{"by": "api", "reason": "Not ours."}]},
        ]
    )

    assert changed(before, settled, "ic", works=True) == {
        "escalations_for_you_to_settle": {"removed": ["e1"]}
    }
    assert changed(before, passed, "ic", works=True) == {
        "escalations_for_you_to_settle": {
            "changed": [passed["escalations_for_you_to_settle"][1]]
        }
    }


def test_an_ic_is_not_told_the_status_its_own_acts_gave_its_team():
    before, done = context(status="at work"), context(status="done")

    assert changed(before, done, "ic", works=True) == {}
    assert changed(before, done, "manager", works=False) == {"status": "done"}


def test_a_long_text_is_cut_in_an_update():
    block = update_block({"spec": "x" * (CUT + 500), "sub_teams": [{"summary": "ok"}]})
    told = json.loads(block[block.index("{") :])

    assert told["spec"].startswith("x" * CUT) and "read_team_context" in told["spec"]
    assert len(told["spec"]) < CUT + 100
    assert told["sub_teams"] == [{"summary": "ok"}]


async def test_an_ic_is_not_sent_back_what_it_wrote_itself(
    repo: Path, launcher: FakeLauncher
):
    account = "It adds the checkout endpoint and its tests. " * 200
    launcher.agent_answers = {WRITE_UP: account}
    service = await started(repo, launcher)
    await split_root(service)
    api = ic(service, "services.api")
    assert api is not None

    commit(repo, "services/api/app.py", "order_id = '1'\n")
    await use(service, api, SubmitReviewTool, DONE)
    await use(service, ic(service), ReviewTeamTool, reject())
    commit(repo, "services/api/app.py", "order_id = 1\n")
    await use(service, api, SubmitReviewTool, DONE)
    await reviewer_wakes(service, launcher)
    await use(service, ic(service), ReviewTeamTool, accept())
    await service.settled()

    updates = [
        told for told in map(update_of, launcher.messages_to(api)) if told is not None
    ]
    assert updates
    for told in updates:
        assert account[:60] not in json.dumps(told)
        assert len(json.dumps(told)) < 600
        assert set(told.get("review_of_your_work") or {}) <= {"state", "round"}

    # A spec it wrote itself is not told back to it either.
    await use(
        service,
        await woken(service, "apps.web"),
        WriteSpecTool,
        WriteSpecAction(file_text="# Web\nThe router is server-side."),
    )
    assert "spec" not in (update_of(await tell(service, launcher, "apps/web")) or {})
    await service.aclose()


async def test_a_briefing_carries_the_teams_whole_context(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)

    [start] = [s for s in launcher.derived if "IC of the team 'Web'" in s.task]

    assert "Contracts your team must keep" in start.task
    assert "read_team_context` is not needed in normal work" in start.task
    block = start.task[start.task.index("lost track.\n") :]
    rest = json.loads(block[block.index("{") : block.rindex("}") + 1])
    assert rest["you"] == {
        "role": "ic",
        "team_id": node(service, "apps.web").id,
        "team": "Web",
        "type": "software-engineer",
        "session": "Checkout",
    }
    assert rest["team_above"] == node(service, "root").id
    # What is empty is left out.
    assert "sub_teams" not in rest and "pending_split" not in rest
    # Said in the briefing's own words above, so not a second time.
    assert "spec" not in rest and "boundary" not in rest
    await service.aclose()


async def test_a_change_sends_nothing_and_comes_with_the_next_message(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await tell(service, launcher, "apps/web")
    heard = len(launcher.messages_to(ic(service, "apps.web")))

    await use(
        service,
        await woken(service, "apps.web"),
        ReportTool,
        ReportAction(text="The router is server-side."),
    )

    assert len(launcher.messages_to(ic(service, "apps.web"))) == heard

    first = await tell(service, launcher, "apps/web")
    second = await tell(service, launcher, "apps/web")

    assert first.startswith("Go on.\n\n[director] context update")
    told = update_of(first)
    assert told is not None and list(told) == ["reports_of_your_team"]
    [report] = told["reports_of_your_team"]["added"]
    assert (report["by"], report["text"]) == ("manager", "The router is server-side.")
    assert second == "Go on."
    await service.aclose()


async def test_teams_that_start_and_end_turns_cause_no_update(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await tell(service, launcher, "")
    web = ic(service, "apps.web")
    assert web is not None

    launcher.report(web, ConversationExecutionStatus.RUNNING)
    working = await tell(service, launcher, "")
    launcher.report(web, ConversationExecutionStatus.FINISHED)
    idle = await tell(service, launcher, "")

    assert (working, idle) == ("Go on.", "Go on.")
    await service.aclose()


async def test_reviews_and_escalations_below_come_with_what_wakes_the_ic(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await tell(service, launcher, "")
    commit(repo, "services/api/app.py", "order_id = 1\n")
    await use(service, ic(service, "services.api"), SubmitReviewTool, DONE)

    await use(
        service,
        ic(service, "apps.web"),
        RaiseEscalationTool,
        RaiseEscalationAction(category="design_decision", title="Which router?"),
    )

    # The review request went when the work was submitted and the
    # escalation after it: each carries what had changed by then.
    updates = [
        told
        for told in map(update_of, launcher.messages_to(ic(service))[-2:])
        if told is not None
    ]
    [[to_settle]] = [
        told["escalations_for_you_to_settle"]["added"]
        for told in updates
        if "escalations_for_you_to_settle" in told
    ]
    assert to_settle["title"] == "Which router?"
    by_team = {
        team["team_id"]: team
        for told in updates
        for team in told.get("sub_teams", {}).get("changed", [])
    }
    assert by_team[node(service, "apps.web").id]["status"] == "blocked"
    assert by_team[node(service, "services.api").id]["review"] == "pending"
    # The Web IC raised it itself and is not told so.
    web = await tell(service, launcher, "apps/web")
    assert "escalations_you_raised" not in (update_of(web) or {})
    await service.aclose()


async def test_what_an_agent_was_told_outlasts_the_server(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    await tell(first, launcher, "")
    await first.aclose()

    second = await again(repo, launcher)
    await second.resume()

    assert await tell(second, launcher, "") == "Go on."
    await second.aclose()


async def test_an_agent_never_told_its_context_gets_all_of_it_once(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    root = ic(service)
    assert root is not None
    service._runs().unmark(TOLD, root)

    first = await tell(service, launcher, "")
    second = await tell(service, launcher, "")

    told = update_of(first)
    assert told is not None and told["you"]["team_id"] == node(service, "root").id
    assert "boundary" in told and second == "Go on."
    await service.aclose()


async def test_what_the_agents_of_a_removed_team_were_told_is_forgotten(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    gone = {ic(service, "services.api"), await woken(service, "services.api")}
    assert gone <= set(service._runs().marked(TOLD))

    await use(
        service,
        manager(service),
        PruneTeamTool,
        PruneTeamAction(team="services.api", reason="No way forward."),
    )

    assert not gone & set(service._runs().marked(TOLD))
    assert ic(service, "apps.web") in service._runs().marked(TOLD)
    await service.aclose()


async def test_the_directors_own_notices_bring_what_changed_too(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)

    await use(service, manager(service), MsgIcTool, MsgIcAction(text="Carry on."))

    approved, written = launcher.messages_to(ic(service))[-2:]
    assert approved.startswith("[director] Your team was split: your Manager approved")
    told = update_of(approved)
    assert told is not None
    assert [team["team_id"] for team in told["sub_teams"]["added"]] == [
        node(service, "services.api").id,
        node(service, "apps.web").id,
    ]
    # The teams were given their tasks between the two: the one change.
    assert written.startswith("[director] Your Manager writes:\nCarry on.")
    assert [team["status"] for team in told["sub_teams"]["added"]] == [
        "not_started",
        "not_started",
    ]
    changed = update_of(written)
    assert changed is not None and list(changed) == ["sub_teams"]
    assert [
        (team["team_id"], team["status"]) for team in changed["sub_teams"]["changed"]
    ] == [
        (node(service, "services.api").id, "at work"),
        (node(service, "apps.web").id, "at work"),
    ]
    await service.aclose()


def test_a_part_that_is_gone_is_told_and_another_order_is_not():
    api = {"team_id": "api", "title": "API"}
    web = {"team_id": "web", "title": "Web"}
    other = [{"session": "s2", "title": "Refunds"}]
    before = context(sub_teams=[api, web], other_sessions_of_your_team=other)

    assert changed(before, context(sub_teams=[web, api]), "ic", works=True) == {
        "other_sessions_of_your_team": None
    }
    assert (
        changed(
            before,
            context(sub_teams=[web, api], other_sessions_of_your_team=other),
            "ic",
            works=True,
        )
        == {}
    )


async def test_an_agent_whose_history_was_shortened_is_told_of_the_org_again(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    conversation = ic(service)
    assert conversation is not None
    await tell(service, launcher, "")

    launcher.condensations[UUID(conversation)] = "first"
    after = await tell(service, launcher, "")

    assert "history was shortened" in after
    assert "<ORG_PROTOCOL>" in after and "Your team's context now" in after
    # It is said once for each shortening, not with every message.
    assert "<ORG_PROTOCOL>" not in await tell(service, launcher, "")
    launcher.condensations[UUID(conversation)] = "second"
    assert "<ORG_PROTOCOL>" in await tell(service, launcher, "")
