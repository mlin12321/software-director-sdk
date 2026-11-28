"""The director over a live agent server: an org is run, paused, reviewed,
split and pruned through the HTTP routes, with a scripted model."""

import json
import re
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from litellm.types.utils import Choices, Message as LiteLLMMessage, ModelResponse

from openhands.sdk import LLM, Message, TextContent
from openhands.sdk.git.utils import run_git_command
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.llm.utils.metrics import MetricsSnapshot
from tests.cross.conftest import live_server_env


@pytest.fixture
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[dict]:
    with live_server_env(tmp_path, monkeypatch) as env:
        yield env


def _scripted_org_llm(
    monkeypatch: pytest.MonkeyPatch,
    api_ic: list[tuple[str, dict]] | None = None,
    web_ic: list[tuple[str, dict]] | None = None,
    top_ic: list[tuple[str, dict]] | None = None,
    tasked: tuple[str, ...] = ("services.api", "apps.web"),
) -> None:
    """Agents of an org that split the root once: the IC proposes when it is
    briefed, its Manager approves when it is told, the IC then gives each
    team its task, and everyone else just replies.

    The top IC accepts every team it is asked to review, and tells its Manager
    of what the API team escalates; the Manager then removes that team, and
    the top IC submits the work that is left. The IC of a session opened
    later to add refunds just replies.

    Args:
        api_ic: Tool calls the IC of the API sub-team makes, one per turn.
        web_ic: The same for the IC of the Web sub-team.
        top_ic: The same for the IC of the top team, in place of its proposal.
        tasked: The sub-teams the top IC gives a task, which starts them.
    """
    scripted = {"API": api_ic or [], "Web": web_ic or [], "Checkout": top_ic or []}

    def tool_call(name: str, arguments: dict) -> dict:
        # A value the script can know only when the call is made is given
        # as a function.
        arguments = {
            key: value() if callable(value) else value
            for key, value in arguments.items()
        }
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{name}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    def text_of(message: Message) -> str:
        return "".join(
            content.text
            for content in message.content
            if isinstance(content, TextContent)
        )

    def reply_to(messages: list[Message]) -> dict:
        last = messages[-1]
        if last.role == "user" and "has already been delivered" in text_of(last):
            # A message came in the middle of a turn: the SDK points back
            # at it, and it is the one to answer.
            last = next(
                (
                    message
                    for message in reversed(messages[:-1])
                    if message.role == "user"
                ),
                last,
            )
        text = text_of(last)
        if "You label the work of software teams" in text_of(messages[0]):
            return {
                "role": "assistant",
                "content": "Title: Checkout\nHeadline: Builds the checkout flow.",
            }
        later = any("Add refunds" in text_of(m) for m in messages)
        for team, calls in scripted.items():
            if later:
                break
            if any(f"the IC of the team {team!r}" in text_of(m) for m in messages):
                done = sum(message.role == "tool" for message in messages)
                if done < len(calls):
                    return tool_call(*calls[done])
        if any("your Manager approved your sub-teams" in text_of(m) for m in messages):
            # Approved teams wait for their task; one call a turn.
            given = sum(
                call.name == "msg_team"
                for message in messages
                for call in message.tool_calls or []
            )
            if given < len(tasked):
                return tool_call("msg_team", {"team": tasked[given], "text": "Begin."})
        asked = [at for at, m in enumerate(messages) if "Review request." in text_of(m)]
        if asked:
            # One call a turn: the teams of the request, in its order.
            request = text_of(messages[asked[-1]])
            waiting = re.findall(r"`review_team` for team (\S+):", request)
            # Only its own answers count: a request may arrive in the
            # middle of a turn, before the answer to another tool.
            reviewed = sum(
                call.name == "review_team"
                for m in messages[asked[-1] :]
                for call in m.tool_calls or []
            )
            if reviewed < len(waiting):
                return tool_call(
                    "review_team", {"team": waiting[reviewed], "decision": "accept"}
                )
        if last.role == "user" and "in hand-over mode" in text:
            return {"role": "assistant", "content": "Nothing to pass on."}
        if last.role == "user" and "Your hand-over mode has ended" in text:
            return tool_call("submit_review", {"summary": "Shown, not served."})
        if last.role == "user" and "You are the IC of the team 'Checkout'" in text:
            # The contracts first: sub-teams are not proposed without them.
            return tool_call(
                "write_contracts",
                {
                    "command": "create",
                    "file_text": "POST /checkout returns an order id\n",
                },
            )
        if last.role == "tool" and "the ones you propose read these contracts" in text:
            return tool_call(
                "propose_teams",
                {
                    "text": "Two teams can build this side by side.",
                    "teams": [
                        {
                            "name": "API",
                            "path": "services/api",
                            "instruction": "Serve it",
                        },
                        {"name": "Web", "path": "apps/web", "instruction": "Show it"},
                    ],
                },
            )
        if (
            last.role == "user"
            and "Your IC writes:" in text
            and "Proposed sub-teams:" in text
        ):
            return tool_call("approve_teams", {})
        if last.role == "user" and "The IC of team 'API'" in text:
            return tool_call(
                "msg_manager",
                {"text": "The API team cannot go on. Its IC escalated: too much."},
            )
        if last.role == "user" and "The API team cannot go on" in text:
            return tool_call(
                "prune_team",
                {
                    "team": "services.api",
                    "reason": "The file grew past what one agent can hold.",
                },
            )
        return {"role": "assistant", "content": "Understood."}

    def fake_completion(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        litellm_msg = LiteLLMMessage.model_validate(reply_to(messages))
        return LLMResponse(
            message=Message.from_llm_chat_message(litellm_msg),
            metrics=MetricsSnapshot(
                model_name="test-model",
                accumulated_cost=0.0,
                max_budget_per_task=None,
                accumulated_token_usage=None,
            ),
            raw_response=ModelResponse(
                id="test-resp",
                created=int(time.time()),
                model="test-model",
                choices=[Choices(index=0, finish_reason="stop", message=litellm_msg)],
            ),
        )

    async def fake_acompletion(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        return fake_completion(self, messages, tools, **kwargs)

    monkeypatch.setattr(LLM, "completion", fake_completion, raising=True)
    monkeypatch.setattr(LLM, "acompletion", fake_acompletion, raising=True)


def _work_places(client: httpx.Client, session_id: str) -> dict[str, dict]:
    """What a session has at each team it works at, by the team's id."""
    response = client.get(f"/api/director/org/sessions/{session_id}")
    teams = response.json()["teams"] if response.status_code == 200 else []
    return {team["team_id"]: team for team in teams if not team["division_lead"]}


def test_director_runs_an_org_whose_root_splits_into_sub_teams(
    server_env, monkeypatch: pytest.MonkeyPatch
):
    _scripted_org_llm(monkeypatch)
    repo = server_env["workspace_path"]
    run_git_command(["git", "init", "-b", "main"], repo)
    run_git_command(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "commit"]
        + ["--allow-empty", "-m", "init"],
        repo,
    )

    with httpx.Client(base_url=server_env["host"], timeout=30) as client:
        before = client.get("/api/director/org").json()
        created = client.post(
            "/api/director/org",
            json={
                "title": "Checkout",
                "task": "Build the checkout flow",
                "conversation": {
                    "agent": {
                        "llm": {"model": "gpt-4o-mini", "api_key": "test"},
                        "tools": [],
                    },
                    "workspace": {"working_dir": str(repo)},
                    "autotitle": False,
                },
            },
        )
        assert created.status_code == 201, created.text
        opened = created.json()
        session_id = opened["session"]["id"]

        deadline = time.time() + 30
        places = _work_places(client, session_id)
        while time.time() < deadline and (
            len(places) < 3
            or places["root"]["pending_split"]
            or not all(place["agents"] for place in places.values())
        ):
            time.sleep(0.2)
            places = _work_places(client, session_id)

        assert (before["root_id"], before["is_git_repository"]) == (None, True)
        assert list(places) == ["root", "services.api", "apps.web"]
        top = places["root"]
        assert top["id"] == f"root@{session_id}"
        assert top["child_ids"] == [
            f"services.api@{session_id}",
            f"apps.web@{session_id}",
        ]
        assert top["pending_split"] is None
        # The org has a session's sub-teams once its work is accepted, and
        # its teams have no agents: those are the session's.
        tree = client.get("/api/director/org/nodes/root").json()
        assert tree["children"] == [] and "agents" not in tree["node"]
        threads = client.get(f"/api/director/org/nodes/{top['id']}/messages").json()
        # The IC gave each team its task in a thread with that team.
        assert sum(thread["ic_to_ic"] for thread in threads) == 2
        [thread] = [thread for thread in threads if not thread["ic_to_ic"]]
        assert (thread["manager_node_id"], thread["ic_node_id"]) == (
            top["id"],
            top["id"],
        )
        [proposal] = thread["messages"]
        assert proposal["sender"] == "ic" and "Proposed sub-teams:" in proposal["text"]
        # A team starts with its IC; its Manager came with the proposal.
        [first] = opened["session"]["teams"][0]["agents"]
        assert (first["role"], first["conversation_id"]) == (
            "ic",
            opened["conversation_id"],
        )
        root_ic, root_manager = (agent["conversation_id"] for agent in top["agents"])
        assert root_ic == opened["conversation_id"]
        manager = client.get(f"/api/conversations/{root_manager}").json()
        assert manager["parent_conversation_id"] == root_ic
        assert [
            client.get(f"/api/conversations/{root_ic}").json()["title"],
            manager["title"],
        ] == ["Checkout (IC)", "Checkout (Manager)"]
        for team_id in ("services.api", "apps.web"):
            [child_ic] = (
                agent["conversation_id"] for agent in places[team_id]["agents"]
            )
            ic = client.get(f"/api/conversations/{child_ic}")
            # A team's IC is on the node a moment before its conversation is.
            while ic.status_code == 404 and time.time() < deadline:
                time.sleep(0.2)
                ic = client.get(f"/api/conversations/{child_ic}")
            assert ic.status_code == 200, team_id
            assert ic.json()["parent_conversation_id"] == root_ic
        specs = client.get(
            f"/api/director/org/nodes/services.api@{session_id}/memory"
        ).json()
        assert [(found["name"], found["own"]) for found in specs["documents"]] == [
            ("contracts", False)
        ]
        assert "POST /checkout" in specs["documents"][0]["text"]
        # Everything the org keeps is in hidden refs, not in the working tree.
        assert run_git_command(["git", "status", "--porcelain"], repo) == ""


def test_director_holds_a_paused_org_and_takes_it_up_again(
    server_env, monkeypatch: pytest.MonkeyPatch
):
    _scripted_org_llm(
        monkeypatch,
        top_ic=[
            ("raise_escalation", {"category": "design_decision", "title": "REST?"})
        ],
    )
    repo = server_env["workspace_path"]
    run_git_command(["git", "init", "-b", "main"], repo)
    run_git_command(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "commit"]
        + ["--allow-empty", "-m", "init"],
        repo,
    )

    with httpx.Client(base_url=server_env["host"], timeout=30) as client:
        refused = client.post("/api/director/org/pause")
        created = client.post(
            "/api/director/org",
            json={
                "title": "Checkout",
                "task": "Build the checkout flow",
                "conversation": {
                    "agent": {
                        "llm": {"model": "gpt-4o-mini", "api_key": "test"},
                        "tools": [],
                    },
                    "workspace": {"working_dir": str(repo)},
                    "autotitle": False,
                },
            },
        )
        assert created.status_code == 201, created.text
        root_ic = created.json()["conversation_id"]
        session_id = created.json()["session"]["id"]

        def said_to_the_ic() -> list[str]:
            events = client.get(
                f"/api/conversations/{root_ic}/events/search", params={"limit": 100}
            ).json()["items"]
            return [
                content["text"]
                for event in events
                if event["kind"] == "MessageEvent" and event["source"] == "user"
                for content in event["llm_message"]["content"]
            ]

        def told_the_answer() -> bool:
            return any("Use REST." in text for text in said_to_the_ic())

        def waiting() -> list[dict]:
            top = _work_places(client, session_id).get("root", {})
            return [
                escalation
                for escalation in top.get("escalations", [])
                if escalation["held_by"] is None
            ]

        deadline = time.time() + 30
        while time.time() < deadline and not waiting():
            time.sleep(0.2)
        [asked] = waiting()

        paused = client.post("/api/director/org/pause")
        sent = client.post("/api/director/org/messages", json={"text": "Add a cart"})
        answered = client.post(
            f"/api/director/org/sessions/{session_id}/teams/root"
            f"/escalations/{asked['id']}/answer",
            json={"answer": "Use REST."},
        )
        held = client.get("/api/director/org/recovery").json()
        time.sleep(1)
        before = told_the_answer()
        resumed = client.post("/api/director/org/resume")
        deadline = time.time() + 30
        while time.time() < deadline and not told_the_answer():
            time.sleep(0.2)

        assert refused.status_code == 409, refused.text
        assert paused.status_code == 200 and paused.json()["paused"] == "human"
        # A paused org opens no session, and holds what is said to its agents.
        assert sent.status_code == 409, sent.text
        assert answered.status_code == 200, answered.text
        assert answered.json()["escalations"] == []
        assert held["messages_held"] >= 1
        assert not before
        assert resumed.status_code == 200 and resumed.json()["paused"] is None
        assert told_the_answer()
        assert client.get("/api/director/org").json()["paused"] is None
        # What is said in a session goes on in it and opens no other.
        said = client.post(
            "/api/director/org/messages",
            json={"text": "And a wishlist.", "session_id": session_id},
        )
        assert said.status_code == 200, said.text
        assert said.json()["conversation_id"] == created.json()["conversation_id"]
        assert [
            found["id"] for found in client.get("/api/director/org/sessions").json()
        ] == [session_id]
    # What the director was in the middle of is beside the repository, never
    # in it.
    assert (repo / ".git" / "director" / "state.db").is_file()
    assert run_git_command(["git", "status", "--porcelain"], repo) == ""


def test_director_has_submitted_work_reviewed_once_all_teams_submitted(
    server_env, monkeypatch: pytest.MonkeyPatch
):
    repo = server_env["workspace_path"]
    _scripted_org_llm(
        monkeypatch,
        api_ic=[("submit_review", {"summary": "Served."})],
        web_ic=[("submit_review", {"summary": "Shown."})],
    )
    run_git_command(["git", "init", "-b", "main"], repo)
    run_git_command(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "commit"]
        + ["--allow-empty", "-m", "init"],
        repo,
    )

    with httpx.Client(base_url=server_env["host"], timeout=30) as client:
        created = client.post(
            "/api/director/org",
            json={
                "title": "Checkout",
                "task": "Build the checkout flow",
                "conversation": {
                    "agent": {
                        "llm": {"model": "gpt-4o-mini", "api_key": "test"},
                        "tools": [],
                    },
                    "workspace": {"working_dir": str(repo)},
                    "autotitle": False,
                },
            },
        )
        assert created.status_code == 201, created.text
        session_id = created.json()["session"]["id"]

        def api() -> dict:
            return _work_places(client, session_id).get("services.api", {})

        states: list[str] = []
        deadline = time.time() + 30
        while time.time() < deadline and states[-1:] != ["accepted"]:
            review = api().get("review")
            if review and states[-1:] != [review["state"]]:
                states.append(review["state"])
            time.sleep(0.05)

        assert states[-1:] == ["accepted"], states
        # The team is done a moment after its review says accepted.
        while time.time() < deadline and api().get("status") != "done":
            time.sleep(0.05)
        node = api()
        assert node["status"] == "done"
        assert node["review"]["summary"] == "Served."
        assert node["base_commit"] == run_git_command(
            ["git", "rev-parse", "HEAD"], repo
        )

        # Their accepted work is the whole of the session's, so it goes up
        # as it is and the org has the session's teams from then on.
        def teams() -> list[dict]:
            return client.get("/api/director/org/nodes/root").json()["children"]

        while time.time() < deadline and len(teams()) < 2:
            time.sleep(0.05)
        assert [team["id"] for team in teams()] == ["services.api", "apps.web"]
        top = _work_places(client, session_id)["root"]
        assert (top["review"]["state"], top["review"]["passed_up"]) == (
            "accepted",
            True,
        )


def test_director_opens_a_session_beside_a_team_and_closes_it(
    server_env, monkeypatch: pytest.MonkeyPatch
):
    repo = server_env["workspace_path"]
    _scripted_org_llm(
        monkeypatch,
        api_ic=[("submit_review", {"summary": "Served."})],
        tasked=("services.api",),
    )
    run_git_command(["git", "init", "-b", "main"], repo)
    run_git_command(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "commit"]
        + ["--allow-empty", "-m", "init"],
        repo,
    )

    with httpx.Client(base_url=server_env["host"], timeout=30) as client:
        created = client.post(
            "/api/director/org",
            json={
                "title": "Checkout",
                "task": "Build the checkout flow",
                "conversation": {
                    "agent": {
                        "llm": {"model": "gpt-4o-mini", "api_key": "test"},
                        "tools": [],
                    },
                    "workspace": {"working_dir": str(repo)},
                    "autotitle": False,
                },
            },
        )
        assert created.status_code == 201, created.text
        first = created.json()["session"]["id"]

        def node(node_id: str) -> dict:
            response = client.get(f"/api/director/org/nodes/{node_id}")
            return response.json()["node"] if response.status_code == 200 else {}

        def until(reached, seconds: float = 30) -> None:
            deadline = time.time() + seconds
            while time.time() < deadline and not reached():
                time.sleep(0.05)

        # The org has the teams of its first session once their work is
        # accepted, and a session can be opened at one of them from then on.
        until(lambda: node("services.api"))
        unknown = client.post(
            "/api/director/org/sessions", json={"text": "x", "node_id": "no.such"}
        )
        opened = client.post(
            "/api/director/org/sessions",
            json={
                "text": "Add refunds to the API.",
                "title": "Refunds",
                "node_id": "services.api",
            },
        )
        assert unknown.status_code == 404, unknown.text
        assert opened.status_code == 201, opened.text
        session = opened.json()["session"]

        def top() -> dict:
            response = client.get(f"/api/director/org/sessions/{session['id']}")
            return response.json()["teams"][0] if response.status_code == 200 else {}

        # Its IC works in a git worktree of its own, on a branch of its own.
        until(lambda: top().get("base_commit"))
        started = top()
        worktree = Path(started["worktree_path"])
        assert (session["team_id"], session["above_id"]) == ("services.api", "root")
        assert (session["title"], started["team_id"]) == ("Refunds", "services.api")
        assert worktree.is_dir() and worktree.resolve() != Path(repo).resolve()
        assert started["branch"] not in ("", "main")
        assert started["branch"] == run_git_command(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], worktree
        )
        events = client.get(
            f"/api/conversations/{opened.json()['conversation_id']}/events/search",
            params={"limit": 100},
        ).json()["items"]
        said = "".join(
            content["text"]
            for event in events
            if event["kind"] == "MessageEvent" and event["source"] == "user"
            for content in event["llm_message"]["content"]
        )
        assert "Add refunds to the API." in said and "session 'Refunds'" in said
        listed = client.get("/api/director/org/sessions").json()
        assert [found["id"] for found in listed] == [first, session["id"]]
        # The human wrote in its conversation, so it is listed with theirs.
        talk = opened.json()["conversation_id"]
        until(
            lambda: (
                client.get(f"/api/conversations/{talk}").json()["tags"]
                == {"startedby": "director", "humanjoined": "true"}
            )
        )
        place = client.get(f"/api/director/org/conversations/{talk}").json()
        assert (place["node_id"], place["session"], place["human_joined"]) == (
            "services.api",
            "Refunds",
            True,
        )
        # Its reviewer is started when it submits, so nobody is above it yet.
        assert place["links"] == []
        again_joined = client.post(f"/api/director/org/conversations/{talk}/joined")
        assert again_joined.status_code == 200, again_joined.text
        missing = "00000000-0000-0000-0000-000000000000"
        assert (
            client.get(f"/api/director/org/conversations/{missing}").status_code == 404
        )
        # The org tree has the teams it had, and the session is none of them.
        root = client.get("/api/director/org/nodes/root").json()
        assert [child["id"] for child in root["children"]] == [
            "services.api",
            "apps.web",
        ]
        assert not node(started["id"])

        closed = client.delete(f"/api/director/org/sessions/{session['id']}")
        until(lambda: not top())
        again = client.delete(f"/api/director/org/sessions/{session['id']}")

        assert closed.status_code == 202, closed.text
        assert again.status_code == 404, again.text
        assert [
            found["id"] for found in client.get("/api/director/org/sessions").json()
        ] == [first]
        until(lambda: not worktree.exists(), 10)
        assert not worktree.exists()
        assert _work_places(client, first)["services.api"]["agents"]


def test_director_prunes_a_blocked_team_and_maps_who_owns_what(
    server_env, monkeypatch: pytest.MonkeyPatch
):
    repo = server_env["workspace_path"]
    _scripted_org_llm(
        monkeypatch,
        api_ic=[
            (
                "raise_escalation",
                {"category": "critical_issue", "title": "Too much for one team"},
            )
        ],
        web_ic=[("submit_review", {"summary": "Shown."})],
    )
    run_git_command(["git", "init", "-b", "main"], repo)
    for path in ("services/api/app.py", "apps/web/index.html"):
        (repo / path).parent.mkdir(parents=True)
        (repo / path).write_text("x\n")
    run_git_command(["git", "add", "."], repo)
    run_git_command(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "commit"]
        + ["-m", "init"],
        repo,
    )

    with httpx.Client(base_url=server_env["host"], timeout=30) as client:
        created = client.post(
            "/api/director/org",
            json={
                "title": "Checkout",
                "task": "Build the checkout flow",
                "conversation": {
                    "agent": {
                        "llm": {"model": "gpt-4o-mini", "api_key": "test"},
                        "tools": [],
                    },
                    "workspace": {"working_dir": str(repo)},
                    "autotitle": False,
                },
            },
        )
        assert created.status_code == 201, created.text
        session_id = created.json()["session"]["id"]

        def teams() -> list[str]:
            return list(_work_places(client, session_id))

        def landed() -> list[str]:
            detail = client.get("/api/director/org/nodes/root").json()
            return [child["id"] for child in detail["children"]]

        seen: set[str] = set()
        deadline = time.time() + 30
        while time.time() < deadline and (
            "services.api" not in seen or "services.api" in teams()
        ):
            seen.update(teams())
            time.sleep(0.1)

        assert teams() == ["root", "apps.web"]
        # The work of the team that is left is all of the session's, and
        # with it accepted the org has that team and never had the other.
        while time.time() < deadline and not landed():
            time.sleep(0.1)
        assert landed() == ["apps.web"]
        ownership = client.get("/api/director/org/nodes/root/ownership")
        assert ownership.status_code == 200, ownership.text
        body = ownership.json()
        assert [owner["node_id"] for owner in body["owners"]] == ["apps.web"]
        assert body["files"] == [
            {"path": "apps/web/index.html", "owner": "apps.web"},
            {"path": "services/api/app.py", "owner": None},
        ]
        # Nobody works at a team of the org, so no work can collide there;
        # the session's place is where collisions are looked for.
        assert (body["collisions"], body["collisions_checked"]) == ([], False)
        place = client.get(
            f"/api/director/org/nodes/root@{session_id}/ownership"
        ).json()
        assert [owner["node_id"] for owner in place["owners"]] == [
            f"apps.web@{session_id}"
        ]
        assert (place["collisions"], place["collisions_checked"]) == ([], True)
        titles = [
            item["title"]
            for item in client.get("/api/conversations/search").json()["items"]
        ]
        # An agent's conversation is named by its team and its session.
        assert "Web · Checkout (IC)" in titles
        assert not any(title.startswith("API") for title in titles)


def test_director_starts_an_org_from_a_message_and_holds_teams_to_their_directories(
    server_env, monkeypatch: pytest.MonkeyPatch
):
    repo = server_env["workspace_path"]

    def checkout() -> Path:
        """The worktree the session works in."""
        listed = run_git_command(["git", "worktree", "list", "--porcelain"], repo)
        [other] = [
            Path(line.removeprefix("worktree "))
            for line in listed.splitlines()
            if line.startswith("worktree ")
            and Path(line.removeprefix("worktree ")).resolve() != repo.resolve()
        ]
        return other

    _scripted_org_llm(
        monkeypatch,
        api_ic=[
            (
                "file_editor",
                {
                    "command": "create",
                    "path": lambda: str(checkout() / "by-editor.txt"),
                    "file_text": "x",
                },
            ),
            (
                "file_editor",
                {
                    "command": "create",
                    "path": lambda: str(
                        checkout() / "services" / "api" / "by-editor.txt"
                    ),
                    "file_text": "x",
                },
            ),
            (
                "terminal",
                {
                    "command": (
                        "echo x > by-shell.txt; echo x > ../../by-shell.txt; "
                        "echo x > ../../apps/by-shell.txt; pwd"
                    )
                },
            ),
            (
                "terminal",
                {
                    "command": (
                        "git add . && git -c user.name=T -c user.email=t@example.com "
                        "commit -q -m Served"
                    )
                },
            ),
            ("submit_review", {"summary": "Served."}),
        ],
        tasked=("services.api",),
    )
    run_git_command(["git", "init", "-b", "main"], repo)
    run_git_command(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "commit"]
        + ["--allow-empty", "-m", "init"],
        repo,
    )

    with httpx.Client(base_url=server_env["host"], timeout=30) as client:
        started = client.post(
            "/api/director/org/messages",
            json={
                "text": "Build the checkout flow",
                "conversation": {
                    "agent": {
                        "llm": {"model": "gpt-4o-mini", "api_key": "test"},
                        "tools": [{"name": "terminal"}, {"name": "file_editor"}],
                    },
                    "workspace": {"working_dir": str(repo)},
                },
            },
        )
        assert started.status_code == 200, started.text
        [session] = client.get("/api/director/org/sessions").json()
        assert (started.json()["node_id"], started.json()["started_org"]) == (
            f"root@{session['id']}",
            True,
        )
        assert started.json()["conversation_id"] == session["conversation_id"]
        confined = client.get("/api/director/org").json()["boundary_enforcement"]

        def api_ic_events() -> list[dict]:
            api = _work_places(client, session["id"]).get("services.api")
            # A team has no agent until its IC above gives it its task.
            if api is None or not api["agents"]:
                return []
            api_ic = api["agents"][0]["conversation_id"]
            page = client.get(
                f"/api/conversations/{api_ic}/events/search", params={"limit": 100}
            )
            return page.json()["items"] if page.status_code == 200 else []

        def finished(events: list[dict]) -> bool:
            return any(
                event.get("kind") == "ObservationEvent"
                and event.get("tool_name") == "submit_review"
                for event in events
            )

        deadline = time.time() + 60
        events = api_ic_events()
        while time.time() < deadline and not finished(events):
            time.sleep(0.3)
            events = api_ic_events()
        assert finished(events), [event.get("kind") for event in events]

        root = _work_places(client, session["id"])["root"]
        assert (root["title"], root["headline"]) == (
            "Checkout",
            "Builds the checkout flow.",
        )
        tree = client.get("/api/director/org/nodes/root").json()["node"]
        assert (tree["title"], tree["headline"]) == (root["title"], root["headline"])
        root_ic = root["agents"][0]["conversation_id"]
        assert client.get(f"/api/conversations/{root_ic}").json()["title"] == (
            "Checkout (IC)"
        )

        # The file editor is refused outside the team's directory and works in it.
        rejections = [
            event["rejection_reason"]
            for event in events
            if event.get("kind") == "UserRejectObservation"
        ]
        assert len(rejections) == 1
        assert "outside your team's boundary (services/api)" in rejections[0]
        work = checkout()
        assert not (work / "by-editor.txt").exists()
        assert (work / "services" / "api" / "by-editor.txt").read_text() == "x"

        if confined != "sandbox":
            pytest.skip(
                "this machine cannot confine a shell; only file tools are checked"
            )
        # The shell starts in the team's directory and can write nowhere else.
        assert (work / "services" / "api" / "by-shell.txt").exists()
        assert not (work / "by-shell.txt").exists()
        assert not (work / "apps" / "by-shell.txt").exists()

        # What is said from a team's directory opens a session at that team,
        # which the org has once the work of its first session is accepted.
        while time.time() < deadline and (
            client.get("/api/director/org/nodes/services.api").status_code != 200
        ):
            time.sleep(0.3)
        routed = client.post(
            "/api/director/org/messages",
            json={"text": "Add refunds", "path": "services/api/handlers"},
        ).json()
        [_, opened] = client.get("/api/director/org/sessions").json()
        assert (opened["team_id"], opened["request"]) == (
            "services.api",
            "Add refunds\n\n(The human sent this from services/api/handlers.)",
        )
        assert (routed["node_id"], routed["started_org"]) == (
            f"services.api@{opened['id']}",
            False,
        )
