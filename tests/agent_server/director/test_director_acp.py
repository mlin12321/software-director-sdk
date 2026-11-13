"""Agents that run their own tools (Claude Code, Codex) as agents of an org."""

import json
import os
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

from openhands.agent_server.api import create_app
from openhands.agent_server.director import mcp_shim
from openhands.agent_server.director.acp import (
    MCP_SERVER_NAME,
    agent_token,
    as_org_agent,
    conversation_of,
    tools_server,
)
from openhands.agent_server.director.models import (
    CreateOrgRequest,
    DirectorSettingsUpdate,
    RouteMessageRequest,
    SplitChild,
)
from openhands.agent_server.director.prompts import POD_PROTOCOL
from openhands.agent_server.director.sandbox import (
    command_script,
    find_bwrap,
    write_shell,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    DecideSplitTool,
    ProposeBoundarySplitTool,
)
from openhands.agent_server.director.tools.definition import (
    DecideSplitAction,
    ProposeBoundarySplitAction,
)
from openhands.sdk.agent import ACPAgent
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import FakeLauncher, root_conversation
from tests.agent_server.director.test_director_router import server_config
from tests.agent_server.director.test_director_tools import ic, manager, node, use


APPROVE = DecideSplitAction(decision="approve")
CLAUDE = ["npx", "-y", "@agentclientprotocol/claude-agent-acp"]


async def started(repo: Path, launcher: FakeLauncher, **options) -> DirectorService:
    service = DirectorService(
        repo,
        save_delay=0,
        run_limit_retry=0,
        remembered=repo.parent / "state" / "director.json",
        server_url="http://127.0.0.1:8000",
        **options,
    )
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    await service.settled()
    return service


def split(*children: SplitChild) -> ProposeBoundarySplitAction:
    return ProposeBoundarySplitAction(children=list(children), contracts="")


async def carry_out(service: DirectorService, *children: SplitChild) -> None:
    await use(service, ic(service), ProposeBoundarySplitTool, split(*children))
    await use(service, manager(service), DecideSplitTool, APPROVE)


def test_a_token_names_its_agent_only_with_the_key_it_was_made_with():
    conversation_id = str(uuid4())
    token = agent_token(b"key", conversation_id)

    assert conversation_of(b"key", token) == conversation_id
    assert conversation_of(b"other", token) is None
    assert conversation_of(b"key", f"{uuid4()}.{token.partition('.')[2]}") is None
    assert conversation_of(b"key", "") is None
    assert conversation_of(b"key", conversation_id) is None


def test_an_acp_agent_gets_the_tools_and_is_started_by_its_teams_launcher(
    tmp_path: Path,
):
    launchers = tmp_path / "director-shells"
    tools = tools_server("http://127.0.0.1:8000/", "token-1")
    agent = ACPAgent(acp_command=CLAUDE)

    first = as_org_agent(
        agent, tools=tools, launcher=launchers / "a.acp.sh", launchers=launchers
    )
    # An agent derived from it carries its launcher and its token.
    second = as_org_agent(
        first,
        tools=tools_server("http://127.0.0.1:8000", "token-2"),
        launcher=launchers / "b.acp.sh",
        launchers=launchers,
    )
    unconfined = as_org_agent(first, tools=tools, launcher=None, launchers=launchers)

    assert first.acp_command == [str(launchers / "a.acp.sh"), *CLAUDE]
    assert second.acp_command == [str(launchers / "b.acp.sh"), *CLAUDE]
    assert unconfined.acp_command == CLAUDE
    server = second.mcp_config[MCP_SERVER_NAME]
    assert server.command == sys.executable
    assert server.args == [str(Path(mcp_shim.__file__).resolve())]
    assert server.env is not None
    assert {name: value.get_secret_value() for name, value in server.env.items()} == {
        "DIRECTOR_TOOLS_URL": "http://127.0.0.1:8000/api/director/tools",
        "DIRECTOR_AGENT_TOKEN": "token-2",
    }


def test_a_launcher_without_a_sandbox_runs_the_command_with_the_teams_git_first(
    tmp_path: Path,
):
    programs = tmp_path / "bin"
    programs.mkdir()
    (programs / "git").write_text("#!/bin/sh\necho own git\n")
    (programs / "git").chmod(0o755)
    script = tmp_path / "launch.sh"
    write_shell(
        script,
        command_script(
            None,
            checkouts=[tmp_path],
            private=tmp_path,
            directory="svc",
            first_on_path=programs,
        ),
    )

    ran = subprocess.run(
        [str(script), "sh", "-c", "git; echo $1", "sh", "passed on"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert ran.stdout.splitlines() == ["own git", "passed on"]


def test_a_launcher_holds_the_whole_process_to_the_teams_directory(repo: Path):
    bwrap = find_bwrap()
    if bwrap is None:
        pytest.skip("bubblewrap cannot make a sandbox on this machine")
    (repo / "svc" / "web").mkdir(parents=True)
    script = repo / ".git" / "director-shells" / "team.acp.sh"
    write_shell(
        script,
        command_script(
            bwrap,
            checkouts=[repo],
            private=script.parent,
            directory="svc/api",
        ),
    )

    def run(command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(script), "sh", "-c", command],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=60,
        )

    # The agent is told its directory by its session, so the launcher stays
    # where it was started.
    assert run("pwd").stdout.strip() == str(repo)
    assert run("echo x > svc/api/pay.py").returncode == 0
    refused = run("echo x > svc/web/page.tsx")
    assert refused.returncode != 0 and "Read-only file system" in refused.stderr
    assert run("echo x > README.md").returncode != 0


@pytest.fixture
def tools_route() -> Iterator[tuple[str, list[tuple[str, str, object]]]]:
    """A stand-in for the director's tool routes, and what it was asked."""
    asked: list[tuple[str, str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _answer(self, body: object) -> None:
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            asked.append((self.path, self.headers[mcp_shim.TOKEN_HEADER], None))
            self._answer(
                [{"name": "set_status", "description": "Report", "inputSchema": {}}]
            )

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            asked.append((self.path, self.headers[mcp_shim.TOKEN_HEADER], body))
            if self.path.endswith("/missing"):
                self.send_error(404, "No such tool")
                return
            self._answer({"text": "Recorded.", "is_error": False})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/tools", asked
    server.shutdown()
    server.server_close()


def test_the_mcp_server_lists_the_directors_tools_and_passes_calls_on(tools_route):
    url, asked = tools_route
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "set_status", "arguments": {"status": "done"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        },
        {"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
    ]

    ran = subprocess.run(
        [sys.executable, str(Path(mcp_shim.__file__))],
        input="".join(json.dumps(request) + "\n" for request in requests),
        env={
            **os.environ,
            mcp_shim.URL_VARIABLE: url,
            mcp_shim.TOKEN_VARIABLE: "token-1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    replies = {reply["id"]: reply for reply in map(json.loads, ran.stdout.splitlines())}
    assert sorted(replies) == [1, 2, 3, 4, 5]
    assert replies[1]["result"]["capabilities"] == {"tools": {}}
    assert [tool["name"] for tool in replies[2]["result"]["tools"]] == ["set_status"]
    assert replies[3]["result"] == {
        "content": [{"type": "text", "text": "Recorded."}],
        "isError": False,
    }
    # A call the director refuses is something the agent reads.
    assert replies[4]["result"]["isError"] is True
    assert "404" in replies[4]["result"]["content"][0]["text"]
    assert replies[5]["error"]["code"] == -32601
    assert asked == [
        ("/tools", "token-1", None),
        ("/tools/set_status", "token-1", {"status": "done"}),
        ("/tools/missing", "token-1", {}),
    ]


@pytest.fixture
async def running(
    repo: Path,
) -> AsyncIterator[tuple[httpx.AsyncClient, DirectorService]]:
    app = create_app(server_config(repo, session_api_keys=["secret"]))
    service: DirectorService = app.state.director_service
    await service.start(FakeLauncher())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, service
    await service.aclose()


async def test_an_agent_calls_the_directors_tools_with_its_token_alone(
    running, repo: Path
):
    client, service = running
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    await service.settled()
    token = agent_token(service._key(), str(ic(service)))
    headers = {mcp_shim.TOKEN_HEADER: token}

    listed = await client.get("/api/director/tools", headers=headers)
    spec = await client.post(
        "/api/director/tools/write_spec", headers=headers, json={"content": "# Spec"}
    )
    # Only a Manager decides a split, and this is the IC.
    refused = await client.post(
        "/api/director/tools/decide_split",
        headers=headers,
        json={"decision": "approve"},
    )
    malformed = await client.post(
        "/api/director/tools/write_spec", headers=headers, json={}
    )

    assert listed.status_code == 200
    tools = {tool["name"]: tool for tool in listed.json()}
    assert "propose_boundary_split" in tools and "review_team" in tools
    assert "children" in tools["propose_boundary_split"]["inputSchema"]["properties"]
    assert spec.json()["is_error"] is False
    assert service.blackboard.applies_to("", None)[0][0].spec == "# Spec"
    assert refused.status_code == 200 and refused.json()["is_error"] is True
    assert "Manager" in refused.json()["text"]
    assert malformed.json()["is_error"] is True


async def test_the_tool_routes_refuse_whoever_has_no_agents_token(running, repo: Path):
    client, service = running
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    forged = f"{ic(service)}.{'0' * 64}"
    stranger = agent_token(service._key(), str(uuid4()))

    responses = [
        await client.get("/api/director/tools"),
        await client.get(
            "/api/director/tools", headers={mcp_shim.TOKEN_HEADER: forged}
        ),
        # The server's own key is not an agent's.
        await client.get(
            "/api/director/tools", headers={"X-Session-API-Key": "secret"}
        ),
    ]
    not_in_the_org = await client.post(
        "/api/director/tools/set_status",
        headers={mcp_shim.TOKEN_HEADER: stranger},
        json={"status": "done", "summary": "x"},
    )
    unknown_tool = await client.post(
        "/api/director/tools/format_disk",
        headers={mcp_shim.TOKEN_HEADER: agent_token(service._key(), str(ic(service)))},
        json={},
    )

    assert [response.status_code for response in responses] == [401, 401, 401]
    assert not_in_the_org.json()["is_error"] is True
    assert unknown_tool.status_code == 404
    # The other routes of the director still need the server's key.
    assert (await client.get("/api/director/org")).status_code == 401


async def test_an_acp_agent_may_lead_an_org(repo: Path):
    launcher = FakeLauncher(root_kind="acp")
    service = await started(repo, launcher)

    root = node(service, "root")
    assert [(agent.role, agent.kind, agent.acp_server) for agent in root.agents] == [
        ("ic", "acp", "claude-code"),
        ("manager", "acp", "claude-code"),
    ]
    # No sandbox here, and its tool calls never pass the server.
    assert root.boundary_enforcement == "merge"
    await service.aclose()


async def test_the_agent_of_an_org_reaches_the_tools_as_itself(repo: Path):
    launcher = FakeLauncher(root_kind="acp")
    service = await started(repo, launcher)
    assert launcher.root_adapt is not None

    adapted = launcher.root_adapt(ACPAgent(acp_command=CLAUDE))

    assert isinstance(adapted, ACPAgent)
    server = adapted.mcp_config[MCP_SERVER_NAME]
    assert server.env is not None
    token = server.env["DIRECTOR_AGENT_TOKEN"].get_secret_value()
    assert str(service.agent_of(token)) == ic(service)
    assert (
        server.env["DIRECTOR_TOOLS_URL"].get_secret_value()
        == "http://127.0.0.1:8000/api/director/tools"
    )
    # The token outlives the server, as the agent's configuration does.
    again = DirectorService(repo, remembered=repo.parent / "state" / "director.json")
    assert str(again.agent_of(token)) == ic(service)
    await service.aclose()


async def test_an_acp_team_without_a_sandbox_gets_its_own_worktree(repo: Path):
    launcher = FakeLauncher(root_kind="acp")
    service = await started(repo, launcher)

    await carry_out(service, SplitChild(title="API", path="svc/api", task="Serve"))

    team = node(service, "svc.api")
    starts = {start.title: start for start in launcher.derived}
    assert starts["API (IC)"].isolation == "worktree"
    assert team.isolation == "worktree" and team.boundary_enforcement == "merge"
    assert starts["API (Manager)"].isolation == "shared"
    await service.aclose()


async def test_an_acp_team_in_a_sandbox_works_where_the_split_put_it(repo: Path):
    launcher = FakeLauncher(root_kind="acp")
    service = await started(repo, launcher, find_sandbox=lambda: "/usr/bin/bwrap")

    await carry_out(service, SplitChild(title="API", path="svc/api", task="Serve"))

    team = node(service, "svc.api")
    assert team.isolation == "shared" and team.boundary_enforcement == "sandbox"
    conversation_id = ic(service, "svc.api")
    launch = repo / ".git" / "director-shells" / f"{conversation_id}.acp.sh"
    assert "/usr/bin/bwrap" in launch.read_text()
    start = next(start for start in launcher.derived if start.title == "API (IC)")
    assert start.adapt_agent is not None
    adapted = start.adapt_agent(ACPAgent(acp_command=CLAUDE))
    assert isinstance(adapted, ACPAgent)
    assert adapted.acp_command == [str(launch), *CLAUDE]
    await service.aclose()


async def test_a_sub_team_runs_on_another_agent_profile_when_the_org_allows_it(
    repo: Path,
):
    launcher = FakeLauncher(profiles={"codex": "acp", "reviewer": "openhands"})
    service = await started(repo, launcher)
    codex = SplitChild(title="API", path="svc/api", task="Serve", agent_profile="codex")

    refused = await use(service, ic(service), ProposeBoundarySplitTool, split(codex))
    service.update_settings(DirectorSettingsUpdate(allowed_profiles=["codex"]))
    await carry_out(service, codex, SplitChild(title="Web", path="web", task="Pages"))

    assert refused.is_error and "leave `agent_profile` out" in refused.text
    starts = {start.title: start for start in launcher.derived}
    assert starts["API (IC)"].profile == "codex"
    assert starts["API (IC)"].suffixes == [POD_PROTOCOL]
    assert starts["API (IC)"].parent_id == UUID(str(ic(service)))
    # Its Manager is derived from it, so it runs on the same agent.
    assert starts["API (Manager)"].profile is None
    assert starts["API (Manager)"].parent_id == UUID(str(ic(service, "svc.api")))
    assert starts["Web (IC)"].profile is None
    team = node(service, "svc.api")
    assert team.agent_profile == "codex"
    assert [agent.kind for agent in team.agents] == ["acp", "acp"]
    assert [agent.kind for agent in node(service, "web").agents] == [
        "openhands",
        "openhands",
    ]
    await service.aclose()


async def test_the_profiles_a_sub_team_may_run_on_are_kept_and_told_to_the_ic(
    repo: Path,
):
    launcher = FakeLauncher(profiles={"codex": "acp"})
    service = await started(repo, launcher)

    service.update_settings(DirectorSettingsUpdate(allowed_profiles=["codex"]))
    other = await use(
        service,
        ic(service),
        ProposeBoundarySplitTool,
        split(SplitChild(title="A", path="a", task="x", agent_profile="claude")),
    )

    assert other.is_error and "The org allows: codex." in other.text
    kept = json.loads((repo.parent / "state" / "director.json").read_text())
    assert kept["settings"]["allowed_profiles"] == ["codex"]
    # Limits that are not named stay.
    assert service.update_settings(DirectorSettingsUpdate(max_depth=3)).model_dump(
        include={"allowed_profiles", "max_depth", "max_agents"}
    ) == {"allowed_profiles": ["codex"], "max_depth": 3, "max_agents": 64}
    await service.aclose()


def said_to(launcher: FakeLauncher, conversation_id: str | None) -> list[str]:
    """The director's notices and the human's words that reached an agent."""
    return [
        text
        for text in launcher.messages_to(conversation_id)
        if "merged" in text or "human" in text.lower()
    ]


async def test_a_notice_waits_for_the_turn_of_an_agent_that_runs_its_own_tools(
    repo: Path,
):
    launcher = FakeLauncher(root_kind="acp")
    service = await started(repo, launcher)
    agent = ic(service)
    assert agent is not None
    launcher.report(agent, ConversationExecutionStatus.RUNNING)

    service.notify(agent, "The work of 'API' is merged into your checkout.")
    service.notify(agent, "The work of 'Web' is merged into your checkout.")
    await service.settled()
    waiting = said_to(launcher, agent)

    launcher.report(agent, ConversationExecutionStatus.FINISHED)
    await service.settled()

    assert waiting == []
    assert [text.split("'")[1] for text in said_to(launcher, agent)] == ["API", "Web"]
    await service.aclose()


async def test_what_the_human_says_reaches_such_an_agent_in_the_middle_of_a_turn(
    repo: Path,
):
    launcher = FakeLauncher(root_kind="acp")
    service = await started(repo, launcher)
    agent = ic(service)
    assert agent is not None
    service.store.raise_escalation(
        "root", "jwt", kind="design_decision", title="JWT or opaque tokens?"
    )
    launcher.report(agent, ConversationExecutionStatus.RUNNING)

    await service.route_message(RouteMessageRequest(text="The human: stop, use Go."))
    service.answer_escalation("root", "jwt", "Opaque.")
    await service.settled()

    said = said_to(launcher, agent)
    assert len(said) == 2
    assert "stop, use Go" in said[0] and "Opaque." in said[1]
    await service.aclose()


async def test_a_notice_does_not_wait_for_an_agent_whose_tools_the_server_runs(
    repo: Path,
):
    launcher = FakeLauncher()
    service = await started(repo, launcher)
    agent = ic(service)
    assert agent is not None
    launcher.report(agent, ConversationExecutionStatus.RUNNING)

    service.notify(agent, "The work of 'API' is merged into your checkout.")
    await service.settled()

    assert len(said_to(launcher, agent)) == 1
    await service.aclose()
