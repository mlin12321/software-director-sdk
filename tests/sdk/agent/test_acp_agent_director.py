"""What the director changed in the ACP agent: a side question asked in a
session of its own, a launcher in front of ``npx``, a login file for
Cursor, and a close that ends everything the command started."""

import asyncio
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.exceptions import RequestError as ACPRequestError
from acp.schema import PromptResponse, UsageUpdate

import openhands.sdk.agent.acp_agent as acp_agent_module
from openhands.sdk.agent.acp_agent import (
    ACPAgent,
    _npx_packages,
    _OpenHandsACPBridge,
    _select_auth_method,
)
from tests.sdk.agent.test_acp_agent import _agent_conn, _make_agent


def test_npx_packages_looks_past_a_launcher():
    assert _npx_packages(
        ["/sandbox/launcher.sh", "npx", "-y", "@agentclientprotocol/x@1.2.3"]
    ) == ["@agentclientprotocol/x@1.2.3"]
    assert _npx_packages(["/sandbox/launcher.sh", "codex-acp"]) == []


def test_ask_agent_counts_what_the_side_session_reports_it_cost():
    """A UsageUpdate of the fork session reaches the metrics."""
    agent = _make_agent()
    client = _OpenHandsACPBridge()
    agent._client = client
    agent._conn = MagicMock()
    agent._session_id = "main-session"
    agent._working_dir = "/workspace"
    response = MagicMock()
    response.usage = None

    async def _fake_prompt(*args, **kwargs):
        update = MagicMock(spec=UsageUpdate)
        update.size, update.used = 200000, 1000
        update.cost = MagicMock(amount=0.25)
        await client.session_update("fork-session", update)
        return response

    def _fake_run_async(coro_fn, **_kwargs):
        conn = _agent_conn(agent)
        conn.fork_session = AsyncMock(return_value=MagicMock(session_id="fork-session"))
        conn.prompt = _fake_prompt
        return asyncio.run(coro_fn())

    agent._executor = MagicMock(run_async=_fake_run_async)

    agent.ask_agent("How far along are you?")

    assert agent.llm.metrics.accumulated_cost == pytest.approx(0.25)


def test_ask_agent_gives_up_a_side_session_that_never_answers():
    """A hung side question is cancelled alone and frees the lock."""
    agent = _make_agent()
    agent._client = _OpenHandsACPBridge()
    agent._conn = MagicMock()
    agent._session_id = "main-session"
    agent._working_dir = "/workspace"

    async def _never(*args, **kwargs):
        await asyncio.sleep(3600)

    def _fake_run_async(coro_fn, **_kwargs):
        conn = _agent_conn(agent)
        conn.fork_session = AsyncMock(return_value=MagicMock(session_id="fork-session"))
        conn.prompt = _never
        conn.cancel = AsyncMock()
        return asyncio.run(coro_fn())

    agent._executor = MagicMock(run_async=_fake_run_async)

    with patch("openhands.sdk.agent.acp_agent._SIDE_QUESTION_TIMEOUT", 0.05):
        with pytest.raises(TimeoutError):
            agent.ask_agent("Still there?")

    _agent_conn(agent).cancel.assert_awaited_once_with("fork-session")
    assert agent._client._fork_session_id is None
    assert agent._client._fork_lock.acquire(blocking=False)


def test_ask_agent_says_plainly_that_a_server_cannot_fork():
    """A server without session/fork is not a failure to retry."""
    agent = _make_agent()
    agent._client = _OpenHandsACPBridge()
    agent._conn = MagicMock()
    agent._session_id = "main-session"
    agent._working_dir = "/workspace"

    def _fake_run_async(coro_fn, **_kwargs):
        conn = _agent_conn(agent)
        conn.fork_session = AsyncMock(
            side_effect=ACPRequestError(-32601, "Method not found")
        )
        conn.prompt = AsyncMock()
        return asyncio.run(coro_fn())

    agent._executor = MagicMock(run_async=_fake_run_async)

    with pytest.raises(NotImplementedError, match="cannot fork"):
        agent.ask_agent("Why did you stop?")

    _agent_conn(agent).prompt.assert_not_awaited()
    assert agent._client._fork_lock.acquire(blocking=False)


def test_ask_agent_asks_the_fork_on_the_conversations_model():
    """A fork is put on the agent's model before it is prompted."""
    agent = _make_agent(acp_model="haiku")
    client = _OpenHandsACPBridge()
    agent._client = client
    agent._conn = MagicMock()
    agent._session_id = "main-session"
    agent._working_dir = "/workspace"
    applied: list[tuple[str, str]] = []

    async def _apply(conn, session_id, model, **kwargs):
        applied.append((session_id, model))
        return model

    async def _fake_prompt(*args, **kwargs):
        assert applied == [("fork-session", "haiku")]
        return PromptResponse(stop_reason="end_turn")

    def _fake_run_async(coro_fn, **_kwargs):
        conn = _agent_conn(agent)
        conn.fork_session = AsyncMock(return_value=MagicMock(session_id="fork-session"))
        conn.prompt = _fake_prompt
        return asyncio.run(coro_fn())

    agent._executor = MagicMock(run_async=_fake_run_async)

    with (
        patch.object(acp_agent_module, "_apply_acp_model", _apply),
        patch.object(acp_agent_module, "_USAGE_UPDATE_TIMEOUT", 0.01),
    ):
        agent.ask_agent("What changed?")

    assert applied == [("fork-session", "haiku")]


def test_ask_aside_asks_the_new_session_on_the_conversations_model():
    """A session beside the conversation is put on the agent's model too."""
    agent = _make_agent(acp_model="haiku")
    agent._client = _OpenHandsACPBridge()
    agent._conn = MagicMock()
    agent._working_dir = "/workspace"
    applied: list[tuple[str, str]] = []

    async def _apply(conn, session_id, model, **kwargs):
        applied.append((session_id, model))
        return model

    async def _fake_prompt(*args, **kwargs):
        assert applied == [("side-session", "haiku")]
        return PromptResponse(stop_reason="end_turn")

    def _fake_run_async(coro_fn, **_kwargs):
        conn = _agent_conn(agent)
        conn.new_session = AsyncMock(return_value=MagicMock(session_id="side-session"))
        conn.close_session = AsyncMock()
        conn.prompt = _fake_prompt
        return asyncio.run(coro_fn())

    agent._executor = MagicMock(run_async=_fake_run_async)

    with (
        patch.object(acp_agent_module, "_apply_acp_model", _apply),
        patch.object(acp_agent_module, "_USAGE_UPDATE_TIMEOUT", 0.01),
    ):
        agent.ask_aside("What is this team for?")

    assert applied == [("side-session", "haiku")]


def _asked_aside(
    agent: ACPAgent,
    question: str,
    *,
    text: str = "An answer.",
    closes: Exception | None = None,
) -> Any:
    """Ask ``agent`` aside against a fake connection; returns the connection."""
    client = _OpenHandsACPBridge()
    agent._client = client
    agent._conn = MagicMock()
    agent._working_dir = "/workspace"
    conn = _agent_conn(agent)
    conn.new_session = AsyncMock(return_value=MagicMock(session_id="side-1"))
    conn.fork_session = AsyncMock()
    conn.close_session = AsyncMock(side_effect=closes)
    response = MagicMock()
    response.usage = None

    async def prompt(*, session_id: str, prompt: list[Any]) -> Any:
        assert client._fork_session_id == session_id
        client._fork_accumulated_text.append(text)
        return response

    conn.prompt = AsyncMock(side_effect=prompt)

    def run_async(coro_fn, **_kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_fn())
        finally:
            loop.close()

    executor = MagicMock()
    executor.run_async = run_async
    agent._executor = executor
    conn.answer = agent.ask_aside(question)
    return conn


def test_a_side_question_is_asked_in_a_new_session_without_the_conversation():
    agent = _make_agent()
    # No turn was taken: there is no session of the conversation's to fork.
    agent._session_id = None

    conn = _asked_aside(agent, "Name this team.", text="Title: Checkout")

    assert conn.answer == "Title: Checkout"
    conn.fork_session.assert_not_awaited()
    opened = conn.new_session.await_args.kwargs
    assert (opened["cwd"], opened["mcp_servers"]) == ("/workspace", [])
    assert conn.prompt.await_args.kwargs["session_id"] == "side-1"
    conn.close_session.assert_awaited_once_with(session_id="side-1")
    assert agent._client._fork_session_id is None
    assert agent._client._fork_accumulated_text == []


def test_a_side_question_answers_where_a_session_cannot_be_closed():
    conn = _asked_aside(
        _make_agent(), "Anything.", closes=RuntimeError("Method not found")
    )

    assert conn.answer == "An answer."
    conn.close_session.assert_awaited_once()


def test_a_side_question_needs_a_connection():
    agent = _make_agent()
    with pytest.raises(RuntimeError, match="no ACP connection"):
        agent.ask_aside("Anything.")


def test_cursor_login_when_login_file_present(tmp_path):
    """The Cursor CLI's login is selected when `agent login` left its file,
    under ~/.config or under XDG_CONFIG_HOME."""
    method = MagicMock()
    method.id = "cursor_login"
    methods = [method]
    with patch("openhands.sdk.agent.acp_agent.Path.home", return_value=tmp_path):
        assert _select_auth_method(methods, {}) is None

        login = tmp_path / ".config" / "cursor"
        login.mkdir(parents=True)
        (login / "auth.json").write_text("{}", encoding="utf-8")
        assert _select_auth_method(methods, {}) == "cursor_login"

        elsewhere = tmp_path / "xdg"
        assert _select_auth_method(methods, {"XDG_CONFIG_HOME": str(elsewhere)}) is None
        (elsewhere / "cursor").mkdir(parents=True)
        (elsewhere / "cursor" / "auth.json").write_text("{}", encoding="utf-8")
        assert (
            _select_auth_method(methods, {"XDG_CONFIG_HOME": str(elsewhere)})
            == "cursor_login"
        )


def _process_is_gone(pid: int) -> bool:
    """Whether ``pid`` no longer runs; a zombie nobody reaped counts as gone."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return True
    return state == "Z"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
@pytest.mark.parametrize("ignores_sigterm", [False, True])
def test_close_ends_everything_the_acp_command_started(
    monkeypatch: pytest.MonkeyPatch, ignores_sigterm: bool
):
    import subprocess

    monkeypatch.setattr(acp_agent_module, "_ACP_GROUP_EXIT_GRACE_SECONDS", 0.3)
    trap = 'trap "" TERM; ' if ignores_sigterm else ""
    # The command ends on SIGTERM; what it started is a process of its own.
    command = subprocess.Popen(
        ["sh", "-c", f"sh -c '{trap}sleep 300' & echo $!; wait"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert command.stdout is not None
    started = int(command.stdout.readline())
    executor = MagicMock()
    executor.run_async.side_effect = lambda *_args, **_kwargs: command.wait(timeout=5)
    agent = _make_agent()
    agent._process = cast(Any, command)
    agent._process_group = command.pid
    agent._executor = executor
    agent._conn = None

    agent.close()

    assert command.poll() is not None
    deadline = time.monotonic() + 3
    while not _process_is_gone(started) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _process_is_gone(started)


def test_a_process_that_leads_no_group_has_none_to_signal():
    import subprocess

    # Not started in a session of its own, so no group has its id.
    bystander = subprocess.Popen(["sleep", "300"])
    try:
        assert acp_agent_module._signal_process_group(bystander.pid, 0) is False
        assert acp_agent_module._signal_process_group(None, 0) is False
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()
