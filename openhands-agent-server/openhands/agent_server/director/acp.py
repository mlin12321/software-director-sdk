"""Lets an external agent that runs its own tools be an agent of an org.

Such an agent (Claude Code, Codex, any ACP server) cannot be given the
director's tools in-process and its tool calls never pass the server. It
reaches the director through an MCP server it starts itself, which names the
agent by a token, and it is held to its team's directory by starting its
whole process in the team's sandbox.
"""

import hashlib
import hmac
import sys
from pathlib import Path

from pydantic import SecretStr

from openhands.agent_server.director import mcp_shim
from openhands.sdk.agent import ACPAgent
from openhands.sdk.mcp.config import MCPServer


# The name the director's tools appear under in the agent.
MCP_SERVER_NAME = "director"
TOOLS_PATH = "/api/director/tools"


def agent_token(key: bytes, conversation_id: str) -> str:
    """What an agent shows the director to act as ``conversation_id``."""
    signature = hmac.new(key, conversation_id.encode(), hashlib.sha256).hexdigest()
    return f"{conversation_id}.{signature}"


def conversation_of(key: bytes, token: str) -> str | None:
    """The conversation ``token`` was made for; None when it was not made
    with ``key``."""
    conversation_id, _, _ = token.partition(".")
    if not conversation_id:
        return None
    expected = agent_token(key, conversation_id)
    return conversation_id if hmac.compare_digest(expected, token) else None


def tools_server(server_url: str, token: str) -> MCPServer:
    """The MCP server an agent starts to reach the director's tools."""
    return MCPServer(
        command=sys.executable,
        args=[str(Path(mcp_shim.__file__).resolve())],
        env={
            mcp_shim.URL_VARIABLE: SecretStr(f"{server_url.rstrip('/')}{TOOLS_PATH}"),
            mcp_shim.TOKEN_VARIABLE: SecretStr(token),
        },
    )


def as_org_agent(
    agent: ACPAgent, *, tools: MCPServer, launcher: Path | None, launchers: Path
) -> ACPAgent:
    """``agent`` with the director's tools, started through ``launcher``.

    Args:
        launcher: A script that runs the command it is given inside the
            team's sandbox; None to start the agent as it is.
        launchers: Where the director keeps such scripts. An agent derived
            from another agent of the org still carries that agent's script
            in front of its command, and it is taken off here.
    """
    command = list(agent.acp_command)
    if command and Path(command[0]).parent == launchers:
        command = command[1:]
    if launcher is not None:
        command = [str(launcher), *command]
    return agent.model_copy(
        update={
            "acp_command": command,
            "mcp_config": {**agent.mcp_config, MCP_SERVER_NAME: tools},
        }
    )
