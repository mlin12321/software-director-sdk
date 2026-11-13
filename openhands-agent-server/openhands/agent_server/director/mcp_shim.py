"""The director's tools as an MCP server, for an agent that runs its own.

An external agent (Claude Code, Codex) is started with this file as a stdio
MCP server. It keeps nothing itself: it lists the tools the director has for
the agent and passes every call on to the server, which knows the agent by
the token in ``DIRECTOR_AGENT_TOKEN``.

It is run by its path and uses only the standard library, so that it starts
at once and in any environment the agent gives it.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


TOKEN_HEADER = "X-Director-Agent-Token"
URL_VARIABLE = "DIRECTOR_TOOLS_URL"
TOKEN_VARIABLE = "DIRECTOR_AGENT_TOKEN"
PROTOCOL_VERSION = "2025-06-18"
_TIMEOUT_SECONDS = 300
_METHOD_NOT_FOUND = -32601
_INTERNAL_ERROR = -32603


def ask_director(url: str, token: str, body: dict[str, Any] | None) -> Any:
    """What the director answers at ``url``; a POST when given a body.

    Raises:
        OSError: The server could not be reached or refused the request.
    """
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={TOKEN_HEADER: token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise OSError(f"The director answered {error.code}: {detail}") from error


def answer(message: dict[str, Any], url: str, token: str) -> dict[str, Any] | None:
    """The reply to one JSON-RPC message; None for a notification."""
    method, request_id = message.get("method"), message.get("id")
    if request_id is None:
        return None
    params = message.get("params") or {}
    try:
        if method == "initialize":
            result: Any = {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "director", "version": "1"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": ask_director(url, token, None)}
        elif method == "tools/call":
            said = ask_director(
                f"{url}/{params.get('name', '')}", token, params.get("arguments") or {}
            )
            result = {
                "content": [{"type": "text", "text": said["text"]}],
                "isError": bool(said["is_error"]),
            }
        else:
            return _error(request_id, _METHOD_NOT_FOUND, f"Unknown method {method}")
    except (OSError, ValueError, KeyError) as error:
        if method == "tools/call":
            # The agent reads this and can try again; a protocol error would
            # only tell it that the tool is broken.
            result = {
                "content": [{"type": "text", "text": str(error)}],
                "isError": True,
            }
        else:
            return _error(request_id, _INTERNAL_ERROR, str(error))
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, text: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": text},
    }


def main() -> None:
    url, token = os.environ[URL_VARIABLE], os.environ[TOKEN_VARIABLE]
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        reply = answer(message, url, token) if isinstance(message, dict) else None
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
