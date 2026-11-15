"""Runs a director tool for an agent that calls it from outside the server.

An agent that runs its own tools reaches the director's through an MCP
server of its own (``director/mcp_shim.py``), which calls these. The same
functions change the org as for an agent whose tools run in-process.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from openhands.agent_server.director.git import GitError
from openhands.agent_server.director.pod_actions import caller_of
from openhands.agent_server.director.service import (
    HAND_OVER_TOOLS,
    TOOLS_OF,
    DirectorService,
)
from openhands.agent_server.director.tools.definition import DIRECTOR_TOOLS
from openhands.sdk.errors import DomainError, NotFound
from openhands.sdk.tool.tool import Action


class RemoteTool(BaseModel):
    """A tool of the director as an MCP server lists it."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(serialization_alias="inputSchema")


class ToolAnswer(BaseModel):
    """What the director answered to a tool call."""

    text: str
    is_error: bool = Field(
        description="The director refused; the text says why, for the agent."
    )


def list_tools(service: DirectorService, conversation_id: UUID) -> list[RemoteTool]:
    """The director's tools the agent of ``conversation_id`` is offered.

    Raises:
        NotFound: The conversation is not an agent of the org.
    """
    caller = caller_of(service, conversation_id)
    offered = HAND_OVER_TOOLS if caller.hand_over else TOOLS_OF[caller.role]
    return [
        RemoteTool(
            name=name,
            description=tool.summary,
            input_schema=tool.action().to_mcp_schema(),
        )
        for name, tool in DIRECTOR_TOOLS.items()
        if name in offered
    ]


def wrong_arguments(name: str, action: type[Action], error: ValidationError) -> str:
    """What an agent is told when a call's arguments do not fit the tool:
    that nothing was done, what was wrong with each argument, and what the
    tool takes."""
    schema = action.to_mcp_schema()
    takes: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    said = {
        "missing": " is required and was not given",
        "extra_forbidden": " is not an argument of this tool, so it was not read",
    }
    lines = [f"`{name}` was not carried out: its arguments do not fit."]
    for wrong in error.errors():
        where = ".".join(str(part) for part in wrong["loc"]) or "the arguments"
        lines.append(f"- `{where}`{said.get(wrong['type'], ': ' + wrong['msg'])}.")
    listed = ", ".join(
        f"`{argument}`" + (" (required)" if argument in required else "")
        for argument in takes
    )
    lines.append(f"`{name}` takes: {listed or 'no arguments'}. Call it again.")
    return "\n".join(lines)


def call_tool(
    service: DirectorService, conversation_id: UUID, name: str, arguments: Any
) -> ToolAnswer:
    """Run the tool ``name`` for the agent of ``conversation_id``.

    Raises:
        NotFound: The director has no such tool.
    """
    tool = DIRECTOR_TOOLS.get(name)
    if tool is None:
        raise NotFound(f"The director has no tool named {name!r}")
    try:
        action = tool.action().model_validate(arguments)
    except ValidationError as error:
        return ToolAnswer(
            text=wrong_arguments(name, tool.action(), error), is_error=True
        )
    try:
        caller = caller_of(service, conversation_id)
        return ToolAnswer(text=tool.act()(service, caller, action), is_error=False)
    except (DomainError, GitError) as error:
        return ToolAnswer(text=str(error), is_error=True)
