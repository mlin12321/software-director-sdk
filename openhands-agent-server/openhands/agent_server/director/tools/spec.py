"""What a tool of the director is, whatever an agent reaches it through."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolSpec[ArgumentsT: BaseModel]:
    """One tool: its name, what it is for, what it takes and what it does.

    Attributes:
        name: What agents call it by.
        summary: What it is for, as agents are told.
        arguments: What it takes.
        act: Does it for the agent that called: given the director, the
            caller and the arguments, returns what the agent is answered.
        read_only: Whether it changes nothing.
    """

    name: str
    summary: str
    arguments: type[ArgumentsT]
    act: Callable[[Any, Any, ArgumentsT], str]
    read_only: bool = False
