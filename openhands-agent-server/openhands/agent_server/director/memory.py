"""Keeps the rules of a boundary from growing into repeats of each other.

A model rewrites a boundary's rules into fewer when enough have been added
since it last did. The blackboard records what the rules were then, so a
boundary whose rules did not change is never read to a model again, on this
machine or on one that pulled the blackboard.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter, ValidationError

from openhands.agent_server.director.git_blackboard import (
    CONSOLIDATED_FILE,
    MEMORY_FILE,
    BoardChange,
    BoardReader,
    GitBlackboard,
)
from openhands.agent_server.director.models import BoundaryMemory


USAGE_ID = "director-memory"
# Fewer rules than this are left alone.
MIN_RULES = 12
# How much a boundary's rules must have grown since they were consolidated.
MIN_GROWTH = 1.2

_INSTRUCTIONS = """\
You maintain the standing rules of one directory of a software repository. \
Agents and people added them one at a time, so some repeat or overlap. \
Rewrite the list: merge rules that say the same thing, and where one rule \
is a special case of another keep the broader one with the detail that \
matters. Do not drop a rule that says something no other does, do not add \
any, and keep each rule's own wording where you can. If two rules \
contradict each other, keep both.

Answer with one JSON array of strings and nothing else."""

_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_RULES = TypeAdapter(list[str])


class _Mark(BaseModel):
    digest: str
    rules: int


@dataclass(frozen=True)
class Due:
    """A boundary whose rules are to be consolidated, as they were read."""

    directory: str
    raw: str
    rules: tuple[str, ...]


def _file(directory: str, name: str) -> str:
    return f"{directory}/{name}" if directory else name


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _mark(raw: str | None) -> _Mark | None:
    try:
        return _Mark.model_validate_json(raw) if raw else None
    except ValidationError:
        return None


def due(board: GitBlackboard) -> list[Due]:
    """The boundaries with enough new rules to be worth a model call."""
    found = []
    for directory in board.directories_with(MEMORY_FILE):
        raw = board.read(_file(directory, MEMORY_FILE))
        if raw is None:
            continue
        try:
            rules = BoundaryMemory.model_validate_json(raw).rules
        except ValidationError:
            continue
        mark = _mark(board.read(_file(directory, CONSOLIDATED_FILE)))
        if len(rules) <= MIN_RULES or (
            mark is not None
            and (
                mark.digest == _digest(raw)
                or len(rules) < math.ceil(mark.rules * MIN_GROWTH)
            )
        ):
            continue
        found.append(Due(directory, raw, tuple(rules)))
    return found


def prompt(item: Due) -> tuple[str, str]:
    """The system and the user message that ask for the rules rewritten."""
    return _INSTRUCTIONS, json.dumps(list(item.rules), indent=2)


def parse_rules(answer: str, item: Due) -> list[str] | None:
    """The rewritten rules; None unless they are a usable, shorter-or-equal
    list."""
    match = _ARRAY.search(answer)
    if match is None:
        return None
    try:
        rules = [rule.strip() for rule in _RULES.validate_json(match.group(0))]
    except ValidationError:
        return None
    rules = list(dict.fromkeys(rule for rule in rules if rule))
    return rules if 0 < len(rules) <= len(item.rules) else None


def apply(board: GitBlackboard, item: Due, rules: list[str]) -> bool:
    """Replace a boundary's rules, unless they changed since they were read.

    Returns:
        Whether the rules were as read, and so are consolidated now.
    """
    path = _file(item.directory, MEMORY_FILE)
    content = json.dumps(BoundaryMemory(rules=rules).model_dump(), indent=2) + "\n"
    mark = _Mark(digest=_digest(content), rules=len(rules))
    applied = False

    def mutate(read: BoardReader) -> list[BoardChange]:
        nonlocal applied
        applied = read(path) == item.raw
        if not applied:
            return []
        return [
            BoardChange(path, content),
            BoardChange(
                _file(item.directory, CONSOLIDATED_FILE),
                json.dumps(mark.model_dump(), indent=2) + "\n",
            ),
        ]

    board.commit(mutate, f"blackboard: consolidate {path}")
    return applied
