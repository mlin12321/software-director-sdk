"""Names a team and says in one line what it works on.

A model call of its own, apart from the agents' conversations: the first
message of a request is turned into a short title and a headline, the way
chat clients title a conversation from its first message.
"""

import logging
import re
from dataclasses import dataclass

from openhands.sdk.conversation.title_utils import strip_reasoning_blocks
from openhands.sdk.llm import LLM, Message, TextContent


logger = logging.getLogger(__name__)

MAX_TITLE_LENGTH = 50
MAX_HEADLINE_LENGTH = 120
# As much of a task as the model is shown.
_MAX_TASK_LENGTH = 4000

_INSTRUCTIONS = """\
You label the work of software teams for an org chart. You are given what a \
team was asked to do, and sometimes what it last reported. Answer with \
exactly two lines and nothing else:
Title: <a name for the team's work, 2 to 5 words, no quotes, no final period>
Headline: <one sentence of at most 15 words on what the team is delivering, \
or its latest state if it reported one>"""

_LINE = re.compile(r"^\s*(title|headline)\s*:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class TeamSummary:
    title: str
    headline: str


def clip(text: str, length: int) -> str:
    """The first line of ``text``, cut to ``length`` with an ellipsis."""
    stripped = text.strip()
    line = stripped.splitlines()[0].strip() if stripped else ""
    return line if len(line) <= length else f"{line[: length - 1].rstrip()}…"


def provisional_summary(task: str) -> TeamSummary:
    """What a team is called until the model has summarized its task."""
    return TeamSummary(
        title=clip(task, MAX_TITLE_LENGTH), headline=clip(task, MAX_HEADLINE_LENGTH)
    )


def parse_summary(answer: str) -> TeamSummary | None:
    """Read the model's two lines; None when either is missing."""
    found: dict[str, str] = {}
    for line in strip_reasoning_blocks(answer).splitlines():
        match = _LINE.match(line)
        if match is not None:
            found.setdefault(match.group(1).lower(), match.group(2).strip("\"'`* "))
    title, headline = found.get("title"), found.get("headline")
    if not title or not headline:
        return None
    return TeamSummary(
        title=clip(title, MAX_TITLE_LENGTH),
        headline=clip(headline, MAX_HEADLINE_LENGTH),
    )


def summary_question(task: str) -> str:
    """The same request as one message, for an agent that is asked itself."""
    return (
        f"{_INSTRUCTIONS}\n\nWhat the team was asked to do:\n{task[:_MAX_TASK_LENGTH]}"
    )


def summarize_team(llm: LLM, task: str) -> TeamSummary | None:
    """Ask ``llm`` for a title and a headline of ``task``.

    Returns:
        None when the call fails or the answer cannot be read; the caller
        keeps what it had.
    """
    messages = [
        Message(role="system", content=[TextContent(text=_INSTRUCTIONS)]),
        Message(role="user", content=[TextContent(text=task[:_MAX_TASK_LENGTH])]),
    ]
    try:
        response = llm.generate(messages, store=False)
    except Exception:
        logger.warning("Could not summarize a team's task", exc_info=True)
        return None
    text = "".join(
        part.text for part in response.message.content if isinstance(part, TextContent)
    )
    return parse_summary(text)
