"""A model's own pass over a team's diff, before the IC above reviews it.

The instructions are OpenHands' public code review skill, the one its pull
request reviewer runs with. Here the model is given the diff directly and
answers with findings instead of commenting on a pull request.
"""

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from openhands.agent_server.director.git import GitError, git, pathspec
from openhands.agent_server.director.models import ReviewFinding
from openhands.sdk.conversation.title_utils import strip_reasoning_blocks
from openhands.sdk.skills import KeywordTrigger, Skill, TaskTrigger, load_public_skills


logger = logging.getLogger(__name__)

# As much of a diff as the model is shown.
MAX_DIFF_LENGTH = 60_000
USAGE_ID = "director-bugbot"
_SKILL_NAMES = ("codereview", "code-review")
_SKILL_TRIGGER = "/codereview"

_ANSWER_FORMAT = """\
You are given the diff of one team's work, the spec the team wrote for it \
and the contracts it must keep with other teams. Review the diff by the \
instructions above. You cannot run anything or open other files: report only \
what the diff itself shows, and nothing you would have to guess.

Answer with one JSON object and nothing else:
{"findings": [{"path": "<file>", "line": <line in the new file, or null>, \
"issue": "<what is wrong and what it breaks>", "evidence": "<the code that \
shows it>"}]}
An empty list means you found nothing that must change."""

_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class _Answer(BaseModel):
    findings: list[ReviewFinding]


def review_instructions() -> str | None:
    """OpenHands' public code review skill; None when it cannot be loaded."""
    try:
        # Every public skill, whatever the default marketplace lists.
        skills = load_public_skills(marketplace_path=None)
    except Exception:
        logger.warning("Could not load the public skills", exc_info=True)
        return None
    for skill in skills:
        if skill.name in _SKILL_NAMES or _SKILL_TRIGGER in _keywords(skill):
            return skill.content
    return None


def _keywords(skill: Skill) -> list[str]:
    trigger = skill.trigger
    if isinstance(trigger, KeywordTrigger):
        return trigger.keywords
    if isinstance(trigger, TaskTrigger):
        return trigger.triggers
    return []


def team_diff(checkout: Path, base: str, directory: str) -> str:
    """What a team committed under ``directory`` since ``base``.

    Raises:
        GitError: ``base`` is not a commit of the checkout.
    """
    return git(checkout, ["diff", base, "HEAD", "--", pathspec(directory)])


def review_prompt(
    instructions: str, spec: str | None, contracts: list[str], diff: str
) -> tuple[str, str]:
    """The system and the user message of the review call."""
    shown = diff[:MAX_DIFF_LENGTH]
    parts = [
        "The team's spec:",
        spec or "(the team wrote none)",
        "",
        "Contracts the team must keep:",
        "\n\n".join(contracts) or "(none)",
        "",
        "The diff:"
        if len(diff) <= MAX_DIFF_LENGTH
        else f"The diff (the first {MAX_DIFF_LENGTH} of {len(diff)} characters):",
        shown,
    ]
    return f"{instructions.strip()}\n\n{_ANSWER_FORMAT}", "\n".join(parts)


def parse_findings(answer: str) -> list[ReviewFinding] | None:
    """Read the model's findings; None when the answer cannot be read."""
    text = strip_reasoning_blocks(answer)
    match = _OBJECT.search(text)
    if match is None:
        return None
    try:
        return _Answer.model_validate(json.loads(match.group(0))).findings
    except (ValueError, ValidationError):
        return None


def diff_or_none(checkout: Path, base: str, directory: str) -> str | None:
    """:func:`team_diff`, or None when it cannot be made."""
    try:
        return team_diff(checkout, base, directory)
    except (GitError, OSError):
        logger.warning("Could not diff %s against %s", checkout, base, exc_info=True)
        return None
