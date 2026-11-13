"""What a team's IC is asked about its own work before the IC above reviews
it: to write up what changed, and to review it.

Both are questions put on top of the IC's conversation, which already holds
the work, so no diff is sent to a model. The review instructions are
OpenHands' public code review skill, the one its pull request reviewer runs
with; the IC answers with findings instead of commenting on a pull request.
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

_SKILL_NAMES = ("codereview", "code-review")
_SKILL_TRIGGER = "/codereview"

_WRITE_UP = """\
Your team reported its work done, and the IC of the team above reviews it \
next. Write the pull request description of everything your team changed \
since it started, for that reviewer. It has not seen your conversation.

Cover, in this order:
- What changed and why, in a few sentences.
- Each file that matters, with what now behaves differently in it. Files \
that only follow from another change can share one line.
- Anything that touches a contract with another team, or that another team \
will notice.
- How the work was tested, and what was not.
- What the reviewer should look at first.

Say only what you know from your work. git lists these changed files; \
account for all of them:
"""

_REVIEW = """\
Review the change your team just reported done by the instructions below, \
as its reviewer would. Report only what you know is in the committed code, \
and nothing you would have to guess.

Answer with one JSON object and nothing else:
{"findings": [{"path": "<file>", "line": <line in the file, or null>, \
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


def changed_files(checkout: Path, base: str, directory: str) -> str | None:
    """git's list of what a team committed under ``directory`` since
    ``base``, with the lines added and removed in each file.

    Returns:
        '' when nothing changed; None when git cannot tell.
    """
    try:
        return git(
            checkout, ["diff", "--stat=200", base, "HEAD", "--"] + [pathspec(directory)]
        ).rstrip()
    except (GitError, OSError):
        logger.warning("Could not diff %s against %s", checkout, base, exc_info=True)
        return None


def write_up_question(files: str) -> str:
    """Asks a team's IC to describe its team's changes for the reviewer."""
    return f"{_WRITE_UP}\n{files}"


def review_question(
    instructions: str, spec: str | None, contracts: list[str], files: str
) -> str:
    """Asks a team's IC to review its team's changes."""
    parts = [
        _REVIEW,
        "",
        "The review instructions:",
        instructions.strip(),
        "",
        "The team's spec:",
        spec or "(the team wrote none)",
        "",
        "Contracts the team must keep:",
        "\n\n".join(contracts) or "(none)",
        "",
        "The changed files:",
        files,
    ]
    return "\n".join(parts)


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
