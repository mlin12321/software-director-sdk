"""Chooses the team a request goes to without passing down the tree.

A wrong team beside the right one costs more than a team too high: too high
is the slow way a request took before, beside is a team at work on what is
not its own. So every doubt moves the choice up, never sideways, and a
choice nobody confirms is taken only when it is sure.

Two things say which teams a request concerns: the files and directories
its text names, which is exact, and a model that is shown a card of each
team and is asked twice with the cards in a different order. The choice is
the lowest team that holds every team named.
"""

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from openhands.agent_server.director.models import ChosenBy
from openhands.sdk.conversation.title_utils import strip_reasoning_blocks


USAGE_ID = "director-routing"
# Above this many teams, only the ones whose cards share most words with the
# request are shown to the model, with every team above them.
MAX_CARDS = 30
CARDS_KEPT = 12
MAX_NAMES = 25
MAX_REQUEST_LENGTH = 4000
_MAX_PURPOSE_LENGTH = 300
# How far the best card must be ahead of the next for the words to count.
_CLEAR_LEAD = 1.5

_INSTRUCTIONS = """\
You route a request to the teams of a software org. Each team owns one \
directory of a repository; a team listed under another owns a directory \
inside that team's. You are given the teams and one request. Name every \
team whose own files the request asks to change or to explain. Name a team \
only when the request clearly concerns what that team holds; name the team \
above when it concerns what several teams below it share, or how they fit \
together. When the request concerns the whole repository, or you cannot \
tell, answer none. Answer with exactly two lines and nothing else:
Teams: <the ids of the teams, separated by commas, or none>
Reason: <one sentence of at most 20 words>"""

_LINE = re.compile(r"^\s*(teams|reason)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_EDGES = "\"'`()[]{}<>,;:!?*"
_COMMON = frozenset(
    "the and for that this with from have has are was were will would should "
    "could can not but you your our its all any into out about when what "
    "which who how why please need needs make made add fix use using does "
    "team teams file files code change changes".split()
)


@dataclass(frozen=True)
class Card:
    """What the model is shown of one team: only what the director holds of
    it, so that no team can write itself into more requests."""

    node_id: str
    parent_id: str | None
    title: str
    directory: str
    purpose: str = ""
    names: Sequence[str] = ()
    """Files and directories at the top of what the team holds."""
    took: Sequence[str] = ()
    """Requests it was sent directly before."""


@dataclass(frozen=True)
class Choice:
    """The team a request goes to, and how it was found."""

    node_id: str
    by: ChosenBy
    sure: bool
    reason: str
    named: Sequence[str] = ()
    answers: Sequence[Sequence[str] | None] = field(default_factory=tuple)
    worded: str | None = None

    def log_line(self) -> str:
        return (
            f"routed to {self.node_id} by {self.by} "
            f"({'sure' if self.sure else 'unsure'}): named={list(self.named)} "
            f"answers={[a if a is None else list(a) for a in self.answers]} "
            f"words={self.worded} reason={self.reason!r}"
        )


def paths_named(text: str, tracked: Collection[str]) -> list[str]:
    """The tracked files and directories the text names, in its order.

    A word counts when it is a path of the repository with a ``/`` or a
    ``.`` in it, or the name of exactly one tracked file. A bare word that
    happens to be a directory's name does not.
    """
    files = set(tracked)
    directories = {file[:end] for file in files for end in _slashes(file) if file[:end]}
    by_name: dict[str, list[str]] = {}
    for file in files:
        by_name.setdefault(file.rpartition("/")[2], []).append(file)
    found: list[str] = []
    for raw in text.split():
        word = raw.strip(_EDGES).rstrip(".")
        if word.startswith("./"):
            word = word[2:]
        word = word.rstrip("/")
        if not word or ("/" not in word and "." not in word):
            continue
        if word in files or word in directories:
            path = word
        elif "/" not in word and len(by_name.get(word, ())) == 1:
            path = by_name[word][0]
        else:
            continue
        if path not in found:
            found.append(path)
    return found


def _slashes(path: str) -> list[int]:
    return [index for index, char in enumerate(path) if char == "/"]


def names_at_top(
    directory: str, tracked: Collection[str], below: Collection[str]
) -> list[str]:
    """The files and directories directly in a team's directory, without
    the directories of its sub-teams."""
    prefix = f"{directory}/" if directory else ""
    names: set[str] = set()
    for file in tracked:
        if not file.startswith(prefix):
            continue
        name, slash, _ = file[len(prefix) :].partition("/")
        if f"{prefix}{name}" in below:
            continue
        names.add(f"{name}/" if slash else name)
    return sorted(names)[:MAX_NAMES]


def _line_up(node_id: str, parents: Mapping[str, str | None]) -> list[str]:
    """A team and every team above it, nearest first."""
    line = [node_id]
    while (above := parents.get(line[-1])) is not None:
        line.append(above)
    return line


def lowest_holding(
    node_ids: Collection[str], parents: Mapping[str, str | None]
) -> str | None:
    """The lowest team that is, or is above, every one of ``node_ids``."""
    lines = [_line_up(node_id, parents) for node_id in node_ids]
    if not lines:
        return None
    return next(
        (team for team in lines[0] if all(team in line for line in lines)),
        None,
    )


def on_one_line(one: str, other: str, parents: Mapping[str, str | None]) -> bool:
    """Whether one team is the other, or above it, or below it."""
    return one in _line_up(other, parents) or other in _line_up(one, parents)


def _words(text: str) -> set[str]:
    return {
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 2 and word not in _COMMON
    }


def _card_words(card: Card) -> set[str]:
    return _words(
        " ".join([card.title, card.directory, card.purpose, *card.names, *card.took])
    )


def word_scores(text: str, cards: Sequence[Card]) -> dict[str, float]:
    """How much of the request's words each card shares, rarer words more."""
    asked = _words(text)
    held = {card.node_id: _card_words(card) for card in cards}
    scores: dict[str, float] = {}
    for word in asked:
        having = [node_id for node_id, words in held.items() if word in words]
        if not having:
            continue
        weight = math.log(1 + len(cards) / len(having))
        for node_id in having:
            scores[node_id] = scores.get(node_id, 0.0) + weight
    return scores


def best_by_words(text: str, cards: Sequence[Card]) -> str | None:
    """The team whose card clearly shares most with the request, if one does."""
    ranked = sorted(word_scores(text, cards).items(), key=lambda item: -item[1])
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] < ranked[1][1] * _CLEAR_LEAD:
        return None
    return ranked[0][0]


def cards_to_show(text: str, cards: Sequence[Card]) -> list[Card]:
    """The cards the model is shown: all of a small org, and of a large one
    the best by words with every team above them."""
    if len(cards) <= MAX_CARDS:
        return list(cards)
    parents = {card.node_id: card.parent_id for card in cards}
    scores = word_scores(text, cards)
    best = sorted(scores, key=lambda node_id: -scores[node_id])[:CARDS_KEPT]
    kept = {above for node_id in best for above in _line_up(node_id, parents)}
    kept.update(card.node_id for card in cards if card.parent_id is None)
    return [card for card in cards if card.node_id in kept]


def _clip(text: str, length: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= length else f"{flat[: length - 1].rstrip()}…"


def outline(cards: Sequence[Card], *, turned: bool = False) -> str:
    """The teams as an indented list, sub-teams under their team.

    Args:
        turned: List the teams of each level in the opposite order.
    """
    shown = {card.node_id for card in cards}
    below: dict[str | None, list[Card]] = {}
    for card in cards:
        above = card.parent_id if card.parent_id in shown else None
        below.setdefault(above, []).append(card)
    lines: list[str] = []

    def write(above: str | None, depth: int) -> None:
        level = below.get(above, [])
        for card in reversed(level) if turned else level:
            pad = "  " * depth
            lines.append(
                f"{pad}- id: {card.node_id} | {card.title} | "
                f"directory: {card.directory or '. (the whole repository)'}"
            )
            if card.purpose:
                lines.append(f"{pad}  for: {_clip(card.purpose, _MAX_PURPOSE_LENGTH)}")
            if card.names:
                lines.append(f"{pad}  holds: {', '.join(card.names)}")
            for request in card.took:
                lines.append(f"{pad}  took before: {_clip(request, 120)}")
            write(card.node_id, depth + 1)

    write(None, 0)
    return "\n".join(lines)


def question(
    text: str, cards: Sequence[Card], not_theirs: Sequence[str], *, turned: bool
) -> tuple[str, str]:
    """The system and the user message that ask which teams a request is for.

    Args:
        not_theirs: Earlier requests with the team that said each was not
            its own, as lines.
    """
    parts = ["The teams:", outline(cards, turned=turned)]
    if not_theirs:
        parts += ["", "Requests that were sent to the wrong team before:", *not_theirs]
    parts += ["", "The request:", text[:MAX_REQUEST_LENGTH]]
    return _INSTRUCTIONS, "\n".join(parts)


def parse_answer(answer: str | None, known: Collection[str]) -> list[str] | None:
    """The teams the model named; [] when it named none.

    Returns:
        None when the answer cannot be read or names a team that is not
        among ``known``: such an answer is no answer.
    """
    if not answer:
        return None
    found: dict[str, str] = {}
    for line in strip_reasoning_blocks(answer).splitlines():
        match = _LINE.match(line)
        if match is not None:
            found.setdefault(match.group(1).lower(), match.group(2))
    teams = found.get("teams")
    if teams is None:
        return None
    named = [part.strip("\"'`*[] ") for part in teams.split(",")]
    named = [part for part in named if part]
    if not named or [part.lower() for part in named] == ["none"]:
        return []
    if any(part not in known for part in named):
        return None
    return list(dict.fromkeys(named))


def reason_of(answer: str | None) -> str:
    """The model's one sentence of why, if it gave one."""
    for line in strip_reasoning_blocks(answer or "").splitlines():
        match = _LINE.match(line)
        if match is not None and match.group(1).lower() == "reason":
            return _clip(match.group(2).strip("\"'`* "), 200)
    return ""


def choose(
    *,
    start: str,
    parents: Mapping[str, str | None],
    takes: Collection[str],
    named_owners: Sequence[str],
    answers: Sequence[list[str] | None],
    worded: str | None,
    reason: str = "",
) -> Choice:
    """Decide where a request goes from what was found.

    Args:
        start: The team the request goes to when no lower one is sure.
        parents: Every team of the org with the team above it.
        takes: The teams that can take a request now.
        named_owners: The owners of the paths the text names.
        answers: What the model answered each time it was asked: the teams
            it named, [] for none, None for no usable answer.
        worded: The team whose card clearly shares most words with the
            request, if any.
        reason: The model's own.
    """

    def settled(node_ids: Collection[str]) -> str | None:
        """The lowest team holding them all that can take a request."""
        lowest = lowest_holding(node_ids, parents) if node_ids else start
        if lowest is None:
            return None
        return next((n for n in _line_up(lowest, parents) if n in takes), None)

    def unsure(why: str) -> Choice:
        return Choice(start, "top", False, why, named_owners, tuple(answers), worded)

    def sure(node_id: str, by: ChosenBy, why: str) -> Choice:
        if start not in _line_up(node_id, parents):
            return unsure("the choice is not below where the request was made")
        return Choice(node_id, by, True, why, named_owners, tuple(answers), worded)

    by_name = settled(named_owners) if named_owners else None
    given = [answer for answer in answers if answer is not None]
    if not given or len(given) < len(answers):
        if by_name is None:
            return unsure("no answer to go by")
        return sure(by_name, "named", "the request names files this team holds")
    picks = [settled(answer) for answer in given]
    pick = picks[0]
    if pick is None or any(other != pick for other in picks):
        return unsure("asked twice, the answers differ")
    if by_name is not None and pick not in _line_up(by_name, parents):
        return unsure("the files the request names are not this team's")
    if worded is not None and not on_one_line(worded, pick, parents):
        return unsure("the request's words point at another team")
    return sure(pick, "model", reason or "the request concerns what this team holds")
