"""Sending a request straight to the team it concerns."""

import asyncio
import logging
from pathlib import Path

from openhands.agent_server.director import routing
from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.git import GitError, git
from openhands.agent_server.director.models import OrgNode
from openhands.agent_server.director.org_store import AGENT_LOST
from openhands.agent_server.director.prompts import misrouted_notice
from openhands.agent_server.director.routing import Card, Choice
from openhands.agent_server.director.service.teams import Teams
from openhands.sdk.errors import Invalid


logger = logging.getLogger(__name__)

# How many of the requests a team took before are on its card.
_TOOK_SHOWN = 3
_NOT_THEIRS_SHOWN = 10


def _tracked(checkout: Path) -> list[str]:
    try:
        listed = git(checkout, ["ls-files", "-z", "--full-name", "--", ":(top)"])
    except (GitError, OSError):
        return []
    return [path for path in listed.split("\0") if path]


def takes_requests(node: OrgNode) -> bool:
    """Whether a request may be sent straight to a team now.

    Not to one that was only set up, which has not had its task from the IC
    above; not to one that is stopped or whose IC is gone; and not to one
    whose work waits for its review, which would otherwise go on changing
    what is being reviewed.
    """
    ic = next((agent for agent in node.agents if agent.role == "ic"), None)
    if ic is None or ic.conversation_id is None or ic.status == AGENT_LOST:
        return False
    if node.stopped is not None:
        return False
    return node.review is None or node.review.state != "pending"


class Requests(Teams):
    """Chooses the team a request goes straight to."""

    async def choose_team(
        self, text: str, start: OrgNode, *, asker: OrgNode | None = None
    ) -> Choice:
        """The team at or below ``start`` that what is asked in ``text``
        goes to. It is ``start`` unless a lower team is sure.

        Args:
            asker: The team whose IC asks, when it is not the human. What
                a team asks for is told in the words of its own work, so
                its card and those of the teams above and below it say
                nothing about whose the request is.
        """
        store = self.store
        # A session takes what the human writes to it and no request more.
        nodes = store.teams()
        parents = {node.id: node.parent_id for node in nodes}
        below = [
            node
            for node in nodes
            if node.id == start.id
            or any(above.id == start.id for above in store.ancestors(node.id))
        ]
        takes = {node.id for node in below if takes_requests(node)} | {start.id}
        if len(takes) == 1:
            return Choice(start.id, "top", False, "no team below can take it")
        root = store.get(store.root_id) if store.root_id else None
        checkout = Path((root or start).worktree_path)
        tracked = await asyncio.to_thread(_tracked, checkout)
        named = routing.paths_named(text, tracked)
        owners = [
            max(holding, key=lambda node: len(node.boundary.path)).id
            for path in named
            if (holding := [node for node in nodes if owns(node, path)])
        ]
        # A started team that cannot take a request now is shown all the
        # same: what is its own then goes to the team above it.
        cards = await asyncio.to_thread(
            self._cards,
            [node for node in below if node.agents or node.id == start.id],
            tracked,
        )
        own = (
            {
                node.id
                for node in below
                if routing.on_one_line(node.id, asker.id, parents)
            }
            if asker
            else set()
        )
        shown = routing.cards_to_show(text, cards)
        answers, reason = await self._ask_which(text, shown)
        choice = routing.choose(
            start=start.id,
            parents=parents,
            takes=takes,
            named_owners=owners,
            answers=answers,
            worded=routing.best_by_words(
                text, [card for card in cards if card.node_id not in own]
            ),
            reason=reason,
        )
        logger.info("A request was %s", choice.log_line())
        return choice

    def _cards(self, teams: list[OrgNode], tracked: list[str]) -> list[Card]:
        directories = {node.boundary.path for node in self.store.teams()}
        took: dict[str, list[str]] = {}
        for request in self.routed_requests():
            if request.kind == "asked":
                took.setdefault(request.team_id, []).append(request.text)
        return [
            Card(
                node_id=node.id,
                parent_id=node.parent_id,
                title=node.title,
                directory=node.boundary.path,
                purpose=self._purpose(node),
                names=routing.names_at_top(
                    node.boundary.path, tracked, directories - {node.boundary.path}
                ),
                took=took.get(node.id, [])[-_TOOK_SHOWN:],
            )
            for node in teams
        ]

    async def _ask_which(
        self, text: str, cards: list[Card]
    ) -> tuple[list[list[str] | None], str]:
        """Ask the model twice, with the cards in a different order, which
        teams a request concerns.

        Returns:
            Each answer as ``routing.parse_answer`` reads it, and the reason
            the model gave first; no answers when no model can be asked.
        """
        asker = self._asker()
        if asker is None or self._paused is not None:
            return [], ""
        not_theirs = [
            f"- {' '.join(request.text.split())[:200]} (not for {request.team!r})"
            for request in await asyncio.to_thread(self.routed_requests)
            if request.kind == "misrouted"
        ][-_NOT_THEIRS_SHOWN:]
        known = {card.node_id for card in cards}

        async def ask(turned: bool) -> str | None:
            system, user = routing.question(text, cards, not_theirs, turned=turned)
            async with self._model_calls:
                return await self._ask(asker, system, user, routing.USAGE_ID)

        raw = await asyncio.gather(ask(False), ask(True))
        reason = next((r for r in map(routing.reason_of, raw) if r), "")
        return [routing.parse_answer(answer, known) for answer in raw], reason

    def not_ours(self, node: OrgNode, why: str) -> str:
        """A team's IC says the request the human's words brought it
        directly is not its team's: hand it to the IC of the team above.
        Safe from any thread.

        Raises:
            Invalid: Nothing was sent to the team directly, or it was
                handed on already.
        """
        mine = [
            request for request in self.routed_requests() if request.team_id == node.id
        ]
        above = self.store.get(node.parent_id) if node.parent_id else None
        if above is None or not mine or mine[-1].kind != "asked":
            raise Invalid(
                "Nothing was sent to your team directly that is still with "
                "it. Raise what you cannot settle as a `critical_issue` or "
                "a `design_decision`."
            )
        request = mine[-1]
        self.keep_routed("misrouted", node, request.text, request.chosen_by, why)
        self.tell_ic(above.id, misrouted_notice(node, request.text, why))
        return (
            f"Handed to the IC of the team above ({above.title}). It is no "
            "longer yours; go on with your own work."
        )
