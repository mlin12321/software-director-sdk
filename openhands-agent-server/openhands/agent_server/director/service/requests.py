"""Sending a request straight to the team it concerns."""

import asyncio
import logging
from pathlib import Path

from openhands.agent_server.director import routing
from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.git import GitError, git
from openhands.agent_server.director.git_store import team_key
from openhands.agent_server.director.models import OrgNode, RoutedRequest, SessionTeam
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    is_division_lead,
)
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


def takes_requests(place: SessionTeam) -> bool:
    """Whether a request may be sent straight to a team now.

    Not to one that was only set up, which has not had its task from the IC
    above; not to one that is stopped or whose IC is gone; and not to one
    whose work waits for its review, which would otherwise go on changing
    what is being reviewed.
    """
    ic = next((agent for agent in place.agents if agent.role == "ic"), None)
    if ic is None or ic.conversation_id is None or ic.status == AGENT_LOST:
        return False
    if place.stopped is not None:
        return False
    return place.review is None or place.review.state != "pending"


class Requests(Teams):
    """Chooses the team a request goes straight to."""

    async def choose_team(
        self,
        text: str,
        start: OrgNode,
        *,
        within: str = "",
    ) -> Choice:
        """The team at or below ``start`` that what is asked in ``text``
        goes to. It is ``start`` unless a lower team is sure.

        Args:
            within: The session the human says it in, when it does not
                begin a new one.
        """
        store = self.store
        started: set[str] = set()
        able: set[str] | None = None
        checkout = self._repo_dir
        if not within:
            # What the human asks anew opens a session of its own, which
            # any team of the org can have.
            nodes = store.teams()
        else:
            # What is said in a session stays in it: the teams are the ones
            # the session has, those it set up itself included, each as it
            # is there, under the team's own id.
            places = [
                place for place in store.session(within) if not is_division_lead(place)
            ]
            ids = {place.id: place.team_id for place in places}
            rebased = [
                place.model_copy(
                    update={
                        "id": ids[place.id],
                        "parent_id": ids.get(place.parent_id or ""),
                    }
                )
                for place in places
            ]
            nodes = list[OrgNode](rebased)
            start = nodes[0] if nodes else start
            started = {place.id for place in rebased if place.agents}
            by_id = {place.id: place for place in rebased}

            def at_work(place: SessionTeam | None) -> bool:
                # A team under a stopped one is stopped with it.
                while place is not None:
                    if place.stopped is not None:
                        return False
                    place = by_id.get(place.parent_id or "")
                return True

            able = {
                place.id
                for place in rebased
                if takes_requests(place) and at_work(place)
            }
            # A session's own checkout has the directories of the teams it
            # set up.
            if places:
                checkout = Path(places[0].worktree_path)
        parents = {node.id: node.parent_id for node in nodes}

        def under(team_id: str | None) -> bool:
            seen: set[str] = set()
            while team_id is not None and team_id not in seen:
                if team_id == start.id:
                    return True
                seen.add(team_id)
                team_id = parents.get(team_id)
            return False

        below = [node for node in nodes if under(node.id)]
        takes = {node.id for node in below if able is None or node.id in able} | {
            start.id
        }
        if len(takes) == 1:
            return Choice(start.id, "top", False, "no team below can take it")
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
            [
                node
                for node in below
                if not within or node.id == start.id or node.id in started
            ],
            tracked,
        )
        shown = routing.cards_to_show(text, cards)
        answers, reason = await self._ask_which(text, shown)
        choice = routing.choose(
            start=start.id,
            parents=parents,
            takes=takes,
            named_owners=owners,
            answers=answers,
            worded=routing.best_by_words(text, cards),
            reason=reason,
        )
        logger.info("A request was %s", choice.log_line())
        return choice

    def _cards(self, teams: list[OrgNode], tracked: list[str]) -> list[Card]:
        directories = {node.boundary.path for node in self.store.teams()} | {
            node.boundary.path for node in teams
        }
        took: dict[str, list[str]] = {}
        for request in self.routed_requests():
            if request.kind == "asked":
                took.setdefault(team_key(request.team_id), []).append(request.text)
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

    def not_ours(self, place: SessionTeam, why: str) -> str:
        """A team's IC says the request the human's words brought it
        directly is not its team's: hand it to the IC of the team above.
        Safe from any thread.

        Raises:
            Invalid: Nothing was sent to the team directly, or it was
                handed on already.
        """
        mine = [
            request for request in self.routed_requests() if request.team_id == place.id
        ]
        above = self.store.place(place.parent_id) if place.parent_id else None
        if above is None or not mine or mine[-1].kind != "asked":
            raise Invalid(
                "Nothing was sent to your team directly that is still with "
                "it. Raise what you cannot settle as a `critical_issue` or "
                "a `design_decision`."
            )
        request = mine[-1]
        self.keep_routed("misrouted", place, request.text, request.chosen_by, why)
        return self._hand_on(place, above, request, why)

    def _hand_on(
        self, place: SessionTeam, above: SessionTeam, request: RoutedRequest, why: str
    ) -> str:
        """Give a request a team said is not its own to the IC above it."""
        self.tell_ic(above.id, misrouted_notice(place, request.text, why))
        return (
            f"Handed to the IC of the team above ({above.title}). It is no "
            "longer yours; go on with your own work."
        )
