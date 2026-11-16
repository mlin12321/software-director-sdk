#!/usr/bin/env python3
"""Measure how the director chooses a team for a request.

Runs labelled requests against the sample org with a real model and counts
where each one went: to the right team, to a team above it (the slow way a
request took before), to a team below it, or to a team beside it (a team at
work on what is not its own).

It makes model calls only; no agent runs. The sample org has no files, so a
few are made up for each team to give the named paths something to match.

    OPENAI_API_KEY=... uv run python scripts/director_routing_measure.py
    ROUTING_MODEL=openai/gpt-5.6  (the default)
"""

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import litellm
from litellm.types.utils import Choices, ModelResponse

from openhands.agent_server.director import routing
from openhands.agent_server.director.fixture import build_fixture_nodes
from openhands.agent_server.director.routing import Card


MODEL = os.environ.get("ROUTING_MODEL", "openai/gpt-5.6")

# The request, and the team it is for.
REQUESTS: list[tuple[str, str]] = [
    # One service, said in words.
    (
        "Card refunds are processed twice when the payments worker retries a "
        "job. Make the retry idempotent.",
        "payments.worker",
    ),
    (
        "The search gateway returns 502 under load; add a circuit breaker in "
        "the search gateway.",
        "search.gateway",
    ),
    (
        "Add a page to the identity console that lists a user's active sessions.",
        "identity.console",
    ),
    ("The growth api should expose an endpoint for referral codes.", "growth.api"),
    ("Compact old partitions in the data storage service nightly.", "data.storage"),
    ("The platform worker leaks file handles; fix it.", "platform.worker"),
    ("Payments storage needs an index on invoice date.", "payments.storage"),
    ("Rate-limit login attempts at the identity gateway.", "identity.gateway"),
    ("Show index freshness in the search console.", "search.console"),
    ("The data api should paginate export listings.", "data.api"),
    ("Growth worker: send the weekly digest email in batches.", "growth.worker"),
    ("Platform console: add a dark theme toggle.", "platform.console"),
    (
        "The search worker's nightly reindex job is slow; speed it up.",
        "search.worker",
    ),
    (
        "Customers are charged twice when they double-click pay; the "
        "background job that captures charges must dedupe.",
        "payments.worker",
    ),
    # What several services of one domain share.
    (
        "Change the contract between the payments api and the payments worker "
        "so every job carries an idempotency key.",
        "payments",
    ),
    (
        "The identity services disagree on the session token format; settle "
        "one and update the api and the gateway.",
        "identity",
    ),
    ("Search api and search storage must agree on a new index format.", "search"),
    ("Rename the event schema shared by the growth services.", "growth"),
    (
        "Data domain: write down the retention rule every data service follows.",
        "data",
    ),
    (
        "Platform: the gateway and the console need the same auth header.",
        "platform",
    ),
    (
        "The payments gateway should support Apple Pay, and the payments "
        "console should show it.",
        "payments",
    ),
    ("Password reset tokens should be stored hashed.", "identity"),
    # The whole org.
    (
        "Payments must call identity to verify the user before charging; "
        "define that interface.",
        "root",
    ),
    ("Move every service to the new logging format.", "root"),
    ("What is the release plan for next week?", "root"),
    ("How far along are we?", "root"),
    (
        "Search results should rank paying customers first, using data from payments.",
        "root",
    ),
    ("Cut scope: drop the growth domain from this release.", "root"),
    ("Thanks, looks good.", "root"),
    ("Add tracing across platform, search and data.", "root"),
    # Named paths.
    (
        "Fix the off-by-one in services/payments/api/module-03/handler.ts.",
        "payments.api.module-03",
    ),
    (
        "services/search/worker/module-11/handler.ts times out on big "
        "batches; stream instead.",
        "search.worker.module-11",
    ),
    (
        "Update services/identity/gateway/index.ts to export the new middleware.",
        "identity.gateway",
    ),
    (
        "services/data/contracts/README.md is out of date; bring it in line.",
        "data",
    ),
    (
        "Make services/growth/console/module-02/handler.ts and "
        "services/growth/console/module-07/handler.ts share a helper.",
        "growth.console",
    ),
    (
        "services/platform/api/index.ts and services/payments/api/index.ts "
        "should use the same error shape.",
        "root",
    ),
    (
        "Do not touch services/payments/storage/index.ts; instead fix the "
        "caller in the payments api.",
        "payments.api",
    ),
    (
        "In services/search/api/module-05, add input validation.",
        "search.api.module-05",
    ),
    ("module-04 of the identity api should log less.", "identity.api.module-04"),
    (
        "Tidy the README in services/platform/storage/module-09.",
        "platform.storage.module-09",
    ),
]


def _made_up_files(directory: str, depth: int) -> list[str]:
    under = f"{directory}/" if directory else ""
    names = {
        0: ["README.md"],
        1: ["README.md", "contracts/README.md"],
        2: ["README.md", "index.ts"],
        3: ["README.md", "handler.ts"],
    }[depth]
    return [under + name for name in names]


def _ask(system: str, user: str) -> str | None:
    try:
        response = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=120,
        )
    except litellm.exceptions.APIError as error:
        print(f"model call failed: {error}", file=sys.stderr)
        return None
    if not isinstance(response, ModelResponse):
        return None
    first = response.choices[0]
    return first.message.content if isinstance(first, Choices) else None


def main() -> int:
    nodes = build_fixture_nodes()
    parents = {node.id: node.parent_id for node in nodes}
    directory = {node.id: node.boundary.path for node in nodes}
    depth = {
        node.id: node.id.count(".") + 1 if node.id != "root" else 0 for node in nodes
    }
    tracked = [
        path
        for node in nodes
        for path in _made_up_files(directory[node.id], depth[node.id])
    ]
    every = set(directory.values())
    cards = [
        Card(
            node_id=node.id,
            parent_id=node.parent_id,
            title=node.title,
            directory=directory[node.id],
            purpose=node.spec_summary,
            names=routing.names_at_top(
                directory[node.id], tracked, every - {directory[node.id]}
            ),
        )
        for node in nodes
    ]
    known_all = set(parents)

    def owner(path: str) -> str:
        holding = [
            node_id
            for node_id, held in directory.items()
            if not held or path == held or path.startswith(held + "/")
        ]
        return max(holding, key=lambda node_id: len(directory[node_id]))

    def line_up(node_id: str) -> list[str]:
        line = []
        at: str | None = node_id
        while at is not None:
            line.append(at)
            at = parents[at]
        return line

    def run(labelled: tuple[str, str]) -> dict[str, object]:
        text, right = labelled
        shown = routing.cards_to_show(text, cards)
        known = {card.node_id for card in shown}
        raw = [
            _ask(*routing.question(text, shown, [], turned=turned))
            for turned in (False, True)
        ]
        choice = routing.choose(
            start="root",
            parents=parents,
            takes=known_all,
            named_owners=[owner(path) for path in routing.paths_named(text, tracked)],
            answers=[routing.parse_answer(answer, known) for answer in raw],
            worded=routing.best_by_words(text, cards),
            reason=next((r for r in map(routing.reason_of, raw) if r), ""),
        )
        if choice.node_id == right:
            outcome = "right"
        elif choice.node_id in line_up(right):
            outcome = "too high"
        elif right in line_up(choice.node_id):
            outcome = "too low"
        else:
            outcome = "beside"
        return {
            "request": text,
            "for": right,
            "went_to": choice.node_id,
            "outcome": outcome,
            "by": choice.by,
            "why": choice.reason,
            "named": list(choice.named),
            "answers": [a if a is None else list(a) for a in choice.answers],
            "words": choice.worded,
            "cards_shown": len(shown),
        }

    assert all(right in parents for _, right in REQUESTS)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, REQUESTS))
    for result in results:
        print(json.dumps(result))
    counts = Counter(str(result["outcome"]) for result in results)
    print(
        f"\n{MODEL}, {len(nodes)} teams, {len(results)} requests: "
        + ", ".join(
            f"{counts[name]} {name}"
            for name in ("right", "too high", "too low", "beside")
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
