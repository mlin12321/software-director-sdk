"""A deterministic 487-node org tree with four open escalations.

Used to develop clients and exercise the event stream before real pods exist.
No agent in it has a live conversation.
"""

from openhands.agent_server.director.models import (
    AgentRef,
    Boundary,
    Escalation,
    NodeHealth,
    OrgNodeInput,
)


_DOMAINS = ("platform", "payments", "identity", "search", "growth", "data")
_SERVICES = ("api", "worker", "storage", "gateway", "console")
_MODULES_PER_SERVICE = 15
_SEED = 20261111

# Weighted so most of the tree is healthy and a few nodes need attention.
_HEALTH_WHEEL: tuple[NodeHealth, ...] = (
    "converging",
    "converging",
    "converging",
    "converging",
    "done",
    "done",
    "negotiating_split",
    "needs_attention",
    "collision",
    "pruned",
)


# The few things in the tree that wait on the human, by the node that raised
# them. Everything else an agent team settles on its own.
_ESCALATIONS: dict[str, list[Escalation]] = {
    "identity": [
        Escalation(
            id="session-tokens",
            kind="design_decision",
            title="Session tokens: signed JWTs or opaque server-side sessions?",
            detail=(
                "The api and gateway teams disagree. JWTs avoid a lookup per "
                "request; opaque sessions can be revoked immediately."
            ),
            raised_at="2026-11-10T09:12:00Z",
        )
    ],
    "payments.gateway.module-07": [
        Escalation(
            id="idempotency-collision",
            kind="critical_issue",
            title="Idempotency keys collide across retries",
            detail=(
                "Two retries of one charge can both succeed. The fix needs a "
                "schema change outside this module's boundary."
            ),
            raised_at="2026-11-11T16:40:00Z",
        )
    ],
    "search.storage": [
        Escalation(
            id="index-format",
            kind="design_decision",
            title="Keep the v1 index format readable for one more release?",
            detail="Dropping it removes 4 modules; keeping it delays the migration.",
            raised_at="2026-11-09T13:05:00Z",
        )
    ],
    "data.worker.module-03": [
        Escalation(
            id="pii-in-logs",
            kind="critical_issue",
            title="Export job writes customer emails to debug logs",
            detail="Found by the reviewer; three attempts to fix it were pruned.",
            raised_at="2026-11-11T08:20:00Z",
        )
    ],
}


class _Wheel:
    """Picks healths with a fixed linear congruential generator.

    ``random.Random`` is avoided on purpose: the fixture must stay the same
    across Python versions, because clients and tests refer to its contents.
    """

    def __init__(self, seed: int) -> None:
        self._state = seed

    def pick(self) -> NodeHealth:
        self._state = (self._state * 1103515245 + 12345) % 2**31
        return _HEALTH_WHEEL[(self._state >> 16) % len(_HEALTH_WHEEL)]


def _pod(*, leaf: bool) -> list[AgentRef]:
    agents = [
        AgentRef(role="lead_ic", status="running"),
        AgentRef(role="manager", status="idle"),
        AgentRef(role="reviewer", status="idle"),
    ]
    if leaf:
        agents.append(AgentRef(role="worker", status="running"))
    return agents


def build_fixture_nodes() -> list[OrgNodeInput]:
    """Build the fixture tree, parents before children."""
    wheel = _Wheel(_SEED)
    nodes = [
        OrgNodeInput(
            id="root",
            parent_id=None,
            title="Root organization",
            boundary=Boundary(path="", readonly=["**"]),
            isolation="worktree",
            worktree_path=".",
            branch="main",
            agents=_pod(leaf=False),
            health="converging",
            spec_summary="Owns cross-domain contracts and the release plan.",
        )
    ]
    for domain in _DOMAINS:
        domain_path = f"services/{domain}"
        nodes.append(
            OrgNodeInput(
                id=domain,
                parent_id="root",
                title=f"{domain} domain",
                boundary=Boundary(
                    path=domain_path,
                    writable=[f"{domain_path}/*.md"],
                    readonly=["src/types/**"],
                ),
                isolation="worktree",
                worktree_path=f".worktrees/{domain}",
                branch=f"agents/{domain}",
                agents=_pod(leaf=False),
                health=wheel.pick(),
                spec_summary=f"Interface contracts between the {domain} services.",
                escalations=_ESCALATIONS.get(domain, []),
            )
        )
        for service in _SERVICES:
            service_id = f"{domain}.{service}"
            service_path = f"{domain_path}/{service}"
            nodes.append(
                OrgNodeInput(
                    id=service_id,
                    parent_id=domain,
                    title=f"{domain}/{service}",
                    boundary=Boundary(
                        path=service_path,
                        writable=[f"{service_path}/*.ts"],
                        readonly=[f"{domain_path}/contracts/**"],
                    ),
                    isolation="worktree",
                    worktree_path=f".worktrees/{domain}-{service}",
                    branch=f"agents/{domain}/{service}",
                    agents=_pod(leaf=False),
                    health=wheel.pick(),
                    spec_summary=f"Implements the {service} service for {domain}.",
                    escalations=_ESCALATIONS.get(service_id, []),
                )
            )
            for index in range(1, _MODULES_PER_SERVICE + 1):
                module = f"module-{index:02d}"
                module_path = f"{service_path}/{module}"
                nodes.append(
                    OrgNodeInput(
                        id=f"{service_id}.{module}",
                        parent_id=service_id,
                        title=module,
                        boundary=Boundary(
                            path=module_path,
                            writable=[f"{module_path}/**"],
                            readonly=[f"{service_path}/*.ts"],
                        ),
                        # Module workers collaborate inside their service's
                        # worktree.
                        isolation="shared",
                        worktree_path=f".worktrees/{domain}-{service}",
                        branch=f"agents/{domain}/{service}",
                        agents=_pod(leaf=True),
                        health=wheel.pick(),
                        spec_summary=f"Leaf module {index} of {domain}/{service}.",
                        escalations=_ESCALATIONS.get(f"{service_id}.{module}", []),
                    )
                )
    return nodes
