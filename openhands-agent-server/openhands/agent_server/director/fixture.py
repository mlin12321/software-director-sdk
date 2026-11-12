"""A deterministic 487-node org tree with four open escalations, one of
which has reached the human.

Used to develop clients and exercise the event stream before real pods exist.
No agent in it has a live conversation.
"""

from openhands.agent_server.director.models import (
    AgentRef,
    Boundary,
    Escalation,
    EscalationHandoff,
    NodeStatus,
    OrgNodeInput,
)


_DOMAINS = ("platform", "payments", "identity", "search", "growth", "data")
_SERVICES = ("api", "worker", "storage", "gateway", "console")
_MODULES_PER_SERVICE = 15
_SEED = 20261111

# Weighted so most of the tree is working and a few nodes are blocked.
_STATUS_WHEEL: tuple[NodeStatus, ...] = (
    "working",
    "working",
    "working",
    "working",
    "done",
    "done",
    "idle",
    "idle",
    "blocked",
    "blocked",
)


# What teams in the tree could not settle themselves, by the node that raised
# it. Each is held by a node above; only the last has been passed up by every
# one of them and waits on the human.
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
            held_by="root",
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
            held_by="payments.gateway",
        )
    ],
    "search.storage": [
        Escalation(
            id="index-format",
            kind="design_decision",
            title="Keep the v1 index format readable for one more release?",
            detail="Dropping it removes 4 modules; keeping it delays the migration.",
            raised_at="2026-11-09T13:05:00Z",
            held_by="search",
        )
    ],
    "data.worker.module-03": [
        Escalation(
            id="pii-in-logs",
            kind="critical_issue",
            title="Export job writes customer emails to debug logs",
            detail="Found in review; three attempts to fix it were pruned.",
            raised_at="2026-11-11T08:20:00Z",
            held_by=None,
            handoffs=[
                EscalationHandoff(
                    node_id="data.worker",
                    reason=(
                        "The logger is shared by every data service; changing "
                        "it is outside this team's boundary."
                    ),
                    at="2026-11-11T09:02:00Z",
                ),
                EscalationHandoff(
                    node_id="data",
                    reason=(
                        "The shared logger belongs to the platform domain, "
                        "which this domain cannot write to."
                    ),
                    at="2026-11-11T09:40:00Z",
                ),
                EscalationHandoff(
                    node_id="root",
                    reason=(
                        "Logs already shipped contain the emails. Whether to "
                        "purge them and notify customers is not a call an "
                        "agent can make."
                    ),
                    at="2026-11-11T10:15:00Z",
                ),
            ],
        )
    ],
}


class _Wheel:
    """Picks statuses with a fixed linear congruential generator.

    ``random.Random`` is avoided on purpose: the fixture must stay the same
    across Python versions, because clients and tests refer to its contents.
    """

    def __init__(self, seed: int) -> None:
        self._state = seed

    def pick(self) -> NodeStatus:
        self._state = (self._state * 1103515245 + 12345) % 2**31
        return _STATUS_WHEEL[(self._state >> 16) % len(_STATUS_WHEEL)]


def _pod() -> list[AgentRef]:
    return [
        AgentRef(role="ic", status="working"),
        AgentRef(role="manager", status="idle"),
    ]


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
            agents=_pod(),
            status="working",
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
                agents=_pod(),
                status=wheel.pick(),
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
                    agents=_pod(),
                    status=wheel.pick(),
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
                        agents=_pod(),
                        status=wheel.pick(),
                        spec_summary=f"Leaf module {index} of {domain}/{service}.",
                        escalations=_ESCALATIONS.get(f"{service_id}.{module}", []),
                    )
                )
    return nodes
