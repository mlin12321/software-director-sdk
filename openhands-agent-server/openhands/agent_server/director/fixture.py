"""A deterministic 487-node org tree.

Used to develop clients and exercise the event stream before real pods exist.
No agent in it has a live conversation.
"""

from openhands.agent_server.director.models import (
    AgentRef,
    Boundary,
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
                    )
                )
    return nodes
