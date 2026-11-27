"""Settings: the limits the server sets on new teams, what an org sets
for itself, and the kind of org it is."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openhands.agent_server.director.agent_types import TeamTemplate
from openhands.agent_server.director.documents.store import DocumentStore
from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.environment import CodingEnvironment
from openhands.agent_server.director.models import (
    AgentTypeInfo,
    DirectorSettings,
    DirectorSettingsState,
    DirectorSettingsUpdate,
    MemberInfo,
    OrgSettings,
    OrgSettingsState,
)
from openhands.agent_server.director.org import OrgBase
from openhands.agent_server.director.org_config import load_org_config
from openhands.agent_server.director.org_persistence import (
    own_settings,
    save_own_settings,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.orgs import ORG_KINDS
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.sdk.errors import Invalid, NotFound
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# In the remembered file: the settings as last saved, and the ones changed
# through the settings route, which hold over the org's ``director.yaml``
# field by field.
_SETTINGS = "settings"
_CHANGED_SETTINGS = "settings_changed"


@dataclass
class OrgSettingsService:
    """Reads and changes what holds for new teams.

    Two things are set. The server's settings (:class:`DirectorSettings`)
    come from ``director.yaml`` and the settings route, and are kept
    beside the server's conversations. What an org sets for itself
    (:class:`OrgSettings`) is kept with it in its repository, so another
    org on the same server has its own, and holds over the server's.

    Raises:
        InvalidOrgConfig: ``config_file`` does not hold settings.
    """

    config_file: Path | None
    """The org's ``director.yaml``; None for the director's defaults."""
    environment: CodingEnvironment
    """Where the org's records are kept."""
    remembered: RememberedFile
    """Where the server's settings are kept between starts."""
    team_types: TeamTypeService
    """The types there are, for what may be allowed."""
    tree: TeamTree
    documents: DocumentStore
    """The store the org's documents are kept in."""
    """Where a team stands among the others, for the kind of org."""
    org: Callable[[], OrgStore]
    """The org whose settings these are."""
    repo_dir: Callable[[], Path]
    """The repository the org is in."""
    max_concurrent_runs: int | None = None
    """How many conversations the server runs at once, to tell the human;
    None when not known."""
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _configured: dict[str, Any] = field(default_factory=dict, init=False)
    _settings: DirectorSettings | None = field(default=None, init=False)
    # What the org sets for itself, by the repository it was read from.
    _own_settings: tuple[str, OrgSettings] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # A server that cannot read its settings does not start.
        if self.config_file:
            self._configured = load_org_config(self.config_file)

    @property
    def settings(self) -> DirectorSettings:
        """The limits on new teams, as last saved."""
        with self._lock:
            if self._settings is None:
                # With a ``director.yaml`` only what was changed since
                # holds over it; without one, the settings as last saved.
                saved = (
                    {**self._configured, **self._changed()}
                    if self.config_file
                    else self.remembered.read().get(_SETTINGS)
                )
                try:
                    self._settings = DirectorSettings.model_validate(saved or {})
                except ValueError:
                    logger.warning("Ignoring the saved director settings: %r", saved)
                    self._settings = DirectorSettings.model_validate(self._configured)
            return self._settings

    def _changed(self) -> dict[str, Any]:
        """The settings changed through the settings route, as kept."""
        kept = self.remembered.read().get(_CHANGED_SETTINGS)
        return kept if isinstance(kept, dict) else {}

    @property
    def own_settings(self) -> OrgSettings:
        """What the org sets for itself."""
        repo_dir = self.repo_dir()
        kept = self._own_settings
        if kept is None or kept[0] != str(repo_dir):
            found = (
                own_settings(self.environment.org_tree(repo_dir))
                if self.org().root_id is not None
                else OrgSettings()
            )
            kept = self._own_settings = (str(repo_dir), found)
        return kept[1]

    @property
    def org_kind(self) -> OrgBase:
        """The kind of organisation this is, as its settings name it:
        what it allows. One the director does not know is the default."""
        named = self.settings.org_kind
        kind = ORG_KINDS.get(named)
        if kind is None:
            logger.warning("There is no kind of org named %r", named)
            kind = ORG_KINDS[DirectorSettings().org_kind]
        return kind(self.tree)

    def default_type(self) -> str:
        """The type of a new team whose making names none."""
        return self.own_settings.default_agent_type or self.settings.default_agent_type

    def _allowed_type_names(self) -> list[str]:
        own = self.own_settings.allowed_agent_types
        return self.settings.allowed_agent_types if own is None else own

    def types_allowed(self) -> list[TeamTemplate]:
        """The types a new team may be. A team that exists keeps its type
        whatever this says."""
        allowed = self._allowed_type_names()
        return [
            found
            for name, found in self.team_types.agent_types.items()
            if not allowed or name in allowed
        ]

    def org_settings_state(self) -> OrgSettingsState:
        """What holds for new teams of the org, and the types there are."""
        allowed = {kind.name for kind in self.types_allowed()}
        return OrgSettingsState(
            default_agent_type=self.default_type(),
            allowed_agent_types=self._allowed_type_names(),
            own=self.own_settings,
            agent_types=[
                AgentTypeInfo(
                    name=name,
                    description=kind.description,
                    allowed=name in allowed,
                    sas_profile=kind.sas_profile,
                    members=[
                        MemberInfo(
                            name=member,
                            title=kind.title_of(member),
                            writes=member == kind.writer,
                            reviews=member == kind.reviewer,
                            oversees=member == kind.manager,
                        )
                        for member in kind.agents
                    ],
                )
                for name, kind in self.team_types.agent_types.items()
            ],
        )

    def update_org_settings(self, update: OrgSettings) -> OrgSettingsState:
        """Set what the org sets for itself. Only what ``update`` names is
        changed; an empty default goes back to the one for every org.

        It holds for teams made from now on: no team that exists changes
        its type or is removed.

        Raises:
            NotFound: There is no org to set it for.
            Invalid: A type is named that the director does not find, or
                the default is not among the allowed ones.
        """
        if self.org().root_id is None:
            raise NotFound("No org exists")
        named = update.model_dump(exclude_unset=True)
        changed = self.own_settings.model_copy(update=named)
        if changed.default_agent_type == "":
            changed.default_agent_type = None
        types = self.team_types.agent_types
        asked = [*(changed.allowed_agent_types or [])]
        if changed.default_agent_type:
            asked.append(changed.default_agent_type)
        unknown = sorted(set(asked) - set(types))
        if unknown:
            raise Invalid(
                f"The director finds no agent type named {', '.join(unknown)}. "
                f"The types are: {', '.join(types)}."
            )
        allowed = (
            self.settings.allowed_agent_types
            if changed.allowed_agent_types is None
            else changed.allowed_agent_types
        )
        default = changed.default_agent_type or self.settings.default_agent_type
        if allowed and default not in allowed:
            raise Invalid(
                f"The default agent type {default!r} is not among the allowed "
                f"ones: {', '.join(allowed)}"
            )
        repo_dir = self.repo_dir()
        save_own_settings(self.environment.org_tree(repo_dir), changed)
        self._own_settings = (str(repo_dir), changed)
        return self.org_settings_state()

    def settings_state(self) -> DirectorSettingsState:
        """The settings, with how large the org is now."""
        levels: dict[str, int] = {}
        for node in self.org().teams():
            levels[node.id] = levels.get(node.parent_id or "", 0) + 1
        return DirectorSettingsState(
            **self.settings.model_dump(),
            teams=len(levels),
            depth=max(levels.values(), default=0),
            max_concurrent_runs=self.max_concurrent_runs,
        )

    def update_settings(self, update: DirectorSettingsUpdate) -> DirectorSettingsState:
        """Change the limits on new teams. Existing teams are left alone.

        An open session follows a new cost limit, unless the human let it
        go on past the one it had.

        Raises:
            Invalid: A kind of org is named that the director does not know.
        """
        values = update.model_dump(exclude_none=True)
        if values.get("org_kind", self.settings.org_kind) not in ORG_KINDS:
            raise Invalid(
                f"There is no kind of org named {values['org_kind']!r}. "
                f"The kinds are: {', '.join(ORG_KINDS)}."
            )
        before = self.settings.session_cost_limit
        now = self.settings.model_dump()
        # The org page sends every setting when one is changed: only what
        # differs was changed, and only that holds over ``director.yaml``.
        values = {key: value for key, value in values.items() if now[key] != value}
        changed = self.settings.model_copy(update=values)
        with self._lock:
            self._settings = changed
        if changed.session_cost_limit != before:
            store = self.org()
            for session in store.sessions():
                if session.cost_allowance == before:
                    store.set_session_use(
                        session.id, session.usage, changed.session_cost_limit
                    )
        earlier = self._changed()
        self.remembered.keep(
            **{
                _SETTINGS: changed.model_dump(),
                _CHANGED_SETTINGS: {**earlier, **values},
            }
        )
        return self.settings_state()

    def team_documents(self) -> TeamDocuments:
        """The documents of the org as each team reads, writes and leaves
        them."""
        return TeamDocuments(self.documents, self.team_types.protocols, self.org_kind)
