"""An org's settings as a person writes them: ``director.yaml``.

The file holds the fields of :class:`DirectorSettings` and nothing else.
It is read when the server starts. What is changed afterwards through the
settings route is kept by the server and holds over the file, field by
field.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from openhands.agent_server.director.models import DirectorSettings


class InvalidOrgConfig(ValueError):
    """``director.yaml`` does not hold an org's settings."""


def load_org_config(path: Path) -> dict[str, Any]:
    """The settings ``path`` sets, as written; {} when there is no file.

    Raises:
        InvalidOrgConfig: The file is not YAML, sets something that is no
            setting, or sets one to a value it cannot have.
    """
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise InvalidOrgConfig(f"{path}: cannot be read as YAML: {error}") from error
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise InvalidOrgConfig(f"{path}: the settings are a mapping")
    unknown = sorted(set(loaded) - set(DirectorSettings.model_fields))
    if unknown:
        raise InvalidOrgConfig(
            f"{path}: not a setting of the director: {', '.join(map(str, unknown))}. "
            f"The settings are: {', '.join(DirectorSettings.model_fields)}"
        )
    try:
        DirectorSettings.model_validate(loaded)
    except ValidationError as error:
        raise InvalidOrgConfig(f"{path}: {error}") from error
    return loaded
