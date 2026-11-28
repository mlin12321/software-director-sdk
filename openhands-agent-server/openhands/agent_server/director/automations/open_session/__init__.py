"""The automation that opens a session of the org for an event."""

import base64
import gzip
import io
import json
import tarfile
from collections.abc import Mapping
from pathlib import Path

from openhands.agent_server.director.automations.open_session.main import SETTINGS
from openhands.agent_server.director.models import AutomationBundle


NAME = "Open a session of the org"
SCRIPT = "main.py"
ENTRYPOINT = f"python3 {SCRIPT}"


def bundle(settings: Mapping[str, str] | None = None) -> AutomationBundle:
    """The script as the automation service takes it: the same bytes each time.

    Args:
        settings: What the script is set to (its ``DIRECTOR_EVENT_...``
            names), kept beside it in the bundle. The automation service
            sets nothing for one automation alone, so this is where what
            differs between two automations is.
    """
    files = {SCRIPT: Path(__file__).with_name(SCRIPT).read_bytes()}
    kept = {name: value for name, value in (settings or {}).items() if value}
    if kept:
        files[SETTINGS] = json.dumps(kept, indent=2, sort_keys=True).encode()
    packed = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=packed, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as tar,
    ):
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(content))
    return AutomationBundle(
        name=NAME,
        entrypoint=ENTRYPOINT,
        tarball_base64=base64.b64encode(packed.getvalue()).decode(),
    )
