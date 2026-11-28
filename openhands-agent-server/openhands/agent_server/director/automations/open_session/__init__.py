"""The automation that opens a session of the org for an event."""

import base64
import gzip
import io
import tarfile
from pathlib import Path

from openhands.agent_server.director.models import AutomationBundle


NAME = "Open a session of the org"
SCRIPT = "main.py"
ENTRYPOINT = f"python3 {SCRIPT}"


def bundle() -> AutomationBundle:
    """The script as the automation service takes it: the same bytes each time."""
    script = Path(__file__).with_name(SCRIPT).read_bytes()
    packed = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=packed, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as tar,
    ):
        entry = tarfile.TarInfo(SCRIPT)
        entry.size = len(script)
        entry.mode = 0o644
        tar.addfile(entry, io.BytesIO(script))
    return AutomationBundle(
        name=NAME,
        entrypoint=ENTRYPOINT,
        tarball_base64=base64.b64encode(packed.getvalue()).decode(),
    )
