"""What the director keeps between starts of the server: one file beside
the server's conversations, never in the org's repository."""

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from openhands.sdk.errors import Unavailable
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# The shape of the file. A file with no version is of the first.
REMEMBERED_VERSION = 1
_VERSION_KEY = "schema_version"


@dataclass
class RememberedFile:
    """A few values by name, kept in one JSON file.

    The file also holds the key the tokens of the org's agents are signed
    with, so it is written in place and never replaced: an agent's shell
    is kept from that one file.
    """

    path: Path | None
    """Where it is kept; None to keep nothing."""
    # Held to read it too: it is written in place.
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def read(self) -> dict[str, object]:
        """Everything that is kept; nothing when the file cannot be read.

        Raises:
            Unavailable: The file was written by a director of another
                version.
        """
        if self.path is None:
            return {}
        try:
            with self._lock:
                kept = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(kept, dict):
            return {}
        version = kept.get(_VERSION_KEY, REMEMBERED_VERSION)
        if version != REMEMBERED_VERSION:
            raise Unavailable(
                f"{self.path} was written by a director of version "
                f"{version}; this one reads version {REMEMBERED_VERSION}"
            )
        return kept

    def keep(self, **changes: object) -> None:
        """Keep ``changes`` for the next start of the server. A file that
        cannot be written, or is of another version, is left as it is."""
        if self.path is None:
            return
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # One that cannot be read is left as it is: writing over
                # it would lose the key of the tool tokens.
                before = json.loads(self.path.read_text()) if self.path.exists() else {}
                written = before.get(_VERSION_KEY, REMEMBERED_VERSION)
                if written != REMEMBERED_VERSION:
                    logger.error(
                        "%s is of version %s and is not written over",
                        self.path,
                        written,
                    )
                    return
                versioned = {_VERSION_KEY: REMEMBERED_VERSION}
                kept = json.dumps({**versioned, **before, **changes}) + "\n"
                self.path.touch(mode=0o600)
                self.path.write_text(kept)
            except (OSError, ValueError, TypeError):
                logger.exception("Could not write %s", self.path)
