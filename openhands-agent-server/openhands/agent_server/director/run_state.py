"""What the director is in the middle of, kept beside the org's repository.

The org tree is in git. What changes too fast for commits, or matters only
to the machine that runs the agents, is here: which agents were running,
which teams wait for a run, the steps the director had begun and the
messages it had not delivered yet. Every change is written as it happens,
so a server that stops at any point can take its work up again.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openhands.sdk.errors import DomainError


SCHEMA_VERSION = 1

# Marks: one row per (kind, key), kept in the order they were first made.
RUNNING = "running"  # conversation id -> node id
WAITING = "waiting"  # node id, waiting for the server to have room
HELD = "held"  # node id, its review request waits for the teams beside it
TEAM = "team"  # node id -> what the team was made of at the last save
TOLD = "told"  # conversation id -> the context its agent was last told

_TABLES = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS marks (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    UNIQUE (kind, key)
);
CREATE TABLE IF NOT EXISTS intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    node_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    steps TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    text TEXT NOT NULL
);
"""


class RunStateError(DomainError):
    """The run state on disk cannot be used."""


@dataclass(frozen=True)
class Intent:
    """Something of several steps the director began for a team."""

    id: int
    kind: str
    node_id: str
    payload: dict[str, Any]
    steps: frozenset[str]


@dataclass(frozen=True)
class HeldMessage:
    """A message to an agent that has not been delivered."""

    seq: int
    conversation_id: str
    node_id: str
    text: str


class RunState:
    """A small database of the director's unfinished work. Thread-safe."""

    def __init__(self, path: Path | None) -> None:
        """
        Args:
            path: The file to keep it in; None to keep it in memory only.

        Raises:
            RunStateError: The file was written by a newer director.
        """
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(
            str(path) if path else ":memory:",
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_TABLES)
        version = self.get("schema_version")
        if version is None:
            self.put("schema_version", str(SCHEMA_VERSION))
        elif version != str(SCHEMA_VERSION):
            self._db.close()
            raise RunStateError(
                f"{path} has run state of version {version}; this director "
                f"reads version {SCHEMA_VERSION}"
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def get(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key = ?", (key,))
        return row[0] if row else None

    def put(self, key: str, value: str | None) -> None:
        if value is None:
            self._run("DELETE FROM meta WHERE key = ?", (key,))
        else:
            self._run(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def mark(self, kind: str, key: str, value: str = "") -> None:
        self._run(
            "INSERT INTO marks (kind, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT (kind, key) DO UPDATE SET value = excluded.value",
            (kind, key, value),
        )

    def unmark(self, kind: str, key: str) -> None:
        self._run("DELETE FROM marks WHERE kind = ? AND key = ?", (kind, key))

    def marked(self, kind: str) -> dict[str, str]:
        """The marks of one kind, oldest first."""
        return dict(
            self._all(
                "SELECT key, value FROM marks WHERE kind = ? ORDER BY seq", (kind,)
            )
        )

    def replace_marks(self, kind: str, marks: dict[str, str]) -> None:
        with self._lock:
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM marks WHERE kind = ?", (kind,))
            self._db.executemany(
                "INSERT INTO marks (kind, key, value) VALUES (?, ?, ?)",
                [(kind, key, value) for key, value in marks.items()],
            )
            self._db.execute("COMMIT")

    def begin(self, kind: str, node_id: str, payload: dict[str, Any]) -> Intent:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO intents (kind, node_id, payload, steps) "
                "VALUES (?, ?, ?, '[]')",
                (kind, node_id, json.dumps(payload)),
            )
            intent_id = cursor.lastrowid
        assert intent_id is not None
        return Intent(intent_id, kind, node_id, payload, frozenset())

    def step_done(self, intent: Intent, step: str, **learned: Any) -> Intent:
        """Record that a step of ``intent`` is done, with what it found out."""
        changed = Intent(
            intent.id,
            intent.kind,
            intent.node_id,
            {**intent.payload, **learned},
            intent.steps | {step},
        )
        self._run(
            "UPDATE intents SET payload = ?, steps = ? WHERE id = ?",
            (json.dumps(changed.payload), json.dumps(sorted(changed.steps)), intent.id),
        )
        return changed

    def finish(self, intent_id: int) -> None:
        """Forget an intent: it was carried out, or will never be."""
        self._run("DELETE FROM intents WHERE id = ?", (intent_id,))

    def open_intents(self) -> list[Intent]:
        """What was begun and not finished, oldest first."""
        return [
            Intent(
                intent_id,
                kind,
                node_id,
                json.loads(payload),
                frozenset(json.loads(steps)),
            )
            for intent_id, kind, node_id, payload, steps in self._all(
                "SELECT id, kind, node_id, payload, steps FROM intents ORDER BY id"
            )
        ]

    def hold(self, conversation_id: str, node_id: str, text: str) -> int:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO outbox (conversation_id, node_id, text) VALUES (?, ?, ?)",
                (conversation_id, node_id, text),
            )
            seq = cursor.lastrowid
        assert seq is not None
        return seq

    def delivered(self, seq: int) -> None:
        """Forget a message: it was delivered, or will never be."""
        self._run("DELETE FROM outbox WHERE seq = ?", (seq,))

    def is_held(self, seq: int) -> bool:
        return self._one("SELECT 1 FROM outbox WHERE seq = ?", (seq,)) is not None

    def held(self) -> list[HeldMessage]:
        """The messages not delivered yet, oldest first."""
        return [
            HeldMessage(*row)
            for row in self._all(
                "SELECT seq, conversation_id, node_id, text FROM outbox ORDER BY seq"
            )
        ]

    def forget(self) -> None:
        """Drop everything but the schema version."""
        with self._lock:
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM marks")
            self._db.execute("DELETE FROM intents")
            self._db.execute("DELETE FROM outbox")
            self._db.execute("DELETE FROM meta WHERE key != 'schema_version'")
            self._db.execute("COMMIT")

    def _run(self, statement: str, values: tuple[Any, ...]) -> None:
        with self._lock:
            self._db.execute(statement, values)

    def _one(self, statement: str, values: tuple[Any, ...]) -> tuple[Any, ...] | None:
        with self._lock:
            return self._db.execute(statement, values).fetchone()

    def _all(
        self, statement: str, values: tuple[Any, ...] = ()
    ) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._db.execute(statement, values).fetchall()
