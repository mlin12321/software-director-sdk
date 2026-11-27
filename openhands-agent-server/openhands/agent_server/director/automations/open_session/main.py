"""Open a session of the org for an event an automation received.

Upload this file as a script automation (entrypoint ``python3 main.py``) and
give it any trigger: a webhook source (a ticket filed, an alert) or a
schedule. It knows nothing of where the event comes from. It says what is
asked in words, posts that with the event to the director
(``POST /api/director/org/events``), and reports how it went to the
automation's run. It uses the standard library only.

What the automation service gives it:

- ``AUTOMATION_EVENT_PAYLOAD``: the event, as JSON. None for a schedule.
- ``AGENT_SERVER_URL`` and ``SESSION_API_KEY`` (or ``OH_SESSION_API_KEYS_0``).
- ``AUTOMATION_CALLBACK_URL``, ``AUTOMATION_CALLBACK_API_KEY`` and
  ``AUTOMATION_RUN_ID``, to report to.
- ``OPENHANDS_OBSERVABILITY_METADATA``: the automation's id and name.

What you set on the automation, each a template in which ``{a.b.c}`` is that
field of the event (``{issue.key}``, ``{items.0.name}``) and a field the
event lacks is left empty:

- ``DIRECTOR_EVENT_TEXT``: what is asked of the org. Required for a
  schedule; for an event it defaults to asking the org to handle it.
- ``DIRECTOR_EVENT_TITLE``: what to call the session.
- ``DIRECTOR_EVENT_KEY``: names what happened, so that an event delivered
  twice opens one session (``{issue.key}``).
- ``DIRECTOR_EVENT_PATH``: the directory of the repository it is about.
- ``DIRECTOR_EVENT_SOURCE``: what sent it, in words ("Jira").
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any


EVENTS_ROUTE = "/api/director/org/events"
_FIELD = re.compile(r"\{([\w.\-]+)\}")
_TIMEOUT_SECONDS = 120.0


def field_of(event: Any, path: str) -> str:
    """The field of ``event`` that ``path`` names with dots; '' when the
    event has none."""
    found = event
    for part in path.split("."):
        if isinstance(found, Mapping):
            found = found.get(part)
        elif isinstance(found, list) and part.isdigit() and int(part) < len(found):
            found = found[int(part)]
        else:
            return ""
    if found is None:
        return ""
    return found if isinstance(found, str) else json.dumps(found)


def filled(template: str, event: Any) -> str:
    """``template`` with each ``{a.b}`` replaced by that field of ``event``."""
    return _FIELD.sub(lambda named: field_of(event, named.group(1)), template).strip()


def request_for(env: Mapping[str, str]) -> dict[str, Any]:
    """What to ask of the org for the event in ``env``.

    Raises:
        ValueError: The event is not JSON, or nothing says what to ask.
    """
    raw = env.get("AUTOMATION_EVENT_PAYLOAD", "").strip()
    event = json.loads(raw) if raw else None
    try:
        about = json.loads(env.get("OPENHANDS_OBSERVABILITY_METADATA", "") or "{}")
    except ValueError:
        about = {}
    if not isinstance(about, Mapping):
        about = {}
    name = env.get("DIRECTOR_EVENT_SOURCE", "") or str(
        about.get("automation.name") or ""
    )
    text = filled(env.get("DIRECTOR_EVENT_TEXT", ""), event)
    if not text:
        if event is None:
            raise ValueError("DIRECTOR_EVENT_TEXT must say what to ask of the org")
        text = (
            f"Handle this event from {name or 'an automation'}: work out what "
            "it asks for from what was received, and do it."
        )
    return {
        "text": text,
        "title": filled(env.get("DIRECTOR_EVENT_TITLE", ""), event),
        "path": env.get("DIRECTOR_EVENT_PATH") or None,
        "payload": event if isinstance(event, Mapping) else None,
        "source": {
            "name": name,
            "automation_id": str(about.get("automation.id") or ""),
            "run_id": env.get("AUTOMATION_RUN_ID", "")
            or str(about.get("automation.run_id") or ""),
            "event_key": filled(env.get("DIRECTOR_EVENT_KEY", ""), event),
        },
    }


def _post(url: str, body: Mapping[str, Any], headers: Mapping[str, str]) -> Any:
    sent = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(sent, timeout=_TIMEOUT_SECONDS) as answer:
        return json.loads(answer.read() or b"null")


def open_session(env: Mapping[str, str]) -> dict[str, Any]:
    """Post the event in ``env`` to the director.

    Returns:
        The session it opened, or had opened for the same event already.

    Raises:
        ValueError: ``env`` does not say where the server is or what to ask.
        RuntimeError: The director refused; its reason is the message.
    """
    server = env.get("AGENT_SERVER_URL", "").rstrip("/")
    if not server:
        raise ValueError("AGENT_SERVER_URL is not set")
    key = env.get("SESSION_API_KEY") or env.get("OH_SESSION_API_KEYS_0") or ""
    try:
        return _post(
            server + EVENTS_ROUTE,
            request_for(env),
            {"X-Session-API-Key": key} if key else {},
        )
    except urllib.error.HTTPError as refused:
        said = refused.read().decode(errors="replace")
        try:
            said = json.loads(said).get("detail") or said
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"The director refused ({refused.code}): {said}") from None


def report(
    env: Mapping[str, str], opened: Mapping[str, Any] | None, error: str
) -> None:
    """Tell the automation's run how it went; nothing where no run asks."""
    url = env.get("AUTOMATION_CALLBACK_URL")
    if not url:
        return
    body: dict[str, Any] = {"status": "FAILED" if error else "COMPLETED"}
    if env.get("AUTOMATION_RUN_ID"):
        body["run_id"] = env["AUTOMATION_RUN_ID"]
    if opened and opened.get("conversation_id"):
        body["conversation_id"] = opened["conversation_id"]
    if error:
        body["error"] = {
            "source": "environment",
            "code": "EventRefused",
            "detail": error,
        }
    key = env.get("AUTOMATION_CALLBACK_API_KEY")
    try:
        _post(url, body, {"Authorization": f"Bearer {key}"} if key else {})
    except Exception as failed:
        print(f"The run could not be told: {failed}", file=sys.stderr)


def main(env: Mapping[str, str] = os.environ) -> int:
    opened, error = None, ""
    try:
        opened = open_session(env)
    except Exception as failed:
        error = str(failed)
    report(env, opened, error)
    if error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(opened))
    return 0


if __name__ == "__main__":
    sys.exit(main())
