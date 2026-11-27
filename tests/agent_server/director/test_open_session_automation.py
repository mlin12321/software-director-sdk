"""Tests for the script an automation runs to open a session for an event."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from openhands.agent_server.director.automations.open_session.main import (
    EVENTS_ROUTE,
    field_of,
    filled,
    main,
    request_for,
)


TICKET = {
    "issue": {
        "key": "SHOP-7",
        "fields": {"summary": "Orders are slow", "labels": ["db", "perf"]},
    }
}


class _Received(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"null")
        server: Any = self.server
        headers = {name.lower(): value for name, value in self.headers.items()}
        server.received.append((self.path, headers, body))
        status, answer = server.answers.get(self.path, (200, {}))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(answer).encode())

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[Any]:
    """Stands for the agent server and for the automation's run."""
    found: Any = HTTPServer(("127.0.0.1", 0), _Received)
    found.received = []
    found.answers = {EVENTS_ROUTE: (200, {"session_id": "s1", "conversation_id": "c1"})}
    found.url = f"http://127.0.0.1:{found.server_port}"
    thread = threading.Thread(target=found.serve_forever, daemon=True)
    thread.start()
    yield found
    found.shutdown()
    found.server_close()
    thread.join()


def env_for(server: Any, **more: str) -> dict[str, str]:
    return {
        "AGENT_SERVER_URL": server.url + "/",
        "SESSION_API_KEY": "the-key",
        "AUTOMATION_CALLBACK_URL": server.url + "/callback",
        "AUTOMATION_CALLBACK_API_KEY": "run-key",
        "AUTOMATION_RUN_ID": "run-1",
        "AUTOMATION_EVENT_PAYLOAD": json.dumps(TICKET),
        **more,
    }


def test_a_template_takes_fields_of_the_event_and_leaves_missing_ones_empty():
    assert field_of(TICKET, "issue.fields.labels.1") == "perf"
    assert field_of(TICKET, "issue.fields.labels") == '["db", "perf"]'
    assert field_of(TICKET, "issue.nothing.here") == ""
    assert (
        filled("{issue.key}: {issue.fields.summary} {issue.assignee.name}", TICKET)
        == "SHOP-7: Orders are slow"
    )


def test_the_request_says_what_is_asked_and_names_the_event():
    request = request_for(
        {
            "AUTOMATION_EVENT_PAYLOAD": json.dumps(TICKET),
            "AUTOMATION_RUN_ID": "run-1",
            "OPENHANDS_OBSERVABILITY_METADATA": json.dumps(
                {"automation.id": "a1", "automation.name": "Tickets"}
            ),
            "DIRECTOR_EVENT_TEXT": "Fix {issue.key}: {issue.fields.summary}",
            "DIRECTOR_EVENT_TITLE": "{issue.key}",
            "DIRECTOR_EVENT_KEY": "jira-{issue.key}",
            "DIRECTOR_EVENT_PATH": "services/api",
        }
    )

    assert request == {
        "text": "Fix SHOP-7: Orders are slow",
        "title": "SHOP-7",
        "path": "services/api",
        "payload": TICKET,
        "source": {
            "name": "Tickets",
            "automation_id": "a1",
            "run_id": "run-1",
            "event_key": "jira-SHOP-7",
        },
    }


def test_an_event_nothing_is_said_about_asks_the_org_to_handle_it():
    request = request_for(
        {
            "AUTOMATION_EVENT_PAYLOAD": json.dumps(TICKET),
            "DIRECTOR_EVENT_SOURCE": "Jira",
        }
    )

    assert request["text"].startswith("Handle this event from Jira")
    assert request["source"]["event_key"] == ""


def test_a_schedule_must_say_what_to_ask():
    with pytest.raises(ValueError, match="DIRECTOR_EVENT_TEXT"):
        request_for({})

    assert request_for({"DIRECTOR_EVENT_TEXT": "Update the dependencies."}) == {
        "text": "Update the dependencies.",
        "title": "",
        "path": None,
        "payload": None,
        "source": {"name": "", "automation_id": "", "run_id": "", "event_key": ""},
    }


def test_it_posts_the_event_to_the_director_and_tells_the_run(server, capsys):
    done = main(env_for(server, DIRECTOR_EVENT_KEY="{issue.key}"))

    assert done == 0
    (path, headers, body), (told, told_headers, said) = server.received
    assert path == EVENTS_ROUTE
    assert headers["x-session-api-key"] == "the-key"
    assert (body["payload"], body["source"]["event_key"]) == (TICKET, "SHOP-7")
    assert told == "/callback"
    assert told_headers["authorization"] == "Bearer run-key"
    assert said == {"status": "COMPLETED", "run_id": "run-1", "conversation_id": "c1"}
    assert json.loads(capsys.readouterr().out)["session_id"] == "s1"


def test_a_refusal_fails_the_run_with_the_directors_reason(server, capsys):
    server.answers[EVENTS_ROUTE] = (503, {"detail": "The org is paused"})

    done = main(env_for(server))

    assert done == 1
    _, (_, _, said) = server.received
    assert said["status"] == "FAILED"
    assert said["error"]["detail"] == "The director refused (503): The org is paused"
    assert "The org is paused" in capsys.readouterr().err


def test_it_fails_without_a_server_to_post_to(capsys):
    assert main({"AUTOMATION_EVENT_PAYLOAD": json.dumps(TICKET)}) == 1
    assert "AGENT_SERVER_URL" in capsys.readouterr().err
