"""The one file the director keeps between starts of the server."""

import json
from pathlib import Path

import pytest

from openhands.agent_server.director.remembered import RememberedFile
from openhands.sdk.errors import Unavailable


def test_what_is_kept_is_read_again_and_earlier_values_stay(tmp_path: Path):
    remembered = RememberedFile(tmp_path / "kept" / "director.json")

    remembered.keep(repo_path="/repo")
    remembered.keep(instance="one")

    assert RememberedFile(remembered.path).read() == {
        "schema_version": 1,
        "repo_path": "/repo",
        "instance": "one",
    }


def test_nothing_is_kept_without_a_file():
    remembered = RememberedFile(None)

    remembered.keep(note="lost")

    assert remembered.read() == {}


def test_a_file_that_cannot_be_read_holds_nothing_and_is_not_written_over(
    tmp_path: Path,
):
    file = tmp_path / "director.json"
    file.write_text("{cut off")
    remembered = RememberedFile(file)

    remembered.keep(note="again")

    assert remembered.read() == {}
    assert file.read_text() == "{cut off"


def test_a_file_of_another_version_is_refused_and_left_as_it_is(tmp_path: Path):
    file = tmp_path / "director.json"
    newer = json.dumps({"schema_version": 99, "tool_key": "k"})
    file.write_text(newer)
    remembered = RememberedFile(file)

    remembered.keep(note="again")

    assert file.read_text() == newer
    with pytest.raises(Unavailable):
        remembered.read()


def test_it_stays_the_same_file_when_it_is_written_again(tmp_path: Path):
    file = tmp_path / "director.json"
    remembered = RememberedFile(file)
    remembered.keep(note="first")
    before = file.stat().st_ino

    remembered.keep(note="second")

    assert file.stat().st_ino == before
    assert file.stat().st_mode & 0o777 == 0o600
