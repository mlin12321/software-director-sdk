"""The launcher the director runs agents through, while it runs any."""

import asyncio

import pytest

from openhands.agent_server.director.service.agent_launch_service import (
    AgentLaunchService,
)
from openhands.sdk.errors import Unavailable
from tests.agent_server.director.conftest import FakeLauncher


def test_no_launcher_is_there_before_the_director_runs_agents():
    launch = AgentLaunchService()

    assert not launch.is_running
    with pytest.raises(Unavailable):
        launch.require()


async def test_the_launcher_and_the_loop_are_held_from_begin_to_end():
    launch = AgentLaunchService()
    launcher = FakeLauncher()

    launch.begin(launcher)

    assert launch.is_running
    assert launch.require() is launcher
    assert launch.loop is asyncio.get_running_loop()

    launch.end()

    assert not launch.is_running and launch.loop is None
    with pytest.raises(Unavailable):
        launch.require()
