"""When an agent's reports are rewritten shorter, and what may replace them."""

from pathlib import Path

from openhands.agent_server.director import compaction
from openhands.agent_server.director.models import (
    CreateOrgRequest,
    DirectorSettingsUpdate,
    Report,
)
from openhands.agent_server.director.service import DirectorService
from tests.agent_server.director.conftest import FakeLauncher, root_conversation


def reports(*lengths: int, compacted: int = 0) -> list[Report]:
    return [
        Report(
            at=f"2026-11-0{number + 1}T00:00:00+00:00",
            role="ic",
            text="x" * length,
            compacted=number < compacted,
        )
        for number, length in enumerate(lengths)
    ]


def test_reports_are_due_once_they_pass_the_limit():
    assert not compaction.due(reports(500, 500), 1_000)
    assert compaction.due(reports(500, 501), 1_000)


def test_reports_just_rewritten_are_not_due_until_enough_is_new():
    assert not compaction.due(reports(1_100, 100, compacted=1), 1_000)
    assert compaction.due(reports(1_100, 250, compacted=1), 1_000)


def test_the_rewrite_must_be_shorter_than_what_it_replaces():
    given = reports(40, 60)

    assert compaction.compacted(" " + "y" * 100, given, "now") is None
    assert compaction.compacted("  ", given, "now") is None
    assert compaction.compacted(" Short. ", given, "now") == [
        Report(at="now", role="ic", text="Short.", compacted=True)
    ]


async def reporting_service(repo: Path, launcher: FakeLauncher) -> DirectorService:
    service = DirectorService(repo, save_delay=0, run_limit_retry=0)
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    service.update_settings(DirectorSettingsUpdate(report_compaction_chars=1_000))
    await service.settled()
    return service


async def test_long_reports_are_rewritten_by_a_model_once(repo: Path):
    launcher = FakeLauncher(answers={compaction.USAGE_ID: "Chose SQLite; ORM failed."})
    service = await reporting_service(repo, launcher)
    try:
        root = service.store.get("root")
        assert root is not None
        service.report(root, "ic", "a" * 600)
        service.report(root, "manager", "m" * 600)
        await service.settled()
        asked_before = len(launcher.asked)
        service.report(root, "ic", "b" * 600)
        await service.settled()
        service.report(root, "ic", "Then this.")
        await service.settled()
        kept = service.reports_of(root)
    finally:
        await service.aclose()

    assert asked_before == 0
    assert [usage_id for _, usage_id, _, _ in launcher.asked] == [compaction.USAGE_ID]
    assert [(report.role, report.text, report.compacted) for report in kept] == [
        ("manager", "m" * 600, False),
        ("ic", "Chose SQLite; ORM failed.", True),
        ("ic", "Then this.", False),
    ]


async def test_an_answer_that_is_no_shorter_leaves_the_reports_as_they_are(
    repo: Path,
):
    launcher = FakeLauncher(answers={compaction.USAGE_ID: "z" * 5_000})
    service = await reporting_service(repo, launcher)
    try:
        root = service.store.get("root")
        assert root is not None
        service.report(root, "ic", "a" * 1_200)
        await service.settled()
        kept = service.reports_of(root)
    finally:
        await service.aclose()

    assert len(launcher.asked) == 1
    assert [report.text for report in kept] == ["a" * 1_200]
