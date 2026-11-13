"""Titles and headlines a model writes for a team."""

from typing import cast

import pytest

from openhands.agent_server.director.headline import (
    MAX_HEADLINE_LENGTH,
    MAX_TITLE_LENGTH,
    TeamSummary,
    parse_summary,
    provisional_summary,
    summarize_team,
)
from openhands.sdk.llm import LLM, Message, TextContent


class ScriptedLLM:
    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.asked: list[list[Message]] = []

    def generate(self, messages: list[Message], **_: object):
        self.asked.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        message = Message(role="assistant", content=[TextContent(text=self.answer)])
        return type("Response", (), {"message": message})()


def summarize(llm: ScriptedLLM, task: str) -> TeamSummary | None:
    return summarize_team(cast(LLM, llm), task)


def test_a_team_is_first_named_after_the_first_line_of_its_task():
    task = "Build the checkout flow for the shop, with payment and tax\nand tests"

    summary = provisional_summary(task)

    assert summary.title == "Build the checkout flow for the shop, with paymen…"
    assert len(summary.title) == MAX_TITLE_LENGTH
    assert (
        summary.headline == "Build the checkout flow for the shop, with payment and tax"
    )


def test_the_models_two_lines_become_the_title_and_the_headline():
    llm = ScriptedLLM(
        "Title: Checkout Flow\nHeadline: Builds the shop's checkout with payments."
    )

    summary = summarize(llm, "Build the checkout flow")

    assert summary == TeamSummary(
        "Checkout Flow", "Builds the shop's checkout with payments."
    )
    [[system, user]] = llm.asked
    assert system.role == "system"
    assert user.content == [TextContent(text="Build the checkout flow")]


@pytest.mark.parametrize(
    "answer",
    [
        '<think>short and clear</think>\nTitle: "Checkout Flow"\n'
        "headline: `Builds the checkout.`",
        "Sure!\n\nTitle: Checkout Flow\nHeadline: Builds the checkout.\nTitle: Other",
    ],
)
def test_reasoning_quotes_and_chatter_around_the_lines_are_dropped(answer):
    assert parse_summary(answer) == TeamSummary("Checkout Flow", "Builds the checkout.")


def test_an_overlong_answer_is_cut():
    summary = parse_summary(f"Title: {'t' * 80}\nHeadline: {'h' * 300}")

    assert summary is not None
    assert len(summary.title) == MAX_TITLE_LENGTH
    assert len(summary.headline) == MAX_HEADLINE_LENGTH


@pytest.mark.parametrize(
    "answer", ["", "Checkout Flow", "Title: Checkout Flow", RuntimeError("no model")]
)
def test_nothing_comes_back_when_the_model_fails_or_leaves_a_line_out(answer):
    assert summarize(ScriptedLLM(answer), "Build the checkout flow") is None
