"""Tests for how the team a request goes straight to is chosen."""

from openhands.agent_server.director.routing import (
    CARDS_KEPT,
    MAX_CARDS,
    Card,
    best_by_words,
    cards_to_show,
    choose,
    lowest_holding,
    names_at_top,
    outline,
    parse_answer,
    paths_named,
    question,
)


TRACKED = [
    "README.md",
    "services/api/checkout.py",
    "services/api/db/schema.sql",
    "apps/web/cart.tsx",
    "apps/web/pages/index.tsx",
    "apps/admin/pages/index.tsx",
]
PARENTS = {
    "root": None,
    "api": "root",
    "db": "api",
    "web": "root",
}
ALL = set(PARENTS)


def decide(named=(), answers=(), worded=None, start="root", takes=ALL):
    return choose(
        start=start,
        parents=PARENTS,
        takes=takes,
        named_owners=list(named),
        answers=list(answers),
        worded=worded,
    )


def test_a_path_is_named_by_its_full_name_or_by_a_file_name_only_it_has():
    text = (
        "Fix `services/api/checkout.py`, look at ./apps/web/ and schema.sql. "
        "The api and README are fine; index.tsx is not (see docs/none.md)."
    )

    assert paths_named(text, TRACKED) == [
        "services/api/checkout.py",
        "apps/web",
        "services/api/db/schema.sql",
    ]


def test_a_teams_card_lists_what_is_at_the_top_of_its_own_directory():
    assert names_at_top("services/api", TRACKED, {"services/api/db"}) == ["checkout.py"]
    assert names_at_top("", TRACKED, {"services/api", "apps/web"}) == [
        "README.md",
        "apps/",
        "services/",
    ]


def test_the_lowest_team_that_holds_every_team_named():
    assert lowest_holding(["db"], PARENTS) == "db"
    assert lowest_holding(["db", "api"], PARENTS) == "api"
    assert lowest_holding(["db", "web"], PARENTS) == "root"
    assert lowest_holding([], PARENTS) is None


def test_an_answer_is_read_and_one_that_names_an_unknown_team_is_none():
    assert parse_answer("Teams: api, db\nReason: both", ALL) == ["api", "db"]
    assert parse_answer("Teams: none\nReason: all of it", ALL) == []
    assert parse_answer("Teams: `api`\nReason: x", ALL) == ["api"]
    assert parse_answer("Teams: payments\nReason: x", ALL) is None
    assert parse_answer("I think the API team.", ALL) is None
    assert parse_answer(None, ALL) is None


def test_two_answers_that_agree_are_sure():
    choice = decide(answers=[["db"], ["db"]])

    assert (choice.node_id, choice.by, choice.sure) == ("db", "model", True)


def test_two_answers_that_differ_leave_the_request_where_it_was_made():
    choice = decide(answers=[["db"], ["web"]])

    assert (choice.node_id, choice.by, choice.sure) == ("root", "top", False)


def test_teams_named_together_give_the_team_above_them():
    assert decide(answers=[["db", "api"], ["api", "db"]]).node_id == "api"
    across = decide(answers=[["db", "web"], ["web", "db"]])
    assert (across.node_id, across.sure) == ("root", True)


def test_no_usable_answer_leaves_the_named_paths_to_decide():
    named = decide(named=["db"], answers=[None, ["db"]])
    nothing = decide(answers=[None, None])
    unasked = decide(named=["api", "db"])

    assert (named.node_id, named.by, named.sure) == ("db", "named", True)
    assert (nothing.node_id, nothing.by, nothing.sure) == ("root", "top", False)
    assert (unasked.node_id, unasked.by) == ("api", "named")


def test_a_named_path_outside_the_team_the_model_chose_makes_it_unsure():
    beside = decide(named=["web"], answers=[["db"], ["db"]])
    above = decide(named=["api"], answers=[["db"], ["db"]])
    below = decide(named=["db"], answers=[["api"], ["api"]])

    assert (beside.node_id, beside.sure) == ("root", False)
    assert (above.node_id, above.sure) == ("root", False)
    assert (below.node_id, below.sure) == ("api", True)


def test_words_that_point_at_another_branch_make_it_unsure():
    assert decide(answers=[["db"], ["db"]], worded="web").sure is False
    assert decide(answers=[["db"], ["db"]], worded="api").sure is True
    assert decide(answers=[["api"], ["api"]], worded="db").sure is True


def test_a_team_that_cannot_take_a_request_gives_the_one_above():
    choice = decide(answers=[["db"], ["db"]], takes={"root", "api", "web"})

    assert (choice.node_id, choice.sure) == ("api", True)


def test_the_choice_never_leaves_what_is_below_where_the_request_was_made():
    choice = decide(answers=[["web"], ["web"]], start="api")

    assert (choice.node_id, choice.by, choice.sure) == ("api", "top", False)


def card(node_id: str, parent: str | None, **rest) -> Card:
    return Card(node_id, parent, rest.pop("title", node_id), **rest)


CARDS = [
    card("root", None, title="Shop", directory=""),
    card("api", "root", directory="services/api", purpose="Serve the checkout"),
    card("db", "api", directory="services/api/db", names=["schema.sql"]),
    card("web", "root", directory="apps/web", took=["Make the cart page faster"]),
]


def test_the_teams_are_shown_as_a_tree_in_either_order():
    plain = outline(CARDS)
    turned = outline(CARDS, turned=True)

    assert plain.index("id: api") < plain.index("  - id: db") < plain.index("id: web")
    assert turned.index("id: web") < turned.index("id: api")
    assert "    holds: schema.sql" in plain
    assert "took before: Make the cart page faster" in plain
    _, asked = question(
        "Speed up the cart", CARDS, ["- x (not for 'API')"], turned=True
    )
    assert asked.endswith("The request:\nSpeed up the cart")
    assert "sent to the wrong team before:\n- x (not for 'API')" in asked


def test_the_words_of_a_request_name_a_team_only_when_one_is_clearly_ahead():
    assert best_by_words("the schema.sql migration", CARDS) == "db"
    assert best_by_words("something unrelated entirely", CARDS) is None
    # `services` is on two cards alike.
    assert best_by_words("the services", CARDS) is None


def test_a_large_org_shows_the_model_the_likeliest_teams_and_those_above():
    many = [card("root", None, directory="")] + [
        card(f"t{n}", "root", directory=f"pkg/t{n}", purpose=f"area{n}")
        for n in range(MAX_CARDS + 10)
    ]
    many.append(card("deep", "t3", directory="pkg/t3/billing", purpose="invoices"))

    shown = [c.node_id for c in cards_to_show("fix the invoices", many)]

    assert {"root", "t3", "deep"} <= set(shown)
    assert len(shown) <= CARDS_KEPT * 2 + 1
    assert len(cards_to_show("fix the invoices", CARDS)) == len(CARDS)


def test_sub_teams_do_not_crowd_out_the_team_beside_theirs():
    """A team's sub-teams share its words; two teams a request names are
    both shown before the sub-teams of either."""
    services = ["gateway", "console"]
    many = [Card("root", None, "Shop", "")]
    for service in services:
        many.append(Card(service, "root", service, f"pay/{service}"))
        many += [
            Card(f"{service}.{n}", service, f"part {n}", f"pay/{service}/part{n}")
            for n in range(MAX_CARDS)
        ]

    shown = {c.node_id for c in cards_to_show("gateway and console take cards", many)}

    assert {"gateway", "console"} <= shown


def test_a_request_naming_a_file_of_a_team_outside_stays_where_it_was_made():
    # Made at API, it names a file of Web: the model's pick of DB is not taken.
    below_api = {"api", "db"}
    choice = decide(
        named=["web"], answers=[["db"], ["db"]], start="api", takes=below_api
    )

    assert (choice.node_id, choice.sure) == ("api", False)
    assert "outside" in choice.reason
    # One that names only what is below goes down as before.
    assert (
        decide(
            named=["db"], answers=[["db"], ["db"]], start="api", takes=below_api
        ).node_id
        == "db"
    )
