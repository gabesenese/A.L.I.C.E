"""A sentence that mentions a project is not a goal.

Read from the live goal stack, nine "active goals" were recorded — and six of
them were raw utterances:

    finish the routing refactor                            <- a goal
    continue our project roadmap                           <- a goal
    build automation plan                                  <- a goal
    alright lets focus on my ai project
    ready to work on our ai project
    learn something simple to understand how it works
    I've been stuck on this project for a while
    honestly this whole project has gotten away from me
    danger delete all project files                        <- a safety test string

They arrive through a keyword gate. CompanionRuntimeLoop._extract_project_hints
matches any sentence containing project / feature / milestone / roadmap / repo /
build / test suite / automation and keeps the fragment verbatim; those hints are
merged into memory_domains.projects; contract_pipeline merges *that* into
active_goals; and sync_from_active_goals stores whatever it is handed. The loop
closes on itself — a hint becomes a goal becomes a hint.

The damage is user-visible on every turn. llm_engine._build_companion_context
injects the top three as "Active goals:" into the system prompt, so Alice spends
each turn being told that one of Gabriel's goals is "danger delete all project
files", and another is that he is stuck and losing heart. Frustration he voiced
once is re-read back to her as an objective, forever.

This pins the gate at the one door all of it comes through.
"""

import pytest

from ai.goals.goal_engine import GoalEngine

# Verbatim from data/goals/goal_stack.json.
REAL_GOALS = [
    "finish the routing refactor",
    "continue our project roadmap",
    "build automation plan",
]

REAL_JUNK = [
    "alright lets focus on my ai project",
    "ready to work on our ai project",
    "learn something simple to understand how it works",
    "I've been stuck on this project for a while",
    "honestly this whole project has gotten away from me",
    "danger delete all project files",
]


@pytest.fixture
def engine(tmp_path):
    return GoalEngine(goals_file=tmp_path / "goal_stack.json")


# -- the gate -----------------------------------------------------------------


@pytest.mark.parametrize("description", REAL_GOALS)
def test_a_real_goal_is_kept(engine, description):
    engine.sync_from_active_goals([description])
    assert [g.description for g in engine.active()] == [description]


@pytest.mark.parametrize("description", REAL_JUNK)
def test_an_utterance_that_merely_mentions_a_project_is_not_stored(engine, description):
    engine.sync_from_active_goals([description])
    assert engine.active() == [], f"stored as a goal: {description!r}"


def test_the_live_stack_would_keep_three_of_nine(engine):
    engine.sync_from_active_goals(REAL_GOALS + REAL_JUNK)
    assert sorted(g.description for g in engine.active()) == sorted(REAL_GOALS)


# -- what reaches the prompt --------------------------------------------------


def test_junk_already_on_disk_does_not_reach_the_prompt(engine):
    """Filtering at the door is not enough on its own: nine of these are already
    stored, and data/goals/goal_stack.json is the user's file, not mine to
    rewrite. So the read is filtered too."""
    engine.sync_from_active_goals(REAL_GOALS)
    for description in REAL_JUNK:
        engine._goals.append(type(engine._goals[0])(description=description))

    surfaced = [g.description for g in engine.active()]
    assert sorted(surfaced) == sorted(REAL_GOALS), surfaced


def test_the_session_summary_is_filtered_too(engine):
    engine.sync_from_active_goals(REAL_GOALS)
    engine._goals.append(type(engine._goals[0])(description="honestly this whole project has gotten away from me"))
    assert "gotten away from me" not in engine.session_summary()


# -- the shape it is testing for ----------------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "ship the memory migration",
        "fix the schema drift in the notes table",
        "write the trust tier documentation",
        "refactor the routing layer",
        "migrate embeddings off pickle",
    ],
)
def test_an_action_phrase_is_a_goal(engine, description):
    engine.sync_from_active_goals([description])
    assert [g.description for g in engine.active()] == [description]


@pytest.mark.parametrize(
    "description",
    [
        "I'm worried about the repo",
        "this feature is confusing",
        "so anyway the build broke again",
        "do something about it",
        "work on stuff",
        "maybe we should look at the roadmap at some point",
    ],
)
def test_commentary_is_not_a_goal(engine, description):
    engine.sync_from_active_goals([description])
    assert engine.active() == [], f"stored as a goal: {description!r}"


# -- the explicit paths must keep working -------------------------------------


def test_an_explicitly_stated_goal_still_lands(engine):
    """extract_from_text already required a real intent phrase; nothing here
    should narrow that."""
    extracted = engine.extract_from_text("I want to finish the routing refactor this week")
    assert extracted
    assert "routing refactor" in extracted


def test_adding_a_goal_directly_is_unaffected(engine):
    """A goal Gabriel states outright is his to phrase however he likes — the
    gate is for text the extractor guessed at, not for an explicit request."""
    goal = engine.add("alright lets focus on my ai project")
    assert goal is not None
    assert goal.description in [g.description for g in engine.active()]
