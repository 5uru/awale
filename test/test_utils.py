import pytest
import jax.numpy as jnp
from jax import random
import chex
from typing import NamedTuple

from AwaleEnv.utils import (
    State,
    distribute_seeds,
    capture_seeds,
    update_game_state,
    get_action_space,
)


@pytest.fixture
def initial_state() -> State:
    """Create a fresh game state for testing"""
    return State(
        board=jnp.array([4] * 12, dtype=jnp.int8),
        action_space=jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int8),
        key=random.PRNGKey(0),
        score=jnp.array([0, 0], dtype=jnp.int8),
        current_player=jnp.int8(0),
    )


def test_initial_state(initial_state):
    assert (initial_state.board == jnp.array([4] * 12, dtype=jnp.int8)).all()
    assert (initial_state.score == jnp.array([0, 0], dtype=jnp.int8)).all()
    assert initial_state.current_player == 0
    assert (
        initial_state.action_space == jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int8)
    ).all()


def test_distribute_seeds():
    board = jnp.array([4] * 12, dtype=jnp.int8)
    current_pit = jnp.int8(0)
    seeds = board[current_pit]

    new_board, last_pit = distribute_seeds(board, current_pit, seeds)
    expected = jnp.array([0, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int8)
    assert (new_board == expected).all()
    assert last_pit == 3


def test_capture_seeds():
    board = jnp.array([4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 4, 4], dtype=jnp.int8)
    last_pit = jnp.int8(9)
    current_player = jnp.int8(0)

    new_board, captured = capture_seeds(board, last_pit, current_player)
    expected = jnp.array([4, 4, 4, 4, 4, 3, 0, 0, 0, 0, 4, 4], dtype=jnp.int8)
    assert (new_board == expected).all()
    assert captured == 9


def test_update_game_state_score_end():
    board = jnp.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=jnp.int8)
    scores = jnp.array([24, 23], dtype=jnp.int8)
    current_player = jnp.int8(0)

    (
        new_board,
        new_scores,
        new_player,
        done,
        reward,
        winner,
        action_space,
    ) = update_game_state(board, scores, current_player)
    assert done
    assert reward == 100  # Winner reward
    assert winner == 0


def test_update_game_state_empty_side():
    board = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int8)
    scores = jnp.array([10, 10], dtype=jnp.int8)
    current_player = jnp.int8(0)

    (
        new_board,
        new_scores,
        new_player,
        done,
        reward,
        winner,
        action_space,
    ) = update_game_state(board, scores, current_player)
    assert done
    assert (new_board == jnp.zeros_like(board)).all()
    assert new_scores[1] > new_scores[0]  # Player 1 should win


def test_get_action_space():
    player_0_actions = get_action_space(jnp.int8(0))
    player_1_actions = get_action_space(jnp.int8(1))

    assert (player_0_actions == jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int8)).all()
    assert (player_1_actions == jnp.array([6, 7, 8, 9, 10, 11], dtype=jnp.int8)).all()


def test_switch_player():
    board = jnp.array([4] * 12, dtype=jnp.int8)
    scores = jnp.array([0, 0], dtype=jnp.int8)
    current_player = jnp.int8(0)

    (
        new_board,
        new_scores,
        new_player,
        done,
        reward,
        winner,
        action_space,
    ) = update_game_state(board, scores, current_player)
    assert new_player == 1
    assert not done
    assert reward == 0
    assert (action_space == jnp.array([6, 7, 8, 9, 10, 11], dtype=jnp.int8)).all()


# Helper function to check array equality with proper type checking
def check_array_equal(arr1, arr2):
    return (
        isinstance(arr1, chex.Array)
        and isinstance(arr2, chex.Array)
        and (arr1 == arr2).all()
    )
