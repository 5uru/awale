import pytest
from awale.utils import (
    distribute_seeds,
    get_action_space,
    determine_game_over,
    calculate_reward,
)

from jax import numpy as jnp


def test_distribute_seeds():
    board = jnp.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int8)
    new_board, captured_seeds = distribute_seeds(board, 0)
    assert new_board.shape == (12,)
    assert isinstance(captured_seeds, int)


def test_determine_game_over():
    board = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int8)
    scores = jnp.array([25, 23], dtype=jnp.int8)
    done, winner, reason = determine_game_over(board, scores)
    assert done is True
    assert winner == 0
    assert reason == "Joueur 1 a capturé la majorité des graines"


def test_calculate_reward():
    board = jnp.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int8)
    captured_seeds = 2
    reward = calculate_reward(board, captured_seeds, 0, game_over=False)
    assert reward.dtype == jnp.float16


def test_get_action_space():
    board = jnp.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int8)
    action_space = get_action_space(board, 0)
    assert action_space.shape == (12,)
    assert all(action in range(6) for action in action_space)


def test_get_action_space_player_0():
    board = jnp.array([4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0], dtype=jnp.int8)
    action_space = get_action_space(board, 0)
    assert action_space.shape == (12,)
    assert all(
        action in [0, 1, 2, 3, 4, 5]
        for action in range(12)
        if action_space[action] == 1
    )


def test_get_action_space_player_1():
    board = jnp.array([0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 4, 4], dtype=jnp.int8)
    action_space = get_action_space(board, 1)
    assert action_space.shape == (12,)
    assert all(
        action in [6, 7, 8, 9, 10, 11]
        for action in range(12)
        if action_space[action] == 1
    )


def test_get_action_space_opponent_empty():
    board = jnp.array([4, 4, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0], dtype=jnp.int8)
    action_space = get_action_space(board, 0)
    assert action_space.shape == (12,)
    assert all(
        action in [0, 1, 2, 3, 4, 5]
        for action in range(12)
        if action_space[action] == 1
    )


def test_get_action_space_no_valid_actions():
    board = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int8)
    action_space = get_action_space(board, 0)
    assert action_space.shape == (12,)
    assert all(action_space[action] == 0 for action in range(12))


if __name__ == "__main__":
    pytest.main()
