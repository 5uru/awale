import mlx.core as mx
import pytest
from awale.utils import (
    distribute_seeds,
    get_action_space,
    determine_game_over,
    calculate_reward,
)


def test_distribute_seeds():
    board = mx.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=mx.int8)
    new_board, captured_seeds = distribute_seeds(board, 0)
    assert new_board.shape == (12,)
    assert isinstance(captured_seeds, int)


def test_determine_game_over():
    board = mx.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=mx.int8)
    scores = mx.array([25, 23], dtype=mx.int8)
    done, winner, reason = determine_game_over(board, scores)
    assert done is True
    assert winner == 0
    assert reason == "Joueur 1 a capturé la majorité des graines"


def test_calculate_reward():
    board = mx.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=mx.int8)
    captured_seeds = 2
    reward = calculate_reward(board, captured_seeds, 0, game_over=False)
    assert isinstance(reward, float)


def test_get_action_space():
    board = mx.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=mx.int8)
    action_space = get_action_space(board, 0)
    assert action_space.shape == (6,)
    assert all(action in range(6) for action in action_space)


if __name__ == "__main__":
    pytest.main()
