import pytest
import jax.numpy as jnp
from jax import random


from AwaleEnv.utils import (
    State,
    distribute_seeds,
    capture_seeds,
    update_game_state,
    calculate_reward,
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
    board = board.at[0].set(0)

    new_board, last_pit = distribute_seeds(board, current_pit, seeds)
    expected = jnp.array([0, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int8)
    assert (new_board == expected).all()
    assert last_pit == 4


def test_capture_seeds():
    board = jnp.array([4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 4, 4], dtype=jnp.int8)
    last_pit = jnp.int8(9)
    current_player = jnp.int8(0)

    new_board, captured = capture_seeds(board, last_pit, current_player)
    expected = jnp.array([4, 4, 4, 4, 4, 3, 0, 0, 0, 0, 4, 4], dtype=jnp.int8)
    assert jnp.array_equal(new_board, expected)
    assert captured == 9


def test_update_game_state_score_end():
    board = jnp.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=jnp.int8)
    scores = jnp.array([24, 23], dtype=jnp.int8)
    current_player = jnp.int8(0)

    (board, score, done, winner) = update_game_state(board, scores, current_player)
    assert done
    assert winner == 0


def test_update_game_state_empty_side():
    board = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=jnp.int8)
    scores = jnp.array([10, 10], dtype=jnp.int8)
    current_player = jnp.int8(0)

    (new_board, new_scores, done, winner) = update_game_state(
        board, scores, current_player
    )
    assert done
    assert jnp.array_equal(new_board, jnp.zeros_like(board))
    assert new_scores[1] > new_scores[0]  # Player 1 should win


def test_update_game_state_switch_player():
    board = jnp.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=jnp.int8)
    scores = jnp.array([0, 0], dtype=jnp.int8)
    current_player = jnp.int8(0)

    (new_board, new_scores, done, winner) = update_game_state(
        board, scores, current_player
    )
    assert done == False
    assert winner == 1
    assert jnp.array_equal(new_board, board)


@pytest.fixture
def sample_boards():
    # Adjusted for 12 pits (6 per player)
    previous_board = jnp.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int32)
    current_board = jnp.array([4, 0, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4], dtype=jnp.int32)
    return previous_board, current_board


@pytest.fixture
def sample_scores():
    previous_score = jnp.array([0, 0], dtype=jnp.int32)
    current_score = jnp.array([1, 0], dtype=jnp.int32)
    return previous_score, current_score


def test_calculate_reward_basic(sample_boards, sample_scores):
    previous_board, current_board = sample_boards
    previous_score, current_score = sample_scores
    player_id = jnp.int32(0)
    reward = calculate_reward(
        current_board, previous_board, current_score, previous_score, player_id
    )
    # Reward should be:
    # 10.0 (base for capturing 1 seed)
    # Plus position evaluation difference
    assert reward == 8.0


def test_calculate_reward_game_over_win(sample_boards, sample_scores):
    previous_board, current_board = sample_boards
    previous_score, current_score = sample_scores
    player_id = jnp.int32(0)
    reward = calculate_reward(
        current_board,
        previous_board,
        current_score,
        previous_score,
        player_id,
        game_over=True,
    )
    # Expected: 10.0 (capture) + 100.0 (win bonus) = 110.0
    assert reward == 108.0


def test_calculate_reward_game_over_loss(sample_boards, sample_scores):
    previous_board, current_board = sample_boards
    # Create two separate arrays for scores
    previous_score = jnp.array([0, 1], dtype=jnp.int32)
    current_score = jnp.array([1, 2], dtype=jnp.int32)
    player_id = jnp.int32(0)

    reward = calculate_reward(
        current_board,
        previous_board,
        current_score,
        previous_score,
        player_id,
        game_over=True,
    )

    # Expected: 10.0 (capture) - 50.0 (loss penalty) = -40.0
    expected_reward = -42.0
    assert reward == expected_reward, f"Expected {expected_reward}, got {reward}"


def test_calculate_reward_game_over_draw(sample_boards, sample_scores):
    previous_board, current_board = sample_boards
    previous_score = jnp.array([1, 1], dtype=jnp.int32)
    current_score = jnp.array([1, 1], dtype=jnp.int32)
    player_id = jnp.int32(0)

    reward = calculate_reward(
        current_board,
        previous_board,
        current_score,
        previous_score,
        player_id,
        game_over=True,
    )

    # Expected: 25.0 (draw bonus) + position evaluation
    expected_reward = 23.0
    assert reward == expected_reward, f"Expected {expected_reward}, got {reward}"


def test_calculate_reward_illegal_move(sample_boards, sample_scores):
    previous_board, current_board = sample_boards
    # Set an illegal number of seeds (>48) in one pit
    current_board = current_board.at[0].set(49)
    previous_score, current_score = sample_scores
    player_id = jnp.int32(0)

    reward = calculate_reward(
        current_board, previous_board, current_score, previous_score, player_id
    )

    # Expected: -1000 (illegal move penalty) + 10.0 (capture) + position evaluation
    expected_reward = -969.5
    assert reward == expected_reward, f"Expected {expected_reward}, got {reward}"


def test_calculate_reward_vulnerable_position(sample_boards, sample_scores):
    previous_board, current_board = sample_boards
    # Create a board with vulnerable positions (2-3 seeds)
    current_board = jnp.array([2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int32)
    previous_score, current_score = sample_scores
    player_id = jnp.int32(0)

    reward = calculate_reward(
        current_board, previous_board, current_score, previous_score, player_id
    )

    # Should include penalties for the vulnerable positions
    assert (
        reward < 10.0
    )  # Should be less than basic capture reward due to vulnerabilities
