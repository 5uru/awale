import pytest
from awale_env_jax import AwaleJAX
import jax.numpy as jnp


@pytest.fixture
def game():
    return AwaleJAX()


def test_initial_state(game):
    _extracted_from_test_reset_2(game)


def test_is_valid_move(game):
    assert game._is_valid_move(0)
    assert game._is_valid_move(5)
    assert not game._is_valid_move(6)
    assert not game._is_valid_move(11)
    game.current_player = 1
    assert game._is_valid_move(6)
    assert game._is_valid_move(11)
    assert not game._is_valid_move(0)
    assert not game._is_valid_move(5)
    game.state = [0] * 12
    assert not game._is_valid_move(0)


def test_capture(game):
    game.state = jnp.array([4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 4, 4], dtype=jnp.int8)
    game.scores = jnp.zeros(2, dtype=jnp.int8)
    captured = game._capture_seeds(9)
    assert jnp.array_equal(
        game.state, jnp.array([4, 4, 4, 4, 4, 3, 0, 0, 0, 0, 4, 4], dtype=jnp.int8)
    )
    assert captured == 9


def test_step_valid_move(game):
    state, reward, done, info = game.step(0)
    assert jnp.array_equal(
        state, jnp.array([0, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int8)
    )
    assert not done
    assert game.current_player == 1


def test_step_invalid_move(game):
    state, reward, done, info = game.step(6)
    assert jnp.array_equal(game.state, jnp.array([4] * 12, dtype=jnp.int8))
    assert reward == -1
    assert not done
    assert info == {"invalid": True}
    assert game.current_player == 0


def test_capture_after_move(game):
    game.state = jnp.array([4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 4, 4], dtype=jnp.int8)
    state, reward, done, info = game.step(3)
    assert jnp.array_equal(
        game.state, jnp.array([4, 4, 4, 0, 5, 4, 0, 0, 3, 2, 4, 4], dtype=jnp.int8)
    )
    assert jnp.array_equal(game.scores, jnp.array([6, 0], dtype=jnp.int8))


def test_game_end_by_score(game):
    game.scores = jnp.array([24, 23], dtype=jnp.int8)
    game.state = jnp.array(
        [
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=jnp.int8,
    )  # Ensure there's at least one seed to move
    game.current_player = 0  # Ensure it is player 0's turn
    state, reward, done, info = game.step(0)
    assert done
    assert reward > 99
    assert info["winner"] == 0


def test_game_end_by_empty_side(game):
    game.state = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int8)
    game.current_player = 0
    state, reward, done, info = game.step(5)  # Player 0 tries to make a move
    assert done
    assert info["winner"] == 1


def test_reset(game):
    game.step(0)
    game.reset()
    _extracted_from_test_reset_2(game)


def _extracted_from_test_reset_2(game):
    assert jnp.array_equal(game.state, jnp.array([4] * 12, dtype=jnp.int8))
    assert jnp.array_equal(game.scores, jnp.zeros(2, dtype=jnp.int8))
    assert game.current_player == 0


def test_render(game):
    # This is a simple test to ensure render method runs without error
    try:
        game.render()
    except Exception as e:
        pytest.fail(f"render() raised {type(e).__name__} unexpectedly!")
