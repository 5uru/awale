import pytest
import jax.numpy as jnp
from jax.random import PRNGKey
from AwaleEnv.env import AwaleJAX
from AwaleEnv.utils import State

# Constantes pour les tests
INITIAL_SEEDS = 4
BOARD_SIZE = 12


@pytest.fixture
def game():
    """Fixture pour créer une instance du jeu"""
    return AwaleJAX()


@pytest.fixture
def initial_state(game):
    """Fixture pour l'état initial du jeu"""
    return game.reset(PRNGKey(0))


def test_initial_state(initial_state):
    assert jnp.array_equal(initial_state.board, jnp.array([4] * 12, dtype=jnp.int8))
    assert jnp.array_equal(initial_state.score, jnp.zeros(2, dtype=jnp.int8))


def test_step_valid_move(game, initial_state):
    current_player = initial_state.current_player
    state, reward, done = game.step(initial_state, 0)
    assert jnp.array_equal(
        state.board, jnp.array([0, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int8)
    )
    assert not done
    assert state.current_player != current_player


def test_capture_after_move(game, initial_state):
    initial_state = State(
        board=jnp.array([4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 4, 4], dtype=jnp.int8),
        action_space=initial_state.action_space,
        key=initial_state.key,
        score=initial_state.score,
        current_player=0,
    )

    state, reward, done = game.step(initial_state, 3)
    assert jnp.array_equal(
        state.board, jnp.array([4, 4, 4, 0, 5, 4, 0, 0, 3, 2, 4, 4], dtype=jnp.int8)
    )
    assert jnp.array_equal(state.score, jnp.array([6, 0], dtype=jnp.int8))


def test_game_end_by_score(game, initial_state):
    initial_state = State(
        board=jnp.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=jnp.int8),
        action_space=initial_state.action_space,
        key=initial_state.key,
        score=jnp.array([24, 23], dtype=jnp.int8),
        current_player=0,
    )
    state, reward, done = game.step(initial_state, 0)
    assert done
    assert reward > 99
    assert state.current_player == 0


def test_game_end_by_empty_side(game, initial_state):
    initial_state = State(
        board=jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int8),
        action_space=initial_state.action_space,
        key=initial_state.key,
        score=initial_state.score,
        current_player=0,
    )

    game.state = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int8)
    game.current_player = 0
    state, reward, done = game.step(initial_state, 5)
    assert done
    assert state.current_player == 1


def test_action_space_changes(game, initial_state):
    initial_state = State(
        board=jnp.array([4, 0, 4, 0, 4, 0, 3, 3, 0, 3, 0, 3], dtype=jnp.int8),
        action_space=jnp.array([0, 2, 4], dtype=jnp.int8),
        key=initial_state.key,
        score=initial_state.score,
        current_player=0,
    )
    state, reward, done = game.step(initial_state, 0)
    assert jnp.array_equal(state.action_space, jnp.array([6, 7, 9, 11], dtype=jnp.int8))
    assert state.current_player


def test_reset(game, initial_state):
    game.step(initial_state, 0)
    state = game.reset(PRNGKey(0))
    assert jnp.array_equal(state.board, jnp.array([4] * 12, dtype=jnp.int8))
    assert jnp.array_equal(state.score, jnp.zeros(2, dtype=jnp.int8))
    assert state.current_player == 0


def test_render(game, initial_state):
    try:
        game.render(initial_state)
    except Exception as e:
        pytest.fail(f"render() raised {type(e).__name__} unexpectedly!")


def test_render_other_state(game, initial_state):
    initial_state = State(
        board=jnp.array([4, 0, 4, 0, 4, 0, 3, 3, 0, 3, 0, 3], dtype=jnp.int8),
        action_space=initial_state.action_space,
        key=initial_state.key,
        score=jnp.array([24, 23], dtype=jnp.int8),
        current_player=0,
    )
    try:
        game.render(initial_state)
    except Exception as e:
        pytest.fail(f"render() raised {type(e).__name__} unexpectedly!")


def test_repr(game, initial_state):
    try:
        game.__repr__()
    except Exception as e:
        pytest.fail(f"render() raised {type(e).__name__} unexpectedly!")
