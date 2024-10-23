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


def test_step_valid_move(initial_state, game):
    state, reward, done, info = game.step(initial_state, 0)
    assert jnp.array_equal(
        state, jnp.array([0, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4], dtype=jnp.int8)
    )
    assert not done
