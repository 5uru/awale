import pytest
from awale_env.env import Awale, State
from jax import numpy as jnp


def test_reset():
    game = Awale()
    state = game.reset()
    assert isinstance(state, State)
    assert state.board.shape == (12,)
    assert state.action_space.shape == (6,)
    assert state.score.shape == (2,)
    assert state.current_player in [0, 1]


def test_step():
    game = Awale()
    initial_state = game.reset()
    action = initial_state.action_space[0]
    state, reward, done, info = game.step(action)
    assert isinstance(state, State)
    assert reward.dtype == jnp.float16
    assert isinstance(done, bool)
    assert isinstance(info, str)


if __name__ == "__main__":
    pytest.main()
