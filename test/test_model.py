import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple
import numpy as np
from model import AwaleNetwork, compute_gradients


@pytest.fixture
def network():
    """Create a fresh network instance for each test."""
    key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
    return AwaleNetwork(key)


@pytest.fixture
def sample_game_state():
    """Create a sample game state."""
    # 12 pits with random seeds (0-4) + 2 scores
    key = jax.random.PRNGKey(1)
    return jax.random.randint(key, (12,), 0, 5)


@pytest.fixture
def sample_score():
    """Create a sample score."""
    return jnp.array([0, 0])


def test_network_initialization(network):
    """Test that the network initializes with correct structure."""
    assert isinstance(network, AwaleNetwork)
    assert len(network.state_encoder) == 5
    assert isinstance(network.action_embedding, eqx.nn.Embedding)
    assert len(network.combine) == 5


def test_state_encoder_dimensions(network, sample_game_state, sample_score):
    """Test the dimensions of state encoder output."""
    state_features = jnp.concatenate([sample_game_state, sample_score])

    # Pass through state encoder layers
    for layer in network.state_encoder:
        state_features = layer(state_features)

    assert state_features.shape == (64,)


def test_action_embedding_dimensions(network):
    """Test the dimensions of action embeddings."""
    actions = jnp.array([0, 1, 2])  # Sample valid actions
    embeddings = jax.vmap(network.action_embedding)(actions)
    assert embeddings.shape == (3, 32)


def test_forward_pass_dimensions():
    key = jax.random.PRNGKey(0)
    network = AwaleNetwork(key)

    sample_game_state = jnp.array(
        [4, 1, 3, 3, 4, 4, 0, 4, 4, 1, 3, 0], dtype=jnp.int32
    )  # 12 pits + 2 scores
    sample_score = jnp.array([0, 0], dtype=jnp.int32)
    valid_actions = jnp.array([0, 1, 2])  # 3 valid actions

    output = network(sample_game_state, valid_actions, sample_score)
    print(type(output.shape))
    assert output.shape == (3,)


def test_empty_valid_actions(network, sample_game_state, sample_score):
    """Test handling of empty valid actions list."""
    valid_actions = jnp.array([])
    output = network(sample_game_state, valid_actions, sample_score)
    assert len(output) == 0


def test_single_valid_action(network, sample_game_state, sample_score):
    """Test handling of single valid action."""
    valid_actions = jnp.array([0])
    output = network(sample_game_state, valid_actions, sample_score)
    assert output.shape == (1,)


def test_all_valid_actions(network, sample_game_state, sample_score):
    """Test handling of all possible actions."""
    valid_actions = jnp.arange(12)  # All 12 possible actions
    output = network(sample_game_state, valid_actions, sample_score)
    assert output.shape == (12,)


@pytest.mark.parametrize(
    "game_state,valid_actions,expected_shape",
    [
        (jnp.zeros(12), jnp.array([0, 1]), (2,)),
        (jnp.ones(12), jnp.array([0]), (1,)),
        (jnp.full(12, 2), jnp.arange(6), (6,)),
    ],
)
def test_various_inputs(
    network, game_state, valid_actions, expected_shape, sample_score
):
    """Test network with various input combinations."""
    output = network(game_state, valid_actions, sample_score)
    assert output.shape == expected_shape


def test_network_gradients():
    """Test that gradients can be computed through the network."""
    key = jax.random.PRNGKey(0)
    network = AwaleNetwork(key)
    game_state = jnp.ones(12)
    valid_actions = jnp.array([0, 1, 2])
    score = jnp.array([0, 0])

    # Compute gradients using the helper function
    grads = compute_gradients(network, game_state, valid_actions, score)

    # Check that gradients exist and are not None
    assert grads is not None
    # Check that gradients are not all zero
    flat_grads = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0) for g in flat_grads)


def test_network_jit():
    """Test that the network can be JIT-compiled."""
    key = jax.random.PRNGKey(0)
    network = AwaleNetwork(key)
    game_state = jnp.ones(12)
    valid_actions = jnp.array([0, 1, 2])
    score = jnp.array([0, 0])

    @eqx.filter_jit  # Use equinox's filter_jit instead of jax.jit
    def forward(network, state, actions, score):
        return network(state, actions, score)

    # This should not raise any errors
    output = forward(network, game_state, valid_actions, score)
    assert output.shape == (3,)


def test_numerical_stability(network, sample_game_state, sample_score):
    """Test network behavior with extreme input values."""
    # Test with very large values
    large_state = jnp.full(12, 1e5)
    valid_actions = jnp.array([0, 1])
    large_score = jnp.array([1e5, 1e5])
    output_large = network(large_state, valid_actions, large_score)
    assert jnp.all(jnp.isfinite(output_large))

    # Test with very small values
    small_state = jnp.full(12, 1e-5)
    small_score = jnp.array([1e-5, 1e-5])
    output_small = network(small_state, valid_actions, small_score)
    assert jnp.all(jnp.isfinite(output_small))


if __name__ == "__main__":
    pytest.main([__file__])
