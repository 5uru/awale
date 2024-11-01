import pytest
import numpy as np
from model import ModelConfig, AwaleNet, select_action
import jax.numpy as jnp
import equinox as eqx
import jax


@pytest.fixture
def model_config():
    return ModelConfig(input_size=14, hidden_sizes=[64, 32], dropout_rate=0.2)


@pytest.fixture
def model(model_config):
    key = jax.random.PRNGKey(0)
    return AwaleNet(key, model_config)


def test_model_initialization(model):
    """Test model initialization and architecture."""
    assert len(model.layers) == 2

    # Test layer shapes
    assert model.layers[0].weight.shape == (64, 14)
    assert model.layers[1].weight.shape == (32, 64)
    assert model.output_layer.weight.shape == (12, 32)


def test_model_forward_pass(model):
    """Test forward pass with single example."""
    key = jax.random.PRNGKey(1)
    board = jnp.ones((1, 12))
    scores = jnp.zeros((1, 2))
    valid_actions = jnp.ones((1, 12), dtype=jnp.bool_)

    # Test inference mode
    output = model(board, scores, valid_actions, key, training=False)
    assert output.shape == (1, 12)
    assert jnp.allclose(jnp.sum(output), 1.0)
    assert jnp.all(output >= 0)

    # Test training mode
    output_train = model(board, scores, valid_actions, key, training=True)
    assert output_train.shape == (1, 12)
    assert jnp.allclose(jnp.sum(output_train), 1.0)


def test_batch_processing(model):
    """Test forward pass with batch of examples."""
    key = jax.random.PRNGKey(1)
    batch_size = 32
    board = jnp.ones((batch_size, 12))
    scores = jnp.zeros((batch_size, 2))
    valid_actions = jnp.ones((batch_size, 12), dtype=jnp.bool_)

    # Test both training and inference modes
    output_train = model(board, scores, valid_actions, key, training=True)
    output_infer = model(board, scores, valid_actions, key, training=False)

    assert output_train.shape == (batch_size, 12)
    assert output_infer.shape == (batch_size, 12)
    assert jnp.allclose(jnp.sum(output_train, axis=1), 1.0)
    assert jnp.allclose(jnp.sum(output_infer, axis=1), 1.0)


def test_valid_moves_masking(model):
    """Test that invalid moves are properly masked."""
    key = jax.random.PRNGKey(1)
    board = jnp.ones((1, 12))
    scores = jnp.zeros((1, 2))
    valid_actions = jnp.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=jnp.bool_)

    output = model(board, scores, valid_actions, key)

    # Check that invalid moves have zero probability
    assert jnp.all(output[0, 6:] < 1e-6)
    # Check that valid moves sum to 1
    assert jnp.allclose(jnp.sum(output[0, :6]), 1.0)


def test_select_action_with_model(model):
    """Test action selection using model output."""
    key = jax.random.PRNGKey(0)
    board = jnp.ones((1, 12))
    scores = jnp.zeros((1, 2))

    # Get model predictions
    probs = model(
        board=board,
        scores=scores,
        valid_actions=jnp.ones((1, 12), dtype=jnp.bool_),
        key=key,
        training=False,
    )

    # Test with different valid action patterns
    test_cases = [
        # First 6 valid
        jnp.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.bool_),
        # Last 6 valid
        jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=jnp.bool_),
        # Only one move valid
        jnp.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.bool_),
    ]

    for valid_actions in test_cases:
        # Get action using model's probabilities
        action = select_action(probs[0], valid_actions, key)
        # Verify action is valid
        assert valid_actions[action], f"Selected invalid action {action}"
        assert 0 <= action < 12, f"Action {action} out of range"


if __name__ == "__main__":
    pytest.main([__file__])
