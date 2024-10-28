import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from typing import Any, List


class AwaleNetwork(eqx.Module):
    state_encoder: List[Any]
    action_embedding: nn.Embedding
    combine: List[Any]

    def __init__(self, key: jax.random.PRNGKey):
        super().__init__()
        keys = jax.random.split(key, 7)

        # Input layer for state representation (12 pits + 2 scores)
        self.state_encoder = [
            nn.Linear(14, 64, key=keys[0]),  # 12 pits + 2 scores -> 64
            nn.Linear(64, 128, key=keys[1]),  # 64 -> 128
            jax.nn.relu,
            nn.Linear(128, 64, key=keys[2]),  # 128 -> 64
            jax.nn.relu,
        ]

        # Action embedding layer
        self.action_embedding = nn.Embedding(
            12, 32, key=keys[3]
        )  # Can embed any action index 0-11

        # Final layers to combine state and action information
        self.combine = [
            nn.Linear(96, 64, key=keys[4]),  # 64 (state) + 32 (action) = 96
            jax.nn.relu,
            nn.Linear(64, 32, key=keys[5]),
            jax.nn.relu,
            nn.Linear(32, 1, key=keys[6]),
        ]

    def apply_linear_batch(self, layer: nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
        """Apply a linear layer to batched input, handling bias broadcasting correctly."""
        batch_size = x.shape[0]
        # Reshape input to (batch_size, input_dim)
        reshaped_x = x.reshape(batch_size, -1)
        # Apply weight matrix
        output = jnp.matmul(reshaped_x, layer.weight.T)
        # Add bias with correct broadcasting
        output = output + layer.bias[None, :]
        return output

    def __call__(
        self, game_state: jnp.ndarray, valid_actions: jnp.ndarray, score: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Args:
            game_state: Array of shape (12,) representing the game board
            valid_actions: Array of valid action indices
            score: Array of shape (2,) containing scores for both players
        Returns:
            Array of shape (num_valid_actions,) containing value predictions
        """
        # Handle empty actions case
        if valid_actions.size == 0:
            return jnp.array([])

        # Encode state
        state_features = jnp.concatenate([game_state, score])  # Shape: (14,)

        # Pass through state encoder
        for layer in self.state_encoder:
            state_features = layer(state_features)
        # Get action embeddings using vmap
        action_features = jax.vmap(self.action_embedding)(
            valid_actions
        )  # Shape: (num_actions, 32)

        # Broadcast state features to match number of actions
        num_actions = valid_actions.shape[0]
        state_features = jnp.tile(
            state_features[None, :], (num_actions, 1)
        )  # Shape: (num_actions, 64)

        # Combine features
        combined = jnp.concatenate(
            [state_features, action_features], axis=1
        )  # Shape: (num_actions, 96)

        # Pass through combine layers
        values = combined
        for layer in self.combine:
            if isinstance(layer, nn.Linear):
                values = self.apply_linear_batch(layer, values)
            else:
                values = layer(values)
        return values.squeeze(-1)  # Shape: (num_actions,)


# Helper function for JIT compilation
@eqx.filter_jit
def forward(network, game_state, valid_actions, score):
    """JIT-compiled forward pass."""
    return network(game_state, valid_actions, score)


# Helper function for gradient computation
@eqx.filter_value_and_grad
def loss_fn(network, game_state, valid_actions, score):
    """Compute loss and gradients."""
    output = network(game_state, valid_actions, score)
    return jnp.mean(output)


def compute_gradients(network, game_state, valid_actions, score):
    """Compute gradients with respect to network parameters."""
    loss, grads = loss_fn(network, game_state, valid_actions, score)
    return grads


def init_network(key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> AwaleNetwork:
    """Initialize the network with default parameters."""
    return AwaleNetwork(key)
