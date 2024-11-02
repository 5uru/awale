from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_size: int = 14
    hidden_sizes: List[int] = None
    dropout_rate: float = 0.3

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 32]


class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, *, key: jax.random.PRNGKey):
        self.in_features = in_features
        self.out_features = out_features

        wkey, bkey = jax.random.split(key)
        weight = jax.random.normal(wkey, (out_features, in_features))
        bias = jax.random.normal(bkey, (out_features,))

        self.weight = weight / jnp.sqrt(in_features)
        self.bias = bias * 0.1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.matmul(x, self.weight.T)
        return x + self.bias[None, :]


class AwaleNet(eqx.Module):
    """Neural network for Awale game."""

    layers: List[Linear]
    output_layer: Linear
    dropout_rate: float
    config: ModelConfig

    def __init__(self, key: jax.random.PRNGKey, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        # Split keys for layer initialization
        keys = jax.random.split(key, len(self.config.hidden_sizes) + 1)

        # Initialize layers
        self.layers = []
        prev_size = self.config.input_size

        for i, size in enumerate(self.config.hidden_sizes):
            self.layers.append(Linear(prev_size, size, key=keys[i]))
            prev_size = size

        self.output_layer = Linear(prev_size, 12, key=keys[-1])
        self.dropout_rate = self.config.dropout_rate

    def __call__(
        self,
        board: jnp.ndarray,
        scores: jnp.ndarray,
        valid_actions: jnp.ndarray,
        key: jax.random.PRNGKey,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through the network."""
        # Input validation
        if board.shape[-1] != 12:
            raise ValueError(
                f"Expected board shape with 12 positions, got {board.shape}"
            )
        if scores.shape[-1] != 2:
            raise ValueError(f"Expected scores shape with 2 values, got {scores.shape}")

        # Combine inputs
        x = jnp.concatenate([board, scores], axis=-1)  # Shape: (batch_size, 14)

        # Generate dropout keys if training
        dropout_keys = jax.random.split(key, len(self.layers)) if training else None

        # Forward pass through hidden layers
        for i, layer in enumerate(self.layers):
            # Linear layer
            x = layer(x)

            # Layer normalization (using mean and std across features)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            std = jnp.std(x, axis=-1, keepdims=True) + 1e-5
            x = (x - mean) / std

            # Activation
            x = jax.nn.relu(x)

            # Dropout (only during training)
            if training:
                dropout_key = dropout_keys[i]
                mask = jax.random.bernoulli(dropout_key, 1 - self.dropout_rate, x.shape)
                x = x * mask / (1 - self.dropout_rate)

        # Output layer
        logits = self.output_layer(x)

        # Mask invalid actions
        masked_logits = jnp.where(valid_actions, logits, -1e9)

        # Return probabilities
        return jax.nn.softmax(masked_logits, axis=-1)


class AwaleGameState:
    """Class to handle Awale game state and move validation."""

    def __init__(
        self, board: List[int], player1_score: int = 0, player2_score: int = 0
    ):
        self.board = board
        self.player1_score = player1_score
        self.player2_score = player2_score

    def get_valid_moves(self, player: int) -> List[int]:
        """Get valid moves for the current player."""
        start_idx = 0 if player == 1 else 6
        return [i for i in range(start_idx, start_idx + 6) if self.board[i] > 0]

    def to_model_input(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Convert game state to model input format."""
        board = jnp.array(self.board, dtype=jnp.float32).reshape(1, -1)
        scores = jnp.array(
            [[self.player1_score, self.player2_score]], dtype=jnp.float32
        )
        valid_moves = self.get_valid_moves(1)  # Assuming player 1's perspective
        valid_actions = jnp.zeros((1, 12), dtype=jnp.bool_)
        valid_actions = valid_actions.at[:, valid_moves].set(True)
        return board, scores, valid_actions


def select_action(
    probs: jnp.ndarray, valid_actions: jnp.ndarray, key: jax.random.PRNGKey
) -> int:
    """Select an action between 0 and 11."""
    masked_probs = jnp.where(valid_actions, probs, 0.0)
    masked_probs = masked_probs / (jnp.sum(masked_probs) + 1e-8)
    return jax.random.categorical(key, jnp.log(masked_probs + 1e-8)).astype(int)
