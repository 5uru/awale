from typing import Sequence
import flax.linen as nn
import jax.numpy as jnp
import jax.random as random


class AwaleModel(nn.Module):
    """Neural network for learning to play Awale."""

    features: Sequence[int]  # Feature dimensions for each layer

    @nn.compact
    def __call__(
        self,
        board: jnp.ndarray,
        scores: jnp.ndarray,
        valid_actions: jnp.ndarray,  # shape (12,) mask of valid moves
        rng: random.PRNGKey,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Process the game state and output move probabilities.

        Args:
            board: shape (12,) representing seeds in each pit
            scores: shape (2,) representing current scores
            valid_actions: shape (12,) binary mask of valid moves
            training: whether to apply dropout

        Returns:
            probabilities: shape (12,) array of action probabilities
        """
        # Process the board state
        x = board.reshape(-1)
        x = nn.Dense(features=self.features[0])(x)
        x = nn.relu(x)

        # Process the scores
        s = scores.reshape(-1)
        s = nn.Dense(features=self.features[0] // 2)(s)
        s = nn.relu(s)

        # Combine features
        x = jnp.concatenate([x, s], axis=-1)

        # Process through dense layers
        for feat in self.features[1:-1]:
            x = nn.Dense(features=feat)(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(0.1)(x, deterministic=not training, rng=rng)

        # Policy head: outputs logits for all 12 possible positions
        policy_logits = nn.Dense(features=12)(x)

        # Mask invalid moves using the 12-length mask
        policy_logits = jnp.where(
            valid_actions, policy_logits, jnp.full_like(policy_logits, -1e9)
        )

        return nn.softmax(policy_logits)
