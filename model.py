from flax import nnx
from jax import numpy as jnp


class AwaleModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.board_linear = nnx.Linear(12, 64, rngs=rngs)
        self.board_ln = nnx.LayerNorm(64, rngs=rngs)

        self.action_space_linear = nnx.Linear(12, 64, rngs=rngs)
        self.action_space_ln = nnx.LayerNorm(64, rngs=rngs)

        self.player_linear = nnx.Linear(1, 16, rngs=rngs)
        self.player_ln = nnx.LayerNorm(16, rngs=rngs)

        self.score_linear = nnx.Linear(2, 32, rngs=rngs)
        self.score_ln = nnx.LayerNorm(32, rngs=rngs)

        self.linear1 = nnx.Linear(176, 256, rngs=rngs)
        self.linear1_ln = nnx.LayerNorm(256, rngs=rngs)

        self.linear2 = nnx.Linear(256, 128, rngs=rngs)
        self.linear2_ln = nnx.LayerNorm(128, rngs=rngs)

        self.value_linear = nnx.Linear(128, 12, rngs=rngs)

        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.relu = nnx.leaky_relu

    def __call__(self, board, action_space, player, score):
        action_space_mask = action_space
        board = self.board_linear(board)
        board = self.board_ln(board)
        board = self.relu(board)
        board = self.dropout(board)
        action_space = self.action_space_linear(action_space)
        action_space = self.action_space_ln(action_space)
        action_space = self.relu(action_space)
        action_space = self.dropout(action_space)

        player = jnp.expand_dims(player, axis=0)
        player = self.player_linear(player)
        player = self.player_ln(player)
        player = self.relu(player)
        player = self.dropout(player)

        score = self.score_linear(score)
        score = self.score_ln(score)
        score = self.relu(score)
        score = self.dropout(score)

        x = jnp.concatenate([board, action_space, player, score])
        x = self.linear1(x)
        x = self.linear1_ln(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.linear2_ln(x)
        x = self.relu(x)
        x = self.dropout(x)

        value_output = self.value_linear(x)
        return value_output * action_space_mask
