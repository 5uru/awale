from collections import deque
from jax import numpy as jnp
import random


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        board,
        action_space,
        player,
        score,
        action,
        reward,
        next_board,
        next_action_space,
        next_player,
        next_score,
        done,
    ):
        self.buffer.append(
            (
                board,
                action_space,
                player,
                score,
                action,
                reward,
                next_board,
                next_action_space,
                next_player,
                next_score,
                done,
            )
        )

    def sample(self, batch_size):
        (
            board,
            action_space,
            player,
            score,
            action,
            reward,
            next_board,
            next_action_space,
            next_player,
            next_score,
            done,
        ) = zip(*random.sample(self.buffer, batch_size))
        return {
            "board": jnp.array(board),
            "action_space": jnp.array(action_space),
            "player": jnp.array(player),
            "score": jnp.array(score),
            "action": jnp.array(action),
            "reward": jnp.array(reward),
            "next_board": jnp.array(next_board),
            "next_action_space": jnp.array(next_action_space),
            "next_player": jnp.array(next_player),
            "next_score": jnp.array(next_score),
            "done": jnp.array(done),
        }

    def __len__(self):
        return len(self.buffer)
