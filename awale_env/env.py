from typing import NamedTuple
from awale_env.utils import (
    distribute_seeds,
    determine_game_over,
    calculate_reward,
    get_action_space,
)
from typing_extensions import TypeAlias
from jax import numpy as jnp
import random

# Define a type alias for a board, which is an array.
Board: TypeAlias = jnp.array


class State(NamedTuple):
    """The state of the game"""

    board: Board  # The current state of the game
    action_space: jnp.array  # The valid actions for the current player
    score: jnp.array  # The scores for the two players
    current_player: jnp.int8  # The current player (0 or 1)


class Awale:
    def __init__(self):
        """
        Initiates a new game of Awale.
        Creates the initial state of the game using a fixed random key.
        """
        self.state = self.reset()

    def reset(self):
        """
        Resets or starts a new game.

        Returns:
            Initial state of play
        """
        # Randomly determines the first player (0 or 1)
        current_player = jnp.array(random.choice([0, 1]), dtype=jnp.int8)

        # Calculates the space of valid actions for the current player
        action_space = jnp.where(
            current_player == 0,
            jnp.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int8),
            jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=jnp.int8),
        )

        # Creates the initial tray with 4 seeds in each hole
        board = jnp.full((12,), 4, dtype=jnp.int8)

        # Sets scores to zero for both players
        scores = jnp.zeros(2, dtype=jnp.int8)

        state = State(board, action_space, scores, current_player)
        self.state = state
        return state

    def step(self, action: jnp.int8):
        board, captured_seeds = distribute_seeds(self.state.board, action)
        scores = self.state.score.clone()
        current_player = int(self.state.current_player)  # Convert to integer
        scores = scores.at[current_player].add(captured_seeds)
        done, winner, info = determine_game_over(board, scores)
        reward = calculate_reward(board, captured_seeds, current_player, done)
        next_player = 1 - current_player
        next_action_space = get_action_space(board, next_player)
        state = State(board, next_action_space, scores, next_player)
        self.state = state
        return state, reward, done, info

    def render(self):
        state = self.state
        # Prepares the tray lines for display
        top_row = state.board[6:0:-1]  # Inverts the top line for the display
        bottom_row = state.board[6:]

        # Display construction
        board_display = [
            f"Player 2: {state.score[1]:2d}",
            "   ┌────┬────┬────┬────┬────┬────┐",
            f"   │ {' │ '.join(f'{pit:2d}' for pit in top_row)} │",
            "───┼────┼────┼────┼────┼────┼────┤",
            f"   │ {' │ '.join(f'{pit:2d}' for pit in bottom_row)} │",
            "   └────┴────┴────┴────┴────┴────┘",
            f"Player 1: {state.score[0]:2d}",
        ]

        # Displays the set
        print("\n".join(board_display))
        return "\n".join(board_display)
