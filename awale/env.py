import mlx.core as mx
from typing import NamedTuple, Tuple
from awale.utils import (
    distribute_seeds,
    determine_game_over,
    calculate_reward,
    get_action_space,
)
from typing_extensions import TypeAlias


# Define a type alias for a board, which is an array.
Board: TypeAlias = mx.array


class State(NamedTuple):
    """The state of the game"""

    board: Board  # The current state of the game
    action_space: mx.array  # The valid actions for the current player
    score: mx.array  # The scores for the two players
    current_player: mx.int8  # The current player (0 or 1)


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
        current_player = (mx.random.uniform() > 0.5).astype(mx.int8)

        # Calculates the space of valid actions for the current player
        action_space = mx.where(
            current_player == 0,
            mx.array([0, 1, 2, 3, 4, 5], dtype=mx.int8),
            mx.array([6, 7, 8, 9, 10, 11], dtype=mx.int8),
        )

        # Creates the initial tray with 4 seeds in each hole
        board = mx.full(12, 4, dtype=mx.int8)

        # Sets scores to zero for both players
        scores = mx.zeros(2, dtype=mx.int8)

        state = State(board, action_space, scores, current_player)
        self.state = state
        return state

    def step(self, action: mx.int8):
        board, captured_seeds = distribute_seeds(self.state.board, action)
        scores = self.state.score.at[self.state.current_player].add(captured_seeds)
        done, winner, info = determine_game_over(board, scores)
        reward = calculate_reward(
            board, captured_seeds, self.state.current_player, done
        )
        next_player = 1 - self.state.current_player
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
