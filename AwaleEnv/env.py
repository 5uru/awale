import jax.numpy as jnp
from jax import jit, lax
from typing import NamedTuple, Tuple, Dict
from AwaleEnv.utils import (
    distribute_seeds,
    capture_seeds,
    State,
    update_game_state,
    get_action_space,
)
from AwaleEnv.viewer import save_board_svg
import chex
from typing_extensions import TypeAlias
import jax.random as random
from jax.random import PRNGKey

# Constantes du jeu
BOARD_SIZE = 12
SEEDS_PER_PIT = 4
PLAYER_SIDE_SIZE = 6
INITIAL_SCORE = 0
CAPTURE_REWARD_MULTIPLIER = 0.5


class AwaleJAX:
    """
    Implementation of the Awale game using JAX for optimum performance.
    The game follows the traditional Awale rules with seed capture.
    """

    def __init__(self):
        """
        Initiates a new game of Awale.
        Creates the initial state of the game using a fixed random key.
        """
        self.state = self.reset(PRNGKey(0))

    @staticmethod
    @jit
    def reset(key: PRNGKey) -> State:
        """
        Resets or starts a new game.

        Args:
            key: PRNGKey for random number generation

        Returns:
            State: Initial state of play
        """
        # Randomly determines the first player (0 or 1)
        current_player = (random.uniform(key) > 0.5).astype(jnp.int8)

        # Calculates the space of valid actions for the current player
        action_space = get_action_space(current_player)

        # Create the initial tray with 4 seeds in each hole
        board = jnp.full(BOARD_SIZE, SEEDS_PER_PIT, dtype=jnp.int8)

        # Sets scores to zero for both players
        scores = jnp.zeros(2, dtype=jnp.int8)

        return State(
            board=board,
            action_space=action_space,
            key=key,
            score=scores,
            current_player=current_player,
        )

    def step(self, state: State, action: jnp.int8) -> Tuple[State, float, bool]:
        """
        Performs an action in the game.

        Args:
            state: Current state of play
            action: Current state of play

        Returns:
            Tuple containing:
            - New game state
            - Reward obtained
            - End of game indicator
        """
        # Collects and distributes seeds
        seeds = state.board[action]
        board = state.board.at[action].set(0)

        # Seed distribution
        board, final_pit = distribute_seeds(board, action, seeds)

        # Capture seeds if possible
        board, captured = capture_seeds(board, final_pit, state.current_player)

        # Updating the score and calculating the initial reward
        score = state.score.at[state.current_player].add(captured)
        reward = CAPTURE_REWARD_MULTIPLIER * captured

        # Game status update
        (new_board, new_score, done, new_reward, new_player) = update_game_state(
            board, score, state.current_player, reward
        )

        # Filtering of valid actions (non-empty holes)
        # Calculates valid action space for the next player
        new_action_space = get_action_space(new_player)

        # Create a mask for valid actions (positions with stones)
        valid_actions_mask = board[new_action_space] > 0

        # Filter action space to only valid actions
        new_action_space = jnp.where(
            valid_actions_mask, new_action_space, jnp.zeros_like(new_action_space)
        )
        new_action_space = new_action_space[valid_actions_mask]

        # Creating a new state
        new_state = State(
            board=new_board,
            action_space=new_action_space,
            key=state.key,
            score=new_score,
            current_player=new_player,
        )
        state = self.state
        return new_state, new_reward, done

    @staticmethod
    def render(state: State, filename: str = "awale_board.svg") -> None:
        """
        Displays the current state of the game board in the console.

        Args:
            state: Current state of the game to be displayed
            filename: File name for saving the SVG file
        """
        save_board_svg(state.board, state.score, filename)

    def __repr__(self) -> str:
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
