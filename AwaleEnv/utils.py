import jax.numpy as jnp
from jax import jit, lax

from typing import NamedTuple, Tuple

import chex
from typing_extensions import TypeAlias


# Define a type alias for a board, which is an array.
Board: TypeAlias = chex.Array


class State(NamedTuple):
    """The state of the game"""

    board: Board  # The current state of the game
    action_space: chex.Array  # The valid actions for the current player
    key: chex.PRNGKey  # The random key for the game
    score: chex.Array  # The scores for the two players
    current_player: chex.Numeric  # The current player (0 or 1)


@jit
def distribute_seeds(
    board: Board, current_pit: jnp.int8, seeds: jnp.int8
) -> Tuple[Board, jnp.int8]:
    """Distribute seeds in the pits"""

    # We'll create a function that contains the loop body
    def loop_body(carry):
        seeds_carry, current_pit_carry, state_carry = carry
        current_pit_carry = (current_pit_carry + 1) % 12  # Move to the next pit
        # Use lax.cond to conditionally update the state
        state_carry = lax.cond(
            current_pit_carry != action,
            lambda s: s.at[current_pit_carry].add(1),  # Add a seed to the pit
            lambda s: s,
            state_carry,
        )
        seeds_carry = seeds_carry - 1
        return seeds_carry, current_pit_carry, state_carry

    # Define the while loop condition
    def cond_fun(carry):
        seeds_carry, current_pit_carry, state_carry = carry
        return seeds_carry > 0

    # Initial carry values
    action = current_pit
    init_carry = (seeds, current_pit, board)

    # Run the while loop
    final_seeds, final_pit, final_state = lax.while_loop(
        cond_fun, loop_body, init_carry
    )
    return final_state, final_pit


@jit
def capture_seeds(
    board: Board, last_pit: jnp.int8, current_player: jnp.int8
) -> Tuple[Board, jnp.int8]:
    """Capture seeds from the opponent's side"""
    opponent_side = (1 - current_player) * 6

    def condition(carry):
        """Check if the last seed lands in a two or three pit on the opponent's side"""
        _, last_pit, _ = carry
        return jnp.logical_and(
            jnp.logical_and(opponent_side <= last_pit, last_pit < opponent_side + 6),
            jnp.logical_and(2 <= board[last_pit], board[last_pit] <= 3),
        )

    def body(carry):
        """Capture seeds from the opponent's side"""
        state, last_pit, captured = carry
        captured += state[last_pit]
        state = state.at[last_pit].set(0)
        last_pit -= 1
        return state, last_pit, captured

    initial_carry = (  # Initial values for the loop
        board,
        last_pit,
        0,
    )  # 0 is the initially captured value
    final_state, final_last_pit, total_captured = lax.while_loop(
        condition, body, initial_carry
    )
    return final_state, total_captured


@jit
def calculate_end_game_reward(
    scores: chex.Array, current_player: jnp.int8
) -> jnp.float16:
    """Calculate the reward when the game ends"""
    return lax.select(
        scores[current_player] > scores[1 - current_player],
        100,
        lax.select(
            scores[current_player] < scores[1 - current_player],
            -100,
            -50,
        ),
    )


@jit
def determine_winner(scores: chex.Array) -> jnp.int8:
    """Determine the winner of the game"""
    return lax.select(
        scores[0] > scores[1],
        0,
        lax.select(scores[1] > scores[0], 1, -1),
    )


@jit
def is_player_side_empty(player: jnp.int8, board: Board) -> jnp.bool_:
    """Check if the player's side is empty"""
    start = player * 6
    player_side = lax.dynamic_slice(board, (start,), (6,))
    return jnp.all(player_side == 0)


@jit
def handle_empty_side(board: Board, scores: chex.Array) -> Tuple[Board, chex.Array]:
    """Handle the case when one player's side is empty"""
    # If one side is empty, all remaining seeds go to the other player
    scores = scores.at[0].add(jnp.sum(board[:6], dtype=jnp.int8))
    scores = scores.at[1].add(jnp.sum(board[6:], dtype=jnp.int8))
    board = jnp.zeros_like(board)
    return board, scores


@jit
def get_action_space(current_player: jnp.int8) -> chex.Array:
    """Returns pre-computed action spaces for both players."""
    return lax.select(
        current_player == 0,
        jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int8),
        jnp.array([6, 7, 8, 9, 10, 11], dtype=jnp.int8),
    )


@jit
def update_game_state(
    board: jnp.array,
    score: jnp.array,
    current_player: jnp.int8,
    reward: jnp.float16 = 0.0,
) -> Tuple[Board, chex.Array, jnp.bool_, jnp.float16, jnp.int8]:
    """Update the game state after a move"""
    # Check if the game ends by score
    done = jnp.where(
        jnp.max(score) > 23, True, False
    )  # The game ends when a player reaches 24 points
    reward = jnp.where(
        done, reward + calculate_end_game_reward(score, current_player), reward
    )

    # Check if the game ends by empty side
    empty_side_condition = jnp.logical_or(
        is_player_side_empty(0, board), is_player_side_empty(1, board)
    )

    board, score = lax.cond(
        empty_side_condition,
        lambda x: handle_empty_side(*x),
        lambda x: x,
        (board, score),
    )

    done = jnp.where(empty_side_condition, True, done)
    reward = jnp.where(
        empty_side_condition,
        reward + calculate_end_game_reward(score, current_player),
        reward,
    )
    # Switch to the next player
    current_player = jnp.where(done, current_player, 1 - current_player)

    current_player = jnp.where(done, determine_winner(score), current_player)

    return board, score, done, reward, current_player
