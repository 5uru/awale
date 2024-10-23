import jax.numpy as jnp
from jax import jit, lax

from typing import NamedTuple, Tuple

import chex
from typing_extensions import TypeAlias


# Define a type alias for a board, which is an array.
Board: TypeAlias = chex.Array


class State(NamedTuple):
    board: Board
    action_space: chex.Array
    key: chex.PRNGKey
    score: chex.Array
    current_player: chex.Numeric


def distribute_seeds(board, current_pit, seeds) -> Tuple[Board, jnp.int8]:
    """Distribute seeds in the pits"""

    # We'll create a function that contains the loop body
    def loop_body(carry):
        seeds_carry, current_pit_carry, state_carry = carry
        current_pit_carry = (current_pit_carry + 1) % 12
        # Use lax.cond to conditionally update the state
        state_carry = lax.cond(
            current_pit_carry != action,
            lambda s: s.at[current_pit_carry].add(1),
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
def capture_seeds(board, last_pit, current_player) -> jnp.int8:
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
def calculate_end_game_reward(scores, current_player):
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
def determine_winner(scores):
    """Determine the winner of the game"""
    return lax.select(
        scores[0] > scores[1],
        0,
        lax.select(scores[1] > scores[0], 1, -1),
    )


@jit
def is_player_side_empty(player: jnp.int8, board):
    """Check if the player's side is empty"""
    start = player * 6
    player_side = lax.dynamic_slice(board, (start,), (6,))
    return jnp.all(player_side == 0)


@jit
def handle_empty_side(board, scores) -> jnp.array:
    """Handle the case when one player's side is empty"""
    # If one side is empty, all remaining seeds go to the other player
    scores = scores.at[0].add(jnp.sum(board[:6], dtype=jnp.int8))
    scores = scores.at[1].add(jnp.sum(board[6:], dtype=jnp.int8))
    board = jnp.zeros_like(board)
    return board, scores


def get_action_space(current_player):
    """Returns pre-computed action spaces for both players."""
    return lax.select(
        current_player == 0,
        jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int8),
        jnp.array([6, 7, 8, 9, 10, 11], dtype=jnp.int8),
    )


def score_end_game_state(state_scores_player_info):
    board, scores, current_player = state_scores_player_info
    reward = calculate_end_game_reward(scores, current_player)
    winner = determine_winner(scores)
    action_space = get_action_space(current_player)
    return board, scores, current_player, True, reward, winner, action_space


def empty_side_end_game(state_scores_player_info):
    state, scores, current_player = state_scores_player_info
    new_state, new_scores = handle_empty_side(state, scores)
    reward = calculate_end_game_reward(scores, current_player)
    winner = determine_winner(scores)
    action_space = get_action_space(current_player)
    return new_state, new_scores, current_player, True, reward, winner, action_space


def switch_player(state_scores_player_info):
    state, scores, current_player = state_scores_player_info
    new_player = 1 - current_player
    action_space = get_action_space(new_player)
    return state, scores, new_player, False, 0.0, current_player, action_space


def update_game_state(board, scores, current_player):
    return lax.cond(
        jnp.max(scores) > 23,
        score_end_game_state,
        lambda x: lax.cond(
            jnp.logical_or(
                is_player_side_empty(0, board), is_player_side_empty(1, board)
            ),
            empty_side_end_game,
            switch_player,
            x,
        ),
        (board, scores, current_player),
    )
