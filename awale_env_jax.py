import jax.numpy as jnp
from jax import jit, lax


class AwaleJAX:
    """Awale game environment implemented in JAX"""

    def __init__(self, player: jnp.int8 = 0):
        """Initialize the game"""
        self.current_player = player  # 0 or 1
        self.action_space = jnp.arange(
            player * 6, (player + 1) * 6, dtype=jnp.int8
        )  # 0-5 or 6-11
        self.state = jnp.array([4] * 12, dtype=jnp.int8)  # 12 pits with 4 seeds each
        self.scores = jnp.zeros(2, dtype=jnp.int8)  # 2 players with 0 score each

    def step(self, action):
        """Take a step in the game"""
        reward = -0.1  # Reward for each step
        done = False
        info = {}

        @jit
        def is_valid_move(action: jnp.int8) -> jnp.bool_:
            """Check if the action is valid"""
            in_range = jnp.logical_and(  # Check if the action is on the player's side
                self.current_player * 6 <= action,
                action < (self.current_player + 1) * 6,
            )
            has_seeds = self.state[action] > 0  # Check if the pit has seeds
            return jnp.logical_and(in_range, has_seeds)

        @jit
        def capture_seeds(last_pit) -> jnp.int8:
            opponent_side = (1 - self.current_player) * 6

            def condition(carry):
                """Check if the last seed lands in a two or three pit on the opponent's side"""
                _, last_pit, _ = carry
                return jnp.logical_and(
                    jnp.logical_and(
                        opponent_side <= last_pit, last_pit < opponent_side + 6
                    ),
                    jnp.logical_and(
                        2 <= self.state[last_pit], self.state[last_pit] <= 3
                    ),
                )

            def body(carry):
                """Capture seeds from the opponent's side"""
                state, last_pit, captured = carry
                captured += state[last_pit]
                state = state.at[last_pit].set(0)
                last_pit -= 1
                return state, last_pit, captured

            initial_carry = (  # Initial values for the loop
                self.state,
                last_pit,
                0,
            )  # 0 is the initially captured value
            final_state, final_last_pit, total_captured = lax.while_loop(
                condition, body, initial_carry
            )
            return final_state, total_captured

        @jit
        def calculate_end_game_reward():
            """Calculate the reward when the game ends"""
            return lax.select(
                self.scores[self.current_player] > self.scores[1 - self.current_player],
                100,
                lax.select(
                    self.scores[self.current_player]
                    < self.scores[1 - self.current_player],
                    -100,
                    -50,
                ),
            )

        @jit
        def determine_winner():
            """Determine the winner of the game"""
            return lax.select(
                self.scores[0] > self.scores[1],
                0,
                lax.select(self.scores[1] > self.scores[0], 1, -1),
            )

        @jit
        def is_player_side_empty(player: jnp.int8 = 0):
            """Check if the player's side is empty"""
            start = player * 6
            player_side = lax.dynamic_slice(self.state, (start,), (6,))
            return jnp.all(player_side == 0)

        @jit
        def handle_empty_side(state, scores) -> jnp.array:
            """Handle the case when one player's side is empty"""
            # If one side is empty, all remaining seeds go to the other player
            scores = scores.at[0].add(jnp.sum(state[:6], dtype=jnp.int8))
            scores = scores.at[1].add(jnp.sum(state[6:], dtype=jnp.int8))
            state = jnp.zeros_like(state)
            return state, scores

        if jnp.logical_not(is_valid_move(action)):  # Check if the move is valid
            return self.state, -1, False, {"invalid": True}

        # Collect and distribute seeds
        seeds = self.state[action]
        self.state = self.state.at[action].set(0)
        current_pit = action
        # Distribute seeds
        while seeds > 0:
            current_pit = (current_pit + 1) % 12
            if current_pit != action:
                self.state = self.state.at[current_pit].add(1)
                seeds -= 1

        # Capture seeds if possible
        self.state, captured = capture_seeds(current_pit)
        self.scores = self.scores.at[self.current_player].add(captured)
        reward += 0.5 * captured

        # Check if the game is over

        if jnp.max(self.scores) > 23:  # Check if the game ends by score
            done = True
            reward += calculate_end_game_reward()
            info["winner"] = determine_winner()
        elif jnp.logical_or(
            is_player_side_empty(0), is_player_side_empty(1)
        ):  # Check if the game ends by empty side
            self.state, self.scores = handle_empty_side(self.state, self.scores)
            done = True
            reward += calculate_end_game_reward()
            info["winner"] = determine_winner()
        else:  # Switch to the next player
            self.current_player = 1 - self.current_player
            self.action_space = jnp.arange(
                self.current_player * 6, (self.current_player + 1) * 6, dtype=jnp.int8
            )

        return self.state, reward, done, info

    def reset(self, player: jnp.int8 = 0):
        """Reset the game"""
        self.current_player = player
        self.action_space = jnp.arange(player * 6, (player + 1) * 6, dtype=jnp.int8)
        self.state = jnp.array([4] * 12, dtype=jnp.int8)
        self.scores = jnp.zeros(2, dtype=jnp.int8)
        return self.state

    def render(self):
        """Render the game board"""
        top_row = self.state[6:0:-1]
        bottom_row = self.state[6:]
        board = f"Player 2: {self.scores[1]:2d}\n"
        board += "   ┌────┬────┬────┬────┬────┬────┐\n"
        board += f"   │ {' │ '.join(f'{pit:2d}' for pit in top_row)} │\n"
        board += "───┼────┼────┼────┼────┼────┼────┤\n"
        board += f"   │ {' │ '.join(f'{pit:2d}' for pit in bottom_row)} │\n"
        board += "   └────┴────┴────┴────┴────┴────┘\n"
        board += f"Player 1: {self.scores[0]:2d}"
        print(board)
