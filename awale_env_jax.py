import jax.numpy as jnp


class AwaleJAX:
    def __init__(self, player: jnp.int8 = 0):
        self.current_player = player
        self.action_space = jnp.arange(player * 6, (player + 1) * 6, dtype=jnp.int8)
        self.state = jnp.array([4] * 12, dtype=jnp.int8)
        self.scores = jnp.zeros(2, dtype=jnp.int8)

    def step(self, action):
        reward = -0.1
        done = False
        info = {}

        if not self._is_valid_move(action):
            return self.state, -1, False, {"invalid": True}

        # Collect and distribute seeds
        seeds = self.state[action]
        self.state = self.state.at[action].set(0)
        current_pit = action

        while seeds > 0:
            current_pit = (current_pit + 1) % 12
            if current_pit != action:
                self.state = self.state.at[current_pit].add(1)
                seeds -= 1

        # Capture seeds if possible
        captured = self._capture_seeds(current_pit)
        self.scores = self.scores.at[self.current_player].add(captured)
        reward += 0.5 * captured

        # Check if the game is over

        if jnp.max(self.scores) > 23:
            done = True
            reward += self._calculate_end_game_reward()
            info["winner"] = self._determine_winner()
        elif self._is_player_side_empty(0) or self._is_player_side_empty(1):
            self._handle_empty_side()
            done = True
            reward += self._calculate_end_game_reward()
            info["winner"] = self._determine_winner()
        else:
            self.current_player = 1 - self.current_player
            self.action_space = jnp.arange(
                self.current_player * 6, (self.current_player + 1) * 6, dtype=jnp.int8
            )

        return self.state, reward, done, info

    def _is_valid_move(self, action):
        return (
            self.current_player * 6 <= action < (self.current_player + 1) * 6
            and self.state[action] > 0
        )

    def _capture_seeds(self, last_pit):
        captured = 0
        opponent_side = (1 - self.current_player) * 6
        while (
            opponent_side <= last_pit < opponent_side + 6
            and 2 <= self.state[last_pit] <= 3
        ):
            captured += self.state[last_pit]
            self.state = self.state.at[last_pit].set(0)
            last_pit -= 1
        return captured

    def _handle_empty_side(self):
        # If one side is empty, all remaining seeds go to the other player
        self.scores = self.scores.at[0].add(sum(self.state[:6]))
        self.scores = self.scores.at[1].add(sum(self.state[6:]))
        self.state = jnp.zeros_like(self.state)

    def _is_player_side_empty(self, player):
        start = player * 6
        end = start + 6
        return jnp.all(self.state[start:end] == 0)

    def _calculate_end_game_reward(self):
        if self.scores[self.current_player] > self.scores[1 - self.current_player]:
            return 100
        elif self.scores[self.current_player] < self.scores[1 - self.current_player]:
            return -100
        return -50

    def _determine_winner(self):
        if self.scores[0] > self.scores[1]:
            return 0
        elif self.scores[1] > self.scores[0]:
            return 1
        return -1  # Draw

    def reset(self, player: jnp.int8 = 0):
        self.current_player = player
        self.action_space = jnp.arange(player * 6, (player + 1) * 6, dtype=jnp.int8)
        self.state = jnp.array([4] * 12, dtype=jnp.int8)
        self.scores = jnp.zeros(2, dtype=jnp.int8)
        return self.state

    def render(self):
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
