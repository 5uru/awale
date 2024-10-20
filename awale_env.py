class Awale:
    def __init__(self, player: int = 0):
        self.state = [4] * 12
        self.scores = [0, 0]
        self.current_player = player
        self.action_space = list(range(player * 6, (player + 1) * 6))

    def step(self, action):
        reward = -0.1
        done = False
        info = {}

        if not self._is_valid_move(action):
            return self.state, -1, False, {"invalid": True}

        # Collect and distribute seeds
        seeds = self.state[action]
        self.state[action] = 0
        current_pit = action

        while seeds > 0:
            current_pit = (current_pit + 1) % 12
            if current_pit != action:
                self.state[current_pit] += 1
                seeds -= 1

        # Capture seeds if possible
        captured = self._capture_seeds(current_pit)
        self.scores[self.current_player] += captured
        reward += 0.5 * captured

        # Check if the game is over

        if max(self.scores) > 23:
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
            self.state[last_pit] = 0
            last_pit -= 1
        return captured

    def _handle_empty_side(self):
        # If one side is empty, all remaining seeds go to the other player
        self.scores[0] += sum(self.state[:6])
        self.scores[1] += sum(self.state[6:])
        self.state = [0] * 12

    def _is_player_side_empty(self, player):
        start = player * 6
        end = start + 6
        return all(self.state[i] == 0 for i in range(start, end))

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

    def reset(self):
        self.state = [4] * 12
        self.scores = [0, 0]
        self.current_player = 0
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
