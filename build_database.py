import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Tuple, List, Dict, NamedTuple
from collections import defaultdict
import numpy as np
from functools import partial

# Import from your environment
from awale.env import AwaleJAX
from awale.utils import (
    State,
    get_action_space,
    calculate_reward,
)
from awale.viewer import save_board_svg
from model import AwaleModel
import flax

# Constants
BOARD_SIZE = 12
SEEDS_PER_PIT = 4
PLAYER_SIDE_SIZE = 6
CAPTURE_REWARD_MULTIPLIER = 0.5


def create_state_copy(state: State) -> State:
    """Create a deep copy of a State object."""
    return State(
        board=jnp.array(state.board),
        action_space=jnp.array(state.action_space),
        key=state.key,
        score=jnp.array(state.score),
        current_player=state.current_player,
    )


class MCTS:
    """Monte Carlo Tree Search with Neural Network guidance."""

    def __init__(self, model, params, n_playout=1000, c_puct=5.0):
        self.model = model
        self.params = params
        self.n_playout = n_playout
        self.c_puct = c_puct
        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores times edge s,a was visited
        self.Ns = {}  # stores times board s was visited
        self.Ps = {}  # stores initial policy returned by neural network
        self.env = AwaleJAX()
        self.rng_key = jax.random.PRNGKey(0)

    def _get_state_key(self, state: State) -> tuple:
        """Convert state to hashable format."""
        # Convert JAX arrays to numpy and then to tuples
        board_tuple = tuple(np.array(state.board).astype(np.int8).tolist())
        score_tuple = tuple(np.array(state.score).astype(np.int8).tolist())
        player = int(state.current_player)
        return (board_tuple, score_tuple, player)

    def get_action_probs(self, state: State, temp: float = 1.0) -> np.ndarray:
        """Run MCTS and return action probabilities."""
        for _ in range(self.n_playout):
            state_copy = create_state_copy(state)
            self.search(state_copy)

        s = self._get_state_key(state)
        counts = np.zeros(BOARD_SIZE)

        # Convert action_space to numpy array for iteration
        action_space = np.array(state.action_space)
        for action in action_space:
            action_int = int(action)  # Convert to Python int
            counts[action_int] = self.Nsa.get((s, action_int), 0)

        if temp == 0:  # Choose best action deterministically
            best_a = int(np.argmax(counts))
            probs = np.zeros(BOARD_SIZE)
            probs[best_a] = 1.0
        else:  # Sample based on visit counts
            counts = [float(x) ** (1.0 / temp) for x in counts]
            total = sum(counts)
            probs = [x / total for x in counts] if total > 0 else np.zeros(BOARD_SIZE)

        return np.array(probs, dtype=np.float32)

    def search(self, state: State) -> float:
        """Run one iteration of MCTS."""
        s = self._get_state_key(state)

        # Check if game is terminal
        board_sum = float(np.sum(state.board))
        score_max = float(np.max(state.score))

        if board_sum == 0 or score_max > 24:
            if state.score[0] > state.score[1]:
                return 1.0
            elif state.score[1] > state.score[0]:
                return -1.0
            return 0.0

        if s not in self.Ps:
            # Leaf node - evaluate with neural network
            valid_actions = np.zeros(BOARD_SIZE, dtype=np.float32)
            action_space = np.array(state.action_space)
            valid_actions[action_space] = 1

            # Get neural network prediction
            self.rng_key, eval_key = jax.random.split(self.rng_key)
            probs = self.model.apply(
                {"params": self.params},
                state.board,
                state.score,
                jnp.array(valid_actions),
                eval_key,
                training=False,
            )

            self.Ps[s] = np.array(probs) * valid_actions  # Mask invalid actions
            self.Ns[s] = 0
            return 0.0

        # Select action using UCB
        cur_best = -float("inf")
        best_act = -1

        action_space = np.array(state.action_space)
        for a in action_space:
            a_int = int(a)  # Convert to Python int
            if (s, a_int) in self.Qsa:
                u = self.Qsa[(s, a_int)] + self.c_puct * self.Ps[s][a_int] * np.sqrt(
                    self.Ns[s]
                ) / (1 + self.Nsa[(s, a_int)])
            else:
                u = self.c_puct * self.Ps[s][a_int] * np.sqrt(self.Ns[s])

            if u > cur_best:
                cur_best = u
                best_act = a_int

        # Make move and get next state
        next_state, reward, done = self.env.step(state, jnp.int8(best_act))

        v = float(reward) + (0.0 if done else self.search(next_state))

        # Update statistics
        if (s, best_act) in self.Qsa:
            self.Qsa[(s, best_act)] = (
                self.Nsa[(s, best_act)] * self.Qsa[(s, best_act)] + v
            ) / (self.Nsa[(s, best_act)] + 1)
            self.Nsa[(s, best_act)] += 1
        else:
            self.Qsa[(s, best_act)] = v
            self.Nsa[(s, best_act)] = 1

        self.Ns[s] += 1
        return -v


class TrainerMCTS:
    """Training pipeline using MCTS with neural network."""

    def __init__(self, model_config, learning_rate=0.001):
        self.model = AwaleModel(features=model_config["features"])
        self.rng_key = jax.random.PRNGKey(0)

        # Initialize model
        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_board = jnp.zeros(BOARD_SIZE, dtype=jnp.float32)
        dummy_scores = jnp.zeros(2, dtype=jnp.float32)
        dummy_valid = jnp.ones(BOARD_SIZE, dtype=jnp.float32)

        params = self.model.init(
            init_key, dummy_board, dummy_scores, dummy_valid, init_key, training=False
        )

        # Create training state
        tx = optax.adam(learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params["params"],
            tx=tx,
        )

        self.env = AwaleJAX()

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch):
        """Single training step."""

        def loss_fn(params):
            logits = self.model.apply(
                {"params": params},
                batch["board"],
                batch["scores"],
                batch["valid_moves"],
                self.rng_key,
                training=True,
            )

            loss = optax.softmax_cross_entropy(logits, batch["mcts_probs"])
            return loss.mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss

    def self_play(self, num_games=100):
        """Generate self-play data using MCTS."""
        training_data = []
        mcts = MCTS(self.model, self.state.params)

        for game_idx in range(num_games):
            print(f"Playing game {game_idx + 1}/{num_games}")
            self.rng_key, game_key = jax.random.split(self.rng_key)
            state = self.env.reset(game_key)
            game_data = []
            done = False

            while not done:
                # Create valid actions mask
                valid_moves = np.zeros(BOARD_SIZE, dtype=np.float32)
                valid_moves[state.action_space] = 1

                # Get MCTS probabilities
                temp = 1.0 if len(game_data) < 10 else 0.1
                pi = mcts.get_action_probs(state, temp)

                game_data.append(
                    {
                        "board": np.array(state.board),
                        "scores": np.array(state.score),
                        "valid_moves": valid_moves,
                        "pi": pi,
                    }
                )

                # Sample action from probabilities
                legal_moves = np.array(state.action_space)
                legal_probs = pi[legal_moves]
                legal_probs = legal_probs / legal_probs.sum()  # Renormalize
                action = np.random.choice(legal_moves, p=legal_probs)

                # Make move
                state, reward, done = self.env.step(state, jnp.int8(action))

            # Calculate final value
            if state.score[0] > state.score[1]:
                final_value = 1.0
            elif state.score[1] > state.score[0]:
                final_value = -1.0
            else:
                final_value = 0.0

            # Add value to all positions
            for idx, data in enumerate(game_data):
                value = final_value if idx % 2 == 0 else -final_value
                data["value"] = value
                training_data.append(data)

        return training_data

    def train(
        self,
        num_iterations=100,
        games_per_iteration=100,
        batch_size=256,
        epochs_per_iteration=10,
    ):
        """Main training loop."""
        for iteration in range(num_iterations):
            print(f"Starting iteration {iteration+1}/{num_iterations}")

            # Generate self-play data
            training_data = self.self_play(games_per_iteration)

            # Train on collected data
            for epoch in range(epochs_per_iteration):
                np.random.shuffle(training_data)
                total_loss = 0
                num_batches = 0

                for i in range(0, len(training_data), batch_size):
                    batch = self._prepare_batch(training_data[i : i + batch_size])
                    self.state, loss = self.train_step(self.state, batch)
                    total_loss += loss
                    num_batches += 1

                avg_loss = total_loss / num_batches
                print(
                    f"Iteration {iteration+1}, Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}"
                )

            # Save checkpoint
            if (iteration + 1) % 5 == 0:
                self.save_model(f"awale_model_iter_{iteration+1}.pkl")

    def _prepare_batch(self, data):
        """Convert training data to JAX arrays."""
        return {
            "board": jnp.array([d["board"] for d in data]),
            "scores": jnp.array([d["scores"] for d in data]),
            "valid_moves": jnp.array([d["valid_moves"] for d in data]),
            "mcts_probs": jnp.array([d["pi"] for d in data]),
            "values": jnp.array([d["value"] for d in data]),
        }

    def save_model(self, path):
        """Save model parameters."""
        with open(path, "wb") as f:
            params = {"params": self.state.params}
            flax.serialization.to_bytes(params)

    def load_model(self, path):
        """Load model parameters."""
        with open(path, "rb") as f:
            params = flax.serialization.from_bytes(self.state.params, f.read())
            self.state = self.state.replace(params=params["params"])


def create_model_config():
    """Create model configuration."""
    return {"features": [128, 256, 128, 64]}


if __name__ == "__main__":
    config = create_model_config()
    trainer = TrainerMCTS(config)

    trainer.train(
        num_iterations=100,
        games_per_iteration=100,
        batch_size=256,
        epochs_per_iteration=10,
    )
