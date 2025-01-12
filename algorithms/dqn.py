import sys
import os

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import jax
from jax import vmap
from jax import numpy as jnp
from flax import nnx
from awale_env.env import Awale
from utils import ReplayBuffer
from model import AwaleModel
import optax
import random
import copy


def rnd(key, action_space):
    # Obtenir les indices des actions valides en utilisant jnp.where
    valid_actions = jnp.where(action_space == 1)[0]
    if valid_actions.size == 0:
        raise ValueError("No valid actions available in the action space")
    return jax.random.choice(key, valid_actions)


def policy(key, state, model, epsilon):
    prob = jax.random.uniform(key)
    q = jnp.squeeze(
        model(state.board, state.action_space, state.current_player, state.score)
    )
    action_random = rnd(key, state.action_space)
    return jax.lax.cond(
        prob < epsilon,
        lambda _: action_random,  # Branche true
        lambda _: jnp.argmax(q),  # Branche false
        operand=None,  # Aucun paramètre supplémentaire ici
    )


@vmap
def q_learning_loss(q, target_q, action, action_select, reward, done, gamma=0.9):
    td_target = reward + gamma * (1.0 - done) * target_q[action_select]
    td_error = jax.lax.stop_gradient(td_target) - q[action]
    return td_error**2


def train_step(optimizer, model, target_model, batch):
    def loss_fn(model):
        q = vmap(model)(
            batch["board"], batch["action_space"], batch["player"], batch["score"]
        )
        target_q = vmap(target_model)(
            batch["next_board"],
            batch["next_action_space"],
            batch["next_player"],
            batch["next_score"],
        )
        action_select = vmap(model)(
            batch["next_board"],
            batch["next_action_space"],
            batch["next_player"],
            batch["next_score"],
        ).argmax(-1)
        return jnp.mean(
            q_learning_loss(
                q,
                target_q,
                batch["action"],
                action_select,
                batch["reward"],
                batch["done"],
            )
        )

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grad = grad_fn(model)
    optimizer.update(grad)
    return optimizer, loss


def run_experiment(
    num_episodes=1000,
    num_steps=50,
    batch_size=32,
    replay_size=1000,
    target_update_frequency=10,
    gamma=0.9,
):
    print("Running experiment")
    # Create environment
    env = Awale()
    replay_buffer = ReplayBuffer(replay_size)

    # logging
    ep_losses = []
    ep_returns = []
    key = jax.random.PRNGKey(0)

    # Create model
    model = AwaleModel(rngs=nnx.Rngs(key))
    target_model = AwaleModel(rngs=nnx.Rngs(key))

    # Build and initialize optimizer.
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001))

    epsilon = 0.1
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        # Initialize statistics
        ep_return = 1
        loss = 0
        try:
            while not done:
                action = policy(
                    jax.random.PRNGKey(random.randint(0, 10000)), state, model, epsilon
                )

                # Step the environment.
                next_state, reward, done, _ = env.step(jnp.int8(action))

                # Tell the agent about what just happened.
                replay_buffer.push(
                    board=state.board,
                    action_space=state.action_space,
                    player=state.current_player,
                    score=state.score,
                    action=action,
                    reward=reward,
                    next_board=next_state.board,
                    next_action_space=next_state.action_space,
                    next_player=next_state.current_player,
                    next_score=next_state.score,
                    done=done,
                )
                ep_return += reward

                # Update the replay buffer.
                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    optimizer, loss = train_step(optimizer, model, target_model, batch)
                    ep_losses.append(float(loss))
                    if len(ep_losses) % target_update_frequency == 0:
                        target_model = copy.deepcopy(model)

                state = next_state
        except Exception as e:
            print(e)
        epsilon = max(0.1, epsilon * 0.995)
        # Update episodic statistics
        ep_returns.append(ep_return)
        print(f"Episode #{episode}, Return {ep_return}, Loss {loss}")


if __name__ == "__main__":
    run_experiment()
