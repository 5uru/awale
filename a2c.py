# %%
from model import AwaleModel
from jax import numpy as jnp
from jax import random
from typing import Sequence , Dict
import flax.linen as nn
import jax
from jax import jit , vmap , grad , value_and_grad
from jax import lax
from awale.env import AwaleJAX
import optax
import numpy as np

# %%
class AwaleCritic(nn.Module) :
    """Critic (Value) network for Awale."""
    features: Sequence[ int ]

    @nn.compact
    def __call__(self ,

                 board: jnp.ndarray ,
                 scores: jnp.ndarray ,
                 valid_actions: jnp.ndarray ,
                 rng: random.PRNGKey ,
                 training: bool = False) -> jnp.ndarray :
        """
        Estimate the value of the current game state.
        
        Args:
            board: shape (12,) representing seeds in each pit
            scores: shape (2,) representing current scores
            valid_actions: shape (12,) binary mask of valid moves
            training: whether to apply dropout
            
        Returns:
            value: shape (1,) predicted value of the state [-1, 1]
                  where 1 means current player is winning,
                  -1 means opponent is winning
        """
        # Process board state
        x = board.reshape(-1)
        x = nn.Dense(features=self.features[ 0 ])(x)
        x = nn.relu(x)

        # Process scores
        s = scores.reshape(-1)
        s = nn.Dense(features=self.features[ 0 ] // 2)(s)
        s = nn.relu(s)

        # Process valid actions (indicates mobility)
        v = valid_actions.reshape(-1)
        v = nn.Dense(features=self.features[ 0 ] // 4)(v)
        v = nn.relu(v)

        # Combine all features
        x = jnp.concatenate([ x , s , v ] , axis=-1)

        # Process through dense layers
        for feat in self.features[ 1 :-1 ] :
            x = nn.Dense(features=feat)(x)
            x = nn.relu(x)
            if training :
                x = nn.Dropout(0.1)(x , deterministic=not training , rng=rng)

        # Value head
        x = nn.Dense(features=self.features[ -1 ])(x)
        x = nn.relu(x)

        # Final value prediction
        value = nn.Dense(features=1)(x)
        value = nn.tanh(value)  # Scale to [-1, 1]

        return value

# %%
def create_awale_model(rng: random.PRNGKey) :
    """Creates and initializes the Awale model."""
    model = AwaleModel(features=[ 128 , 256 , 128 , 64 ])

    # Create dummy inputs for initialization
    board = jnp.zeros((12 ,))
    scores = jnp.zeros((2 ,))
    valid_actions = jnp.ones((12 ,))  # Only 6 moves per player

    # Initialize parameters

    variables = model.init(rng , board , scores , valid_actions , rng)

    return model , variables

# %%
def create_critic(rng: random.PRNGKey) :
    """Creates and initializes the Critic model."""
    model = AwaleCritic(features=[ 128 , 256 , 128 , 64 , 32 ])

    # Create dummy inputs for initialization
    board = jnp.zeros((12 ,))
    scores = jnp.zeros((2 ,))
    valid_actions = jnp.ones((12 ,))

    # Initialize parameters
    variables = model.init(rng , board , scores , valid_actions , rng)

    return model , variables

# %%
actor_model , actor_variables = create_awale_model(random.PRNGKey(0))
critic_model , critic_variables = create_critic(random.PRNGKey(1))

# %%
critic_variables

# %%
gamma = 0.99

# %%

def pick_action(model: AwaleModel ,
                variables ,
                board: jnp.ndarray ,
                scores: jnp.ndarray ,
                valid_actions: jnp.ndarray ,
                step) :
    """Selects an action using the actor network."""
    # Get the action probabilities
    key = random.PRNGKey(step)
    probabilities = model.apply(variables , board , scores , valid_actions , training=False , rng=key)

    return jnp.argmax(probabilities)

# %%
@jit
def get_valid_actions(positions: jnp.ndarray) :
    possible_actions = jnp.zeros(12)
    possible_actions = possible_actions.at[ positions ].set(1)
    return possible_actions

# %%
def compute_single_loss(params: Dict ,
                        critic: AwaleCritic ,
                        board: jnp.ndarray ,
                        score: jnp.ndarray ,
                        valid_action: jnp.ndarray ,
                        target_value: jnp.ndarray) -> jnp.ndarray :
    """
    Compute loss for a single example.
    
    Args:
        params: Critic parameters
        critic: Critic model
        board: Single board state (12,)
        score: Single score pair (2,)
        valid_action: Single valid action mask (12,)
        target_value: Single target value (1,)
    
    Returns:
        loss: Scalar loss value
    """
    rng = random.PRNGKey(0)
    predicted_value = critic.apply(
            params ,
            board ,
            score ,
            valid_action ,
            training=True ,
            rng=rng
    )
    return jnp.sum(jnp.square(predicted_value - target_value))

# %%
# Vectorize the loss computation over the batch dimension
vmapped_loss = jax.vmap(
        compute_single_loss ,
        in_axes=(None , None , 0 , 0 , 0 , 0)  # Only vectorize over the data, not params
)

# %%

def compute_batch_loss(params: Dict ,
                       critic: AwaleCritic ,
                       batch) :
    """
    Compute loss for a batch using vmap.
    
    Args:
        params: Critic parameters
        critic: Critic model
        batch:
    
    Returns:
        total_loss: Scalar loss value
    """
    # Compute individual losses for each example in the batch
    individual_losses = vmapped_loss(
            params ,
            critic ,
            batch[ 0 ] ,
            batch[ 1 ] ,
            batch[ 2 ] ,
            batch[ 3 ]
    )

    # Average loss across batch
    batch_loss = jnp.mean(individual_losses)

    # Add L2 regularization
    l2_reg = 0.01 * sum(
            jnp.sum(jnp.square(p))
            for p in jax.tree_util.tree_leaves(params)
    )

    total_loss = batch_loss + l2_reg

    return total_loss

# %%
def compute_policy_loss(policy_params: Dict ,
                        policy_network: AwaleModel ,
                        batch: Dict[ str , jnp.ndarray ] ,
                        advantages: jnp.ndarray ,
                        entropy_coef: float = 0.01) :
    """
    Compute the policy loss using the advantage function.
    
    Args:
        policy_params: Parameters of the policy network
        policy_network: The AwaleModel instance
        batch: Dictionary containing:
            - boards: Game states (B, 12)
            - scores: Player scores (B, 2)
            - valid_actions: Valid moves mask (B, 12)
            - actions_taken: Actions that were actually taken (B,)
        advantages: Advantage values from critic (B,)
        entropy_coef: Coefficient for entropy bonus
    
    Returns:
        loss: The scalar loss value

    """

    key = random.PRNGKey(0)
    # Get policy distributions for each state
    action_probs = vmap(lambda board , scores , valid_actions : policy_network.apply(policy_params , board , scores , valid_actions , training=True , rng=key))(batch[ 'boards' ] , batch[ 'scores' ] ,
                                                                                                                                                                batch[ 'valid_actions' ])

    # Create mask for taken actions
    batch_size = action_probs.shape[ 0 ]
    action_indices = jnp.arange(batch_size)
    actions_mask = jax.nn.one_hot(batch[ 'actions_taken' ] , 12)

    # Get probabilities of taken actions
    selected_action_probs = jnp.sum(action_probs * actions_mask , axis=1)

    # Compute log probabilities of taken actions
    log_probs = jnp.log(selected_action_probs + 1e-8)

    # Policy gradient loss
    policy_loss = -jnp.mean(log_probs * advantages)

    # Entropy bonus (to encourage exploration)
    entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-8) , axis=1)
    entropy_loss = -entropy_coef * jnp.mean(entropy)

    # Add L2 regularization
    l2_coef = 0.01
    l2_reg = l2_coef * sum(
            jnp.sum(jnp.square(p))
            for p in jax.tree_util.tree_leaves(policy_params)
    )

    # Total loss
    total_loss = policy_loss + entropy_loss + l2_reg

    return total_loss

# %%
# Define loss function for this batch
def loss_fn(params , policy_network , batch , advantages) :
    loss = compute_policy_loss(
            params , policy_network , batch , advantages
    )
    return loss

# %%
env = AwaleJAX( )


# %%
reward_records = [ ]
solver = optax.chain(
        optax.clip_by_global_norm(1.0) ,  # Gradient clipping
        optax.adam(learning_rate=0.001)  # Adam optimizer
)

# %%
actor_optimizer = solver.init(actor_variables)
critic_optimizer = solver.init(critic_variables)

# %%
for i in range(1000) :
    #
    # Run episode till done
    #
    key = jax.random.PRNGKey(i)
    done = False
    board = [ ]
    scores = [ ]
    valid_actions = [ ]
    actions = [ ]
    rewards = [ ]
    state = env.reset(key)
    while not done :
        board.append(state.board)
        scores.append(state.score)
        valid_actions.append(get_valid_actions(state.action_space))
        action = pick_action(model=actor_model ,
                             variables=actor_variables ,
                             board=state.board ,
                             scores=state.score ,
                             valid_actions=get_valid_actions(state.action_space) ,
                             step=i)

        state , reward , done = env.step(state , action)
        actions.append(action)
        rewards.append(reward)

    #
    # Get cumulative rewards
    #
    cum_rewards = np.zeros_like(rewards)
    reward_len = len(rewards)
    for j in reversed(range(reward_len)) :
        cum_rewards[ j ] = rewards[ j ] + (cum_rewards[ j + 1 ] * gamma if j + 1 < reward_len else 0)

    cum_rewards = jnp.array(cum_rewards)

    critic_loss , critic_grads = jax.value_and_grad(
            compute_batch_loss
    )(critic_variables , critic_model , (jnp.array(board) , jnp.array(scores) , jnp.array(valid_actions) , cum_rewards))

    critic_updates , critic_optimizer = solver.update(critic_grads , critic_optimizer , critic_variables)
    critic_variables = optax.apply_updates(critic_variables , critic_updates)

    values = vmap(
            lambda board , scores , valid_actions : critic_model.apply(critic_variables , board , scores , valid_actions , rng=random.PRNGKey(0))
    )(jnp.array(board) , jnp.array(scores) , jnp.array(valid_actions))

    # Optimize policy loss (Actor)
    advantages = cum_rewards - values
    policy_loss , policy_grads = jax.value_and_grad(
            loss_fn
    )(actor_variables , actor_model , {
            'boards' : jnp.array(board) ,
            'scores' : jnp.array(scores) ,
            'valid_actions' : jnp.array(valid_actions) ,
            'actions_taken' : jnp.array(actions)
    } , advantages)

    actor_updates , actor_optimizer = solver.update(policy_grads , actor_optimizer , actor_variables)
    actor_variables = optax.apply_updates(actor_variables , actor_updates)

    print("Run episode{} with rewards {}".format(i , sum(rewards)) , end="\r")
    reward_records.append(sum(rewards))


# %%
import plotly.graph_objects as go
from typing import List


def plot_rewards(reward_records: List[ float ]) :
    """
    Plot rewards using plotly without numpy dependency
    """
    # Calculate moving average using pure Python
    average_reward = [ ]
    for idx in range(len(reward_records)) :
        if idx < 50 :
            window = reward_records[ :idx + 1 ]
        else :
            window = reward_records[ idx - 49 :idx + 1 ]
        avg = sum(window) / len(window)
        average_reward.append(avg)

    # Create the plot
    fig = go.Figure( )

    # Add raw rewards
    fig.add_trace(
            go.Scatter(
                    y=reward_records ,
                    mode='lines' ,
                    name='Raw Rewards' ,
                    opacity=0.5
            )
    )

    # Add moving average
    fig.add_trace(
            go.Scatter(
                    y=average_reward ,
                    mode='lines' ,
                    name='50-Episode Moving Average' ,
                    line=dict(width=2)
            )
    )

    # Update layout
    fig.update_layout(
            title='Training Rewards over Time' ,
            xaxis_title='Episode' ,
            yaxis_title='Reward' ,

            hovermode='x unified'
    )

    return fig

# %%
fig = plot_rewards(reward_records)
fig.show( )

# %%
cum_rewards

# %%
