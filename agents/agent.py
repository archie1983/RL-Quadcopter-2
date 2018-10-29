import numpy as np
from task import Task

# AE: First of all, my state space is continuous (coordinates are real numbers), so is my action space (speeds are real numbers too).
# AE: That means that it's best to employ one of the methods in the final lectures, i.e. Actor-Critic, 
# AE: with two collections of features- one for state value function and another for action value function. With that I will
# AE: need to create usable function approximations for state value function (Critic) and action value function (Actor). 
# AE: That's of course a job for TensorFlow or Keras.
# AE:
# AE: Now, luckily Udacity guys have already provided an actor-critic implementation in Keras. All that's left is to adapt
# AE: it to the task and tweak.

from agents.ddpg_actor import Actor # Udacity's implented Actor
from agents.ddpg_critic import Critic # Udacity's implented Critic
from agents.ou_noise import OUNoise # Udacity's implented Ormsteihn-Ulenbeck noise sampler.
from agents.replay_buffer import ReplayBuffer # Udacity's implented memory buffer for action replays.

class AE_DDPG_Agent():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        # AE: Although OUNoise gives me a convenient set of randomness for each of the rotors, I still need
        # AE: to make a decision myself on how to apply the randomness and how to manage its magnitude 
        # AE: (i.e. my eplore vs exploit strategy). These variables will do that.
        self.explore_start = 1.0      # AE: exploration probability at start
        self.explore_stop = 0.001     # AE: minimum exploration probability 
        self.decay_rate = 0.003       # AE: exponential decay rate for exploration prob
        self.magnitude_coeff = 0.1    # AE: a coefficient to limit randomness

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0 # AE: additive to the noise. mu * theta will be directly added
        
        self.exploration_theta = 0.15 # AE: old noise will be multiplied by this
        self.exploration_sigma = 0.2  # AE: new noise will be multiplied by this
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        # AE: The learning rate. How much we trust the new values compared to the old ones.
        self.tau = 0.0001  # for soft update of target parameters

        # AE: current reward in learning procedure (for statistics)
        self.score = -np.inf

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state

        self.best_score = -np.inf
        self.score = -np.inf
        self.total_reward = 0.0
        self.count = 0
        
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

        self.total_reward += reward
        self.count += 1

        # AE: Score (average reward in this episode so far) and best score for statistics
        self.score = self.total_reward / float(self.count)
        if self.score > self.best_score:
            self.best_score = self.score

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])

        # AE: directly sampling approximated value from learned action-value function.
        action = self.actor_local.model.predict(state)[0]
        
        # AE: and adding some noise to that for unpredictability.
        # AE: The magnitude of noise has to drop over time.
        explore_p = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.count)
        #self.noise.update_mu(explore_p)
        noise_sample = self.magnitude_coeff * explore_p * self.noise.sample()
        #noise_sample = explore_p * np.random.randn(self.action_size)
        #print("Noi=", s)

        return list(action + noise_sample * self.action_size)  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        # AE: Updating NN weights directly in the passed model (actor or critic).
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)