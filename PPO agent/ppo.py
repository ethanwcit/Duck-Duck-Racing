import torch
from torch.distributions import MultivariateNormal
from ActorCritic import Actor
from ActorCritic import Critic

class PPO:
    def __init__(self, env):

        self._init_hyperparameters()
        self.env = env
        self.inputs = env.observation_space.shape[0]
        self.outputs = env.action_space.shape[0]

        self.actor = Actor(self.inputs, self.outputs)
        self.critic = Critic(self.inputs)

        self.matrix = torch.full(size=(self.outputs,), fill_value=0.5)
        self.diagonals = torch.diag(self.matrix)

    # Initialize hyperparameters
    def _init_hyperparameters(self):
        # Default values
        self.ts_batch = 4800  # timesteps to run per batch
        self.max_ts_ep = 1600  # Max timesteps per episode
        self.gamma = 0.95  # Discount factor

    def ppo(self, k, batch_size=128):

        for time in range(0, k, batch_size):
            
            # Optional: Log progress
            if time % 1000 == 0:
                print("Timestep", time)

    def learn(self, total_ts):
        ts_til_now = 0
        while ts_til_now < total_ts:
            instances, actions, log_probabilities, reward_tg, lengths = self.playout()

    # As this is on-policy, we need new data for every actor/critic update
    def playout(self):
        instances = []          # observations collected this batch
        actions = []            # actions collected this batch
        log_probabilities = []  # log probabilities of each action
        rewards = []            # rewards
        rewards_tg = []         # rewards to go of each timestep
        lengths = []            # length of each episode
        x = 0                   # Keeps track of how many timesteps we've run so far this batch

        while x < self.ts_batch:
            episode_rewards = []  # rewards collected per episode

            # Environment reset
            instance = self.env.reset()
            done = False

            for episode_x in range(self.max_ts_ep):
                x += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                instances.append(instance)

                # Calculate action and make a step in the env.
                action, log_probability = self. mean_action(instance)
                instance, rew, done, _ = self.env.step(action)

                # Track recent reward, action, and action log probability
                episode_rewards.append(rewards)
                actions.append(action)
                log_probabilities.append(log_probability)

                if done:
                    break

            # Collect length and rewards from episode
            lengths.append(episode_x + 1)
            rewards.append(episode_rewards)

        # Reshape data as tensors in the shape specified in function description, before returning
        instances = torch.tensor(instances, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        log_probabilities = torch.tensor(log_probabilities, dtype=torch.float)

        rewards_tg = self.reward_to_go(rewards)

        return instances, actions, log_probabilities, rewards_tg, lengths

    # returns a sampled action and its log probability
    def mean_action(self, instance):
        mean = self.actor(instance)
        distribution = MultivariateNormal(mean, self.matrix)
        action = distribution.sample()
        log_probability = distribution.log_probability(action)
        sampled_action = action.detach().numpty()
        action_log_probability = log_probability.detach()

        return sampled_action, action_log_probability

    # returns rewards to go per episode per batch
    def reward_to_go(self, rewards):
        reward_tg = []

        for episode_rewards in reversed(rewards):
            disc_reward = 0
            for reward in reversed(episode_rewards):
                disc_reward = reward + disc_reward * self.gamma
                reward_tg.insert(0,disc_reward)

        reward_tg = torch.tensor(reward_tg, dtype=torch.float)
        return reward_tg



