import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from ActorCritic import Actor
from ActorCritic import Critic

class PPO:
    def __init__(self,env):

        self.env = env
        self.inputs = env.observation_space.shape[0]
        self.outputs = env.action_space.shape[0]

        self.actor = Actor(self.inputs, self.outputs)
        self.critic = Critic(self.inputs)

        #hyperparameters - CHANGE
        self.episode_size = 1600 
        self.batch_size = 4800
        self.gamma = 0.99
        self.number_of_updates = 5
        self.clipping = 0.3
        self.learning_rate = 0.005

        self.actor_optimiser = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.cov_var = torch.full(size=(self.outputs,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def ppo(self, k, batch_size=128):

        for time in range(0, k, batch_size):
            
            # Optional: Log progress
            if time % 1000 == 0:
                print("Timestep", time)

    def learn(self, episode):
        current_timesteps = 0
        while current_timesteps < episode:

            states, actions, log_probability, reward_to_go, episode_lengths = self.rollout()

            current_timesteps += np.sum(episode_lengths)
            value, _ = self.evaluate(states, actions)

            # Calculate advantage
            advantage = reward_to_go - value.detach()
            small_number = 1e-10
            advantage = (advantage - advantage.mean()) / (advantage.std() + small_number)

            for _ in range(self.number_of_updates):

                value, current_probabilities = self.evaluate(states, actions)

                ratio = torch.exp(current_probabilities - log_probability)

                actor_loss = (-torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.clipping, 1+ self.clipping) * advantage)).mean()
                critic_loss = nn.MSELoss()(value, reward_to_go)

                # Gradients for actor network
                self.actor_optimiser.zero_grad()
                actor_loss.backward()(retain_graph=True)
                self.actor_optimiser.step()

                # Gradients for critic network
                self.critic_optimiser.zero_grad()
                critic_loss.backward()
                self.critic_optimiser.step()

    def evaluate(self, states, actions):
        # Evaluate values using critic network and log probabilities using actor network
        return self.critic(states).squeeze(), MultivariateNormal(self.actor(states), self.cov_mat).log_prob(actions)
    
    def rollout(self):
        #To collect a set of trajectories, we have to rollout the environment by running the policy.
        states, actions, rewards, log_probability, reward_to_go, episode_lengths = [], [], [], [], [], []

        steps = self.batch_size

        for step in range(steps):
            intermediate_rewards = []
            state_space = self.env.reset()
            check = False
            episode = 1

            while episode <= self.episode_size:
                states.append(state_space)
                
                mean = self.actor(state_space)
                dist = MultivariateNormal(mean, self.cov_mat)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                state_space, reward, check, _ = self.env.step(action)

                intermediate_rewards.append(reward)
                actions.append(action.detach.numpy())
                log_probability.append(log_prob.detach())

                if check:
                    break

                episode += 1
            
            episode_lengths.append(episode)
            rewards.append(intermediate_rewards)


        rewards_tg = self.reward_to_go(rewards)

        return torch.tensor(states, dtype=torch.float), torch.tensor(actions, dtype=torch.float), torch.tensor(log_probability, dtype=torch.float), rewards_tg, episode_lengths


    def reward_to_go(self, rewards):
  
        return 0


    """
    TEST CODE 
    
    import gym
    env = gym.make('Pendulum-v0')
    model = PPO(env)  """