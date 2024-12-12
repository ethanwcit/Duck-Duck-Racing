import torch
from torch.distributions import MultivariateNormal
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

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def ppo(self, k, batch_size=128):

        for time in range(0, k, batch_size):
            
            # Optional: Log progress
            if time % 1000 == 0:
                print("Timestep", time)
    
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
