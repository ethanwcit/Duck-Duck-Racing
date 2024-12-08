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