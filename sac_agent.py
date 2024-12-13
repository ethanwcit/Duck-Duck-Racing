import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_action

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=1e-3,alpha = 0.2):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim).cuda()
        self.critic1_target = Critic(state_dim, action_dim).cuda()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = Critic(state_dim, action_dim).cuda()
        self.critic2_target = Critic(state_dim, action_dim).cuda()
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.alpha = alpha

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=64):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).cuda()
        actions = torch.FloatTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).unsqueeze(1).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        dones = torch.FloatTensor(dones).unsqueeze(1).cuda()

        # Train Critic

        with torch.no_grad():

            next_actions = self.actor_target(next_states)
            target_Q1 = self.critic1_target(next_states, next_actions)
            target_Q2 = self.critic2_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * torch.min(target_Q1,target_Q2)


        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        critic1_loss = nn.MSELoss()(current_Q1, target_Q)
        critic2_loss = nn.MSELoss()(current_Q2, target_Q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Train Actor (to be fixed)
        log_probs = torch.log(torch.abs(self.actor(states)) + 1e-6).sum(dim=1)
        actor_loss = (self.alpha * log_probs - 
              torch.min(self.critic1(states, self.actor(states)),
                        self.critic2(states, self.actor(states)))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, directory):
        """
        Save the SAC model (actor, critic1, critic2) to the specified directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save actor and critics
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(directory, "critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(directory, "critic2.pth"))

        # Save actor and critic optimizers
        torch.save(self.actor_optimizer.state_dict(), os.path.join(directory, "actor_optimizer.pth"))
        torch.save(self.critic1_optimizer.state_dict(), os.path.join(directory, "critic1_optimizer.pth"))
        torch.save(self.critic2_optimizer.state_dict(), os.path.join(directory, "critic2_optimizer.pth"))

        print(f"Model saved to {directory}")

    def load_model(self, directory):
        """
        Load the SAC model (actor, critic1, critic2) from the specified directory.
        """
        # Load actor and critics
        self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth")))
        self.critic1.load_state_dict(torch.load(os.path.join(directory, "critic1.pth")))
        self.critic2.load_state_dict(torch.load(os.path.join(directory, "critic2.pth")))

        # Load actor and critic optimizers
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(directory, "actor_optimizer.pth")))
        self.critic1_optimizer.load_state_dict(torch.load(os.path.join(directory, "critic1_optimizer.pth")))
        self.critic2_optimizer.load_state_dict(torch.load(os.path.join(directory, "critic2_optimizer.pth")))

        print(f"Model loaded from {directory}")