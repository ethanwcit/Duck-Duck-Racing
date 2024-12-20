import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

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
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        # Ensure all inputs have consistent shapes
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = float(reward)
        next_state = np.array(next_state, dtype=np.float32)
        done = float(done)

        # Add transition to the buffer
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        # print(f"nextStates:{next_states}")
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

# DDPG Agent
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=8e-5, policy_noise=0.8, noise_clip=1.0, policy_delay=2):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim).cuda()
        self.critic2 = Critic(state_dim, action_dim).cuda()
        self.critic1_target = Critic(state_dim, action_dim).cuda()
        self.critic2_target = Critic(state_dim, action_dim).cuda()

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=lr)

        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.count = 0
    def select_action(self, state, exploration_noise=0.2):
        # Convert state to a PyTorch tensor if it's not already
        state = torch.FloatTensor(state).unsqueeze(0).cuda()  # Ensure it's a 2D tensor for batch input

        # Get action from the actor network
        action = self.actor(state)  

        # Add exploration noise
        action = action.cpu().data.numpy().flatten()  # Convert back to NumPy for further manipulation (optional)
        action = action + np.random.normal(0, exploration_noise, size=action.shape).clip(-self.noise_clip, self.noise_clip)  # Add noise for exploration

        # Clip to valid action range
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    def apply_policy_noise(self, next_states):
        # Get the actions predicted by the actor_target
        next_actions = self.actor_target(next_states)
        
        # Generate noise of the same shape as the actions
        noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        
        # Add noise to the actions and clamp to valid range
        next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
        
        # print(f"Actions shape after noise: {next_actions.shape}")
        return next_actions
        
    def train(self, batch_size=64):
        if len(self.replay_buffer.buffer) < batch_size:
                    return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).cuda()
        actions = torch.FloatTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).unsqueeze(1).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        dones = torch.FloatTensor(dones).unsqueeze(1).cuda()

        # Add noise to actions for target policy smoothing
        next_actions = self.apply_policy_noise(next_states)
        # print(f"State shape: {next_states.shape}, Actions shape: {next_actions.shape}")
        # Compute target Q-values
        target_Q1 = self.critic1_target(next_states, next_actions)
        target_Q2 = self.critic2_target(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q.detach()

        # Update Critic networks
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        critic_loss1 = nn.MSELoss()(current_Q1, target_Q)
        critic_loss2 = nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        # Delayed policy updates
        if self.count % self.policy_delay == 0:
            # Update Actor network
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.count += 1