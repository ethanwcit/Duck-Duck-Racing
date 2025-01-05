import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
# Determine the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.buffer = deque(maxlen=max_size) #Use deque instead of list for faster operation
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        # Detach the tensor, move it to CPU, and convert to numpy array
        action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action, dtype=np.float32)
        reward = reward.detach().cpu().numpy() if isinstance(reward, torch.Tensor) else np.array(reward, dtype=np.float32)
        next_state = next_state.detach().cpu().numpy() if isinstance(next_state, torch.Tensor) else np.array(next_state, dtype=np.float32)
        done = done.detach().cpu().numpy() if isinstance(done, torch.Tensor) else np.array(done, dtype=np.float32)
        # Add transition to the buffer
        self.buffer.append((state, action, reward, next_state, done))
        # if len(self.buffer) > self.max_size:
        #     self.buffer.pop(0)

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
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=1e-3):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        return self.actor(state).cpu().data.numpy().flatten()

    
    def add_to_replay(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    def train(self, batch_size=64):
        if len(self.replay_buffer.buffer) < batch_size:
            return {'actor': 0.0, 'critic': 0.0}, 0.0  # Return default values if there's insufficient data.

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Train Critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q.detach()
        current_Q = self.critic(states, actions)

        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return {'actor': actor_loss.item(), 'critic': critic_loss.item()}, current_Q.mean().item()
    
    def save(self, filename, folder="saved_agents"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, os.path.join(folder, filename))
        print(f"Agent's state saved to {filename}.")
