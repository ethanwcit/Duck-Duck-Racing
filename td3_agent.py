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
    
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu  # Mean of the noise
        self.theta = theta  # Mean reversion rate
        self.sigma = sigma  # Volatility
        self.state = np.ones(action_dim) * mu  # Initial state (noise)
        self.action_dim = action_dim

    def reset(self):
        """Reset the noise state."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        """Generate a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        # print(self.state)
        return self.state


# DDPG Agent
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.9759, tau=0.00864, lr=3e-4, policy_noise=0.393 , noise_clip=1.4493, policy_delay=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.00074661)

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=0.00074661)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=0.00074661)

        self.noise = OUNoise(action_dim)
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.count = 0

        # Noise decay parameters
        self.noise_decay = 0.995  # Decay factor
        self.noise_min = 0.05  # Minimum noise level

    def add_to_replay(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def select_action(self, state, exploration_noise=0.7):
        # Convert state to a PyTorch tensor if it's not already
        state = torch.FloatTensor(state).unsqueeze(0).cuda()  # Ensure it's a 2D tensor for batch input

        # Get action from the actor network
        action = self.actor(state)  

        # Add exploration noise
        action = action.cpu().data.numpy().flatten()  # Convert back to NumPy for further manipulation 
        # action = action + np.random.normal(0, exploration_noise, size=action.shape).clip(-self.noise_clip, self.noise_clip)  # Add noise for exploration
        action = action + np.clip(np.random.normal(0, exploration_noise, size=action.shape), -self.noise_clip, self.noise_clip)


        # Clip to valid action range
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    # def select_action(self, state, exploration_noise=0.1):
    #     state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Ensure it's a 2D tensor for batch input
    #     action = self.actor(state)
    #     # print(f"before {action}")
    #     noise = self.noise.sample() #* exploration_noise
    #     action = action + torch.tensor(noise).float().to(self.device)  # Add noise to action
    #     # print(action)
    #     action = torch.clamp(action, -self.max_action, self.max_action)  # Clip action
    #     action = action.cpu().data.numpy().flatten()  # Convert back to NumPy
    #     # print(action)
    #     return action
    
    def update_noise(self, episode):
        """Decay the noise as the episodes progress."""
        noise_scale = max(self.noise_min, self.noise_decay ** episode)
        self.noise.sigma = noise_scale  # Reduce the noise amplitude

    def apply_policy_noise(self, next_states):
        # Get the actions predicted by the actor_target
        next_actions = self.actor_target(next_states)
        
        noise = self.policy_noise # self.noise.sample() #* self.policy_noise
        next_actions = next_actions + torch.tensor(noise).float().to(self.device)  # Add noise to action
        next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)  # Clip action
        # next_actions = next_actions.cpu().data.numpy().flatten()  # Convert back to NumPy f
        # print(next_actions.shape, next_actions)
        return next_actions

    def update_target_nets(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def train(self, batch_size=64):
        if len(self.replay_buffer.buffer) < batch_size:
            return {'actor': 0.0, 'critic': 0.0}, 0.0  # Return default values if there's insufficient data.

        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute target Q values
        next_actions = self.apply_policy_noise(next_states)
        target_Q1 = self.critic1_target(next_states, next_actions)
        target_Q2 = self.critic2_target(next_states, next_actions)
        target_Q = rewards + (1 - dones) * self.gamma * torch.min(target_Q1, target_Q2).detach()

        # Update critic networks
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

        # Delayed actor updates
        actor_loss_val = None
        if self.count % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self.update_target_nets()
            actor_loss_val = actor_loss.item()
        self.count += 1
        actor_loss = actor_loss_val
        # print(actor_loss)
        # Return metrics
        avg_q_value = (current_Q1.mean().item() +  current_Q1.mean().item())/2  # Compute the average Q-value
        return {'actor': actor_loss, 'critic': (critic_loss1.item() + critic_loss2.item()) / 2}, avg_q_value


    def save(self, filename, folder="saved_agents"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer1_state_dict': self.critic_optimizer1.state_dict(),
            'critic_optimizer2_state_dict': self.critic_optimizer2.state_dict()
        }, os.path.join(folder, filename))
        print(f"Agent's state saved to {filename}.")
