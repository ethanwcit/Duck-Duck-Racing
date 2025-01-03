import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import seaborn as sns
import gym
from ActCrit import Actor, Critic

class PPO:
    def __init__(self,env):
        
        # Initialize the passed in environment
        self.env = env

        #Input and output dimensions
        self.inputs = self.env.observation_space.shape[0]
        self.outputs = self.env.action_space.n

        # Initialize networks from ActCrit.py
        self.actor = Actor(self.inputs, self.outputs)
        self.critic = Critic(self.inputs)
        
        # Hyperparameters
        self.episode_size = 400 #Timesteps per episode
        self.batch_size = 2000 #Timesteps per update
        self.gamma = 0.99 #Discount factor
        self.number_of_updates = 4 #how many updates per batch
        self.clipping = 0.2 #Clipping rate for PPO
        self.learning_rate = 0.001 #gradient descent step size  
        
        # Optimizers
        self.actor_optimiser = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=self.learning_rate)

        # Metrics tracking
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.timesteps_per_episode = []
        self.cumulative_timesteps = []
        self.running_reward = deque(maxlen=100)

    def ppo(self, batch_size):
        current_timesteps = 0
        episode_count = 0
        
        while current_timesteps < batch_size:
            states, actions, rewards_to_go, old_log_probs, episode_length, episode_reward = self.rollout()
            current_timesteps += episode_length
            episode_count += 1

            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.running_reward.append(episode_reward)
            self.timesteps_per_episode.append(episode_length)
            self.cumulative_timesteps.append(current_timesteps)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards_to_go = torch.FloatTensor(rewards_to_go)
            old_log_probs = torch.FloatTensor(old_log_probs)

            # Training loop
            for _ in range(self.number_of_updates):
                values = self.critic(states).squeeze()
                action_probs = self.actor(states)
                dist = Categorical(action_probs)
                current_log_probs = dist.log_prob(actions)

                #Put into other function, understand and REDO
                advantages = rewards_to_go - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                #Put into other function, understand and REDO
                ratios = torch.exp(current_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clipping, 1+self.clipping) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, rewards_to_go)

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())

                # Optimization step
                self.actor_optimiser.zero_grad()
                self.critic_optimiser.zero_grad()
                total_loss = actor_loss + 0.5 * critic_loss
                total_loss.backward()
                self.actor_optimiser.step()
                self.critic_optimiser.step()

            if episode_count % 10 == 0:
                avg_reward = sum(self.running_reward) / len(self.running_reward)
                print(f"Episode {episode_count}, Timesteps: {current_timesteps}, Average Reward (last 100): {avg_reward:.2f}")

    def rollout(self):
        state = self.env.reset()[0]
        states, actions, rewards, log_probs = [], [], [], []
        episode_length = 0
        
        for _ in range(self.episode_size):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = self.actor(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            
            state = next_state
            episode_length += 1

            if done:
                break

        rewards_to_go = []
        running_sum = 0
        for r in reversed(rewards):
            running_sum = r + self.gamma * running_sum
            rewards_to_go.insert(0, running_sum)

        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards_to_go, dtype=np.float32),
                np.array(log_probs, dtype=np.float32),
                episode_length,
                sum(rewards))

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot individual episode rewards
        ax1.scatter(self.cumulative_timesteps, self.episode_rewards, 
                   alpha=0.3, label='Episode Rewards', color='blue')
        
        # Calculate and plot moving average
        df = pd.DataFrame({'timesteps': self.cumulative_timesteps, 
                          'rewards': self.episode_rewards})
        window_size = 50
        rolling_mean = df.rolling(window=window_size, min_periods=1).mean()
        ax1.plot(rolling_mean['timesteps'], rolling_mean['rewards'], 
                'r-', label=f'{window_size}-episode moving average', linewidth=2)
        
        ax1.set_title('Episode Rewards over Time')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Episode Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot smoothed losses
        df_losses = pd.DataFrame({
            'Actor Loss': self.actor_losses,
            'Critic Loss': self.critic_losses
        })
        df_losses = df_losses.rolling(window=100, min_periods=1).mean()
        df_losses.plot(ax=ax2)
        ax2.set_title('Training Losses (Smoothed)')
        ax2.set_xlabel('Update Step')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def test(self, episodes=5):
        # Create a new environment for testing with rendering
        test_env = gym.make('CartPole-v1', render_mode='human')
        test_rewards = []
        
        for ep in range(episodes):
            state = test_env.reset()[0]
            done = False
            total_reward = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action_probs = self.actor(state_tensor)
                    action = torch.argmax(action_probs).item()
                
                state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            test_rewards.append(total_reward)
            print(f"Episode {ep+1}: Reward = {total_reward}")
        
        print(f"\nAverage Test Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
        test_env.close()

# Example usage
if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='none')
    model = PPO(env)
    model.ppo(200000)
    model.plot_results()
    model.test(episodes=5)