import gym
import numpy as np
import torch
import os
from sac_agent import SACAgent


env = gym.make("Pendulum-v1", render_mode="human")  

state,_ = env.reset()
#print(state, state.shape)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0]) 

agent = SACAgent(state_dim, action_dim, max_action)


if not os.path.exists("training_data"):
    os.makedirs("training_data")


def load_model():
    if os.path.exists("sac_pendulum_model"):
        agent.load_model("sac_pendulum_model")
    else:
        print("No model found, starting training from scratch.")


load_model()

sac_rewards = []
episodes = 100
for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(200):
        action = agent.predict(state)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    sac_rewards.append(episode_reward)
    print(f"Episode: {episode} , Reward: {episode_reward}")

print(f"Stable-Baselines3 SAC Average Reward: {sum(sac_rewards) / len(sac_rewards)}")