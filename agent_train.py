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


def save_model():
    agent.save_model("sac_pendulum_model")

def load_model():
    if os.path.exists("sac_pendulum_model.pth"):
        agent.load_model("sac_pendulum_model")
    else:
        print("No model found, starting training from scratch.")




episodes = 100
episode_rewards = []
episode_lengths = []
for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(100):  

        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward

        agent.replay_buffer.add(state, action, reward, next_state, done)

        agent.train(batch_size=64)

        state = next_state
        if done:
            break
    # Save rewards and episode lengths for each episode
    episode_rewards.append(np.array([episode, episode_reward]))

    print(f"Episode: {episode}, Reward: {episode_reward}")

    # Save model after every episode or periodically
    # if episode % 10 == 0:
    #     save_model()
# Save the training data
save_model()
np.save("training_data/episode_rewards.npy", np.array(episode_rewards))
np.save("training_data/episode_lengths.npy", np.array(episode_lengths))

print("Training completed and data saved!")


