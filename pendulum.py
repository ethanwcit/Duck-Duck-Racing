


from stable_baselines3 import SAC
import gym

# Create environment
env = gym.make("Pendulum-v1")

# Stable-Baselines3 SAC
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Test the agent
sb_rewards = []
episodes = 100
for _ in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    for _ in range(200):
        action, _ = model.predict(state)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    sb_rewards.append(episode_reward)

print(f"Stable-Baselines3 SAC Average Reward: {sum(sb_rewards) / len(sb_rewards)}")