from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from sacenv import CarEnv
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets", "track.png"))
if __name__ == "__main__":
    try:
        # Initialize environment
        env = CarEnv()

        # Check the environment
        check_env(env, warn=True)

        # Define SAC model
        model = SAC("MlpPolicy", env, verbose=1)

        # Train the model
        print("Starting training...")
        model.learn(total_timesteps=10000)
        print("Training complete!")

        # Save the model
        model.save("sac_car_model")

        # Test the model
        print("Starting testing...")
        obs = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

        env.close()
    except Exception as e:
        print(f"An error occurred: {e}")
