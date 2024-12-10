import pygame
import os
import math
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from gymnasium import Env, spaces
from gymnasium.utils import seeding


# Screen Constants
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets", "track.png"))


class CarEnv(Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        self.car = Car()
        self.clock = pygame.time.Clock()

        # Action space: Continuous steering in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: 5 radar distances + car speed + angle
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Reset the car state
        self.np_random, seed = seeding.np_random(seed)
        self.car = Car()
        self.car.alive = True
        obs = self.get_state()
        return obs, {}

    def step(self, action):
        if not self.car.alive:
            return self.reset()

        # Apply action: steering direction
        self.car.direction = np.clip(action[0], -1, 1)
        self.car.update()

        # Observation, reward, and done logic
        obs = self.get_state()
        reward = 1  # Reward for staying alive
        done = not self.car.alive
        if done:
            reward = -10  # Penalty for crashing
        truncated = False  # No truncation logic here
        return obs, reward, done, truncated, {}

    def get_state(self):
        radar_distances = [radar[1] / 200.0 for radar in self.car.radars]  # Normalize radar distances
        state = radar_distances + [
            self.car.vel_vector.length() / 10.0,  # Normalize speed
            self.car.angle % 360 / 360.0          # Normalize angle
        ]
        return np.array(state, dtype=np.float32)


    def render(self, mode="human"):
        SCREEN.blit(TRACK, (0, 0))
        rotated_car = pygame.transform.rotozoom(self.car.original_image, self.car.angle, 0.1)
        SCREEN.blit(rotated_car, rotated_car.get_rect(center=self.car.rect.center))
        pygame.display.flip()

    def close(self):
        pygame.quit()


class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def rotate(self):
        if self.direction > 0:  # Turn left
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction < 0:  # Turn right
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x, y = int(self.rect.center[0]), int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        self.radars.append([radar_angle, length])

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) or \
           SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False


# Train SAC Agent
def train_sac():
    env = CarEnv()

    # Validate the environment
    check_env(env, warn=True)

    # Train the SAC model
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("sac_car_model")

    print("Training complete! Model saved as 'sac_car_model'.")


# Test SAC Agent
def test_sac():
    env = CarEnv()
    model = SAC.load("sac_car_model")

    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()


if __name__ == "__main__":
    mode = input("Enter 'train' to train or 'test' to test: ").strip().lower()
    if mode == "train":
        train_sac()
    elif mode == "test":
        test_sac()
    else:
        print("Invalid mode. Enter 'train' or 'test'.")


