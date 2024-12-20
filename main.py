import pygame
import os
import numpy as np
import math
from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent

# Constants
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
FPS = 512

# Initialize pygame
pygame.init()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Duck Duck Racing")
TRACK = pygame.image.load(os.path.join("Assets", "track.png"))

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
        self.target_direction = 0  # Direction based on action
        self.alive = True
        self.radars = []
        self.filter_alpha = 0.7  # Low-pass filter alpha

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self):
        self.direction = (1 - self.filter_alpha) * self.direction + self.filter_alpha * self.target_direction
                # Check if the direction is sufficiently close to the target direction and directly set it to target
        # print(self.direction)
        if abs(self.direction - self.target_direction) < 0.05:  # Threshold to snap to target
            self.direction = self.target_direction

        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                             + math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input

def main():
    clock = pygame.time.Clock()

    # Initialize TD3 agent
    state_dim = 5  # Radar distances
    action_dim = 1  # Direction (-1, 0, 1)
    max_action = 1
    agent = TD3Agent(state_dim, action_dim, max_action)

    # Create the car and car group
    car = Car()
    car_group = pygame.sprite.GroupSingle(car)

    # Training parameters
    total_episodes = 1000  # Total number of episodes to run
    max_timesteps = 1000  # Max timesteps per episode
    noise_std = 0.1  # Standard deviation for exploration noise

    for episode in range(total_episodes):
        car = Car()  # Reset the car at the start of each episode
        car_group = pygame.sprite.GroupSingle(car)
        total_reward = 0
        episode_timesteps = 0
        paused = False
        exploration_noise = 0.1

        while not paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    paused = True

            # Blit background track image
            SCREEN.blit(TRACK, (0, 0))

            # Get current state from the car
            state = car.data()
            state = np.array(state, dtype=np.float32) / 200  # Normalize radar distances
            episode_timesteps += 1
# For early episodes, use higher exploration noise
            if episode < 300:  # During the first few episodes
                exploration_noise = 0.5  # Higher noise for more exploration
            else:
                exploration_noise = 0.1  # Lower noise as the agent learns more

            # Select action using the TD3 agent
            action = agent.select_action(state, exploration_noise)

            # Map action to car direction
            if action > 0.5:
                car.target_direction = 1  # Turn right
            elif action < -0.5:
                car.target_direction = -1  # Turn left
            else:
                car.target_direction = 0  # Go straight

            # Update car and check for collisions
            car_group.update()

            # Compute reward
            if car.alive:
                reward = 1.0  # Reward for staying alive
                total_reward += reward
                next_state = car.data()
                next_state = np.array(next_state, dtype=np.float32) / 200  # Normalize radar distances
                done = episode_timesteps >= max_timesteps
            else:
                reward = -10.0  # Penalty for collision
                done = True
                next_state = np.zeros_like(state, dtype=np.float32)  # Placeholder for next_state

            # Store transition in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # Train the TD3 agent
            agent.train(batch_size=64)

            # Check if episode is done
            if done:
                print(f"Episode {episode+1}/{total_episodes} ended: Total reward = {total_reward}")
                break  # Exit the while loop to start a new episode

            # Draw the car and update the screen
            car_group.draw(SCREEN)

            # Update the display
            pygame.display.update()

            # Control the frame rate
            clock.tick(FPS)

        # Pause screen (optional)
        if paused:
            font = pygame.font.SysFont("Arial", 48)
            text = font.render("Game Paused - Press SPACE to Resume", True, (255, 255, 255))
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            SCREEN.blit(text, text_rect)
            pygame.display.update()  # Update the display for paused screen

    pygame.quit()

if __name__ == "__main__":
    main()
