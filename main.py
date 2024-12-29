import pygame
import os
import numpy as np
import math
from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent
LAP_REWARD = 100  # Reward for completing a lap

SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
FPS = 60
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets", "lake.png"))

# Change based on agent used
DUCK = "td3.png"

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", DUCK))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        self.vel_vector = pygame.math.Vector2(1, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.target_direction = 0  # Direction based on action
        self.alive = True
        self.radars = []
        self.filter_alpha = 0.7  # Low-pass filter alpha
        self.lap_progress = 0  # Track progress around the lap
        self.total_laps = 0# Total laps to complete
        self.checkpoints = [
            (900, 110),
            (140, 500),
            (570, 870)
        ]
        self.last_checkpoint = -1

    def update_lap_progress(self):
        """
        Check if the car has reached the next checkpoint. If all checkpoints are passed
        sequentially, the lap is completed.
        """
        for i, checkpoint in enumerate(self.checkpoints):
            if i == self.last_checkpoint:
                continue  # Skip already passed checkpoints
                
            print( math.dist(self.rect.center, checkpoint), checkpoint)
            if math.dist(self.rect.center, checkpoint) < 10:  # Checkpoint radius threshold
                self.last_checkpoint = i
    
                self.lap_progress += 1
                if i == len(self.checkpoints) - 1:  # If last checkpoint, lap is completed
                    self.total_laps += 1
                    self.lap_progress = 0
                    self.last_checkpoint = -1  # Reset for the next lap
                break
    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()
        self.update_lap_progress()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # Die on Collision
        if SCREEN.get_at(collision_point_right) == pygame.Color(207,230,186,255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(207,230,186,255):
            self.alive = False

        # Draw Collision Points
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

        while not SCREEN.get_at((x, y)) == pygame.Color(207,230,186,255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Draw Radar
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

    # Initialize agents and cars
    num_agents = 3  # Number of agents/cars
    state_dim = 5  # Radar distances
    action_dim = 1  # Direction (-1, 0, 1)
    max_action = 1

    agents = [TD3Agent(state_dim, action_dim, max_action) for _ in range(num_agents)]
    cars = [Car() for _ in range(num_agents)]
    car_groups = pygame.sprite.Group(*cars)

    # Training parameters
    total_episodes = 1000  # Total number of episodes to run
    max_timesteps = 1000  # Max timesteps per episode

    for episode in range(total_episodes):
        # Reset cars at the start of each episode
        cars = [Car() for _ in range(num_agents)]
        car_groups = pygame.sprite.Group(*cars)
        total_rewards = [0] * num_agents
        episode_timesteps = 0
        paused = False

        while not paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    paused = True

            # Blit background track image
            SCREEN.blit(TRACK, (0, 0))
            # Draw checkpoints
            for i, checkpoint in enumerate(cars[0].checkpoints):
                if i == 0:
                    pygame.draw.rect(SCREEN, (107,19,145), (checkpoint[0], checkpoint[1], 5, 160))
                elif i == 1:
                    pygame.draw.rect(SCREEN, (191,19,19), (checkpoint[0], checkpoint[1], 160, 5))

            for i, car in enumerate(cars):
                if not car.alive:
                    continue  # Skip updates for cars that are not alive

                # Get current state
                state = car.data()
                state = np.array(state, dtype=np.float32) / 200  # Normalize radar distances

                # Select action using the agent
                exploration_noise = 0.8 if episode < 300 else 0.1  # Dynamic noise for exploration
                action = agents[i].select_action(state, exploration_noise)
                # Map action to car direction
                if action > 0.7:
                    car.target_direction = 1  # Turn right
                elif action < -0.7:
                    car.target_direction = -1  # Turn left
                else:
                    car.target_direction = 0  # Go straight

                # Update car and check for collisions
                car.update()
                # Compute reward
                if car.alive:
                    reward = 1.0  # Reward for staying alive
                    print(f"Car {i + 1}: lap = {car.lap_progress}")  if car.lap_progress > 0 else None
                    reward += car.lap_progress * LAP_REWARD  # Additional reward for laps completed
                    total_rewards[i] += reward
                    next_state = car.data()
                    next_state = np.array(next_state, dtype=np.float32) / 200  # Normalize radar distances
                    done = episode_timesteps >= max_timesteps
                else:
                    reward = -10.0  # Penalty for collision
                    done = True
                    next_state = np.zeros_like(state, dtype=np.float32)  # Placeholder for next_state
                # Store transition in replay buffer
                agents[i].replay_buffer.add(state, action, reward, next_state, done)

                # Train the agent
                agents[i].train(batch_size=64)

            # Increment timesteps
            episode_timesteps += 1

            # Check if all cars are done
            if all(not car.alive for car in cars) or episode_timesteps >= max_timesteps:
                print(f"Episode {episode + 1}/{total_episodes} ended.")
                for j, reward in enumerate(total_rewards):
                    print(f"Car {j + 1}: Total reward = {reward}")
                break

            # Draw the cars and update the screen
            car_groups.draw(SCREEN)

            # Update the display
            pygame.display.update()

            # Control the frame rate
            clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()