import optuna
import pandas as pd
from datetime import datetime
import numpy as np
import pygame
import os
import math
from sac_agent import SACAgent
from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent

CHECKPOINT_REWARD = 600  # Reward for crossing a checkpoint
LAP_REWARD = 900  # Reward for completing a lap
COIN_REWARD = 400 # Reward for collecting a coin
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
FPS = 60
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


#####
AGENT = "SACAgent"
#####


pygame.init()
pygame.display.set_caption('Duck Duck: RACING')
ICON = pygame.image.load(os.path.join("Assets", "sac.png"))
pygame.display.set_icon(ICON)
TRACK = pygame.image.load(os.path.join("Assets", "lake_duck.png"))

DUCK = f"{AGENT.lower()}.png"

class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.scale = 30
        self.original_image = pygame.image.load(os.path.join("Assets", "coin.png"))
        self.gray_image = pygame.image.load(os.path.join("Assets", "coin_gray.png"))
        self.image = pygame.transform.scale(self.original_image, (self.scale, self.scale))  # Scale the coin down
        self.rect = self.image.get_rect(center=(x, y))
        self.collected = False

    def gray_out(self):
        self.image = pygame.transform.scale(self.gray_image, (self.scale, self.scale))

    def reset_color(self):
        self.image = pygame.transform.scale(self.original_image, (self.scale, self.scale))
        self.collected = False

class DuckRacer(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "sac.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(660, 900))
        self.vel_vector = pygame.math.Vector2(1, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.target_direction = 0  # Direction based on action
        self.target_velocity = 6  # Velocity based on action
        self.current_velocity = 0 
        self.alive = True
        self.radars = []
        self.filter_alpha = 0.4  # Low-pass filter alpha
        self.smoothing_factor = 0.1
        self.lap_progress = 0  # Track progress around the lap
        self.total_laps = 0  # Total laps to complete
        self.checkpoints = [ # Checkpoints coordinates
            (500, 830, 5, 110),
            (970, 475, 5, 110),
            (580, 62, 5, 110),
            (350, 510, 110, 5)
        ]
        self.last_checkpoint_i = -1
        self.next_checkpoint_i = 1
        self.coins_collected_in_lap = set()  # Set to store indices of collected coins
        self.lap_start_time = self.start_new_lap()  # Time when the lap started
        self.radar_min = 0
        self.radar_max = 200

    def update_lap_progress(self):
        """
        Check if the car has reached the next checkpoint. If all checkpoints are passed
        sequentially, the lap is completed. A higher reward is given for faster lap times.
        """
        next_checkpoint = self.checkpoints[self.next_checkpoint_i]
        checkpoint_center = (
            next_checkpoint[0] + next_checkpoint[2] / 2,  # Center X
            next_checkpoint[1] + next_checkpoint[3] / 2   # Center Y
        )
        distance_to_checkpoint = math.dist(self.rect.center, checkpoint_center)

        if distance_to_checkpoint < 80:  # Distance threshold
            self.last_checkpoint_i = self.next_checkpoint_i
            self.next_checkpoint_i = (self.next_checkpoint_i + 1) % len(self.checkpoints)
            self.lap_progress += 1
            # print(f"Lap progress: {self.lap_progress}/3 , Next checkpoint: {self.next_checkpoint_i}")
            # Check if lap is completed
            if self.lap_progress == len(self.checkpoints):
                self.total_laps += 1
                self.lap_progress = 0
                self.last_checkpoint_i = -1
                # Calculate lap time if lap is completed
                lap_time = pygame.time.get_ticks() - self.lap_start_time
                reward = LAP_REWARD + max(0, 1000 * (0.99 ** (lap_time // 100)))  # Exponential decay function for reward
                print(f"Lap completed in {lap_time / 1000:.2f} seconds. Reward: {reward}")
                self.start_new_lap()  # Start a new lap and track the time
                return reward

            return CHECKPOINT_REWARD if self.lap_progress != 0 else 0

        return 0

    def start_new_lap(self):
        """Start a new lap and record the start time."""
        self.coins_collected_in_lap.clear()  # Clear the coins collected for the new lap
        self.lap_start_time = pygame.time.get_ticks()
        return self.lap_start_time

    def check_coin_collision(self, coins):
        reward = 0
        for index, coin in enumerate(coins):
            # Skip if the coin has already been collected in this lap
            if index in self.coins_collected_in_lap:
                continue

            # Calculate the distance between the center of the car and the coin
            distance = math.dist(self.rect.center, coin.rect.center)

            # Set a custom collision tolerance
            collision_tolerance = 50 

            if distance < collision_tolerance:
                coin.collected = True
                coin.gray_out()
                pygame.time.set_timer(pygame.USEREVENT + index, 1000)  # Reset color after half a second
                self.coins_collected_in_lap.add(index)  # Add coin index to the set
                reward += COIN_REWARD
        return reward


    def update(self):
        self.radars.clear()
        self.rotate()
        self.drive()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()

    def drive(self):
        self.current_velocity += (self.target_velocity - self.current_velocity) * self.smoothing_factor
        self.vel_vector = self.vel_vector.normalize() * self.current_velocity  # Apply smoothed velocity
        self.rect.center += self.vel_vector
    def collision(self):
        length = 40
        collision_point_right = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length),
        ]
        collision_point_left = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length),
        ]

        collision_point_right = [
            max(0, min(SCREEN_WIDTH - 1, collision_point_right[0])),
            max(0, min(SCREEN_HEIGHT - 1, collision_point_right[1])),
        ]
        collision_point_left = [
            max(0, min(SCREEN_WIDTH - 1, collision_point_left[0])),
            max(0, min(SCREEN_HEIGHT - 1, collision_point_left[1])),
        ]

        if SCREEN.get_at(collision_point_right) == pygame.Color(207, 230, 186, 255) or \
        SCREEN.get_at(collision_point_left) == pygame.Color(207, 230, 186, 255):
            self.alive = False

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self):
        self.direction = (1 - self.filter_alpha) * self.direction + self.filter_alpha * self.target_direction

        if abs(self.direction - self.target_direction) < 0.05:
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

        while length < self.radar_max:
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

            if not (0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT):
                break

            if SCREEN.get_at((x, y)) == pygame.Color(207, 230, 186, 255):  # Check for collision
                break

            length += 1

        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        normalized_data = [(x - self.radar_min) / (self.radar_max - self.radar_min) for x in input]
        return normalized_data

def cal_checkpoint_reward(duck):
    next_checkpoint = duck.checkpoints[duck.next_checkpoint_i]
    checkpoint_center = (
        next_checkpoint[0] + next_checkpoint[2] / 2,  # Center X
        next_checkpoint[1] + next_checkpoint[3] / 2   # Center Y
    )

    # Vector from the duck to the checkpoint
    checkpoint_vector = pygame.math.Vector2(checkpoint_center) - pygame.math.Vector2(duck.rect.center)
    distance_to_checkpoint = checkpoint_vector.length()  # The distance to the checkpoint

    # Dot product to determine if we are moving towards or away from the checkpoint
    dot_product = duck.vel_vector.dot(checkpoint_vector.normalize())

    # Reward based on both distance and movement direction
    const =  0.5  # Constant for scaling
    epsilon = 0.001  # Small epsilon to avoid division by zero

    # Reward for distance: inversely proportional to distance (closer = higher reward)
    reward_distance = const / (distance_to_checkpoint + epsilon)

    # Reward for direction: proportional to the dot product (moving towards the checkpoint = higher reward)
    reward_direction = dot_product * const
    # Combine both rewards
    reward = reward_distance + reward_direction

    # If moving away from the checkpoint, penalize more
    if dot_product < 0:
        reward = dot_product * 20  # Strong negative reward for moving away

    return reward

def run_training(agents, num_episodes, max_timesteps):
    """Helper function to run training episodes and return rewards"""
    pygame.init()
    SCREEN = pygame.display.set_mode((1244, 1016))
    TRACK = pygame.image.load(os.path.join("Assets", "lake_duck.png"))
    
    all_episode_rewards = []
    num_agents = len(agents)
    
    for episode in range(num_episodes):
        ducks = [DuckRacer() for _ in range(num_agents)]
        duck_groups = pygame.sprite.Group(*ducks)
        total_rewards = [0] * num_agents
        episode_timesteps = 0
        
        coins = pygame.sprite.Group(
            Coin(256, 350),
            Coin(1200, 525),
            Coin(420, 844),
            Coin(675, 125)
        )
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return float('-inf')
                if event.type >= pygame.USEREVENT:
                    coin_index = event.type - pygame.USEREVENT
                    if 0 <= coin_index < len(coins.sprites()):
                        coins.sprites()[coin_index].reset_color()
            
            SCREEN.blit(TRACK, (0, 0))
            coins.draw(SCREEN)
            for i, checkpoint in enumerate(ducks[0].checkpoints):
                pygame.draw.rect(SCREEN, (80, 90, 145), checkpoint)
            
            for i, duck in enumerate(ducks):
                if not duck.alive:
                    continue
                    
                state = np.array(duck.data(), dtype=np.float32)
                action = agents[i].select_action(state)  # Using default noise value
                
                duck.target_direction = 1 if action[0] > 0.5 else -1 if action[0] < -0.5 else 0
                duck.target_velocity = max(2, min(10, action[1]*7))
                
                reward = cal_checkpoint_reward(duck)
                duck.update()
                reward += duck.update_lap_progress()
                reward += duck.check_coin_collision(coins)
                
                if not duck.alive:
                    reward = -1000
                    done = True
                    next_state = np.zeros_like(state)
                else:
                    reward += 0.0001
                    total_rewards[i] += reward
                    next_state = np.array(duck.data(), dtype=np.float32)
                    done = episode_timesteps >= max_timesteps
                
                agents[i].add_to_replay(state, action, reward, next_state, done)
                agents[i].train(batch_size=256)
            
            episode_timesteps += 1
            
            if all(not duck.alive for duck in ducks) or episode_timesteps >= max_timesteps:
                avg_reward = sum(total_rewards) / num_agents
                all_episode_rewards.append(avg_reward)
                print(f"Episode {episode + 1}: Average Reward = {avg_reward:.2f}")
                break
            
            duck_groups.draw(SCREEN)
            pygame.display.update()
    
    return all_episode_rewards

def objective_sac(trial):
    # Hyperparameter search space
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.01)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    alpha = trial.suggest_float("alpha", 0.05, 0.3)
    
    # Initialize agents with trial hyperparameters
    num_agents = 1
    state_dim = 5
    action_dim = 2
    max_action = 1
    
    agents = [SACAgent(state_dim, action_dim, max_action, 
                      gamma=gamma, tau=tau, lr=lr,
                      alpha=alpha) for _ in range(num_agents)]
    
    rewards = run_training(agents, num_episodes=100, max_timesteps=5000)
    return np.mean(rewards)

def objective_td3(trial):
    # Hyperparameter search space
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.01)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    policy_noise = trial.suggest_float("policy_noise", 0.0, 0.5)
    noise_clip =  trial.suggest_float("noise_clip", 1.0, 2.0)
    policy_delay = trial.suggest_int("policy_delay", 1, 4)

    # Initialize agents with trial hyperparameters
    num_agents = 1
    state_dim = 5
    action_dim = 2
    max_action = 1
    
    agents = [TD3Agent(state_dim, action_dim, max_action
                      ) for _ in range(num_agents)]
    
    rewards = run_training(agents, num_episodes=100, max_timesteps=5000)
    return np.mean(rewards)

def objective_ddpg(trial):
    # Hyperparameter search space
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.01)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    
    # Initialize agents with trial hyperparameters
    num_agents = 1
    state_dim = 5
    action_dim = 2
    max_action = 1
    
    agents = [DDPGAgent(state_dim, action_dim, max_action, 
                      gamma=gamma, tau=tau, lr=lr) for _ in range(num_agents)]
    
    rewards = run_training(agents, num_episodes=100, max_timesteps=5000)
    return np.mean(rewards)

def main_with_hyperparameter_optimisation():

    print("\n=== Starting Hyperparameter Optimization ===")
    # Create study and run optimization
    study = optuna.create_study(direction="maximize")
    # SET THE AGENT TO OPTIMISE HERE
    study.optimize(objective_sac, n_trials=20)
    
    # Prepare results for all trials
    optimization_results = []
    for trial in study.trials:
        result = {
            'gamma': trial.params.get('gamma'),
            'tau': trial.params.get('tau'),
            'learning_rate': trial.params.get('lr'),
            'alpha': trial.params.get('alpha'),
            'mean_reward': trial.value,
            'type': 'optimization'
        }
        optimization_results.append(result)
    
    # Combine baseline and optimization results
    all_results = pd.DataFrame(optimization_results)
    
    # Save results to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'hyperparameter_results_{timestamp}.xlsx'
    all_results.to_excel(filename, index=False)
    
    # Print best parameters
    print("\nBest hyperparameters found:")
    print(f"Best mean reward: {study.best_value}")
    print("Parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    return study.best_params, filename

if __name__ == "__main__":
    best_params, results_file = main_with_hyperparameter_optimisation()