import pygame
import os
import numpy as np
import math
from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent
from sac_agent import SACAgent
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import shutil
import csv
CHECKPOINT_REWARD = 600  # Reward for crossing a checkpoint
LAP_REWARD = 900  # Reward for completing a lap
COIN_REWARD = 400 # Reward for collecting a coin
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
FPS = 60
pygame.init()
pygame.display.set_caption('Duck Duck: RACING')
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
ICON = pygame.image.load(os.path.join("Assets", "ddpg.png"))
pygame.display.set_icon(ICON)
# TRACK = pygame.image.load(os.path.join("Assets", "lake.png"))


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
    def __init__(self,map_name, duck):
        super().__init__()
        self.agent = duck.removesuffix(".png")
        self.original_image = pygame.image.load(os.path.join("Assets", duck))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 850))
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
        # Define checkpoints for each map
        self.map_checkpoints = {
            "lake_circle": [(625, 780, 5, 150), (625, 68, 5, 150)],
            "lake": [(570, 800, 5, 140), (900, 760, 5, 150), (900, 115, 5, 150), (150, 500, 150, 5)],
            "lake_duck": [ (500, 830, 5, 110),(580, 62, 5, 110),(350, 510, 110, 5)]
        }
        self.checkpoints = self.map_checkpoints.get(map_name, [])
        self.last_checkpoint_i = -1
        self.next_checkpoint_i = 1
        self.coins_collected_in_lap = set()  # Set to store indices of collected coins
        self.lap_start_time = self.start_new_lap()  # Time when the lap started
        self.radar_min = 0
        self.radar_max = 200
        self.lap_times = []

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
            # print(f"Lap progress: {self.lap_progress}/{len(self.checkpoints)} , Next checkpoint: {self.next_checkpoint_i}")
            # Check if lap is completed
            if self.lap_progress == len(self.checkpoints):
                self.total_laps += 1
                self.lap_progress = 0
                self.last_checkpoint_i = -1
                # Calculate lap time if lap is completed
                lap_time = pygame.time.get_ticks() - self.lap_start_time
                self.lap_times.append(round(lap_time / 1000, 2))
                # reward = LAP_REWARD + max(0, 1000 * (0.99 ** (lap_time // 50)))  # Exponential decay function for reward
                reward = LAP_REWARD + max(0, 1000 * (0.99 ** (lap_time // 100)))  # Exponential decay function for reward
                print(f"Agent: {self.agent} Lap completed in {lap_time / 1000:.2f} seconds. Reward: {reward}")
                self.start_new_lap()  # Start a new lap and track the time
                return reward

            return CHECKPOINT_REWARD if self.lap_progress != 0 else 0

        return 0
    
    def get_lap_times(self):
        times = self.lap_times
        self.lap_progress = [] # Clear the lap times
        return times

    def start_new_lap(self):
        """Start a new  lap and record the start time."""
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

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        if SCREEN.get_at(collision_point_right) == pygame.Color(207,230,186,255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(207,230,186,255):
            self.alive = False

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate_drive(self):
        self.direction = (1 - self.filter_alpha) * self.direction + self.filter_alpha * self.target_direction
        self.current_velocity += (self.target_velocity - self.current_velocity) * self.smoothing_factor
        self.vel_vector = self.vel_vector.normalize() * self.current_velocity  # Apply smoothed velocity
        
        if abs(self.direction - self.target_direction) < 0.05:
            self.direction = self.target_direction

        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect.center += self.vel_vector
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(207,230,186,255) and length < 200:
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
        normalized_data = [(x - self.radar_min) / (self.radar_max - self.radar_min) for x in input]
        return normalized_data
    
def plot_graphs(total_episodes, rewards_per_episode, lap_times_per_episode, actor_losses, critic_losses, q_values, folder = "metrics",agent_name = "td3", map = "lake"):
    path = os.path.join(os.path.join(folder,agent_name), map)
    metrics_path = os.path.join(path, "metric_arrays")
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    
    # Save metrics arrays
    np.save(os.path.join(metrics_path, "rewards_per_episode.npy"), np.array(rewards_per_episode))
    np.save(os.path.join(metrics_path, "lap_times_per_episode.npy"), np.array(lap_times_per_episode))
    np.save(os.path.join(metrics_path, "actor_losses.npy"), np.array(actor_losses))
    np.save(os.path.join(metrics_path, "critic_losses.npy"), np.array(critic_losses))
    np.save(os.path.join(metrics_path, "q_values.npy"), np.array(q_values))
    print(f"Metric arrays saved in {metrics_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_episodes + 1), rewards_per_episode, label="Total Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Per Episode")
    plt.legend()
    plt.savefig(os.path.join(path,f"{agent_name}_{map}_rewards_plot.png"))
    print("Rewards plot saved as rewards_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_episodes + 1), lap_times_per_episode, label="Average Lap Time")
    plt.xlabel("Episode")
    plt.ylabel("Lap Time (ms)")
    plt.title("Lap Time Per Episode")
    plt.legend()
    plt.savefig(os.path.join(path,f"{agent_name}_{map}_lap_times_plot.png"))
    print("Lap times plot saved as lap_times_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_episodes + 1), critic_losses, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Critic Loss Per Episode")
    plt.legend()
    plt.savefig(os.path.join(path,f"{agent_name}_{map}_losses_critic_plot.png"))
    print("Losses plot saved as losses_critic_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_episodes + 1), actor_losses, label="Actor Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Actor Loss Per Episode")
    plt.legend()
    plt.savefig(os.path.join(path,f"{agent_name}_{map}_losses_actor_plot.png"))
    print("Losses plot saved as losses_actor_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_episodes + 1), q_values, label="Average Q-Value")
    plt.xlabel("Episode")
    plt.ylabel("Q-Value")
    plt.title("Average Q-Value Per Episode")
    plt.legend()
    plt.savefig(os.path.join(path,f"{agent_name}_{map}_q_values_plot.png"))
    print("Q-values plot saved as q_values_plot.png")

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

def create_heatmap(save_pos, screen_w, screen_h, track ,output_file="heatmap.png", folder = "metrics", agent_name = "td3", map = "lake" ):
    path = os.path.join(os.path.join(folder,agent_name), map)
    if not os.path.exists(path):
        os.makedirs(path)
    # Initialize a 2D array to count the frequency of positions
    heatmap_arr = np.zeros((screen_h, screen_w))

    for x, y in save_pos:
        if 0 <= x < screen_w and 0 <= y < screen_h:
            heatmap_arr[y, x] += 1

    # Plot the heatmap using matplotlib
    heatmap_arr = np.log1p(heatmap_arr)  # Logarithmic scaling
    smoothed_heatmap = gaussian_filter(heatmap_arr, sigma=2.3)
    # Load the environment image
    env_map = mpimg.imread(track)

    # Make sure environment image matches the heatmap dimensions
    env_map_resized = np.resize(env_map, (screen_h, screen_w, 3))

    # Plot the environment map and overlay the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(env_map_resized, extent=[0, screen_w, 0, screen_h], alpha=0.7)
    plt.imshow(smoothed_heatmap, cmap='hot', extent=[0, screen_w, 0, screen_h], alpha=0.77)
    plt.colorbar(label="Log-Scaled Frequency")
    plt.title("Agent Position Heatmap with Environment Map Overlay")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    # Save the overlay heatmap as an image
    plt.savefig(os.path.join(path,f"{agent_name}_{map}_heatmap.png"))
    print(f"Heatmap with overlay saved as {output_file}")
    # Save the heatmap array
    # np.save("heatmap_array.npy", heatmap_arr)
    # print("Heatmap array saved as heatmap_array.npy")

def save_frame(surface, frame_number, folder="frames"):
    pygame.image.save(surface, os.path.join(folder, f"frame_{frame_number:04d}.png"))

def create_training_gif(folder="frames", output_filename="training_process.gif", fps=120, folder_save = "metrics", agent_name = "td3", map = "lake"):
    path = os.path.join(os.path.join(folder_save,agent_name), map)
    if not os.path.exists(path):
        os.makedirs(path)

    frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])
    if frames:
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_gif(os.path.join(path,f"{agent_name}_{map}_{output_filename}"))
        print(f"GIF saved as {output_filename}") 
        shutil.rmtree(folder)  # Clean up the frames folder   

def display_episode_number(screen, episode, font):
    text = font.render(f"Episode: {episode}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

def main():
    # Define the different maps and agents
    maps = ["lake_circle.png", "lake.png", "lake_duck.png"]  # Add your map file names here
    # agents = ["td3.png", "ddpg.png", "sac.png"]  # Add your agent names here
    # maps = ["lake_duck.png"]  # Add your map file names here
    agents_paths = ["sac.png"]  # Add your agent names here

    num_iterations = 1  # Number of training iterations
    base_output_folder = "training_metrics"
    # Initialize the CSV file
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)
   
    clock = pygame.time.Clock()
    num_agents = len(agents_paths)
    state_dim = 5
    action_dim = 2  
    max_action = 1
    metrics_file = os.path.join(base_output_folder, "generalisation")
    metrics_file = os.path.join(metrics_file, "iteration_metrics.csv")
    trained_agents = ["lake_duck"]
    if not os.path.exists(metrics_file):
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Agent", "Map", "Iteration", "Average Reward", "Average Lap Time", "Fastest Time","Average Number of Laps"])  # Header row
    for trained_agent in trained_agents:
        for map_path in maps:
            agents = []
            # agent_name = agent_path.removesuffix(".png")
            map_name = map_path.removesuffix(".png")
            TRACK = pygame.image.load(os.path.join("Assets", map_path))
            for iteration in range(1, num_iterations + 1):
                iteration_folder = os.path.join(base_output_folder, f"generalisation")
                if not os.path.exists(iteration_folder):
                    os.makedirs(iteration_folder)

                # Create agents based on the specified types
                for i, agent_type in enumerate(agents_paths):
                    agent_type = agent_type.removesuffix(".png")
                    if agent_type == "ddpg":
                        agent = DDPGAgent(state_dim, action_dim, max_action)
                        agent.load(f"{agent_type}_{trained_agent}_fastest_time_state",folder = base_output_folder)
                    elif agent_type == "td3":
                        agent = TD3Agent(state_dim, action_dim, max_action)
                        agent.load(f"{agent_type}_{trained_agent}_fastest_time_state", folder=base_output_folder)
                    elif agent_type == "sac":
                        agent = SACAgent(state_dim, action_dim, max_action)
                        agent.load(f"{agent_type}_{trained_agent}_fastest_time_state", folder=base_output_folder)
                    else:
                        raise ValueError(f"Unknown agent type: {agent_type}")
                    agents.append(agent)
                if map_path == "lake_circle.png":
                    coins = None
                elif map_path == "lake.png":
                    coins = pygame.sprite.Group(
                        Coin(980, 250),
                        Coin(600, 130),
                        Coin(800, 850),
                        Coin(230, 400)
                    )
                else:
                    coins = pygame.sprite.Group(
                        Coin(256, 350),
                        Coin(1000, 525),
                        Coin(420, 844),
                        Coin(675, 125)
                    )
                total_episodes = 5
                max_timesteps = 1500
                position_log = []
                rewards_per_duck = [[] for _ in range(num_agents)]  # Track rewards separately for each duck
                lap_times_per_duck = [[] for _ in range(num_agents)]  # Track lap times separately for each duck
                lap_count_per_duck = [[] for _ in range(num_agents)]  # Track number of laps separately for each duck
                
                fastest_time = [float('inf')] * num_agents
                frame_count = 0
                pygame.font.init()

                for episode in range(total_episodes):
                    ducks = [DuckRacer(map_name=map_name, duck=agent_path) for agent_path in agents_paths[:num_agents]]
                    duck_groups = pygame.sprite.Group(*ducks)
                    total_rewards = [0] * num_agents
                    episode_timesteps = 0
                    lap_times = [[] for _ in range(len(ducks))]  # Store lap times for this episode
                    paused = False
                    episode_updates = 0
                    lap_count = [0]*num_agents


                    while not paused:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                paused = True
                            if event.type >= pygame.USEREVENT:
                                coin_index = event.type - pygame.USEREVENT
                                if coins != None:
                                    if 0 <= coin_index < len(coins.sprites()):
                                        coins.sprites()[coin_index].reset_color()

                        SCREEN.blit(TRACK, (0, 0))
                        if coins:
                            coins.draw(SCREEN)
                        # display_episode_number(SCREEN, episode + 1, font)  # Display the current episode number
                        for i, checkpoint in enumerate(ducks[0].checkpoints):
                            pygame.draw.rect(SCREEN, (80, 90, 145), checkpoint)

                        for i, duck in enumerate(ducks):
                            if not duck.alive:
                                continue
                            reward = 0
                            state = duck.data()
                            state = np.array(state, dtype=np.float32)

                            if duck.agent == "td3":
                                exploration_noise = 0.393
                                action = agents[i].select_action(state, exploration_noise)
                            else:
                                action = agents[i].select_action(state)

                            duck.target_direction = 1 if action[0] > 0.5 else -1 if action[0] < -0.5 else 0
                            duck.target_velocity = max(2, min(10, action[1]*7)) #minimum velocity is 2, max is 10

                            reward = cal_checkpoint_reward(duck)

                            duck.update()
                            reward += duck.update_lap_progress()
                            if coins:
                                reward += duck.check_coin_collision(coins)
                            if not duck.alive:
                                reward = -1000
                                done = True
                                next_state = np.zeros_like(state, dtype=np.float32)
                            else:
                                # reward += 0.0001
                                total_rewards[i] += reward
                                next_state = duck.data()
                                next_state = np.array(next_state, dtype=np.float32)
                                done = episode_timesteps >= max_timesteps
                            agents[i].add_to_replay(state, action, reward, next_state, done)
                            episode_updates += 1
                            # Log the position for heatmap
                            position_log.append(duck.rect.center)
                        episode_timesteps += 1

                        if all(not duck.alive for duck in ducks) or episode_timesteps >= max_timesteps:
                            print(f"\n----------------Episode {episode + 1}/{total_episodes} ended.----------------")
                            for j, reward in enumerate(total_rewards):
                                print(f"Duck {duck.agent}: Total reward = {reward}")
                            break
                        
                        duck_groups.draw(SCREEN)
                        pygame.display.update()

                    # Store episode-level metrics
                    for i in range(num_agents):
                        rewards_per_duck[i].append(total_rewards[i])
                        duck_times = ducks[i].get_lap_times()
                        lap_count_per_duck[i].append(ducks[i].total_laps)
                        if duck_times != []:
                            lap_times[i].extend(duck_times)
                            fastest_time[i] = min(fastest_time[i], min(lap_times[i]))
                            lap_times_per_duck[i].append(sum(lap_times[i]) / len(lap_times[i]))
                        else:
                            lap_times_per_duck[i].append(None)
                lap_count_per_duck
                avg_lap_count = [sum(lap_count) / total_episodes for lap_count in lap_count_per_duck]
                avg_rewards = [sum(rewards) / total_episodes for rewards in rewards_per_duck]
                avg_lap_times = []
                for i, times in enumerate(lap_times_per_duck):
                    total = 0
                    count = 0
                    for time in times:
                        if time != None:
                            total+= time
                            count+=1
                    avg_lap_times.append(total / count if count != 0 else 0)
                for i, duck in enumerate(agents_paths):
                    print(f"Agent {duck}: Average reward = {avg_rewards[i]}, Average lap time = {avg_lap_times[i]},avg_lap_count = {avg_lap_count[i]} Fastest time = {fastest_time[i]}")
                    # print(f"Iteration {iteration}: average reward = {avg_reward}")
                    # print(f"Iteration {iteration}: average lap time = {avg_lap_time}")
                    # print(f"Iteration {iteration}: Fastest time = {fastest_time}")
                    # print(f"Iteration {iteration}: average reward = {avg_reward}")
                    # print(f"Iteration {iteration}: Fastest time = {fastest_time}")
                with open(metrics_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f"{agents_paths[0].removesuffix('.png')}_{trained_agent}", map_name, iteration, avg_rewards[0], avg_lap_times[0], fastest_time[0], avg_lap_count[0]])
                create_heatmap(position_log, SCREEN_WIDTH, SCREEN_HEIGHT, folder=iteration_folder, track = os.path.join("Assets", map_path), agent_name=f"{agents_paths[0].removesuffix('.png')}_{trained_agent}", map=map_name)
                # plot_graphs(total_episodes, rewards_per_episode, lap_times_per_episode, actor_losses, critic_losses, q_values, folder=iteration_folder, agent_name=agent_name, map=map_name)
                # create_training_gif(folder=frames_folder, output_filename="training_process.gif", fps=60, folder_save=iteration_folder, agent_name=agent_name, map=map_name)
    pygame.quit()


if __name__ == "__main__":
    main()
