import pygame
import os
import math

from sac_agent import SACAgent

# Constants
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
FPS = 60

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
        self.alive = True
        self.radars = []

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

    # Initialize DDPG agent
    state_dim = 5  # Radar distances
    action_dim = 1  # Direction (-1, 0, 1)
    max_action = 1
    agent = SACAgent(state_dim, action_dim, max_action)

    car = Car()
    car_group = pygame.sprite.GroupSingle(car)

    run = True
    paused = False

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if paused:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = False
                        main()

        SCREEN.blit(TRACK, (0, 0))

        if not paused:
            state = car.data()

            action = agent.select_action(state)

            # Map action to car direction
            if action > 0.5:
                car.direction = 1  # Turn right
            elif action < -0.5:
                car.direction = -1  # Turn left
            else:
                car.direction = 0  # Go straight

            car_group.update()

            if not car.alive:
                paused = True

        car_group.draw(SCREEN)

        if paused:
            font = pygame.font.SysFont("Arial", 48)
            text = font.render("Game Paused - Press SPACE to Resume", True, (255, 255, 255))
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            SCREEN.blit(text, text_rect)

        # Update the display
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()