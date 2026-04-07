# render.py — simple Pygame visualization for TrafficEnvMixed

import pygame
from utils import SCREEN_W, SCREEN_H, FPS
from env import TrafficEnvMixed

GREY_ROAD = (60, 60, 60)
DARK_GREY = (40, 40, 40)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 200, 0)

class TrafficRenderer:
    def __init__(self, env: TrafficEnvMixed):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

        self.cx, self.cy = SCREEN_W // 2, SCREEN_H // 2
        self.road_w = 200  # width of each road segment

    def draw_intersection(self):
        # Road vertical
        pygame.draw.rect(self.screen, GREY_ROAD,
                         (self.cx - self.road_w//2, 0, self.road_w, SCREEN_H))
        # Road horizontal
        pygame.draw.rect(self.screen, GREY_ROAD,
                         (0, self.cy - self.road_w//2, SCREEN_W, self.road_w))

        # Intersection center
        pygame.draw.rect(self.screen, DARK_GREY,
                         (self.cx - 50, self.cy - 50, 100, 100))

    def draw_phase_indicators(self, phase):
        # Draw four phase boxes on top-left corner
        for i in range(4):
            color = GREEN if i == phase else RED
            pygame.draw.rect(self.screen, color, (10 + i*40, 10, 30, 20))
            txt = self.font.render(f"P{i}", True, WHITE)
            self.screen.blit(txt, (12 + i*40, 12))

    def draw_queues(self, queues):
        """
        queues = {
            'N': ['S','L','S'],
            'S': [...], 
            'E': [...],
            'W': [...]
        }
        We draw characters on each road arm.
        """
        # North queue
        qN = "".join(queues['N'])
        self.screen.blit(self.font.render(f"N: {qN}", True, WHITE),
                         (self.cx - 20, 50))

        # South queue
        qS = "".join(queues['S'])
        self.screen.blit(self.font.render(f"S: {qS}", True, WHITE),
                         (self.cx - 20, SCREEN_H - 70))

        # West queue
        qW = "".join(queues['W'])
        self.screen.blit(self.font.render(f"W: {qW}", True, WHITE),
                         (50, self.cy - 20))

        # East queue
        qE = "".join(queues['E'])
        self.screen.blit(self.font.render(f"E: {qE}", True, WHITE),
                         (SCREEN_W - 150, self.cy - 20))

    def draw_info(self, env):
        txt = self.font.render(f"t={env.t}  passed={env.passed_vehicles}", True, YELLOW)
        self.screen.blit(txt, (10, SCREEN_H - 30))

    def render_step(self, action, info):
        self.screen.fill(BLACK)

        # draw roads
        self.draw_intersection()

        # draw phase indicators
        self.draw_phase_indicators(self.env.phase)

        # draw queues
        snap = self.env.get_snapshot()
        self.draw_queues(snap['queues'])

        # draw time and stats
        self.draw_info(self.env)

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()


def run_visual_demo(agent=None, steps=400, seed=77):
    """
    Run a simple visualization using TrafficEnvMixed.
    If agent is None: always use phase 0.
    If agent not None: use trained agent.best_action().
    """
    env = TrafficEnvMixed(seed=seed)
    env.reset()
    renderer = TrafficRenderer(env)

    running = True
    step = 0

    while running and step < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = env._get_state()
        if agent is not None:
            action = agent.best_action(state)
        else:
            action = 0  # default demo: NS straight green

        next_state, reward, done, info = env.step(action)

        renderer.render_step(action, info)

        if done:
            break

        step += 1

    renderer.close()
