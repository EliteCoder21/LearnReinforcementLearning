import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pygame
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tank_env import (
    TankEnv,
    ARENA_W,
    ARENA_H,
    TANK_RADIUS,
    PROJECTILE_RADIUS,
    MAX_HEALTH,
    N_RAYS,
    N_TANKS_PER_TEAM,
    RAY_MAX_DIST,
)
from dqn_agent import DQNAgent

RED = (200, 50, 50)
BLUE = (50, 70, 220)
DARK_RED = (120, 20, 20)
DARK_BLUE = (20, 40, 140)
GREEN = (60, 200, 60)
GRAY = (80, 80, 80)
DARK_GRAY = (40, 40, 40)
WHITE = (200, 200, 200)
YELLOW = (255, 220, 50)
BLACK = (15, 15, 20)
ORANGE = (240, 160, 30)


def draw_tank(surf, tank, friendly_rays=None):
    if not tank.alive:
        return
    color = RED if tank.team == 0 else BLUE
    turret_color = DARK_RED if tank.team == 0 else DARK_BLUE
    pygame.draw.circle(surf, color, (int(tank.x), int(tank.y)), TANK_RADIUS)
    pygame.draw.circle(surf, turret_color, (int(tank.x), int(tank.y)), TANK_RADIUS - 4)
    ex = tank.x + np.cos(tank.angle) * TANK_RADIUS
    ey = tank.y + np.sin(tank.angle) * TANK_RADIUS
    pygame.draw.line(surf, WHITE, (tank.x, tank.y), (ex, ey), 3)

    bar_w = 30
    bar_h = 4
    bx = tank.x - bar_w // 2
    by = tank.y - TANK_RADIUS - 8
    pygame.draw.rect(surf, (40, 40, 40), (bx, by, bar_w, bar_h))
    fill = bar_w * (tank.health / MAX_HEALTH)
    hp_c = GREEN if tank.health > 2 else ORANGE if tank.health > 1 else RED
    pygame.draw.rect(surf, hp_c, (bx, by, fill, bar_h))

    if friendly_rays is not None:
        for j in range(N_RAYS):
            a = tank.angle + (2 * np.pi * j / N_RAYS)
            d = friendly_rays[j]
            ex = tank.x + np.cos(a) * min(d, RAY_MAX_DIST)
            ey = tank.y + np.sin(a) * min(d, RAY_MAX_DIST)
            pygame.draw.line(surf, (60, 120, 60, 80), (tank.x, tank.y), (ex, ey), 1)
            if d < RAY_MAX_DIST - 1:
                pygame.draw.circle(surf, YELLOW, (int(ex), int(ey)), 2)


def draw_obstacles(surf, obstacles):
    for obs in obstacles:
        pygame.draw.rect(surf, GRAY, (obs.x, obs.y, obs.w, obs.h))
        pygame.draw.rect(surf, DARK_GRAY, (obs.x, obs.y, obs.w, obs.h), 2)


def draw_projectiles(surf, projectiles):
    for p in projectiles:
        color = RED if p.team == 0 else BLUE
        pygame.draw.circle(surf, color, (int(p.x), int(p.y)), PROJECTILE_RADIUS)
        pygame.draw.circle(surf, WHITE, (int(p.x), int(p.y)), PROJECTILE_RADIUS // 2)


def render(env, surf, show_rays=False):
    surf.fill(BLACK)
    draw_obstacles(surf, env.obstacles)
    pygame.draw.rect(surf, WHITE, (0, 0, ARENA_W, ARENA_H), 2)

    for p in env.projectiles:
        c = RED if p.team == 0 else BLUE
        pygame.draw.circle(surf, c, (int(p.x), int(p.y)), PROJECTILE_RADIUS)

    for i, tank in enumerate(env.tanks):
        if not tank.alive:
            continue
        rays = env.ray_debug[i] if show_rays else None
        draw_tank(surf, tank, rays)

    font = pygame.font.Font(None, 22)
    red_alive = sum(1 for t in env.tanks if t.team == 0 and t.alive)
    blue_alive = sum(1 for t in env.tanks if t.team == 1 and t.alive)
    texts = [
        (f"Red alive: {red_alive}/5", RED),
        (f"Blue alive: {blue_alive}/5", BLUE),
        (f"Step: {env.steps}", WHITE),
    ]
    for k, (txt, col) in enumerate(texts):
        s = font.render(txt, True, col)
        surf.blit(s, (10, 10 + k * 22))

    pygame.display.flip()


def plot_rewards(
    red_history, blue_history, save_path="multi-agent-rl/training_curve.png"
):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(red_history, alpha=0.3, color=RED, label="Red Team")
    if len(red_history) >= 20:
        s = np.convolve(red_history, np.ones(50) / 50, mode="valid")
        plt.plot(
            range(49, len(red_history)),
            s,
            color=RED,
            linewidth=2,
            label="Smoothed (50)",
        )
    plt.xlabel("Episode")
    plt.ylabel("Team Reward")
    plt.title("Red Team")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(blue_history, alpha=0.3, color=BLUE, label="Blue Team")
    if len(blue_history) >= 20:
        s = np.convolve(blue_history, np.ones(50) / 50, mode="valid")
        plt.plot(
            range(49, len(blue_history)),
            s,
            color=BLUE,
            linewidth=2,
            label="Smoothed (50)",
        )
    plt.xlabel("Episode")
    plt.ylabel("Team Reward")
    plt.title("Blue Team")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def train(num_episodes=500, render_episodes=None, show_rays=False):
    env = TankEnv()
    obs_dim = env.observation_dim
    act_dim = env.action_dim

    red_policy = DQNAgent(obs_dim, act_dim)
    blue_policy = DQNAgent(obs_dim, act_dim)

    if render_episodes is None:
        render_episodes = []
    pygame.init()
    surf = pygame.display.set_mode((ARENA_W, ARENA_H))
    pygame.display.set_caption("Multi-Agent Tank Battle 5v5")
    clock = pygame.time.Clock()

    red_rewards, blue_rewards = [], []
    all_tanks = list(range(N_TANKS_PER_TEAM * 2))

    for episode in range(num_episodes):
        obs = env.reset()
        ep_red, ep_blue = 0.0, 0.0
        done = False
        step = 0

        info = {"red_alive": 5, "blue_alive": 5}
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return red_rewards, blue_rewards

            actions = [0] * len(all_tanks)
            for i in all_tanks:
                if not env.tanks[i].alive:
                    continue
                policy = red_policy if env.tanks[i].team == 0 else blue_policy
                actions[i] = policy.act(obs[i])

            next_obs, rewards, dones, truncs, info = env.step(actions)
            done = dones[0]

            for i in all_tanks:
                if not env.tanks[i].alive and dones[i]:
                    continue
                policy = red_policy if env.tanks[i].team == 0 else blue_policy
                policy.memory.push(
                    obs[i], actions[i], rewards[i], next_obs[i], dones[i]
                )

            if episode > 5 and step % 5 == 0:
                red_policy.update()
                blue_policy.update()

            obs = next_obs
            ep_red += sum(rewards[i] for i in all_tanks if env.tanks[i].team == 0)
            ep_blue += sum(rewards[i] for i in all_tanks if env.tanks[i].team == 1)

            if episode in render_episodes:
                render(env, surf, show_rays)
                clock.tick(30)

            step += 1

        red_policy.decay_epsilon()
        blue_policy.decay_epsilon()
        red_rewards.append(ep_red)
        blue_rewards.append(ep_blue)

        if episode % 25 == 0:
            avg_r = (
                np.mean(red_rewards[-50:])
                if len(red_rewards) >= 50
                else np.mean(red_rewards)
            )
            avg_b = (
                np.mean(blue_rewards[-50:])
                if len(blue_rewards) >= 50
                else np.mean(blue_rewards)
            )
            ra = info["red_alive"]
            ba = info["blue_alive"]
            print(
                f"Ep {episode:4d} | Red={ep_red:+7.2f} Blue={ep_blue:+7.2f} "
                f"| Alive {ra}v{ba} | eps={red_policy.epsilon:.3f} "
                f"| avg50R={avg_r:.1f} avg50B={avg_b:.1f}"
            )

    pygame.quit()
    return red_rewards, blue_rewards


if __name__ == "__main__":
    print("=" * 65)
    print("Training 5v5 Multi-Agent Tank Battle with DQN (parameter sharing)")
    print("=" * 65)

    render_eps = set(range(0, 200, 50))
    red_hist, blue_hist = train(
        num_episodes=300, render_episodes=render_eps, show_rays=True
    )

    print("\n" + "=" * 65)
    print(f"Episodes: {len(red_hist)}")
    print(f"Red  final avg50: {np.mean(red_hist[-50:]):.2f}")
    print(f"Blue final avg50: {np.mean(blue_hist[-50:]):.2f}")
    plot_rewards(red_hist, blue_hist)
    print("Done!")
