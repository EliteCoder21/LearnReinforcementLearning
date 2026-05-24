import pygame
import torch
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from tank_env import TankEnv, ARENA_W, ARENA_H, TANK_RADIUS, MAX_STEPS

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RED = (200, 40, 40)
BLUE = (40, 100, 220)
GREEN = (40, 200, 40)
GRAY = (100, 100, 100)
DARK = (30, 30, 30)
WHITE = (220, 220, 220)
BG = (20, 20, 30)
PROJ_COLOR = (255, 200, 50)

pygame.init()
font = None


def get_font(size=18):
    global font
    if font is None:
        font = pygame.font.Font(None, size)
    return font


def draw_tank(surf, x, y, angle, color, health, max_hp=5):
    pygame.draw.circle(surf, color, (int(x), int(y)), TANK_RADIUS, 2)
    ex = int(x + TANK_RADIUS * 1.3 * pygame.math.Vector2(1, 0).rotate_rad(-angle).x)
    ey = int(y - TANK_RADIUS * 1.3 * pygame.math.Vector2(1, 0).rotate_rad(-angle).y)
    pygame.draw.line(surf, color, (int(x), int(y)), (ex, ey), 3)
    bw, bh = 24, 4
    bx, by = int(x - bw // 2), int(y - TANK_RADIUS - 10)
    pygame.draw.rect(surf, (60, 60, 60), (bx, by, bw, bh))
    fill = bw * health / max_hp
    pygame.draw.rect(
        surf,
        GREEN if health > 2 else (255, 255, 0) if health > 0 else RED,
        (bx, by, fill, bh),
    )


def main():
    B = 1
    env = TankEnv(batch_size=B)
    obs = env.reset().to(_DEV)

    model = None
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        from ppo_model import PPONetwork

        model = PPONetwork(obs_dim=env.observation_dim, act_dim=env.action_dim).to(_DEV)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=_DEV, weights_only=True)
        )
        model.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")

    screen = pygame.display.set_mode((ARENA_W, ARENA_H + 40))
    pygame.display.set_caption("Tank Battle")
    clock = pygame.time.Clock()
    running, auto = True, True
    step_n = 0
    total_rew = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto = not auto
                elif event.key == pygame.K_r:
                    obs = env.reset().to(_DEV)
                    step_n = 0
                    total_rew = 0.0
                elif event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_RIGHT and not auto:
                    pass

        if auto or (not auto and pygame.key.get_pressed()[pygame.K_RIGHT]):
            with torch.no_grad():
                obs_flat = obs.reshape(-1, obs.shape[-1])
                if model:
                    logits, _ = model(obs_flat)
                else:
                    logits = torch.randn(B * env.T, env.action_dim, device=_DEV)
                actions = torch.distributions.Categorical(logits=logits).sample()
            next_obs, rewards, done, _ = env.step(actions.reshape(B, env.T).cpu())
            next_obs = next_obs.to(_DEV)
            total_rew += rewards.mean().item()
            obs = next_obs
            step_n += 1
            if done.all():
                print(f"Episode done at step {step_n}, total reward={total_rew:.2f}")
                obs = env.reset().to(_DEV)
                step_n = 0
                total_rew = 0.0

        screen.fill(BG)
        ob = env.obstacles[0].cpu()
        for oi in range(len(ob)):
            x, y, w, h = ob[oi].tolist()
            pygame.draw.rect(screen, GRAY, (x, y, w, h))
            pygame.draw.rect(screen, DARK, (x, y, w, h), 1)

        for t in range(env.T):
            if not env.alive[0, t]:
                continue
            x = env.pos[0, t, 0].item()
            y = env.pos[0, t, 1].item()
            a = env.angle[0, t].item()
            hp = env.health[0, t].item()
            team = env.team[t].item()
            color = RED if team == 0 else BLUE
            draw_tank(screen, x, y, a, color, int(hp))

        p_colors = [
            (env.pteam[0, p].item(), env.palive[0, p].item())
            for p in range(min(env.pcnt[0].item(), env.maxp))
        ]
        for pi in range(min(env.pcnt[0].item(), env.maxp)):
            team_id, alive = p_colors[pi]
            if not alive:
                continue
            px = env.px[0, pi].item()
            py = env.py[0, pi].item()
            pcolor = RED if team_id == 0 else BLUE
            pygame.draw.circle(screen, PROJ_COLOR, (int(px), int(py)), 3)
            pygame.draw.circle(screen, pcolor, (int(px), int(py)), 3, 1)

        ra = env.alive[0, :5].sum().item()
        ba = env.alive[0, 5:].sum().item()
        info = f"Step {step_n}/{MAX_STEPS}  Red:{int(ra)} Blue:{int(ba)}  Reward:{total_rew:.2f}"
        if auto:
            info += "  [AUTO]"
        else:
            info += "  [MANUAL - hold RIGHT]"
        ts = get_font(20).render(info, True, WHITE)
        screen.blit(ts, (10, ARENA_H + 10))
        pygame.display.flip()
        clock.tick(30 if auto else 10)

    pygame.quit()


if __name__ == "__main__":
    main()
