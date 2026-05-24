import torch
import math

ARENA_W, ARENA_H = 800, 600
TANK_RADIUS = 16
MAX_HEALTH = 5
COOLDOWN_MAX = 12
MAX_STEPS = 200
N_OBSTACLES = 4
N_TANKS = 10
N_TANKS_PER_TEAM = 5
N_RAYS = 8
RAY_MAX_DIST = 300.0
PROJECTILE_SPEED = 12
EPS = 1e-6

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_N = torch.arange(N_TANKS, device=_DEV)


def _fill_obs_batch(env):
    B, T, R = env.B, env.T, env.R
    own = torch.stack(
        [
            env.pos[..., 0] / ARENA_W,
            env.pos[..., 1] / ARENA_H,
            torch.sin(env.angle),
            torch.cos(env.angle),
            env.health / MAX_HEALTH,
            env.cooldown / COOLDOWN_MAX,
        ],
        dim=-1,
    )
    # Batched raycasting for all tanks at once
    ox = env.pos[:, :, 0:1].unsqueeze(-1)  # (B, T, 1, 1)
    oy = env.pos[:, :, 1:2].unsqueeze(-1)  # (B, T, 1, 1)
    ang = env.angle.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
    a = ang + torch.linspace(0, 2 * math.pi, R + 1, device=_DEV)[:R]  # (B, T, 1, R)
    dx = torch.cos(a)  # (B, T, 1, R)
    dy = torch.sin(a)  # (B, T, 1, R)
    best = torch.full((B, T, 1, R), float("inf"), device=_DEV)

    for wx in (0, ARENA_W):
        ok = dx.abs() > EPS
        t = torch.where(ok, (wx - ox) / dx, float("inf"))
        v = ok & (t > EPS)
        yh = oy + dy * t
        v &= (yh >= -EPS) & (yh <= ARENA_H + EPS)
        best = torch.where(v & (t < best), t, best)
    for wy in (0, ARENA_H):
        ok = dy.abs() > EPS
        t = torch.where(ok, (wy - oy) / dy, float("inf"))
        v = ok & (t > EPS)
        xh = ox + dx * t
        v &= (xh >= -EPS) & (xh <= ARENA_W + EPS)
        best = torch.where(v & (t < best), t, best)

    for oi in range(N_OBSTACLES):
        o = env.obstacles[:, oi]  # (B, 4)
        x = o[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        y = o[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        w = o[:, 2:3].unsqueeze(-1).unsqueeze(-1)
        h = o[:, 3:4].unsqueeze(-1).unsqueeze(-1)
        for ax, ay, bx, by in [
            (x, y, x + w, y),
            (x + w, y, x + w, y + h),
            (x + w, y + h, x, y + h),
            (x, y + h, x, y),
        ]:
            sx = bx - ax
            sy = by - ay
            denom = dx * sy - dy * sx
            ok = denom.abs() > EPS
            t = torch.where(ok, ((ax - ox) * sy - (ay - oy) * sx) / denom, float("inf"))
            u = torch.where(ok, ((ax - ox) * dy - (ay - oy) * dx) / denom, float("inf"))
            v = ok & (t > EPS) & (u >= -EPS) & (u <= 1 + EPS) & (t < best)
            best = torch.where(v, t, best)

    # Tank circles: all-vs-all pairwise, mask self & dead
    cx = env.pos[:, :, 0:1].unsqueeze(1)  # (B, 1, T, 1)
    cy = env.pos[:, :, 1:2].unsqueeze(1)  # (B, 1, T, 1)
    ocx = ox - cx  # (B, T, T, 1)
    ocy = oy - cy
    b = 2 * (ocx * dx + ocy * dy)  # (B, T, T, R)
    c = ocx * ocx + ocy * ocy - TANK_RADIUS * TANK_RADIUS  # (B, T, T, 1)
    disc = b * b - 4 * c  # (B, T, T, R)
    ok = disc >= 0
    sqrt = torch.sqrt(disc.clamp(min=0))
    t1, t2 = (-b - sqrt) / 2, (-b + sqrt) / 2
    tv = torch.where(t1 >= 0, t1, t2)
    tv = torch.where((t1 >= 0) | (t2 >= 0), tv, float("inf"))
    tv = torch.where(ok, tv, float("inf"))
    alive_j = env.alive.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)
    tv = torch.where(alive_j, tv, float("inf"))
    self_m = torch.eye(T, device=_DEV, dtype=torch.bool).reshape(1, T, T, 1)
    tv = torch.where(~self_m, tv, float("inf"))
    tv_best = tv.min(dim=2, keepdim=True).values  # (B, T, 1, R)
    v = (tv_best > EPS) & (tv_best < best)
    best = torch.where(v, tv_best, best)

    rays = (best / RAY_MAX_DIST).clamp(0, 1).squeeze(-2)  # (B, T, R)
    return torch.cat([own, rays], dim=-1).to(torch.float32)


class TankEnv:
    def __init__(self, batch_size=16):
        self.B, self.T, self.R = batch_size, N_TANKS, N_RAYS
        self.team = torch.zeros(N_TANKS, device=_DEV, dtype=torch.long)
        self.team[N_TANKS_PER_TEAM:] = 1
        self.reset()

    def _init_obstacles(self):
        margin, gap = 60, 50
        cols = (ARENA_W - 2 * margin) // gap
        rows = (ARENA_H - 2 * margin) // gap
        n = min(N_OBSTACLES, cols * rows)
        idx = torch.randperm(cols * rows, device=_DEV)[:n]
        cx = (idx % cols).float() * gap + margin + 20
        cy = (idx // cols).float() * gap + margin + 20
        self.obstacles = torch.stack(
            [cx - 20, cy - 20, torch.full_like(cx, 40), torch.full_like(cx, 40)], dim=1
        )
        self.obstacles = self.obstacles.unsqueeze(0).expand(self.B, -1, -1).contiguous()

    def _spawn(self):
        B, T = self.B, self.T
        m, half = 60, ARENA_W // 2
        xs_r = torch.rand(B, N_TANKS_PER_TEAM, device=_DEV) * (half - 2 * m) + m
        ys_r = torch.rand(B, N_TANKS_PER_TEAM, device=_DEV) * (ARENA_H - 2 * m) + m
        xs_b = torch.rand(B, N_TANKS_PER_TEAM, device=_DEV) * (half - 2 * m) + half + m
        ys_b = torch.rand(B, N_TANKS_PER_TEAM, device=_DEV) * (ARENA_H - 2 * m) + m
        self.pos = torch.stack(
            [torch.cat([xs_r, xs_b], 1), torch.cat([ys_r, ys_b], 1)], dim=-1
        )
        self.angle = torch.rand(B, T, device=_DEV) * 2 * math.pi
        self.health = torch.full((B, T), MAX_HEALTH, device=_DEV, dtype=torch.float32)
        self.cooldown = torch.zeros(B, T, device=_DEV, dtype=torch.float32)
        self.alive = torch.ones(B, T, device=_DEV, dtype=torch.bool)
        self.step_count = torch.zeros(B, device=_DEV, dtype=torch.long)
        self.done = torch.zeros(B, device=_DEV, dtype=torch.bool)

    def _init_proj(self):
        self.maxp = 100
        self.px = torch.zeros(self.B, self.maxp, device=_DEV)
        self.py = torch.zeros(self.B, self.maxp, device=_DEV)
        self.pvx = torch.zeros(self.B, self.maxp, device=_DEV)
        self.pvy = torch.zeros(self.B, self.maxp, device=_DEV)
        self.pteam = torch.full((self.B, self.maxp), -1, device=_DEV, dtype=torch.long)
        self.palive = torch.zeros(self.B, self.maxp, device=_DEV, dtype=torch.bool)
        self.pcnt = torch.zeros(self.B, device=_DEV, dtype=torch.long)

    def reset(self):
        self._init_obstacles()
        self._spawn()
        self._init_proj()
        return _fill_obs_batch(self)

    def step(self, actions):
        B, T = self.B, self.T
        rs, ms = 0.12, 4.5
        actions = actions.to(_DEV)

        rot_l = (actions == 3) & self.alive
        rot_r = (actions == 4) & self.alive
        self.angle[rot_l] -= rs
        self.angle[rot_r] += rs

        fwd = (actions == 1) & self.alive
        bwd = (actions == 2) & self.alive
        move = fwd | bwd
        sgn = torch.where(fwd, 1.0, -1.0) * move.float()
        nx = self.pos[..., 0] + torch.cos(self.angle) * ms * sgn
        ny = self.pos[..., 1] + torch.sin(self.angle) * ms * sgn
        valid = (
            move
            & (nx >= TANK_RADIUS)
            & (nx <= ARENA_W - TANK_RADIUS)
            & (ny >= TANK_RADIUS)
            & (ny <= ARENA_H - TANK_RADIUS)
        )

        for oi in range(N_OBSTACLES):
            o = self.obstacles[:, oi]
            cx = nx.clamp(o[:, 0:1], o[:, 0:1] + o[:, 2:3])
            cy = ny.clamp(o[:, 1:2], o[:, 1:2] + o[:, 3:4])
            valid &= (nx - cx).pow(2) + (ny - cy).pow(2) >= TANK_RADIUS**2
        nx_exp = nx.unsqueeze(-1)
        ny_exp = ny.unsqueeze(-1)
        px_exp = self.pos[:, :, 0].unsqueeze(1)
        py_exp = self.pos[:, :, 1].unsqueeze(1)
        d2 = (nx_exp - px_exp).pow(2) + (ny_exp - py_exp).pow(2)
        self_mask = torch.eye(T, device=_DEV, dtype=torch.bool).reshape(1, T, T)
        d2 = torch.where(self_mask, float("inf"), d2)
        collision = (d2 < (TANK_RADIUS * 2 + 2) ** 2) & self.alive.unsqueeze(1)
        valid &= ~collision.any(dim=-1)

        self.pos = torch.where(
            valid.unsqueeze(-1), torch.stack([nx, ny], dim=-1), self.pos
        )
        self.cooldown = (self.cooldown - 1).clamp(min=0)
        shoot = (actions == 5) & (self.cooldown <= 0) & self.alive
        self.cooldown[shoot] = COOLDOWN_MAX

        sb, st = shoot.nonzero(as_tuple=True)
        for k in range(min(len(sb), self.B * self.maxp)):
            b, t = sb[k].item(), st[k].item()
            if self.pcnt[b] < self.maxp:
                a = self.angle[b, t].item()
                ci = self.pcnt[b].item()
                self.px[b, ci] = self.pos[b, t, 0].item() + math.cos(a) * TANK_RADIUS
                self.py[b, ci] = self.pos[b, t, 1].item() + math.sin(a) * TANK_RADIUS
                self.pvx[b, ci] = math.cos(a) * PROJECTILE_SPEED
                self.pvy[b, ci] = math.sin(a) * PROJECTILE_SPEED
                self.pteam[b, ci] = self.team[t].item()
                self.palive[b, ci] = True
                self.pcnt[b] += 1

        self.px += self.pvx
        self.py += self.pvy
        inb = (
            (self.px >= 0)
            & (self.px <= ARENA_W)
            & (self.py >= 0)
            & (self.py <= ARENA_H)
        )
        self.palive &= inb

        for oi in range(N_OBSTACLES):
            o = self.obstacles[:, oi]
            obs_hit = (
                (self.px >= o[:, 0:1])
                & (self.px <= o[:, 0:1] + o[:, 2:3])
                & (self.py >= o[:, 1:2])
                & (self.py <= o[:, 1:2] + o[:, 3:4])
            )
            self.palive &= ~obs_hit

        hit_rew = torch.zeros(2, B, device=_DEV)
        kill_rew = torch.zeros(2, B, device=_DEV)
        px_exp = self.px.unsqueeze(-1)
        py_exp = self.py.unsqueeze(-1)
        tx_exp = self.pos[:, :, 0].unsqueeze(1)
        ty_exp = self.pos[:, :, 1].unsqueeze(1)
        d2 = (px_exp - tx_exp).pow(2) + (py_exp - ty_exp).pow(2)
        tank_team = self.team.reshape(1, 1, T)
        proj_team = self.pteam.unsqueeze(-1)
        hit = (
            (d2 < (TANK_RADIUS + 4) ** 2)
            & (proj_team != tank_team)
            & self.palive.unsqueeze(-1)
            & self.alive.unsqueeze(1)
        )
        hit_count = hit.any(dim=1).float()
        self.health -= hit_count
        self.palive &= ~hit.any(dim=-1)
        is_team0 = (self.pteam == 0).unsqueeze(-1)
        is_team1 = (self.pteam == 1).unsqueeze(-1)
        hit_rew[0] = (hit & is_team0).any(dim=-1).float().sum(dim=-1) * 2.0
        hit_rew[1] = (hit & is_team1).any(dim=-1).float().sum(dim=-1) * 2.0
        pre_alive = self.alive.clone()
        died = (self.health <= 0) & pre_alive
        self.alive = self.health > 0
        tank_team_2d = self.team.unsqueeze(0)
        died_team0 = died & ((1 - tank_team_2d) == 0)
        died_team1 = died & ((1 - tank_team_2d) == 1)
        kill_rew[0] = died_team0.float().sum(dim=-1) * 10.0
        kill_rew[1] = died_team1.float().sum(dim=-1) * 10.0

        ra = self.alive[:, :N_TANKS_PER_TEAM].sum(1)
        ba = self.alive[:, N_TANKS_PER_TEAM:].sum(1)
        self.step_count += 1
        self.done = (ra == 0) | (ba == 0) | (self.step_count >= MAX_STEPS)

        tm = self.team.unsqueeze(0)
        hpt = torch.where(tm == 0, hit_rew[0:1].t(), hit_rew[1:2].t())
        kpt = torch.where(tm == 0, kill_rew[0:1].t(), kill_rew[1:2].t())
        rewards = -0.05 + hpt + kpt * 10.0
        rewards[~self.alive] = -5.0
        red_win = self.done & (ra > 0) & (ba == 0)
        blue_win = self.done & (ba > 0) & (ra == 0)
        rewards[red_win.unsqueeze(-1) & (self.team.unsqueeze(0) == 0)] += 5.0
        rewards[blue_win.unsqueeze(-1) & (self.team.unsqueeze(0) == 1)] += 5.0

        return _fill_obs_batch(self), rewards, self.done, {}

    @property
    def observation_dim(self):
        return 6 + N_RAYS

    @property
    def action_dim(self):
        return 6
