import numpy as np
import random

ARENA_W, ARENA_H = 1000, 800
TANK_RADIUS = 16
PROJECTILE_RADIUS = 4
PROJECTILE_SPEED = 9
MAX_HEALTH = 5
COOLDOWN_MAX = 14
MAX_STEPS = 300
OBSTACLE_SIZE = 40
N_OBSTACLES = 8
N_RAYS = 12
RAY_MAX_DIST = 350.0
N_TANKS_PER_TEAM = 5


def random_rect():
    margin = 60
    cols = (ARENA_W - 2 * margin) // (OBSTACLE_SIZE + 30)
    rows = (ARENA_H - 2 * margin) // (OBSTACLE_SIZE + 30)
    cx = (
        random.randint(0, cols - 1) * (OBSTACLE_SIZE + 30) + margin + OBSTACLE_SIZE // 2
    )
    cy = (
        random.randint(0, rows - 1) * (OBSTACLE_SIZE + 30) + margin + OBSTACLE_SIZE // 2
    )
    return Rect(
        cx - OBSTACLE_SIZE // 2, cy - OBSTACLE_SIZE // 2, OBSTACLE_SIZE, OBSTACLE_SIZE
    )


class Rect:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def minx(self):
        return self.x

    @property
    def maxx(self):
        return self.x + self.w

    @property
    def miny(self):
        return self.y

    @property
    def maxy(self):
        return self.y + self.h


class Tank:
    __slots__ = ("x", "y", "angle", "health", "cooldown", "alive", "team", "idx")

    def __init__(self, x, y, angle, team, idx):
        self.x = x
        self.y = y
        self.angle = angle
        self.health = MAX_HEALTH
        self.cooldown = 0
        self.alive = True
        self.team = team
        self.idx = idx


class Projectile:
    __slots__ = ("x", "y", "vx", "vy", "team")

    def __init__(self, x, y, angle, team):
        self.x = x
        self.y = y
        self.vx = np.cos(angle) * PROJECTILE_SPEED
        self.vy = np.sin(angle) * PROJECTILE_SPEED
        self.team = team

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def is_in_bounds(self):
        return 0 <= self.x <= ARENA_W and 0 <= self.y <= ARENA_H


def _edge_data(obstacles):
    edges = []
    for obs in obstacles:
        edges.append((obs.minx, obs.miny, obs.maxx, obs.miny))
        edges.append((obs.maxx, obs.miny, obs.maxx, obs.maxy))
        edges.append((obs.maxx, obs.maxy, obs.minx, obs.maxy))
        edges.append((obs.minx, obs.maxy, obs.minx, obs.miny))
    return (
        np.array(edges, dtype=np.float32)
        if edges
        else np.empty((0, 4), dtype=np.float32)
    )


def cast_rays(ox, oy, angle, obs_edges, tank_pos, exclude_idx, alive_mask):
    n = N_RAYS
    angles = angle + np.linspace(0, 2 * np.pi, n, endpoint=False)
    dx = np.cos(angles)
    dy = np.sin(angles)
    best_t = np.full(n, np.inf)

    # arena walls
    EPS = 1e-6
    for wx in (0, ARENA_W):
        mask = np.abs(dx) > EPS
        if not np.any(mask):
            continue
        t = np.full(n, np.inf)
        t[mask] = (wx - ox) / dx[mask]
        valid = mask & (t > EPS)
        y_hit = oy + dy * t
        valid &= (y_hit >= -EPS) & (y_hit <= ARENA_H + EPS)
        update = valid & (t < best_t)
        best_t[update] = t[update]
    for wy in (0, ARENA_H):
        mask = np.abs(dy) > EPS
        if not np.any(mask):
            continue
        t = np.full(n, np.inf)
        t[mask] = (wy - oy) / dy[mask]
        valid = mask & (t > EPS)
        x_hit = ox + dx * t
        valid &= (x_hit >= -EPS) & (x_hit <= ARENA_W + EPS)
        update = valid & (t < best_t)
        best_t[update] = t[update]

    # obstacle edges
    for e in range(len(obs_edges)):
        ax, ay, bx, by = obs_edges[e]
        sx = bx - ax
        sy = by - ay
        denom = dx * sy - dy * sx
        mask = np.abs(denom) > EPS
        if not np.any(mask):
            continue
        t = np.full(n, np.inf)
        u = np.full(n, np.inf)
        t[mask] = ((ax - ox) * sy - (ay - oy) * sx) / denom[mask]
        u[mask] = ((ax - ox) * dy[mask] - (ay - oy) * dx[mask]) / denom[mask]
        valid = mask & (t > EPS) & (u >= -EPS) & (u <= 1 + EPS) & (t < best_t)
        best_t[valid] = t[valid]

    # other tanks (circles)
    for i in range(len(tank_pos)):
        if i == exclude_idx or not alive_mask[i]:
            continue
        cx, cy = tank_pos[i]
        ocx = ox - cx
        ocy = oy - cy
        b = 2 * (ocx * dx + ocy * dy)
        c = ocx * ocx + ocy * ocy - TANK_RADIUS * TANK_RADIUS
        disc = b * b - 4 * c  # a = dx^2 + dy^2 = 1 since dx,dy are cos/sin
        mask = disc >= 0
        if not np.any(mask):
            continue
        sqrt_disc = np.sqrt(disc[mask])
        t1 = (-b[mask] - sqrt_disc) * 0.5
        t2 = (-b[mask] + sqrt_disc) * 0.5
        t_vals = np.where(t1 >= 0, t1, t2)
        both_neg = (t1 < 0) & (t2 < 0)
        t_vals[both_neg] = np.inf
        sub_valid = (t_vals > EPS) & (t_vals < best_t[mask])
        valid = np.zeros(n, dtype=bool)
        valid[mask] = sub_valid
        best_t[valid] = t_vals[sub_valid]

    return np.minimum(best_t, RAY_MAX_DIST)


class TankEnv:
    def __init__(self):
        self.obstacles = []
        self.tanks = []
        self.projectiles = []
        self.steps = 0
        self.ray_debug = []
        self._obs_edges = np.empty((0, 4), dtype=np.float32)
        self.reset()

    def _spawn_tanks(self):
        margin = 60
        self.tanks = []
        spawn_zones = [
            (margin, ARENA_W // 2 - margin),
            (ARENA_W // 2 + margin, ARENA_W - margin),
        ]
        for team in range(2):
            x_lo, x_hi = spawn_zones[team]
            placed = 0
            attempts = 0
            while placed < N_TANKS_PER_TEAM and attempts < 200:
                x = random.uniform(x_lo, x_hi)
                y = random.uniform(margin, ARENA_H - margin)
                overlap = False
                for obs in self.obstacles:
                    if (
                        abs(x - obs.cx) < TANK_RADIUS + obs.w / 2 + 8
                        and abs(y - obs.cy) < TANK_RADIUS + obs.h / 2 + 8
                    ):
                        overlap = True
                        break
                if not overlap:
                    for t in self.tanks:
                        dx = x - t.x
                        dy = y - t.y
                        if dx * dx + dy * dy < (TANK_RADIUS * 2 + 6) ** 2:
                            overlap = True
                            break
                if not overlap:
                    angle = random.uniform(0, 2 * np.pi)
                    self.tanks.append(Tank(x, y, angle, team, placed))
                    placed += 1
                attempts += 1

    def reset(self):
        self.obstacles = [random_rect() for _ in range(N_OBSTACLES)]
        self._obs_edges = _edge_data(self.obstacles)
        self._spawn_tanks()
        self.projectiles = []
        self.steps = 0
        self.ray_debug = [[] for _ in range(len(self.tanks))]
        return self._get_obs()

    def _get_obs(self):
        obs_list = []
        tank_pos = np.array([(t.x, t.y) for t in self.tanks], dtype=np.float32)
        alive_mask = np.array([t.alive for t in self.tanks], dtype=bool)
        for i, tank in enumerate(self.tanks):
            if not tank.alive:
                obs_list.append(np.zeros(self._obs_dim(), dtype=np.float32))
                continue
            own = np.array(
                [
                    tank.x / ARENA_W,
                    tank.y / ARENA_H,
                    np.sin(tank.angle),
                    np.cos(tank.angle),
                    tank.health / MAX_HEALTH,
                    tank.cooldown / COOLDOWN_MAX,
                ],
                dtype=np.float32,
            )
            raw = cast_rays(
                tank.x, tank.y, tank.angle, self._obs_edges, tank_pos, i, alive_mask
            )
            self.ray_debug[i] = raw.tolist()
            obs_list.append(
                np.concatenate([own, (raw / RAY_MAX_DIST).astype(np.float32)])
            )
        return obs_list

    def _obs_dim(self):
        return 6 + N_RAYS

    @property
    def observation_dim(self):
        return self._obs_dim()

    action_dim = 6

    def _move_tank(self, tank, action):
        rot_speed = 0.12
        move_speed = 4.5
        if action == 3:
            tank.angle -= rot_speed
        elif action == 4:
            tank.angle += rot_speed
        elif action == 1:
            nx = tank.x + np.cos(tank.angle) * move_speed
            ny = tank.y + np.sin(tank.angle) * move_speed
            if self._can_move_to(nx, ny, tank.idx):
                tank.x, tank.y = nx, ny
        elif action == 2:
            nx = tank.x - np.cos(tank.angle) * move_speed
            ny = tank.y - np.sin(tank.angle) * move_speed
            if self._can_move_to(nx, ny, tank.idx):
                tank.x, tank.y = nx, ny

    def _can_move_to(self, x, y, exclude_idx):
        if not (
            TANK_RADIUS <= x <= ARENA_W - TANK_RADIUS
            and TANK_RADIUS <= y <= ARENA_H - TANK_RADIUS
        ):
            return False
        for obs in self.obstacles:
            cx, cy = max(obs.minx, min(x, obs.maxx)), max(obs.miny, min(y, obs.maxy))
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy < TANK_RADIUS * TANK_RADIUS:
                return False
        for i, t in enumerate(self.tanks):
            if i == exclude_idx or not t.alive:
                continue
            dx = x - t.x
            dy = y - t.y
            if dx * dx + dy * dy < (TANK_RADIUS * 2 + 2) ** 2:
                return False
        return True

    def _shoot(self, tank):
        if tank.cooldown <= 0:
            self.projectiles.append(Projectile(tank.x, tank.y, tank.angle, tank.team))
            tank.cooldown = COOLDOWN_MAX

    def _handle_projectiles(self):
        team_hits = [0, 0]
        team_kills = [0, 0]
        new_proj = []
        for p in self.projectiles:
            p.update()
            if not p.is_in_bounds():
                continue
            hit = False
            # check obstacles
            for obs in self.obstacles:
                if obs.minx <= p.x <= obs.maxx and obs.miny <= p.y <= obs.maxy:
                    hit = True
                    break
            if hit:
                continue
            # check tanks
            for i, tank in enumerate(self.tanks):
                if not tank.alive or tank.team == p.team:
                    continue
                dx = p.x - tank.x
                dy = p.y - tank.y
                if dx * dx + dy * dy < (TANK_RADIUS + PROJECTILE_RADIUS) ** 2:
                    tank.health -= 1
                    team_hits[p.team] += 1
                    if tank.health <= 0:
                        tank.alive = False
                        team_kills[p.team] += 1
                    hit = True
                    break
            if not hit:
                new_proj.append(p)
        self.projectiles = new_proj
        return team_hits, team_kills

    def step(self, actions):
        self.steps += 1
        for i, tank in enumerate(self.tanks):
            if not tank.alive:
                continue
            action = actions[i]
            if action == 5:
                self._shoot(tank)
            elif action != 0:
                self._move_tank(tank, action)
            if tank.cooldown > 0:
                tank.cooldown -= 1

        hits, kills = self._handle_projectiles()

        red_alive = sum(1 for t in self.tanks if t.team == 0 and t.alive)
        blue_alive = sum(1 for t in self.tanks if t.team == 1 and t.alive)
        done = red_alive == 0 or blue_alive == 0 or self.steps >= MAX_STEPS

        rewards = [0.0] * len(self.tanks)
        for i, tank in enumerate(self.tanks):
            if not tank.alive:
                rewards[i] = -5.0 * (1 - tank.health / MAX_HEALTH)
                continue
            r = -0.05
            r += hits[tank.team] * 2.0
            r += kills[tank.team] * 10.0
            if done:
                if tank.team == 0 and blue_alive == 0:
                    r += 5.0
                elif tank.team == 1 and red_alive == 0:
                    r += 5.0
            rewards[i] = r

        obs = self._get_obs()
        truncated = self.steps >= MAX_STEPS
        dones = [done] * len(self.tanks)
        truncs = [truncated] * len(self.tanks)
        return (
            obs,
            rewards,
            dones,
            truncs,
            {"red_alive": red_alive, "blue_alive": blue_alive},
        )
