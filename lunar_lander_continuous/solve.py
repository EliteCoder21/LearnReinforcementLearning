import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


class CEM:
    def __init__(self, pop_size=200, elite_frac=0.1, horizon=8, num_iters=4, env=None):
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.horizon = horizon
        self.num_iters = num_iters
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_mean = np.zeros((self.horizon, self.action_dim))
        self.action_std = np.ones((self.horizon, self.action_dim)) * 0.5
        self.gamma = 0.99

    def plan(self, state):
        for iteration in range(self.num_iters):
            actions = np.random.normal(
                self.action_mean,
                self.action_std,
                (self.pop_size, self.horizon, self.action_dim),
            )
            actions = np.clip(actions, -1, 1)
            returns = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                total_r = 0
                discount = 1.0
                s = state.copy()
                for t in range(self.horizon):
                    s, r, term, trunc, _ = self.env.step(actions[i, t])
                    total_r += discount * r
                    discount *= self.gamma
                    if term or trunc:
                        break
                returns[i] = total_r
            elite_idx = returns.argsort()[-int(self.pop_size * self.elite_frac) :]
            elite_actions = actions[elite_idx]
            self.action_mean = elite_actions.mean(axis=0)
            self.action_std = elite_actions.std(axis=0) + 1e-4
        return self.action_mean[0]

    def reset(self):
        self.action_mean = np.zeros((self.horizon, self.action_dim))
        self.action_std = np.ones((self.horizon, self.action_dim)) * 0.5


def run_mpc(
    env,
    num_episodes=200,
    horizon=8,
    pop_size=200,
    elite_frac=0.1,
    num_iters=4,
    render_every=None,
):
    rewards_history = []
    print(f"MPC: {num_episodes} eps, h={horizon}, pop={pop_size}, iters={num_iters}")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 1000:
            cem = CEM(
                pop_size=pop_size,
                elite_frac=elite_frac,
                horizon=horizon,
                num_iters=num_iters,
                env=env,
            )
            action = cem.plan(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if render_every and episode % render_every == 0:
                env.render()
                time.sleep(0.01)
        rewards_history.append(total_reward)
        if episode % 10 == 0:
            avg_reward = (
                np.mean(rewards_history[-5:])
                if len(rewards_history) >= 5
                else np.mean(rewards_history)
            )
            print(
                f"Episode {episode + 1}/{num_episodes}: reward={total_reward:.1f}, avg={avg_reward:.1f}"
            )
            if avg_reward > 100:
                print(f"SOLVED at episode {episode + 1}!")
                break
    return rewards_history


def plot_training(rewards_history, save_path="training_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.3, label="Episode Reward")
    if len(rewards_history) >= 20:
        smoothed = np.convolve(rewards_history, np.ones(20) / 20, mode="valid")
        plt.plot(range(19, len(rewards_history)), smoothed, label="Smoothed (20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Lunar Lander Continuous - Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    print("Starting MPC training on Lunar Lander Continuous...")
    print("Goal: Solve in < 200 episodes")
    print("=" * 50)
    env = gym.make("LunarLanderContinuous-v3")
    rewards_history = run_mpc(
        env, num_episodes=200, horizon=8, pop_size=200, elite_frac=0.1, num_iters=4
    )
    print("\n" + "=" * 50)
    print(f"Total episodes: {len(rewards_history)}")
    print(f"Final avg (last 20): {np.mean(rewards_history[-20:]):.2f}")
    plot_training(rewards_history)
    print("\n=== Final Visualization ===")
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")
    run_mpc(env, num_episodes=3, render_every=1)
    env.close()
    print("Done!")
