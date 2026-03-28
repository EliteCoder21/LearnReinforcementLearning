import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))


class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer()
        
        self.noise = OUNoise(action_dim)
    
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().detach().numpy()[0]
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def train(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions.detach())
        target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
    
    def sample(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        return self.state
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu


def train(env, agent, num_episodes=1000, max_steps=1000, batch_size=256, reward_threshold=200):
    episode_rewards = []
    episode_losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        total_critic_loss = 0
        total_actor_loss = 0
        loss_count = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            losses = agent.train(batch_size)
            if losses:
                critic_loss, actor_loss = losses
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss
                loss_count += 1
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        avg_loss = (total_critic_loss + total_actor_loss) / loss_count if loss_count > 0 else 0
        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 10): {avg_reward:.2f} | Avg Loss: {avg_loss:.4f}")
        
        if np.mean(episode_rewards[-100:]) >= reward_threshold:
            print(f"Solved! Average reward over last 100 episodes: {np.mean(episode_rewards[-100:]):.2f}")
            break
    
    return episode_rewards, episode_losses


def evaluate(env, agent, num_episodes=10, max_steps=1000):
    eval_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
        print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    return np.mean(eval_rewards), np.std(eval_rewards)


def plot_training(episode_rewards, episode_losses):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(episode_rewards)
    axes[0].axhline(y=200, color='r', linestyle='--', label='Target (200)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(episode_losses)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    agent = DDPGAgent(state_dim, action_dim)
    
    print("Starting training...")
    episode_rewards, episode_losses = train(env, agent, num_episodes=1000)
    
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = evaluate(env, agent, num_episodes=10)
    print(f"Evaluation: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    agent.save("ddpg_lunar_lander.pth")
    plot_training(episode_rewards, episode_losses)
    
    env.close()
