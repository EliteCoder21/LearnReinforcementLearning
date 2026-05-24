import torch
import torch.nn as nn

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPONetwork(nn.Module):
    def __init__(self, obs_dim=14, act_dim=6, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.actor(h), self.critic(h)

    def get_value(self, obs):
        return self.critic(self.encoder(obs))

    def get_dist(self, obs):
        logits = self.actor(self.encoder(obs))
        return torch.distributions.Categorical(logits=logits)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, value, dist.log_prob(action), dist.entropy()
