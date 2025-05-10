import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ===== Helper networks =====

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = MLP(obs_dim, act_dim * 2)
        self.act_dim = act_dim

    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        # Tanh correction
        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1, keepdim=True)
        return tanh_action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1)
        self.q2 = MLP(obs_dim + act_dim, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

# ===== Agent class required by the system =====

class Agent:
    def __init__(self):
        self.device = torch.device("cpu")  # Force CPU
        obs_dim = 67
        act_dim = 21
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.actor.load_state_dict(torch.load("sac_actor.pth", map_location=self.device))
        self.actor.eval()

    def act(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.sample(obs).squeeze(0)
        return action.cpu().numpy()
