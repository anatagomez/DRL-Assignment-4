import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === Distributional Critic Parameters ===
V_MIN = -150
V_MAX = 150
NUM_ATOMS = 101
ATOM_SUPPORT = torch.linspace(V_MIN, V_MAX, NUM_ATOMS).view(1, -1)

# === Actor Network ===
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# === Distributional Critic ===
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ATOMS)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        logits = self.net(x)
        return F.log_softmax(logits, dim=1)

    def get_distribution(self, obs, act):
        logits = self.forward(obs, act)
        return torch.exp(logits)


# === Projection of Distributional Targets ===
def project_distribution(next_distr, rewards, dones, gamma):
    batch_size = rewards.size(0)
    atom_support = ATOM_SUPPORT.to(rewards.device)
    delta_z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)
    projected = torch.zeros_like(next_distr)

    Tz = rewards + (1 - dones) * gamma * atom_support
    Tz = Tz.clamp(V_MIN, V_MAX)
    b = (Tz - V_MIN) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    for i in range(batch_size):
        for j in range(NUM_ATOMS):
            l_idx = l[i][j]
            u_idx = u[i][j]
            if l_idx == u_idx:
                projected[i][l_idx] += next_distr[i][j]
            else:
                projected[i][l_idx] += next_distr[i][j] * (u[i][j] - b[i][j])
                projected[i][u_idx] += next_distr[i][j] * (b[i][j] - l[i][j])
    return projected


# === Submission Agent ===
class Agent:
    def __init__(self):
        obs_dim = 67
        act_dim = 21
        self.device = torch.device("cpu")
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.actor.load_state_dict(torch.load("actor.pth", map_location=self.device))
        self.actor.eval()

    def act(self, observation):
        if isinstance(observation, dict):
            observation = np.concatenate([v.ravel() for v in observation.values()])
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]
        return action


# === Training Agent ===
class StudentAgent:
    def __init__(self, obs_dim=67, act_dim=21):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def act(self, obs, noise_std=0.1):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]
        return np.clip(action + np.random.normal(0, noise_std, size=action.shape), -1, 1)

    def save(self, path="actor.pth"):
        torch.save(self.actor.state_dict(), path)

    def hard_update_targets(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
