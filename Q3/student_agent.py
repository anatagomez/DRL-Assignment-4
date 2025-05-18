# DDPG Agent Implementation with specified hyperparameters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random

# === Hyperparameters ===
LR = 2.5e-4
GAMMA = 0.99
BATCH_SIZE = 64
TAU = 0.005
REPLAY_BUFFER_SIZE = 1_000_000
TOTAL_STEPS = int(1e8)
NOISE_STD = 0.2

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class DDPGAgent:
    def __init__(self, obs_dim, act_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def select_action(self, state, noise_std=NOISE_STD):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        noise = np.random.normal(0, noise_std, size=action.shape)
        action = np.clip(action + noise, -1, 1)
        return action

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(batch.done), dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            y = rewards + (1 - dones) * GAMMA * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        print(f"Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, Q-Value: {current_q.mean().item():.4f}, Shapes: S={states.shape}, A={actions.shape}, Q={current_q.shape}")


class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.obs_dim = 67
        self.act_dim = 21

        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.actor.load_state_dict(torch.load("ddpg_actor_step6450000.pth", map_location=self.device))
        self.actor.eval()

    def act(self, observation):
        if isinstance(observation, dict):
            observation = np.concatenate([v.ravel() for v in observation.values()])

        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs).cpu().squeeze(0).tolist()
        return np.clip(action, -1, 1)
