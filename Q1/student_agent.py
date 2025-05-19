import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
MAX_ACTION = 2.0
MIN_ACTION = -2.0
DEVICE = torch.device("cpu")  

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(state).to(DEVICE),
            torch.FloatTensor(action).to(DEVICE),
            torch.FloatTensor(reward).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(next_state).to(DEVICE),
            torch.FloatTensor(done).unsqueeze(1).to(DEVICE)
        )

    def size(self):
        return len(self.buffer)

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, MIN_ACTION, MAX_ACTION)

    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        self.total_it += 1

        state, action, reward, next_state, done = self.replay_buffer.sample()

        with torch.no_grad():
            noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(MIN_ACTION, MAX_ACTION)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * GAMMA * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % POLICY_DELAY == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

class Agent(object):
    def __init__(self):
        self.env = gym.make("Pendulum-v1", render_mode='rgb_array')
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        self.td3_agent = TD3Agent(state_dim, action_dim, max_action)

        weights_path = "td3_actor_ep1000.pth"
        if os.path.exists(weights_path):
            self.td3_agent.actor.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            self.td3_agent.actor.eval()
        else:
            raise FileNotFoundError(f"Trained weights not found at {weights_path}")

        self.total_steps = 0

    def act(self, observation):
        action = self.td3_agent.select_action(np.array(observation), noise=0.0)
        return action