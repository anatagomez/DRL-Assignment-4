import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from dmc import make_dmc_env

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MAX_EPISODES = 1000
MAX_STEPS = 1000
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
REPLAY_BUFFER_SIZE = int(1e6)
START_TIMESTEPS = 10000
EXPLORATION_NOISE = 0.1
EVAL_FREQ = 5000
SAVE_MODEL = True
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def size(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
  
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        if replay_buffer.size() < BATCH_SIZE:
            return

        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * GAMMA * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

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

def train():
    env = make_dmc_env("cartpole-balance", SEED, flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    total_steps = 0
    episode_rewards = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            if total_steps < START_TIMESTEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state))
                action = (action + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= START_TIMESTEPS:
                agent.train(replay_buffer)

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")

        if SAVE_MODEL and (episode + 1) % 50 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(MODEL_DIR, "td3_actor.pth"))
            print(f"Saved model at episode {episode + 1}")

    if SAVE_MODEL:
        torch.save(agent.actor.state_dict(), os.path.join(MODEL_DIR, "td3_actor.pth"))
        print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
