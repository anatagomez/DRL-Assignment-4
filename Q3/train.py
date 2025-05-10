import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from student_agent import Actor, Critic  # Reuse networks

import dmc  # Your provided dmc.py

# Hyperparameters
ACT_DIM = 21
OBS_DIM = 67
REPLAY_SIZE = int(1e6)
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 0.005
LR = 1e-4
ALPHA = 0.2  # Entropy weight


# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

# SAC agent with update functions
class SACAgent:
    def __init__(self, obs_dim, act_dim, device='cuda'):
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)
        # self.log_alpha = torch.tensor(np.log(ALPHA), requires_grad=True, device=self.device)
        # self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)
        #---Automatic Entropy Tuning---
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = -act_dim

    def update(self, replay, batch_size):
        state, action, reward, next_state, done = replay.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_action)
            min_next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done) * GAMMA * (min_next_q - self.log_alpha.exp() * next_log_prob)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        sampled_action, log_prob = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, sampled_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp() * log_prob - min_q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()


        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# Main training loop
def train():
    env = dmc.make_dmc_env("humanoid-walk", seed=42, flatten=True, use_pixels=False)
    agent = SACAgent(OBS_DIM, ACT_DIM, device='cuda')
    replay = ReplayBuffer(REPLAY_SIZE)

    print(f"Training in {agent.device}")
    total_steps = 10_000_000
    start_steps = 25_000
    updates_per_step = 1
    eval_interval = 100_000
    episode_rewards = []
    state, _ = env.reset()
    episode_reward = 0
    episode_step = 0

    for step in range(1, total_steps + 1):
        if step < start_steps:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device) 
            with torch.no_grad():
                action, _ = agent.actor.sample(obs_tensor)
            action = action.squeeze(0).cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_step += 1

        if done:
            state, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
            episode_step = 0

        if step >= start_steps:
            for _ in range(updates_per_step):
                agent.update(replay, BATCH_SIZE)

        if step % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Step: {step}, AvgReward (last 10): {avg_reward:.2f}")

        if step % 1_000_000 == 0:
            torch.save(agent.actor.state_dict(), f"sac_actor_{step//1_000_000}M.pth")


    # Save trained actor
    torch.save(agent.actor.state_dict(), "sac_actor.pth")
    print("Saved actor weights to sac_actor.pth")

if __name__ == "__main__":
    train()
