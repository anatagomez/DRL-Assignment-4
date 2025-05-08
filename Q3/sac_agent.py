import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dmc import make_dmc_env
from collections import deque
from tensorboardX import SummaryWriter
import os
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ENV_NAME = "humanoid-walk"
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA = 0.2  # Entropy coefficient
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
START_STEPS = 5000
TOTAL_STEPS = 500_000
UPDATE_AFTER = 1000
UPDATE_EVERY = 50
EVAL_EVERY = 5000

# Observation and action space
env = make_dmc_env(ENV_NAME, seed=42, flatten=True, use_pixels=False)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# TensorBoard
logdir = f'runs/sac_{ENV_NAME}_{int(time.time())}'
writer = SummaryWriter(logdir)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=torch.tensor(self.obs_buf[idxs], device=device),
                    obs2=torch.tensor(self.obs2_buf[idxs], device=device),
                    act=torch.tensor(self.act_buf[idxs], device=device),
                    rew=torch.tensor(self.rew_buf[idxs], device=device).unsqueeze(1),
                    done=torch.tensor(self.done_buf[idxs], device=device).unsqueeze(1))


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU())
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        std = log_std.exp()
        pi_distribution = torch.distributions.Normal(mu, std)
        pi_action = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1, keepdim=True)
        action = torch.tanh(pi_action) * self.act_limit
        return action, logp_pi

    def get_deterministic(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        return torch.tanh(mu) * self.act_limit


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


# Initialize networks
actor = MLPActor(obs_dim, act_dim, act_limit).to(device)
critic1 = MLPCritic(obs_dim, act_dim).to(device)
critic2 = MLPCritic(obs_dim, act_dim).to(device)
critic1_target = MLPCritic(obs_dim, act_dim).to(device)
critic2_target = MLPCritic(obs_dim, act_dim).to(device)
critic1_target.load_state_dict(critic1.state_dict())
critic2_target.load_state_dict(critic2.state_dict())

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=CRITIC_LR)
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=CRITIC_LR)

replay_buffer = ReplayBuffer(obs_dim, act_dim, BUFFER_SIZE)


@torch.no_grad()
def evaluate_policy(n_episodes=5):
    eval_env = make_dmc_env(ENV_NAME, seed=999, flatten=True, use_pixels=False)
    returns = []
    for _ in range(n_episodes):
        obs, done, ep_ret = eval_env.reset(), False, 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = actor.get_deterministic(obs_tensor).cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += reward
        returns.append(ep_ret)
    return np.mean(returns)


def update():
    batch = replay_buffer.sample_batch(BATCH_SIZE)
    obs, obs2, act, rew, done = batch['obs'], batch['obs2'], batch['act'], batch['rew'], batch['done']

    with torch.no_grad():
        a2, logp_a2 = actor(obs2)
        q1_targ = critic1_target(obs2, a2)
        q2_targ = critic2_target(obs2, a2)
        q_targ = torch.min(q1_targ, q2_targ) - ALPHA * logp_a2
        backup = rew + GAMMA * (1 - done) * q_targ

    q1 = critic1(obs, act)
    q2 = critic2(obs, act)
    critic1_loss = ((q1 - backup) ** 2).mean()
    critic2_loss = ((q2 - backup) ** 2).mean()

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()

    for p in critic1.parameters(): p.requires_grad = False
    for p in critic2.parameters(): p.requires_grad = False

    a, logp_a = actor(obs)
    q1_pi = critic1(obs, a)
    q2_pi = critic2(obs, a)
    q_pi = torch.min(q1_pi, q2_pi)

    actor_loss = (ALPHA * logp_a - q_pi).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    for p in critic1.parameters(): p.requires_grad = True
    for p in critic2.parameters(): p.requires_grad = True

    with torch.no_grad():
        for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
            target_param.data.mul_(1 - TAU)
            target_param.data.add_(TAU * param.data)
        for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
            target_param.data.mul_(1 - TAU)
            target_param.data.add_(TAU * param.data)

    return actor_loss.item(), critic1_loss.item(), critic2_loss.item()


# Main loop
obs, ep_ret, ep_len = env.reset(), 0, 0
for t in range(1, TOTAL_STEPS + 1):
    if t < START_STEPS:
        act = env.action_space.sample()
    else:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            act = actor(obs_tensor)[0].cpu().numpy()[0]

    next_obs, reward, terminated, truncated, _ = env.step(act)
    done = terminated or truncated
    ep_ret += reward
    ep_len += 1

    replay_buffer.store(obs, act, reward, next_obs, done)
    obs = next_obs

    if done:
        writer.add_scalar("Episode/Return", ep_ret, t)
        obs, ep_ret, ep_len = env.reset(), 0, 0

    if t >= UPDATE_AFTER and t % UPDATE_EVERY == 0:
        for _ in range(UPDATE_EVERY):
            actor_loss, critic1_loss, critic2_loss = update()
            writer.add_scalar("Loss/Actor", actor_loss, t)
            writer.add_scalar("Loss/Critic1", critic1_loss, t)
            writer.add_scalar("Loss/Critic2", critic2_loss, t)

    if t % EVAL_EVERY == 0:
        eval_score = evaluate_policy()
        print(f"Step: {t}, Eval Score: {eval_score}")
        writer.add_scalar("Eval/Return", eval_score, t)
        torch.save(actor.state_dict(), "actor.pth")

writer.close()
