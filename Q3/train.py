import numpy as np
import torch
import random
from collections import deque, namedtuple
from student_agent import StudentAgent, project_distribution, ATOM_SUPPORT
from dmc import make_dmc_env
from env_wrapper import D4PGEnvWrapper  
from torch.utils.tensorboard import SummaryWriter

N_STEP = 5
GAMMA = 0.99
BATCH_SIZE = 256
REPLAY_SIZE = 100000
ALPHA = 0.6
HARD_UPDATE_FREQ = 100
TRAIN_AFTER = 10000
TRAIN_FREQ = 50

Transition = namedtuple("Transition", ["obs", "act", "reward", "next_obs", "done"])

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step_queue = deque(maxlen=N_STEP)

    def add(self, transition):
        self.n_step_queue.append(transition)
        if len(self.n_step_queue) == N_STEP:
            transition = self._get_n_step_transition()
            max_prio = self.priorities.max() if self.buffer else 1.0
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def _get_n_step_transition(self):
        reward, next_obs, done = 0, self.n_step_queue[-1].next_obs, self.n_step_queue[-1].done
        for idx, trans in enumerate(self.n_step_queue):
            reward += (GAMMA ** idx) * trans.reward
            if trans.done:
                break
        return Transition(self.n_step_queue[0].obs, self.n_step_queue[0].act, reward, next_obs, done)

    def sample(self, batch_size):
        probs = self.priorities[:len(self.buffer)] ** ALPHA
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        return batch, indices

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

def train():
    env = D4PGEnvWrapper(make_dmc_env("humanoid-walk", seed=42, flatten=True, use_pixels=False))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = StudentAgent(obs_dim, act_dim)
    buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
    writer = SummaryWriter(log_dir="runs/d4pg_humanoid_walk")

    print("Training in:", agent.device)
    obs = env.reset()
    episode_return, episode_len = 0, 0
    total_steps, train_steps = 0, 0

    while total_steps < 50_000_000:
        sigma = np.random.uniform(1/32, 1.0)
        action = agent.act(obs, noise_std=sigma)
        next_obs, reward, done, _ = env.step(action)
        buffer.add(Transition(obs, action, reward, next_obs, done))
        obs = next_obs
        episode_return += reward
        episode_len += 1
        total_steps += 1

        if done:
            print(f"Step {total_steps} | Return: {episode_return} | Length: {episode_len}")
            writer.add_scalar("Episode/Return", episode_return, total_steps)
            writer.add_scalar("Episode/Length", episode_len, total_steps)
            obs = env.reset()
            episode_return, episode_len = 0, 0

        if total_steps > TRAIN_AFTER and total_steps % TRAIN_FREQ == 0 and len(buffer) >= BATCH_SIZE:
            for _ in range(TRAIN_FREQ):
                train_steps += 1
                batch, indices = buffer.sample(BATCH_SIZE)

                states = torch.from_numpy(np.array([b.obs for b in batch], dtype=np.float32)).to(agent.device)
                actions = torch.from_numpy(np.array([b.act for b in batch], dtype=np.float32)).to(agent.device)
                rewards = torch.tensor([[b.reward] for b in batch], dtype=torch.float32).to(agent.device)
                next_states = torch.from_numpy(np.array([b.next_obs for b in batch], dtype=np.float32)).to(agent.device)
                dones = torch.tensor([[b.done] for b in batch], dtype=torch.float32).to(agent.device)

                with torch.no_grad():
                    next_actions = agent.actor_target(next_states)
                    next_dist = agent.critic_target.get_distribution(next_states, next_actions)
                    target_dist = project_distribution(next_dist, rewards, dones, GAMMA)

                log_probs = agent.critic(states, actions)
                critic_loss = -(target_dist * log_probs).sum(dim=1).mean()

                agent.critic_opt.zero_grad()
                critic_loss.backward()
                agent.critic_opt.step()

                agent.actor_opt.zero_grad()
                policy_actions = agent.actor(states)
                policy_loss = -(agent.critic.get_distribution(states, policy_actions) * ATOM_SUPPORT.to(agent.device)).sum(dim=1).mean()
                policy_loss.backward()
                agent.actor_opt.step()

                td_errors = (target_dist - agent.critic.get_distribution(states, actions)).abs().sum(dim=1).detach().cpu().numpy()
                buffer.update_priorities(indices, td_errors + 1e-6)

                writer.add_scalar("Loss/Critic", critic_loss.item(), train_steps)
                writer.add_scalar("Loss/Actor", policy_loss.item(), train_steps)
                writer.add_scalar("TD_Error/Mean", np.mean(td_errors), train_steps)

            if total_steps % HARD_UPDATE_FREQ == 0:
                agent.hard_update_targets()

    agent.save("actor.pth")
    writer.close()

if __name__ == "__main__":
    train()
