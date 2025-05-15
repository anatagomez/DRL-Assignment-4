import numpy as np
import torch

class D4PGEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        obs, _ = self.env.reset()
        return self._process_obs(obs)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
