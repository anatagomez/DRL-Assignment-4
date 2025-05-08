import gymnasium as gym
import numpy as np
import torch
from sac_agent import MLPActor  # or copy-paste MLPActor here

class Agent(object):
    def __init__(self):
        self.actor = MLPActor(obs_dim=67, act_dim=21, act_limit=1.0)
        self.actor.load_state_dict(torch.load("actor.pth", map_location="cpu"))
        self.actor.eval()

    def act(self, observation):
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs_tensor)
        return action.squeeze(0).numpy()

