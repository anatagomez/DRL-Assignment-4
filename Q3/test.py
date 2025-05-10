from dmc import make_dmc_env

# state-only env, flat vector
env = make_dmc_env("humanoid-walk", seed=42, flatten=True, use_pixels=False)

# # pixel-only env, (84,84,3) uint8 frames
# env = make_dmc_env("hopper-hop", seed=1, flatten=True, use_pixels=True)

# roll one episode
obs, info = env.reset()
# print(obs)
print(obs.shape)
done = False
while not done:
    act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)
    done = terminated or truncated
