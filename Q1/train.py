import gymnasium as gym
import numpy as np
import torch
from student_agent import Agent, TD3Agent  # Assuming TD3Agent is accessible too
import os

project_dir = "Q1"
os.chdir(project_dir)

def train_td3(env_name="Pendulum-v1", episodes=1000, max_steps=200, start_timesteps=10000):
    env = gym.make(env_name, render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action)

    total_steps = 0
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if total_steps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= start_timesteps:
                agent.train()

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Avg (last 10): {np.mean(episode_rewards[-10:]):.2f}")

        if (episode + 1) % 100 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(project_dir, f"td3_actor_ep{episode + 1}.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(project_dir, f"td3_critic_ep{episode + 1}.pth"))


    env.close()

if __name__ == "__main__":
    train_td3()
