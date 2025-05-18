import numpy as np
import torch
from student_agent import DDPGAgent
from dmc import make_dmc_env
from env_wrapper import D4PGEnvWrapper 
from torch.utils.tensorboard import SummaryWriter
import os
import time

# === Setup ===
env = D4PGEnvWrapper(make_dmc_env("humanoid-walk", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False))
eval_env = D4PGEnvWrapper(make_dmc_env("humanoid-walk", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False))
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
agent = DDPGAgent(obs_dim, act_dim)
print(f"Training on: {agent.device}")

obs = env.reset()
episode_return = 0
step_count = 0
episode_count = 0
start_time = time.time()
last_log_time = start_time
last_log_step = 0

writer = SummaryWriter(log_dir="runs2/ddpg_lr_decay")

# === Evaluation Function ===
def evaluate(agent, env, episodes=5):
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_return = 0
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.actor(state_tensor).cpu().numpy()[0]
            obs, reward, done, _ = env.step(action)
            total_return += reward
        returns.append(total_return)
    return np.mean(returns)

# === Training Loop ===
while step_count < int(1e8):
    action = agent.select_action(obs)
    next_obs, reward, done, _ = env.step(action)

    agent.replay_buffer.add(obs, action, reward, next_obs, float(done))
    obs = next_obs
    episode_return += reward
    step_count += 1

    # Update and print training diagnostics every 1000 steps
    if len(agent.replay_buffer) >= 64:
        batch = agent.replay_buffer.sample(64)
        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(agent.device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32).to(agent.device)
        rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32).unsqueeze(1).to(agent.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(agent.device)
        dones = torch.tensor(np.array(batch.done), dtype=torch.float32).unsqueeze(1).to(agent.device)

        with torch.no_grad():
            target_actions = agent.actor_target(next_states)
            target_q = agent.critic_target(next_states, target_actions)
            y = rewards + (1 - dones) * 0.99 * target_q

        current_q = agent.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(current_q, y)

        agent.critic_opt.zero_grad()
        critic_loss.backward()
        agent.critic_opt.step()

        actor_loss = -agent.critic(states, agent.actor(states)).mean()
        agent.actor_opt.zero_grad()
        actor_loss.backward()
        agent.actor_opt.step()

        for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
            target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)
        for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
            target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)

        if step_count % 1000 == 0:
            print(f"Step: {step_count}, Episode: {episode_count}, Return: {episode_return:.3f}, Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, Q-Value: {current_q.mean().item():.4f}, Shapes: S={states.shape}, A={actions.shape}, Q={current_q.shape}")
            writer.add_scalar("Loss/Critic", critic_loss.item(), step_count)
            writer.add_scalar("Loss/Actor", actor_loss.item(), step_count)
            writer.add_scalar("Q-Value", current_q.mean().item(), step_count)
            writer.add_scalar("LR/Actor", agent.actor_scheduler.get_last_lr()[0], step_count)
            writer.add_scalar("LR/Critic", agent.critic_scheduler.get_last_lr()[0], step_count)

    if done:
        if step_count % 1000 == 0:
            writer.add_scalar("Episode/Return", episode_return, step_count)
            print(f"[Episode Done] Step {step_count} | Episode Return: {episode_return:.3f}")
            obs = env.reset()
            episode_return = 0
            episode_count += 1

    # Periodic evaluation and checkpointing
    if step_count % 50000 == 0:
        avg_eval_return = evaluate(agent, eval_env)
        writer.add_scalar("Eval/Return", avg_eval_return, step_count)
        print(f"[Evaluation] Step {step_count} | Eval Avg Return: {avg_eval_return:.3f}")

        os.makedirs("checkpoints_lr", exist_ok=True)
        torch.save(agent.actor.state_dict(), f"checkpoints/ddpg_actor_step{step_count}.pth")
        torch.save(agent.critic.state_dict(), f"checkpoints/ddpg_critic_step{step_count}.pth")

    # Log throughput every 10,000 steps
    if step_count % 10000 == 0:
        now = time.time()
        elapsed = now - last_log_time
        steps = step_count - last_log_step
        throughput = steps / elapsed
        writer.add_scalar("Perf/Steps_per_second", throughput, step_count)
        print(f"[Throughput] Step {step_count} | {throughput:.2f} steps/sec")
        last_log_time = now
        last_log_step = step_count

writer.close()
