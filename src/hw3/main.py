
import gymnasium as gym
import torch
import numpy as np

from agent import Agent
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    # Initialize the environment and the agent
    env = gym.make("Pusher-v5")

    agent = Agent()
    num_episodes = 10000  # Number of episodes to train
    step_size    = 200    # maximum steps per episode

    rews = []  # List to store cumulative rewards for each episode
    best_reward = float('-inf')

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0

        # Run the episode until done OR until you hit step_size
        while not done and episode_steps < step_size:
            action = agent.decide_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.add_reward(reward)
            cumulative_reward += reward
            state = next_state
            episode_steps += 1

        # Log every 100 episodes
        if i % 100 == 0:
            print(f"Episode={i}, steps={episode_steps}, reward={cumulative_reward:.2f}")

        rews.append(cumulative_reward)
        agent.update_model()

        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            torch.save(agent.model.state_dict(), "best_model.pt")
            print(f"✨ New best model saved with reward: {best_reward:.2f}")

        with open("model_total_reward_per_episode.txt", "a") as f:
            f.write(f"{i},{cumulative_reward}\n")

    # Final save
    torch.save(agent.model.state_dict(), "best_model.pt")
    np.save("rews.npy", np.array(rews))

