
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

    rews = []  # List to store cumulative rewards for each episode
    best_reward = float('-inf')  # Initialize the best reward with a very low value

    for i in range(num_episodes):        
        state, _ = env.reset()  # Get the initial state
        done = False  # Flag to indicate the end of an episode
        cumulative_reward = 0.0  # Track the total reward for the episode
        episode_steps = 0  # Step counter for the current episode

        # Run the episode until it is done
        while not done:
            action = agent.decide_action(state)

            # env.step only bir defa çağrılmalı:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.add_reward(reward)
            cumulative_reward += reward
            state = next_state

        # Print the total reward for the current episode
        if i % 100 == 0:
            
            print(f"Episode={i}, reward={cumulative_reward}")
        rews.append(cumulative_reward)  # Save the cumulative reward

        # Update the model after each episode
        agent.update_model()

        # Save the model if the current episode's reward is the best so far
        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            torch.save(agent.model.state_dict(), "best_model.pt")
            print(f"✨ New best model saved with reward: {best_reward}")

        # Log the total reward for the episode in a file
        with open("model_total_reward_per_episode.txt", "a") as file:
            file.write(f"{i},{cumulative_reward}\n")

    # Save the final best model and the rewards list
    torch.save(agent.model.state_dict(), "best_model.pt")
    np.save("rews.npy", np.array(rews))

        

