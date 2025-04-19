# demo_trained.py

import gymnasium as gym
import torch
import time

from agent import Agent

def demo_once(model_path: str,
              env_name: str = "Pusher-v5",
              render_delay: float = 0.02):
    # 1) create and render environment
    env = gym.make(env_name, render_mode="human")
    
    # 2) load your trained agent
    agent = Agent()
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    
    # 3) single rollout
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        # decide and step
        action = agent.decide_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # render and slow down
        env.render()
        time.sleep(render_delay)
    
    print(f"Demo completed – total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    demo_once("best_model.pt", env_name="Pusher-v5", render_delay=0.03)
