# demo_multi_record.py

import gymnasium as gym
import torch
import time
from agent import Agent
from gymnasium.wrappers import RecordVideo

def demo_multi_record(
    model_path: str,
    env_name: str = "Pusher-v5",
    num_runs: int = 3,
    render_delay: float = 0.02,
    video_folder: str = "videos"
):
    # 1) make the env in rgb_array mode so it can be recorded
    env = gym.make(env_name, render_mode="rgb_array")
    # 2) wrap to record the first `num_runs` episodes
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="pusher_demo",
        episode_trigger=lambda ep: ep < num_runs
    )

    # 3) load your agent
    agent = Agent()
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()

    # 4) run `num_runs` rollouts back‑to‑back
    for run in range(num_runs):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.decide_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()           # returns frame for RecordVideo
            time.sleep(render_delay)
            steps += 1

        print(f"Run {run+1}/{num_runs}: reward={total_reward:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    demo_multi_record(
        model_path="best_model.pt",
        env_name="Pusher-v5",
        num_runs=3,
        render_delay=0.03,
        video_folder="videos"
    )
