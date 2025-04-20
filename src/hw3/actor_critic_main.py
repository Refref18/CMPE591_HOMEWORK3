# train_actor_critic.py

import os
import gymnasium as gym
import torch
import numpy as np

from actor_critic_agent import SACAgent   # make sure this file is in your PYTHONPATH

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # 1) Environment and SAC agent
    env = gym.make("Pusher-v5", render_mode=None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim,
                     replay_size=int(1e6),
                     gamma=0.99,
                     tau=0.005,
                     alpha=0.2,
                     lr_pi=5e-5,
                     lr_q=3e-4,
                     lr_v=3e-4,
                     hidden_sizes=[256,256],
                     batch_size=256)

    # 2) Training hyperparams
    num_episodes = 50000
    max_steps     = 200
    log_interval  = 1000

    # 3) Logging buffers
    returns = []
    best_return = -1e9

    # 4) Main loop
    for ep in range(1, num_episodes+1):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            # a) select and perform action
            action = agent.decide_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            # b) store and update
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()

            obs = next_obs
            ep_return += reward
            steps += 1

        # c) logging
        returns.append(ep_return)
        if ep_return > best_return:
            best_return = ep_return
            torch.save(agent.policy.state_dict(), "best_sac_model.pt")
            print(f"✨ New best SAC model @ ep {ep}: return {best_return:.2f}")

        if ep % log_interval == 0:
            avg_last = np.mean(returns[-log_interval:])
            print(f"[SAC] Episode {ep:6d} | steps {steps:3d} | return {ep_return:7.2f}"
                  f" | avg_last_{log_interval} {avg_last:7.2f}")

        # d) write per-episode returns
        with open("sac_total_reward_per_episode.txt", "a") as f:
            f.write(f"{ep},{ep_return:.2f}\n")

    # 5) final save
    torch.save(agent.policy.state_dict(), "best_sac_model.pt")
    np.save("sac_rews.npy", np.array(returns))
    print("Training complete.")
