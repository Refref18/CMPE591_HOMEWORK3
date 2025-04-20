# Homework 3 Report: Vanilla Policy Gradient (REINFORCE)

## Introduction

In this homework, we aim to train a robot to push an object to a desired position using reinforcement learning methods. The primary focus of this task is to implement and train a Vanilla Policy Gradient (REINFORCE) model. I used gymnasium environment for fast training.

## Approach: Vanilla Policy Gradient (REINFORCE)

REINFORCE is a Monte Carlo policy gradient method that optimizes a parameterized policy by following the gradient of expected return. The policy is updated based on the cumulative rewards obtained during each episode. The following components are crucial for the algorithm:

- **Policy Network:** Uses a neural network to predict the mean and standard deviation of the action distribution.
- **Sampling Actions:** Uses the predicted distribution to sample actions, introducing stochasticity in exploration.
- **Policy Update:** Uses the log-probability of the taken action and the obtained reward to update the policy.

## Implementation Details

- **Network Architecture:** The policy network consists of fully connected layers with ReLU activations, predicting both the mean and standard deviation of the action distribution.
- **Optimizer:** Adam optimizer with a learning rate of `5e-5`.
- **Discount Factor (Gamma):** Set to `0.95` to balance immediate and future rewards.
- **Action Sampling:** The action is sampled from a Gaussian distribution whose mean and standard deviation are predicted by the network.

## Experimental Results

We ran four sets of experiments:

### Experiment 1: Learning Rate = 1e-3

- **Episodes:** 1500+
- **Observation:** The agent failed to learn; rewards remained low and noisy.
- ![Plot 1](plots/total_reward_plot_2025-04-06_13-50-22.png)

### Experiment 2: Learning Rate = 1e-4

- **Episodes:** 700
- **Observation:** Rewards began improving around episode 500 but then declined again.
- ![Plot 2](plots/total_reward_plot_2025-04-06_19-30-30.png)

### Experiment 3: Learning Rate = 1e-4 (continued to 2000 episodes)

- **Episodes:** 2000+
- **Observation:** Initial upward trend gave way to instability; no sustained improvement.
- ![Plot 3](plots/total_reward_plot_2025-04-06_21-10-32.png)

### Experiment 4: REINFORCE + Baseline (Advantage)

- **Modification:** Introduced a running baseline (mean of past episode returns) and used advantages = (return – baseline) in the policy gradient.
- **Episodes:** 10 000 then increased to 500 000
- **Observation:** Learning became stable—both raw returns (blue) and 100‑episode moving average (red) rise smoothly and plateau at higher values, with far less variance.

![Final plot](plots/total_reward_plot_2025-04-20_01-00-34.png)

## Discussion

- **High LR (1e-3)** → Too big updates, no convergence.
- **Lower LR (1e-4)** → Some early gains followed by collapse—PG without variance reduction is unstable.
- **Adding Baseline** → Subtracting a learned baseline dramatically reduces variance and stabilizes learning. You can see the raw reward still fluctuates, but the moving average steadily climbs.

## Visualizing the reults

- You can visualize what model learned by running `trained.py` it shows what best model learned on gymnasium environment.
- You can visualize succesfull and failed examples in `videos` folder.
