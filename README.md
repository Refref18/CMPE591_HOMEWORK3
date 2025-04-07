# Homework 3 Report: Vanilla Policy Gradient (REINFORCE)

## Introduction

In this homework, we aim to train a robot to push an object to a desired position using reinforcement learning methods. The primary focus of this task is to implement and train a Vanilla Policy Gradient (REINFORCE) model, which is worth 75 points. The environment used is similar to the previous homework, but now the action space is continuous. We use high-level states instead of raw pixel data for the input.

## Problem Statement

The task is to train a robot arm to push an object to a goal position. The action space is continuous, meaning the agent needs to learn how to control the arm's movements precisely. The goal is to maximize the reward by pushing the object closer to the goal while minimizing unnecessary movements.

## Approach: Vanilla Policy Gradient (REINFORCE)

REINFORCE is a Monte Carlo policy gradient method that optimizes a parameterized policy by following the gradient of expected return. The policy is updated based on the cumulative rewards obtained during each episode. The following components are crucial for the algorithm:

- **Policy Network:** Uses a neural network to predict the mean and standard deviation of the action distribution.
- **Sampling Actions:** Uses the predicted distribution to sample actions, introducing stochasticity in exploration.
- **Policy Update:** Uses the log-probability of the taken action and the obtained reward to update the policy.

## Implementation Details

- **Network Architecture:** The policy network consists of fully connected layers with ReLU activations, predicting both the mean and standard deviation of the action distribution.
- **Optimizer:** Adam optimizer with a learning rate of `1e-4`.
- **Discount Factor (Gamma):** Set to `0.99` to balance immediate and future rewards.
- **Action Sampling:** The action is sampled from a Gaussian distribution whose mean and standard deviation are predicted by the network.

## Experimental Results

Three sets of experiments were conducted with varying learning rates.

### Experiment 1: Learning Rate = 1e-3

- **Number of Episodes:** 1500+
- **Observation:** The agent did not learn effectively, and the reward remained low throughout the episodes.

### Experiment 2: Learning Rate = 1e-4

- **Number of Episodes:** 700
- **Observation:** Initially, the reward increased around episode 500 but started decreasing afterward.

### Experiment 3: Learning Rate = 1e-4 (Continued)

- **Number of Episodes:** 700+
- **Observation:** The reward initially showed an upward trend around episode 500 but then deteriorated again, indicating instability in the learning process.

And after 2000 epoisodes still no sign of stable learning
![Final plot](src/plots/total_reward_plot_2025-04-07_11-50-54.png)

## Visual Results

Below are the plots representing the reward over episodes for each experiment:

- **Plot 1:** Initial training with a high learning rate (1e-3) - No significant learning observed.  
  ![Plot 1](src/plots/total_reward_plot_2025-04-06_13-50-22.png)

- **Plot 2:** Reduced learning rate (1e-4) - Gradual improvement initially.  
  ![Plot 2](src/plots/total_reward_plot_2025-04-06_19-30-30.png)

- **Plot 3:** Continued training with the same setup - Initial improvement followed by instability.  
  ![Plot 3](src/plots/total_reward_plot_2025-04-06_21-10-32.png)

## Observations

- A higher learning rate (1e-3) caused instability, leading to poor learning.
- Lowering the learning rate to `1e-4` helped the agent learn better at first, but the learning was not stable.
- Adjusting the reward function and other hyperparameters or using adaptive learning rates may improve performance.
