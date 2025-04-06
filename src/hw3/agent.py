import torch
from torch import optim
import torch.nn.functional as F
from model import VPG

# Discount factor for future rewards
gamma = 0.99

class Agent():
    def __init__(self):
        # Initialize the model and optimizer
        self.model = VPG()
        self.rewards = []
        self.saved_log_probs = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def decide_action(self, state):
        # Convert state to tensor and add batch dimension
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get mean and standard deviation from the model
        action_mean, action_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(action_std) + 5e-2

        # Sample action from a Gaussian distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Store log probability for policy update
        self.saved_log_probs.append(log_prob)

        return action.squeeze(0)

    def update_model(self):
        R = 0
        returns = []

        # Calculate discounted rewards
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # Normalize returns for stability
        returns = torch.tensor(returns)
        returns = (returns - returns.mean())

        # Calculate policy loss
        policy_loss = [-log_prob * R for log_prob, R in zip(self.saved_log_probs, returns)]

        # Update model
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear stored rewards and log probabilities
        self.rewards = []
        self.saved_log_probs = []

    def add_reward(self, reward):
        self.rewards.append(reward)
