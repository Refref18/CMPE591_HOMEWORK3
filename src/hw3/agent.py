import torch
from torch import optim
import torch.nn.functional as F
from model import VPG

# Discount factor for future rewards
gamma = 0.95

class Agent():
    """def __init__(self):
        # Initialize the model and optimizer
        self.model = VPG()
        self.rewards = []
        self.saved_log_probs = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

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

        return action.squeeze(0)"""

    def __init__(self):
        # Model and optimizer
        self.model = VPG()
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

        # Buffers for the current episode
        self.reward_buffer = []
        self.log_prob_buffer = []

        # History of past-episode mean returns for baseline
        self.return_mean_history = []

    def decide_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_mean, action_std = self.model(state_tensor).chunk(2, dim=-1)
        action_std = F.softplus(action_std) + 5e-2

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        self.log_prob_buffer.append(log_prob)
        return action.squeeze(0)

    def add_reward(self, reward):
        self.reward_buffer.append(reward)

    def update_model(self):
        # 1. Compute returns-to-go
        returns_to_go = []
        running_return = 0.0
        for r in reversed(self.reward_buffer):
            running_return = r + gamma * running_return
            returns_to_go.insert(0, running_return)
        returns_to_go = torch.tensor(returns_to_go)

        # 2. Update baseline history
        current_return_mean = returns_to_go.mean().item()
        self.return_mean_history.append(current_return_mean)

        # 3. Compute running baseline from history
        running_baseline = torch.tensor(self.return_mean_history, dtype=torch.float32).mean()

        # 4. Compute advantage estimates
        advantage_estimates = returns_to_go - running_baseline

        # 5. Compute policy loss
        log_prob_tensor = torch.stack(self.log_prob_buffer)
        loss = -torch.mean(log_prob_tensor * advantage_estimates)

        # 6. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 7. Clear buffers for next episode
        self.reward_buffer = []
        self.log_prob_buffer = []

    def update_model_old(self):
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