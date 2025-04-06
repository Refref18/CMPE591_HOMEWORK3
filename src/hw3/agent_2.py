import torch
from torch import optim
import torch.nn.functional as F
from model import VPG

gamma = 0.99  # Discount factor
alpha = 0.2  # Entropy temperature (controls exploration-exploitation trade-off)
tau = 0.005  # Soft update parameter
lr = 1e-4  # Learning rate for all networks

class SACAgent():
    def __init__(self):
        # Policy network (Actor)
        self.model = VPG()
        self.critic1 = VPG()  # Q1 network
        self.critic2 = VPG()  # Q2 network
        self.value_net = VPG()  # Value network
        self.target_value_net = VPG()  # Target value network
        
        # Copy weights from value network to target value network
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.rewards = []
        self.log_probs = []

    def decide_action(self, state):
        # Convert state to tensor and compute action mean and standard deviation
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_mean, action_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(action_std) + 5e-2

        # Create a Gaussian distribution for the action
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Store log probability for policy update
        self.log_probs.append(log_prob)

        return action.squeeze(0)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def update_model(self):
        # Calculate discounted rewards
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        value_loss = []
        q1_loss = []
        q2_loss = []

        for log_prob, R in zip(self.log_probs, returns):
            # Update Q-value networks
            current_q1 = self.critic1(R.unsqueeze(0))
            current_q2 = self.critic2(R.unsqueeze(0))
            
            # Target value
            target_value = self.target_value_net(R.unsqueeze(0)).detach()
            q_target = R + gamma * target_value
            
            # Q-value loss
            q1_loss.append(F.mse_loss(current_q1, q_target))
            q2_loss.append(F.mse_loss(current_q2, q_target))

            # Value loss
            expected_value = self.value_net(R.unsqueeze(0))
            q_min = torch.min(current_q1, current_q2).detach()
            v_target = q_min - alpha * log_prob
            value_loss.append(F.mse_loss(expected_value, v_target))

            # Policy loss (actor loss)
            policy_loss.append((alpha * log_prob - q_min).mean())

        # Update policy network
        self.policy_optimizer.zero_grad()
        torch.cat(policy_loss).sum().backward()
        self.policy_optimizer.step()

        # Update critic networks
        self.critic1_optimizer.zero_grad()
        torch.cat(q1_loss).sum().backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        torch.cat(q2_loss).sum().backward()
        self.critic2_optimizer.step()

        # Update value network
        self.value_optimizer.zero_grad()
        torch.cat(value_loss).sum().backward()
        self.value_optimizer.step()

        # Soft update target value network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Clear the memory for the next episode
        self.rewards = []
        self.log_probs = []

    def save_best_model(self, best_reward):
        torch.save(self.model.state_dict(), "best_sac_model.pt")
        print(f"✨ New best SAC model saved with reward: {best_reward}")
