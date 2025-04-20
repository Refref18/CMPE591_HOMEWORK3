# sac_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import random
from collections import deque
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1) Simple MLP builder
# ----------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# ----------------------------
# 2) Policy (actor) network
# ----------------------------
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes, nn.ReLU, nn.ReLU)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs):
        mean, std = self(obs)
        dist = D.Normal(mean, std)
        z = dist.rsample()                                    # reparameterized sample
        action = torch.tanh(z)                                # squash to (−1,1)
        logp = dist.log_prob(z).sum(axis=-1, keepdim=True)
        # correction for tanh squashing:
        logp -= (2*(torch.log(torch.tensor(2.0)) 
                  - z 
                  - F.softplus(-2*z))).sum(axis=-1, keepdim=True)
        return action, logp

# ----------------------------
# 3) Q‑network (critic)
# ----------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        # inputs are [obs, action]
        self.q = mlp([obs_dim + act_dim] + hidden_sizes + [1], nn.ReLU)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return torch.squeeze(self.q(x), -1)  # output shape = [batch]

# ----------------------------
# 4) SAC Agent
# ----------------------------
class SACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_range=1.0,
        replay_size=int(1e6),
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr_pi=5e-5,     # <-- policy LR
        lr_q=3e-4,      # <-- critics LR
        lr_v=3e-4,      # <-- value net LR
        hidden_sizes=[256,256],
        batch_size=256
    ):
        # networks
        self.policy     = GaussianPolicy(obs_dim, act_dim, hidden_sizes).to(device)
        self.q1         = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2         = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.v          = mlp([obs_dim] + hidden_sizes + [1], nn.ReLU).to(device)
        self.v_target   = mlp([obs_dim] + hidden_sizes + [1], nn.ReLU).to(device)
        self.v_target.load_state_dict(self.v.state_dict())

        # optimizers with separate LRs
        self.pi_optimizer = optim.Adam(self.policy.parameters(), lr=lr_pi)
        self.q1_optimizer = optim.Adam(self.q1.parameters(),     lr=lr_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(),     lr=lr_q)
        self.v_optimizer  = optim.Adam(self.v.parameters(),      lr=lr_v)

        # replay buffer, hyperparams, etc.
        self.replay_buffer = deque(maxlen=replay_size)
        self.batch_size    = batch_size
        self.gamma         = gamma
        self.tau           = tau
        self.alpha         = alpha
        self.action_range  = action_range

    def store_transition(self, obs, act, rew, next_obs, done):
        self.replay_buffer.append((obs, act, rew, next_obs, done))

    def decide_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
        mean, std = self.policy(obs_t)
        if deterministic:
            z = mean
            action = torch.tanh(z)
        else:
            action, _ = self.policy.sample(obs_t)
        return (action.cpu().detach().numpy()[0] * self.action_range)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough data yet

        # 1) sample a batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs, act, rew, next_obs, done = map(lambda x: torch.as_tensor(x, dtype=torch.float32).to(device), zip(*batch))
        rew      = rew.unsqueeze(-1)
        done_f   = (1 - done).unsqueeze(-1)

        # 2) update Q‑networks
        with torch.no_grad():
            v_next = self.v_target(next_obs)
            q_target = rew + self.gamma * done_f * v_next

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        loss_q1 = F.mse_loss(q1, q_target)
        loss_q2 = F.mse_loss(q2, q_target)

        self.q1_optimizer.zero_grad(); loss_q1.backward(); self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(); loss_q2.backward(); self.q2_optimizer.step()

        # 3) update Value network
        with torch.no_grad():
            a2, logp_a2 = self.policy.sample(obs)
            q1_pi = self.q1(obs, a2)
            q2_pi = self.q2(obs, a2)
            min_q_pi = torch.min(q1_pi, q2_pi)
            v_target_val = min_q_pi - self.alpha * logp_a2

        v = torch.squeeze(self.v(obs), -1)
        loss_v = F.mse_loss(v, v_target_val)

        self.v_optimizer.zero_grad(); loss_v.backward(); self.v_optimizer.step()

        # 4) update Policy network
        a_new, logp_new = self.policy.sample(obs)
        q1_new = self.q1(obs, a_new)
        q2_new = self.q2(obs, a_new)
        min_q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * logp_new - min_q_new).mean()

        self.pi_optimizer.zero_grad(); policy_loss.backward(); self.pi_optimizer.step()

        # 5) soft‐update target value net
        for p, p_targ in zip(self.v.parameters(), self.v_target.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)
