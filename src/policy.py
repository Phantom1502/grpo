"""
Policy network: interface chung de Sampler/Trainer khong can biet obs la
vector/anh, action la discrete/continuous.

Moi policy implement get_distribution(x) -> torch.distributions object.
get_action() dung chung, khong can override rieng cho tung loai policy.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal


class BasePolicy(nn.Module):
    def forward(self, x):
        """Tra ve raw network output: logits (discrete) hoac mean action (continuous)."""
        raise NotImplementedError

    def get_distribution(self, x):
        """Tra ve 1 torch.distributions object tu raw output cua forward()."""
        raise NotImplementedError

    def get_action(self, state, deterministic: bool = False):
        dist = self.get_distribution(state)
        if deterministic:
            # Categorical khong co .mean -> dung argmax cua probs lam fallback.
            action = dist.mean if hasattr(dist, "mean") else dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class MLPPolicy(BasePolicy):
    """Discrete action, observation dang vector (CartPole, LunarLander, robot state...)."""

    def __init__(self, n_observations: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    def get_distribution(self, x):
        logits = self.forward(x)
        return Categorical(logits=logits)


class CNNPolicy(BasePolicy):
    """Discrete action, observation dang anh (Atari: stack 4 frame 84x84 grayscale)."""

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)

    def get_distribution(self, x):
        logits = self.forward(x)
        return Categorical(logits=logits)


class GaussianMLPPolicy(BasePolicy):
    """
    Continuous action, observation dang vector -- dung cho Box action space
    (vd: finrl StockTradingEnv, MuJoCo, robot dieu khien lien tuc...).

    Dung Gaussian doc lap theo tung chieu action (state-independent log_std,
    kieu tiep can pho bien trong PPO continuous-control). action duoc clamp ve
    dung [action_low, action_high] SAU khi sample.

    LUU Y (caveat quan trong): vi action bi clamp sau khi sample tu Gaussian
    khong bi chan, log_prob dung de tinh gradient la log_prob cua gia tri DA
    CLAMP duoi phan phoi Gaussian goc -- day la xap xi don gian hoa, khong hoan
    toan chinh xac ve mat ly thuyet nhu tanh-squashed Gaussian (kieu SAC). Neu
    can chinh xac hon (dac biet khi policy hay sample ra gia tri vuot bien),
    nen thay bang TransformedDistribution voi TanhTransform.
    """

    def __init__(self, n_observations: int, action_dim: int, hidden: int = 128,
                 log_std_init: float = -0.5, action_low: float = -1.0, action_high: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, x):
        h = self.net(x)
        return self.mean_head(h)

    def get_distribution(self, x):
        mean = self.forward(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def get_action(self, state, deterministic: bool = False):
        dist = self.get_distribution(state)
        action_sample = dist.mean if deterministic else dist.sample()
        
        # Tính log_prob & entropy trên sample THỰC TẾ trước khi bị clamp
        log_prob = dist.log_prob(action_sample)
        entropy = dist.entropy()
        
        # Giới hạn action để gửi vào FinRL Environment
        clamped_action = torch.clamp(action_sample, self.action_low, self.action_high)
        
        return clamped_action, log_prob, entropy
