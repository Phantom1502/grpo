"""
Policy network: interface chung de Sampler/Trainer khong can biet obs la
vector/anh, action la discrete/continuous.

Moi policy implement get_distribution(x) -> torch.distributions object.
get_action() dung chung, khong can override rieng cho tung loai policy.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


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

    Dung Tanh-Squashed Gaussian (kieu SAC) thay vi Gaussian tho + clamp don
    gian. LY DO QUAN TRONG: neu chi sample tu Gaussian khong bi chan roi
    clamp(action, low, high) SAU cung (cach lam truoc day), khi mean bi day ra
    xa khoi [-1,1] (vi du do trong so mean_head bi cung co qua manh qua nhieu
    lan update lien tiep -- rat de xay ra khi review lai cac doan cu nhieu
    lan), MOI rollout deu bi clamp ve dung 1 gia tri bien -> action thuc thi
    ra moi truong gan nhu XAC DINH du entropy cua Gaussian goc (truoc clamp)
    van hien thi binh thuong khong doi. Day la "diem mu" khien group tro nen
    degenerate (khong con gradient signal) MA KHONG HE THAY canh bao qua log
    entropy, va policy bi "khoa cung" vinh vien o dung 1 hanh vi cho moi state
    moi gap phai -- day chinh la trieu chung "hoc rat cham/dung hinh" khi
    curriculum chuyen sang doan moi.

    Tanh-Squashed Gaussian giai quyet dung goc: log_prob/entropy duoc tinh
    TREN PHAN PHOI DA SQUASH (qua TransformedDistribution + TanhTransform,
    tu dong cong dao ham Jacobian dung chuan), nen khi sap bao hoa, entropy
    THAT giam manh (am rat sau) -- entropy_coef luc nay moi phat huy dung tac
    dung "phanh" lai truoc khi bi khoa cung, thay vi vo tri nhu Gaussian+clamp.
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
        # TransformedDistribution khong co .entropy() dang dong (vi TanhTransform
        # phi tuyen) -- danh dau de batched_ops biet fallback sang uoc luong
        # entropy qua -log_prob(action) (cach chuan trong cac implementation SAC).
        self.has_closed_form_entropy = False

    def forward(self, x):
        h = self.net(x)
        return self.mean_head(h)

    def get_distribution(self, x):
        mean = self.forward(x)
        std = torch.exp(self.log_std).expand_as(mean)
        base = Independent(Normal(mean, std), 1)
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])

    def get_action(self, state, deterministic: bool = False):
        mean = self.forward(state)
        std = torch.exp(self.log_std).expand_as(mean)
        base = Independent(Normal(mean, std), 1)
        dist = TransformedDistribution(base, [TanhTransform(cache_size=1)])

        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = base.rsample()

        # QUAN TRỌNG: clamp giá trị TRƯỚC khi qua tanh (không phải sau). Nếu để
        # mean_head cho ra giá trị quá lớn (vd bị đẩy mạnh sau nhiều lần update
        # liên tiếp -- dễ xảy ra khi review lại các đoạn cũ nhiều lần), tanh() ở
        # float32 sẽ làm tròn về ĐÚNG 1.0 khi |x| >= ~10, khiến atanh(1.0) = inf
        # lúc tính lại log_prob (TransformedDistribution cần atanh để lấy log-det
        # Jacobian) -- toàn bộ log_prob/entropy nổ tung thành NaN/số vô nghĩa.
        # Clamp trước tanh với biên an toàn (|x|<=6, tanh(6)~0.9999877, còn xa 1.0)
        # tránh lỗi số học này mà không cần đụng tới giá trị action sau squash.
        pre_tanh = torch.clamp(pre_tanh, -6.0, 6.0)
        action = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(action)
        entropy = -log_prob  # uoc luong entropy qua -log_prob (TransformedDistribution
        # khong co entropy() dang dong) -- day la cach lam chuan trong SAC.
        return action, log_prob, entropy