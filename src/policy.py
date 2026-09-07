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
                 log_std_init: float = -0.5, action_low: float = -1.0, action_high: float = 1.0,
                 log_std_min: float = -2.0, log_std_max: float = 2.0):
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
        # SÀN/TRẦN cho log_std -- đảm bảo exploration KHÔNG BAO GIỜ tắt hẳn dù
        # gradient có đẩy log_std xuống thấp tới đâu (chống "ngừng khám phá" khi
        # policy quá tự tin vào 1 hành vi), đồng thời cũng chặn std nổ quá lớn
        # theo chiều ngược lại (log_std_max). std thực tế luôn nằm trong
        # [exp(log_std_min), exp(log_std_max)] = [e^-2, e^2] ~ [0.135, 7.39] mặc định.
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # TransformedDistribution khong co .entropy() dang dong (vi TanhTransform
        # phi tuyen) -- danh dau de batched_ops biet fallback sang uoc luong
        # entropy qua -log_prob(action) (cach chuan trong cac implementation SAC).
        self.has_closed_form_entropy = False

    def forward(self, x):
        h = self.net(x)
        raw_mean = self.mean_head(h)
        # SỬA LẠI: dùng clamp thay vì tanh ở đây. Bound mean bằng tanh (thử
        # trước đó) gây tác dụng phụ nghiêm trọng: mean đã bị squash 1 lần rồi
        # còn bị TanhTransform bên ngoài squash THÊM 1 lần nữa (2 lớp tanh liên
        # tiếp) -- kết quả là action THỰC THI gần như không bao giờ vươn tới
        # gần +-1 được nữa (đo thực nghiệm: chỉ ~16% action vượt 0.9 dù mean đã
        # "cố hết sức" = 0.9, so với ~99% của thiết kế 1 lớp tanh chuẩn). Policy
        # mất khả năng hành động quyết đoán (mua/bán mạnh) -- chính là nguyên
        # nhân reward đứng yên phẳng lì dù chạy rất nhiều iteration.
        #
        # clamp() KHÔNG làm méo giá trị bên trong biên (identity map trong
        # [-clip_mean, clip_mean]), chỉ tự giới hạn khi vượt ngưỡng -- giữ
        # nguyên độ biểu đạt của raw_mean trong vùng bình thường, đồng thời vẫn
        # ngăn được runaway (không thể vượt quá clip_mean dù trọng số lớn cỡ
        # nào).
        #
        # BIÊN = 3.0 (không phải 6.0): tanh(3)=0.995 vẫn đủ "quyết đoán" (gần
        # như tối đa), NHƯNG quan trọng hơn -- ở gần biên 3.0, đạo hàm tanh còn
        # đủ lớn để dù std đã bị ép về sàn tối thiểu (log_std_min), nhiễu Gaussian
        # SAU tanh vẫn còn spread đáng kể (~400 lần lớn hơn so với biên 6.0). Đây
        # là lớp phòng vệ THỨ 2 chống collapse -- sàn std (log_std_min) chỉ đảm
        # bảo nhiễu TRƯỚC tanh không tắt hẳn, nhưng nếu mean nằm quá sâu trong
        # vùng bão hoà hình học của tanh, nhiễu đó vẫn bị "nén" gần về 0 SAU tanh
        # dù trước đó lớn thế nào -- hẹp biên mean lại là cách duy nhất giải
        # quyết đúng cơ chế collapse THỨ HAI này (khác hẳn nguyên nhân sàn std
        # đang giải quyết).
        return torch.clamp(raw_mean, -3.0, 3.0)

    def get_distribution(self, x):
        mean = self.forward(x)
        # Clamp log_std vào [log_std_min, log_std_max] TRƯỚC khi exp() -- đây là
        # sàn/trần cho std. Quan trọng: phải làm ở ĐÂY (nguồn dùng chung cho cả
        # get_distribution lẫn get_action bên dưới), không tính riêng std ở 2 nơi
        # khác nhau -- nếu không dễ bị lệch (sửa 1 chỗ quên chỗ kia).
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        base = Independent(Normal(mean, std), 1)
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])

    def get_action(self, state, deterministic: bool = False):
        dist = self.get_distribution(state)
        base = dist.base_dist  # Independent(Normal(mean, std), 1) -- std đã qua sàn/trần ở get_distribution()

        if deterministic:
            pre_tanh = base.mean
        else:
            pre_tanh = base.rsample()

        # Lưới an toàn số học: dù std đã có trần (log_std_max), nhiễu Gaussian
        # vẫn có thể đẩy sample ra khá xa ở phần đuôi phân phối. Clamp nhẹ trước
        # tanh (biên rộng, hiếm khi chạm tới trong vận hành bình thường) để tránh
        # atanh(±1)=inf ở float32 khi |pre_tanh|>=10.
        pre_tanh = torch.clamp(pre_tanh, -6.0, 6.0)
        action = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(action)
        entropy = -log_prob  # uoc luong entropy qua -log_prob (TransformedDistribution
        # khong co entropy() dang dong) -- day la cach lam chuan trong SAC.
        return action, log_prob, entropy