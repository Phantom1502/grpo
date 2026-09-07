"""
Advantage estimator: biến G reward thô (1 số / trajectory) thành advantage dùng
để nhân với log_prob trong loss. Đây là module dễ thay đổi nhất khi bạn muốn thử
nghiệm các biến thể GRPO khác nhau, nên tách hẳn thành các hàm độc lập.

Tất cả nhận vào 1 tensor 1-D `rewards` (shape [G]) — mỗi phần tử là total_reward
của 1 trajectory trong group — và trả về tensor advantage cùng shape.
"""

import torch


def is_degenerate_group(rewards: torch.Tensor, eps: float = 1e-8) -> bool:
    """
    Group được coi là 'degenerate' nếu mọi rollout cho cùng 1 kết quả
    (std ~ 0) — nghĩa là advantage sẽ ~0 cho tất cả, không có gradient signal.
    Sampler nên phát hiện case này và resample thay vì lãng phí batch update.
    """
    return bool(rewards.std(unbiased=False).item() < eps)


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO gốc: chuẩn hoá theo mean và std trong group."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False) + eps
    return (rewards - mean) / std


def dr_grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    Dr.GRPO: chỉ trừ mean, KHÔNG chia std.
    Lý do: chia std làm advantage phụ thuộc vào độ "dễ đoán" của group (nếu reward
    ít biến thiên, advantage bị khuếch đại quá mức) — đây là 1 nguồn bias được
    paper Dr.GRPO chỉ ra. Dùng khi bạn muốn loại bias độ khó của từng group.
    """
    return rewards - rewards.mean()


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    Leave-One-Out: advantage của rollout i = reward_i - mean(reward của G-1 rollout còn lại).
    Giảm bias so với dùng group mean (vốn bao gồm cả chính rollout i), đặc biệt
    hữu ích khi G nhỏ (group mean bị chính rollout i kéo lệch đáng kể).
    """
    G = rewards.shape[0]
    total = rewards.sum()
    # mean của (G-1) phần tử còn lại = (total - r_i) / (G-1)
    leave_one_out_mean = (total - rewards) / (G - 1)
    return rewards - leave_one_out_mean


ADVANTAGE_FNS = {
    "grpo": grpo_advantage,
    "dr_grpo": dr_grpo_advantage,
    "rloo": rloo_advantage,
}


def compute_advantage(rewards: torch.Tensor, method: str = "grpo") -> torch.Tensor:
    if method not in ADVANTAGE_FNS:
        raise ValueError(f"Unknown advantage method '{method}'. Options: {list(ADVANTAGE_FNS)}")
    return ADVANTAGE_FNS[method](rewards)
