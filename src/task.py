"""
Task abstraction: bọc một gymnasium env + định nghĩa "thế nào là hoàn thành task".

Ý tưởng: GRPO quan tâm outcome cuối, nhưng "outcome" là khái niệm phụ thuộc bài toán
(CartPole: sống sót lâu; LunarLander: đáp thành công; Atari: điểm số cao; agent
làm task: hoàn thành đúng mục tiêu). Task class tách phần "định nghĩa thành công/
thất bại" ra khỏi phần thu thập rollout, để Sampler/Trainer không cần biết chi tiết
domain.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import numpy as np
import gymnasium as gym


@dataclass
class Trajectory:
    """
    Toàn bộ dữ liệu của một rollout, đủ để tính reward/advantage và update policy.

    LƯU Ý: states/actions được thu thập KHÔNG gradient (torch.no_grad() trong Sampler).
    log_probs/entropies KHÔNG được điền lúc rollout nữa -- chúng được tính lại sau,
    batched trên toàn bộ group, bởi batched_ops.recompute_log_probs_and_entropy().
    Tách 2 pha này ra là tối ưu quan trọng nhất để tận dụng GPU (xem batched_ops.py).
    """
    rewards: list              # reward thô từng step (float)
    states: list                # state tensor thô (cpu, không gradient) từng step
    actions: list                # action (int) đã chọn từng step
    log_probs: list             # log_prob (float) của action đã chọn từng step
    duration: int
    success: bool                # task có hoàn thành theo tiêu chí riêng của domain hay không
    info: dict = field(default_factory=dict)  # metadata tự do (vd: điểm số Atari, lý do fail...)

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))


class Task:
    """
    Bọc một gymnasium env và cung cấp:
      - reset() / step() chuẩn gymnasium
      - is_success(trajectory) -> bool: tiêu chí hoàn thành task, đặc thù theo domain

    success_fn nhận vào (total_reward, duration, terminated, truncated, last_info)
    và trả về True/False. Đây là điểm cắm quan trọng nhất khi bạn chuyển sang bài
    toán "task completion" khác (robot, agent, workflow...) — bạn chỉ cần viết lại
    success_fn, không cần đụng vào Sampler/Trainer.
    """

    def __init__(
        self,
        env_id: str,
        success_fn: Optional[Callable[[float, int, bool, bool, dict], bool]] = None,
        max_steps: int = 1000,
        make_env_fn: Optional[Callable[[], gym.Env]] = None,
    ):
        self.env_id = env_id
        self.max_steps = max_steps
        self._make_env_fn = make_env_fn or (lambda: gym.make(env_id))
        self.env = self._make_env_fn()

        # Mặc định: "thành công" = không bị truncate do fail sớm (terminated=False khi hết reward âm)
        # Với hầu hết task, bạn NÊN override cái này bằng logic đúng của domain.
        self.success_fn = success_fn or self._default_success_fn

    @staticmethod
    def _default_success_fn(total_reward, duration, terminated, truncated, last_info) -> bool:
        # Heuristic ngây thơ: coi là thành công nếu không kết thúc do "terminated"
        # (terminated thường = fail/chết trong nhiều env), truncated = hết time an toàn.
        return truncated and not terminated

    @property
    def n_actions(self) -> int:
        """Chỉ dùng cho discrete action space (CartPole, Atari...)."""
        return self.env.action_space.n

    @property
    def is_discrete(self) -> bool:
        return hasattr(self.env.action_space, "n")

    @property
    def action_dim(self) -> int:
        """Số chiều action -- dùng cho continuous action space (Box), ví dụ finrl."""
        space = self.env.action_space
        return space.n if hasattr(space, "n") else space.shape[0]

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Một vài success_fn dựng sẵn cho các benchmark phổ biến, để dùng ngay hoặc
# tham khảo cách viết success_fn cho task riêng của bạn.
# ---------------------------------------------------------------------------

def cartpole_success_fn(total_reward, duration, terminated, truncated, last_info) -> bool:
    """CartPole-v1: coi là 'thành công' nếu trụ được hết max_steps (không bị terminate sớm)."""
    return not terminated


def lunarlander_success_fn(total_reward, duration, terminated, truncated, last_info) -> bool:
    """
    LunarLander-v3: env trả reward +100 khi đáp an toàn, -100 khi rơi/crash.
    Coi thành công nếu tổng reward vượt ngưỡng (chuẩn phổ biến: >= 200 là "solved").
    """
    return total_reward >= 200


def atari_score_success_fn(threshold: float):
    """Factory: trả về success_fn coi là thành công nếu tổng điểm >= threshold."""
    def fn(total_reward, duration, terminated, truncated, last_info):
        return total_reward >= threshold
    return fn


def stock_trading_success_fn(min_return_pct: float = 0.0):
    """
    Factory cho finrl.StockTradingEnv: 'thành công' nếu lợi nhuận cuối episode
    (tổng reward, vì reward mỗi step của env này chính là delta tài sản)
    vượt ngưỡng min_return_pct * initial_amount.

    Lưu ý: total_reward ở đây là tổng các reward THÔ trả về từ env (đã nhân
    reward_scaling, ví dụ 1e-4) -- không phải % lợi nhuận thực. Muốn so sánh
    theo % chính xác, nên chia total_reward cho (initial_amount * reward_scaling).
    Hàm dưới coi threshold = 0 nghĩa là "không lỗ" (baseline hợp lý để bắt đầu).
    """
    def fn(total_reward, duration, terminated, truncated, last_info):
        return total_reward > min_return_pct
    return fn
