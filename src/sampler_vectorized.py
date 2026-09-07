"""
VectorizedGroupSampler: thu thập G rollout SONG SONG (lockstep) bằng
gym.vector.SyncVectorEnv / AsyncVectorEnv.

Khác biệt cốt lõi so với GroupSampler (single-env, tuần tự):
  - GroupSampler: G rollout chạy lần lượt, mỗi step gọi policy.forward() batch=1
    -> G*T lần forward nhỏ lẻ.
  - Sampler này: G rollout chạy đồng thời trong 1 vòng lặp thời gian, mỗi step
    chỉ gọi policy.forward() MỘT LẦN với batch=G -> chỉ T lần forward, mỗi lần
    xử lý cả group. Đây là đòn bẩy lớn nhất để tận dụng GPU khi group_size lớn.

Lưu ý về autoreset của gymnasium vector env (mode mặc định NEXT_STEP):
  - Khi env i kết thúc (terminated/truncated) ở step t, obs[i]/reward[i] tại
    đúng step t vẫn là dữ liệu THẬT (hợp lệ, phải ghi nhận).
  - Ở step t+1, env i đã tự động reset ngầm -> reward=0, obs là obs mới sau reset.
    Dữ liệu này KHÔNG thuộc trajectory cũ, phải bỏ qua.
  - Sampler dùng mảng `active` để đánh dấu: sau khi ghi nhận step kết thúc, set
    active[i]=False để các step tiếp theo của env i bị bỏ qua (dù vector env vẫn
    tiếp tục chạy nó cho tới khi cả group xong hoặc hết max_steps).

AsyncVectorEnv (multi-process) hữu ích khi env.step() nặng CPU (Atari emulation);
SyncVectorEnv (single-process) đủ dùng cho env nhẹ (CartPole, LunarLander) và
tránh overhead spawn process.
"""

from typing import Callable, List, Optional
import numpy as np
import torch
import gymnasium as gym

from .task import Task, Trajectory
from .advantage import is_degenerate_group


class VectorizedGroupSampler:
    def __init__(
        self,
        task: Task,
        policy,
        device: torch.device,
        group_size: int = 8,
        max_resample_attempts: int = 3,
        use_async: bool = False,
    ):
        self.task = task
        self.policy = policy
        self.device = device
        self.group_size = group_size
        self.max_resample_attempts = max_resample_attempts

        vec_cls = gym.vector.AsyncVectorEnv if use_async else gym.vector.SyncVectorEnv
        self.vec_env = vec_cls([task._make_env_fn for _ in range(group_size)])

    @torch.no_grad()
    def _collect_once(self) -> List[Trajectory]:
        G = self.group_size
        obs, _ = self.vec_env.reset()
        states = torch.tensor(np.asarray(obs), dtype=torch.float32, device=self.device)

        traj_states: List[list] = [[] for _ in range(G)]
        traj_actions: List[list] = [[] for _ in range(G)]
        traj_rewards: List[list] = [[] for _ in range(G)]
        active = np.ones(G, dtype=bool)
        term_flags = np.zeros(G, dtype=bool)
        trunc_flags = np.zeros(G, dtype=bool)

        for _ in range(self.task.max_steps):
            if not active.any():
                break

            # 1 forward pass DUY NHẤT xử lý toàn bộ group -> đây là điểm tối ưu chính.
            # get_distribution() tổng quát cho cả discrete (Categorical) lẫn
            # continuous (Independent Normal) -- Sampler không cần biết loại nào.
            dist = self.policy.get_distribution(states)
            actions = dist.sample()
            actions_np = actions.detach().cpu().numpy()

            next_obs, rewards, terminated, truncated, _infos = self.vec_env.step(actions_np)

            for i in range(G):
                if active[i]:
                    traj_states[i].append(states[i].cpu())
                    traj_actions[i].append(actions[i].detach().cpu())
                    traj_rewards[i].append(float(rewards[i]))
                    if terminated[i] or truncated[i]:
                        active[i] = False
                        term_flags[i] = bool(terminated[i])
                        trunc_flags[i] = bool(truncated[i])

            states = torch.tensor(np.asarray(next_obs), dtype=torch.float32, device=self.device)

        trajectories = []
        for i in range(G):
            duration = len(traj_rewards[i])
            total_reward = float(sum(traj_rewards[i]))
            success = self.task.success_fn(total_reward, duration, term_flags[i], trunc_flags[i], {})
            trajectories.append(
                Trajectory(
                    rewards=traj_rewards[i],
                    states=traj_states[i],
                    actions=traj_actions[i],
                    duration=duration,
                    success=success,
                )
            )
        return trajectories

    def collect_group(self) -> List[Trajectory]:
        for _ in range(self.max_resample_attempts):
            trajectories = self._collect_once()
            rewards = torch.tensor([t.total_reward for t in trajectories], dtype=torch.float32)
            if not is_degenerate_group(rewards):
                return trajectories
        return trajectories

    def close(self):
        self.vec_env.close()
