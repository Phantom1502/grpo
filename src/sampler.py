"""
Sampler (single-env): thu thap G rollout tuan tu, moi rollout dung 1 env rieng.

Rollout chay duoi torch.no_grad(), chi luu state/action THO (tensor, khong
gradient). log_prob/entropy duoc tinh lai sau, batched tren ca group, boi
batched_ops.recompute_log_probs_and_entropy() (xem file do de biet ly do).

Action duoc luu nguyen dang tensor (khong ep .item()) de dung chung cho ca
discrete (Categorical -> tensor int scalar) va continuous (Independent Normal
-> tensor vector) -- Sampler khong can biet action space la loai gi.
"""

from typing import Callable, List, Optional
import torch

from .task import Task, Trajectory
from .advantage import is_degenerate_group


def default_obs_to_tensor(obs, device):
    import numpy as np
    arr = np.asarray(obs, dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def action_to_env(action: torch.Tensor):
    """Chuyen action tensor (output cua policy) thanh dinh dang env.step() can:
    int cho Discrete (CartPole, Atari), numpy array cho Box (finrl, MuJoCo...)."""
    if action.dim() == 0:
        return int(action.item())
    return action.detach().cpu().numpy()


class GroupSampler:
    def __init__(
        self,
        task: Task,
        policy,
        device: torch.device,
        group_size: int = 8,
        max_resample_attempts: int = 3,
        obs_to_tensor: Optional[Callable] = None,
    ):
        self.task = task
        self.policy = policy
        self.device = device
        self.group_size = group_size
        self.max_resample_attempts = max_resample_attempts
        self.obs_to_tensor = obs_to_tensor or default_obs_to_tensor

    @torch.no_grad()
    def _run_single_rollout(self) -> Trajectory:
        obs, _ = self.task.reset()
        state = self.obs_to_tensor(obs, self.device)

        states, actions, rewards = [], [], []
        terminated = truncated = False
        last_info = {}

        for _ in range(self.task.max_steps):
            action, _log_prob, _entropy = self.policy.get_action(state.unsqueeze(0))
            action = action.squeeze(0)  # bo batch dim: scalar (discrete) hoac (action_dim,) (continuous)

            next_obs, reward, terminated, truncated, info = self.task.step(action_to_env(action))

            states.append(state.cpu())
            actions.append(action.detach().cpu())
            rewards.append(reward)
            last_info = info

            if terminated or truncated:
                break
            state = self.obs_to_tensor(next_obs, self.device)

        duration = len(rewards)
        total_reward = float(sum(rewards))
        success = self.task.success_fn(total_reward, duration, terminated, truncated, last_info)

        return Trajectory(
            rewards=rewards,
            states=states,
            actions=actions,
            duration=duration,
            success=success,
            info=last_info,
        )

    def collect_group(self) -> List[Trajectory]:
        for _ in range(self.max_resample_attempts):
            trajectories = [self._run_single_rollout() for _ in range(self.group_size)]
            rewards = torch.tensor(
                [t.total_reward for t in trajectories], dtype=torch.float32
            )
            if not is_degenerate_group(rewards):
                return trajectories
        return trajectories

    def close(self):
        self.task.close()
