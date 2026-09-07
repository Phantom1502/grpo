"""
GRPOTrainer: vong lap huan luyen chinh, ghep Sampler + AdvantageEstimator + Policy update.

Thay doi quan trong so voi ban dau: Trainer khong tu tao Sampler nua ma NHAN
sampler tu ben ngoai (dependency injection). Ly do: ban co the doi giua
GroupSampler (don gian, tuan tu) va VectorizedGroupSampler (nhanh hon nhieu,
dac biet quan trong khi group_size lon / chay tren GPU) ma khong dung gi den
code Trainer.

Trainer dung batched_ops.recompute_log_probs_and_entropy() de tinh log_prob/entropy
CO GRADIENT trong 1 lan forward duy nhat cho ca group, thay vi build graph tang
dan trong luc rollout (xem batched_ops.py de biet ly do).
"""

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.optim as optim

from .advantage import compute_advantage, is_degenerate_group
from .batched_ops import recompute_log_probs_and_entropy, batched_kl_penalty


@dataclass
class GRPOConfig:
    lr: float = 1e-3
    num_iterations: int = 200
    advantage_method: str = "grpo"       # "grpo" | "dr_grpo" | "rloo"
    entropy_coef: float = 0.0
    kl_coef: float = 0.0                 # 0 = tat KL penalty
    max_grad_norm: float = 1.0
    log_every: int = 10


class GRPOTrainer:
    def __init__(self, policy, sampler, config: GRPOConfig, device: torch.device):
        self.policy = policy.to(device)
        self.sampler = sampler
        self.config = config
        self.device = device

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=config.lr)

        self.reference_policy: Optional[torch.nn.Module] = None
        if config.kl_coef > 0:
            self.reference_policy = copy.deepcopy(self.policy)
            for p in self.reference_policy.parameters():
                p.requires_grad_(False)

    def train_step(self):
        trajectories = self.sampler.collect_group()
        rewards = torch.tensor(
            [t.total_reward for t in trajectories], dtype=torch.float32, device=self.device
        )
        degenerate = is_degenerate_group(rewards)
        advantages = compute_advantage(rewards, method=self.config.advantage_method)

        # 1 lan forward batched duy nhat cho ca group -> co gradient.
        log_prob_sums, entropy_means = recompute_log_probs_and_entropy(
            self.policy, trajectories, self.device
        )

        policy_loss_terms = [
            -lp * adv for lp, adv in zip(log_prob_sums, advantages)
        ]
        policy_loss = torch.stack(policy_loss_terms).mean()
        entropy_bonus = torch.stack(entropy_means).mean()

        loss = policy_loss - self.config.entropy_coef * entropy_bonus

        if self.config.kl_coef > 0:
            kl = batched_kl_penalty(self.policy, self.reference_policy, trajectories, self.device)
            loss = loss + self.config.kl_coef * kl
        else:
            kl = torch.tensor(0.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        success_rate = sum(t.success for t in trajectories) / len(trajectories)
        durations = [t.duration for t in trajectories]

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy_bonus.item(),
            "kl": kl.item() if isinstance(kl, torch.Tensor) else kl,
            "avg_reward": rewards.mean().item(),
            "avg_duration": sum(durations) / len(durations),
            "success_rate": success_rate,
            "degenerate_group": degenerate,
        }

    def sync_reference_policy(self):
        if self.reference_policy is not None:
            self.reference_policy.load_state_dict(self.policy.state_dict())

    def train(self):
        history = []
        for it in range(self.config.num_iterations):
            stats = self.train_step()
            history.append(stats)
            if (it + 1) % self.config.log_every == 0:
                flag = " [degenerate group]" if stats["degenerate_group"] else ""
                print(
                    f"Iter {it+1}/{self.config.num_iterations} | "
                    f"reward={stats['avg_reward']:.2f} | "
                    f"success_rate={stats['success_rate']:.2f} | "
                    f"duration={stats['avg_duration']:.1f} | "
                    f"loss={stats['loss']:.4f} | entropy={stats['entropy']:.3f}{flag}"
                )
        return history
