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
    kl_coef: float = 0.0                 # 0 = tat KL penalty (tat ca co che "phanh" ben duoi deu dua vao kl_coef > 0)
    reference_ema_decay: Optional[float] = None
    # "Doi tuong de so sanh, phanh lai update qua manh" ma ban mo ta -- reference
    # policy duoc cap nhat MOI step theo cong thuc:
    #     ref_param = decay*ref_param + (1-decay)*policy_param
    # decay cang gan 1.0 (vd 0.99, 0.999) -> reference "tut hau" cang xa, phanh
    # cang MANH (policy kho di qua xa so voi 1 ban than no lau ve truoc). decay
    # cang gan 0 -> reference bam sat policy hien tai, phanh gan nhu khong co
    # tac dung. None = tat EMA, dung sync_reference_policy() thu cong (hard sync
    # dinh ky) neu muon, hoac khong sync gi (reference dong bang vinh vien tai
    # checkpoint khoi tao -- phanh RAT MANH, gan nhu khong cho policy doi hanh vi
    # nhieu so voi luc bat dau, chi phu hop khi fine-tune tu 1 policy da tot san).
    max_grad_norm: float = 1.0
    ppo_epochs: int = 1                  # so lan gradient update TREN CUNG 1 group da thu thap.
    # =1 (mac dinh): dung het 1 group cho DUNG 1 buoc update -> log_prob luc tinh
    # loss luon khop voi policy vua tao rollout (ratio luon =1), KHONG can clip.
    # >1: tai su dung group cho nhieu epoch (tiet kiem compute, khong can rollout
    # moi moi lan) -- nhung sau vai epoch, policy da troi xa so voi luc tao du
    # lieu -> can PPO-clip (clip_ratio) de tranh 1 epoch nao do day update qua xa
    # dua tren du lieu da "cu" (off-policy nhe trong noi bo cac epoch nay).
    clip_ratio: float = 0.2              # PPO clip epsilon, chi co tac dung khi ppo_epochs > 1
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

        if self.config.ppo_epochs <= 1:
            stats = self._single_epoch_update(trajectories, advantages)
        else:
            stats = self._ppo_clipped_update(trajectories, advantages)

        self._update_reference_ema()  # neu bat EMA: "phanh" tut hau 1 chut sau MOI step

        success_rate = sum(t.success for t in trajectories) / len(trajectories)
        durations = [t.duration for t in trajectories]
        stats.update({
            "avg_reward": rewards.mean().item(),
            "avg_duration": sum(durations) / len(durations),
            "success_rate": success_rate,
            "degenerate_group": degenerate,
        })
        return stats

    def _single_epoch_update(self, trajectories, advantages):
        """Hành vi GỐC (ppo_epochs=1): 1 lần forward batched có gradient duy
        nhất cho cả group, y hệt như trước khi thêm PPO-clip -- không có ratio,
        không có clip, vì chưa từng có epoch nào khác để "lệch" khỏi policy vừa
        tạo ra rollout."""
        log_prob_sums, entropy_means = recompute_log_probs_and_entropy(
            self.policy, trajectories, self.device
        )
        policy_loss_terms = [-lp * adv for lp, adv in zip(log_prob_sums, advantages)]
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

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy_bonus.item(),
            "kl": kl.item() if isinstance(kl, torch.Tensor) else kl,
            "clip_frac": 0.0,
        }

    def _ppo_clipped_update(self, trajectories, advantages):
        """ppo_epochs>1: tái sử dụng CÙNG 1 group cho nhiều lần gradient update.
        old_log_prob chụp lại 1 LẦN DUY NHẤT trước epoch đầu tiên (dưới no_grad,
        đóng vai trò "π cũ" cố định) -- mỗi epoch sau đó so sánh log_prob MỚI
        (policy đang cập nhật dần) với mốc cố định này qua ratio = exp(new-old),
        rồi clip ratio về [1-eps, 1+eps] trước khi nhân với advantage, chặn 1
        epoch nào đó đẩy update quá xa dựa trên dữ liệu đã "cũ" dần qua từng epoch."""
        with torch.no_grad():
            old_log_prob_sums, _ = recompute_log_probs_and_entropy(
                self.policy, trajectories, self.device
            )
            old_log_prob_sums = [lp.detach() for lp in old_log_prob_sums]

        last_stats = {}
        for _ in range(self.config.ppo_epochs):
            log_prob_sums, entropy_means = recompute_log_probs_and_entropy(
                self.policy, trajectories, self.device
            )

            ratios = [torch.exp(lp - old_lp) for lp, old_lp in zip(log_prob_sums, old_log_prob_sums)]
            surrogate_unclipped = [r * adv for r, adv in zip(ratios, advantages)]
            surrogate_clipped = [
                torch.clamp(r, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv
                for r, adv in zip(ratios, advantages)
            ]
            # min(unclipped, clipped): PPO lay can duoi (pessimistic bound) --
            # neu clip lam surrogate NHO hon (advantage duong, ratio vuot 1+eps)
            # hoac neu clip lam no LON hon so voi khong-clip theo huong co loi
            # gia (advantage am), min() luon chon phuong an "than trong" hon.
            policy_loss_terms = [-torch.min(u, c) for u, c in zip(surrogate_unclipped, surrogate_clipped)]
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

            with torch.no_grad():
                clip_frac = torch.stack([
                    ((r < 1 - self.config.clip_ratio) | (r > 1 + self.config.clip_ratio)).float()
                    for r in ratios
                ]).mean().item()

            last_stats = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "entropy": entropy_bonus.item(),
                "kl": kl.item() if isinstance(kl, torch.Tensor) else kl,
                "clip_frac": clip_frac,  # % trajectory bị clip -- theo dõi để biết clip có "ăn" không
            }
        return last_stats

    def sync_reference_policy(self):
        """Hard sync thu cong: reference = ban sao CHINH XAC cua policy hien tai
        tai thoi diem goi. Goi dinh ky (vd moi N iteration) neu muon kieu 'phanh
        theo bac thang' thay vi EMA lien tuc."""
        if self.reference_policy is not None:
            self.reference_policy.load_state_dict(self.policy.state_dict())

    def _update_reference_ema(self):
        """Cap nhat reference theo EMA -- goi tu dong sau MOI train_step() neu
        reference_ema_decay duoc dat. Day la co che 'phanh deu' ma khong co buoc
        nhay dot ngot nhu hard sync dinh ky."""
        if self.reference_policy is None or self.config.reference_ema_decay is None:
            return
        decay = self.config.reference_ema_decay
        with torch.no_grad():
            for ref_p, cur_p in zip(self.reference_policy.parameters(), self.policy.parameters()):
                ref_p.mul_(decay).add_(cur_p, alpha=1 - decay)

    def train(self):
        history = []
        for it in range(self.config.num_iterations):
            stats = self.train_step()
            history.append(stats)
            if (it + 1) % self.config.log_every == 0:
                flag = " [degenerate group]" if stats["degenerate_group"] else ""
                clip_info = f" | clip_frac={stats['clip_frac']:.2f}" if self.config.ppo_epochs > 1 else ""
                print(
                    f"Iter {it+1}/{self.config.num_iterations} | "
                    f"reward={stats['avg_reward']:.2f} | "
                    f"success_rate={stats['success_rate']:.2f} | "
                    f"duration={stats['avg_duration']:.1f} | "
                    f"loss={stats['loss']:.4f} | entropy={stats['entropy']:.3f}{clip_info}{flag}"
                )
        return history