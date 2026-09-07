"""
Tach 2 pha:

  Pha 1 (rollout): chay duoi torch.no_grad(), chi luu state/action THO (tensor,
  khong gradient). Xem sampler.py / sampler_vectorized.py.

  Pha 2 (file nay): ghep TOAN BO (state, action) cua ca group, moi timestep,
  thanh 1 batch lon, forward CO gradient dung 1 lan de lay log_prob/entropy.

Tong quat hoa cho ca discrete (Categorical) va continuous (Independent Normal):
khong con gia dinh cung kieu phan phoi nao -- goi policy.get_distribution(states)
va dung dist.log_prob(actions_batch)/dist.entropy() chung cho moi loai policy.
"""

from typing import List, Tuple
import torch


def recompute_log_probs_and_entropy(
    policy, trajectories, device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    lengths = [len(t.actions) for t in trajectories]
    if sum(lengths) == 0:
        raise ValueError("Tat ca trajectory deu rong (0 step) - kiem tra lai rollout.")

    all_states = torch.stack(
        [s for traj in trajectories for s in traj.states]
    ).to(device)
    # actions da duoc luu dung dtype/shape tu luc sample (long cho discrete,
    # float vector cho continuous) -- chi can stack lai, khong ep kieu thu cong.
    all_actions = torch.stack(
        [a for traj in trajectories for a in traj.actions]
    ).to(device)

    dist = policy.get_distribution(all_states)
    log_probs = dist.log_prob(all_actions)
    entropies = dist.entropy()

    log_prob_sums = [lp.sum() for lp in torch.split(log_probs, lengths)]
    entropy_means = [e.mean() for e in torch.split(entropies, lengths)]
    return log_prob_sums, entropy_means


def batched_kl_penalty(policy, reference_policy, trajectories, device: torch.device) -> torch.Tensor:
    """KL penalty (k3 estimator, Schulman) giua policy hien tai va reference,
    tinh batched tren toan bo group. Dung chung cho discrete/continuous vi chi
    dua vao log_prob cua distribution, khong dua vao logits truc tiep."""
    all_states = torch.stack([s for traj in trajectories for s in traj.states]).to(device)
    all_actions = torch.stack([a for traj in trajectories for a in traj.actions]).to(device)

    with torch.no_grad():
        ref_dist = reference_policy.get_distribution(all_states)
        ref_logp = ref_dist.log_prob(all_actions)
    cur_dist = policy.get_distribution(all_states)
    cur_logp = cur_dist.log_prob(all_actions)

    log_ratio = ref_logp - cur_logp
    kl_terms = torch.exp(log_ratio) - 1 - log_ratio  # k3 estimator, >= 0
    return kl_terms.mean()
