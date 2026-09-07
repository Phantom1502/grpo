import torch
from .task import Task, cartpole_success_fn
from .policy import MLPPolicy
from .sampler_vectorized import VectorizedGroupSampler
from .trainer import GRPOTrainer, GRPOConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task = Task(env_id="CartPole-v1", success_fn=cartpole_success_fn, max_steps=500)
n_obs = task.observation_space.shape[0]
policy = MLPPolicy(n_observations=n_obs, n_actions=task.n_actions)

# VectorizedGroupSampler: G=8 rollout chay song song (lockstep), 1 forward pass/step
# xu ly ca group -> nhanh hon han GroupSampler tuan tu khi group_size lon.
sampler = VectorizedGroupSampler(task, policy, device, group_size=8, use_async=False)

config = GRPOConfig(
    lr=1e-3,
    num_iterations=100,
    advantage_method="grpo",
    entropy_coef=0.01,
    log_every=10,
)

trainer = GRPOTrainer(policy, sampler, config, device)
trainer.train()
sampler.close()
