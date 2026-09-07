import torch
from .task import Task, lunarlander_success_fn
from .policy import MLPPolicy
from .sampler_vectorized import VectorizedGroupSampler
from .trainer import GRPOTrainer, GRPOConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task = Task(env_id="LunarLander-v3", success_fn=lunarlander_success_fn, max_steps=1000)
n_obs = task.observation_space.shape[0]
policy = MLPPolicy(n_observations=n_obs, n_actions=task.n_actions)

sampler = VectorizedGroupSampler(task, policy, device, group_size=8, use_async=False)

config = GRPOConfig(
    lr=1e-3,
    num_iterations=60,
    advantage_method="rloo",
    entropy_coef=0.01,
    log_every=5,
)

trainer = GRPOTrainer(policy, sampler, config, device)
trainer.train()
sampler.close()
