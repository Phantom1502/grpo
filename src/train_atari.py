import ale_py
import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from .task import Task, atari_score_success_fn
from .policy import CNNPolicy
from .sampler_vectorized import VectorizedGroupSampler
from .trainer import GRPOTrainer, GRPOConfig

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_ID = "ALE/Pong-v5"    # doi thanh ALE/Breakout-v5, ALE/SpaceInvaders-v5, ...
FRAME_STACK = 4


def make_atari_env():
    env = gym.make(ENV_ID, frameskip=1)
    env = AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True,
        scale_obs=False, terminal_on_life_loss=True,
    )
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    return env


task = Task(
    env_id=ENV_ID,
    success_fn=atari_score_success_fn(threshold=0.0),
    max_steps=2000,
    make_env_fn=make_atari_env,
)

policy = CNNPolicy(n_actions=task.n_actions, in_channels=FRAME_STACK)

# use_async=True: moi env chay trong 1 process rieng -> quan trong voi Atari vi
# env.step() (gia lap game) nang CPU; chay song song thuc su thay vi gia lap
# tuan tu trong 1 process nhu SyncVectorEnv. Doi lai co overhead khoi tao process
# va serialize du lieu qua IPC, nen chi loi neu group_size du lon / step du nang.
sampler = VectorizedGroupSampler(task, policy, device, group_size=4, use_async=True)

config = GRPOConfig(
    lr=2.5e-4,
    num_iterations=2000,
    advantage_method="grpo",
    entropy_coef=0.01,
    log_every=5,
)

trainer = GRPOTrainer(policy, sampler, config, device)
history = trainer.train()

torch.save(policy.state_dict(), f"policy_{ENV_ID.split('/')[-1]}_final.pt")
sampler.close()
