"""
Train GRPO tren finrl.StockTradingEnv dung local CSV chi co 5 cot OHLCV.

Khac biet quan trong so voi CartPole/LunarLander/Atari: action space cua
StockTradingEnv la Box(-1,1,(stock_dim,)) -- LIEN TUC, khong phai Discrete.
Vi vay dung GaussianMLPPolicy thay vi MLPPolicy/CNNPolicy.

LUU Y VE PATH: file nay import truc tiep `env_stocktrading.py` (source code cua
finrl.meta.env_stock_trading.env_stocktrading) thay vi `pip install finrl` day
du, de tranh keo theo dependency nang (yfinance, alpaca-trade-api, elegantrl,
ray...). Neu ban da co san package `finrl` (vi du dang chay trong tutorial
notebook goc cua finrl), chi can doi dong import ben duoi thanh:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
"""

import sys
import numpy as np
import torch
import gymnasium as gym

# --- doi duong dan nay thanh noi ban dat file env_stocktrading.py, hoac xoa
# dong nay neu da `pip install finrl` day du ---
sys.path.insert(0, "/path/to/folder/containing/env_stocktrading")
from env_stocktrading import StockTradingEnv  # hoặc: from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from .task import Task, stock_trading_success_fn
from .policy import GaussianMLPPolicy
from .sampler_vectorized import VectorizedGroupSampler
from .trainer import GRPOTrainer, GRPOConfig
from .finrl_data import prepare_finrl_dataframe, train_trade_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "your_ohlcv.csv"   # CSV local: cot Open, High, Low, Close, Volume (+ index ngay)
TIC = "STOCK"                  # ten ma co phieu tuy chon (chi co 1 ma trong file nay)

df, indicator_cols = prepare_finrl_dataframe(CSV_PATH, tic=TIC)
train_df, trade_df = train_trade_split(df, split_ratio=0.8)

stock_dim = 1  # file nay xu ly 1 ma co phieu; nhieu ma thi stock_dim = so ma
state_space = 1 + 2 * stock_dim + len(indicator_cols) * stock_dim

env_kwargs = dict(
    hmax=100,
    initial_amount=100_000,
    num_stock_shares=[0] * stock_dim,
    buy_cost_pct=[0.001] * stock_dim,
    sell_cost_pct=[0.001] * stock_dim,
    state_space=state_space,
    stock_dim=stock_dim,
    tech_indicator_list=indicator_cols,
    action_space=stock_dim,
    reward_scaling=1e-4,
)


def make_stock_env():
    env = StockTradingEnv(df=train_df, **env_kwargs)
    # QUAN TRỌNG: state của StockTradingEnv gồm cash (~hàng trăm nghìn), giá cổ
    # phiếu, indicator... không cùng thang đo. Đưa thẳng vào MLP mới khởi tạo sẽ
    # cho ra action "nổ" (vd mean ~ -3000, ngoài khoảng [-1,1] rất xa) vì trọng số
    # random nhân với input hàng trăm nghìn. NormalizeObservation (chuẩn hoá
    # chạy, kiểu running mean/std) giải quyết vấn đề này mà không cần đổi policy.
    env = gym.wrappers.NormalizeObservation(env)
    return env


task = Task(
    env_id="finrl-stock-local",
    success_fn=stock_trading_success_fn(min_return_pct=0.0),  # "thanh cong" = khong lo
    max_steps=len(train_df) - 1,   # 1 episode = di het du lieu train
    make_env_fn=make_stock_env,
)

policy = GaussianMLPPolicy(
    n_observations=state_space,
    action_dim=stock_dim,
    log_std_init=-0.5,
)

# SyncVectorEnv: StockTradingEnv nhe (khong emulate game nang nhu Atari) nen
# khong can AsyncVectorEnv (multi-process) -- overhead spawn process se lon
# hon loi ich vi moi step chi la vai phep tinh pandas/numpy don gian.
sampler = VectorizedGroupSampler(task, policy, device, group_size=8, use_async=False)

config = GRPOConfig(
    lr=3e-4,
    num_iterations=200,
    advantage_method="grpo",
    entropy_coef=0.001,   # continuous action nen entropy_coef nho hon discrete
    log_every=10,
)

trainer = GRPOTrainer(policy, sampler, config, device)
trainer.train()
sampler.close()

torch.save(policy.state_dict(), f"policy_finrl_{TIC}_final.pt")
