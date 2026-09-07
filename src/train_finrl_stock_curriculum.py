"""
Train GRPO trên finrl.StockTradingEnv theo CURRICULUM THEO THỜI GIAN: học tuần
tự từng đoạn dữ liệu (segment), thỉnh thoảng ôn lại đoạn cũ để tránh quên.

Khác với train_finrl_stock.py (dùng 1 window cố định suốt quá trình huấn
luyện), file này:
  - Chia train_df thành nhiều đoạn liên tiếp theo thời gian (SegmentCurriculum).
  - Mỗi iteration: hỏi curriculum "dùng đoạn nào", dựng Task+Sampler MỚI cho
    đúng đoạn đó (mỗi group luôn dùng 1 đoạn duy nhất -- xem curriculum.py để
    biết lý do), chạy 1 train_step(), rồi báo kết quả lại cho curriculum để nó
    quyết định có chuyển sang đoạn tiếp theo hay chưa.
  - Dùng FixedScaleObservation (scale cố định, tính 1 lần từ toàn bộ train_df)
    thay vì NormalizeObservation adaptive, vì env bị tạo lại liên tục mỗi khi
    đổi đoạn/ôn lại (xem finrl_data.compute_state_scale để biết lý do).

CHẾ ĐỘ THÍCH ỨNG (advance_on_success_rate) thay vì số iteration cố định:
  Chế độ cũ (iterations_per_segment cố định) có nhược điểm: nếu đoạn đã "pass"
  (policy học tốt) sớm hơn dự kiến, vẫn bị ép train thêm cho đủ số iteration ->
  lãng phí compute, và tệ hơn là DỄ OVERFIT vào đúng đoạn đó (policy học thuộc
  lòng chuỗi giá cụ thể thay vì học quy luật tổng quát). Chế độ thích ứng cho
  phép rời đoạn NGAY khi đạt ngưỡng success_rate ổn định (require_consecutive
  lần liên tiếp, tránh advance nhầm vì may mắn), với max_iterations_per_segment
  làm lưới an toàn chống kẹt vô hạn ở đoạn quá khó.
"""

import sys
import torch

sys.path.insert(0, "/path/to/folder/containing/env_stocktrading")
from env_stocktrading import StockTradingEnv  # hoặc: from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from .task import Task, stock_trading_success_fn
from .policy import GaussianMLPPolicy
from .sampler_vectorized import VectorizedGroupSampler
from .trainer import GRPOTrainer, GRPOConfig
from .finrl_data import prepare_finrl_dataframe, train_trade_split, compute_state_scale, FixedScaleObservation
from .curriculum import SegmentCurriculum, make_time_segments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "your_ohlcv.csv"
TIC = "STOCK"
SEGMENT_LENGTH = 60              # ~3 tháng giao dịch mỗi đoạn -- chỉnh theo đặc tính dữ liệu của bạn
GROUP_SIZE = 8

SUCCESS_RATE_THRESHOLD = 0.5      # tỉ lệ rollout "thắng" (không lỗ) cần đạt để coi 1 đoạn là "pass"
REQUIRE_CONSECUTIVE = 3           # phải đạt ngưỡng trên LIÊN TIẾP 3 lần mới cho advance -- tránh
                                   # advance nhầm vì 1 group ăn may (group_size nhỏ, dễ nhiễu)
MAX_ITERATIONS_PER_SEGMENT = 40   # lưới an toàn: đoạn quá khó cũng không train quá số này
REVIEW_PROB = 0.2                 # xác suất 1 iteration ôn lại đoạn cũ thay vì đoạn hiện tại
SAFETY_MAX_TOTAL_ITERATIONS = 2000  # chặn cứng phòng trường hợp curriculum kẹt bất thường

df, indicator_cols = prepare_finrl_dataframe(CSV_PATH, tic=TIC)
train_df, trade_df = train_trade_split(df, split_ratio=0.8)

stock_dim = 1
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

# Scale cố định tính 1 LẦN từ toàn bộ train_df, dùng chung cho mọi đoạn/mọi lần
# review -- KHÔNG tính lại theo từng đoạn (mất tính nhất quán giữa các đoạn).
state_scale = compute_state_scale(
    train_df, indicator_cols, env_kwargs["initial_amount"], env_kwargs["hmax"], stock_dim
)

segments = make_time_segments(train_df, segment_length=SEGMENT_LENGTH)
print(f"Chia train_df ({len(train_df)} dòng) thành {len(segments)} đoạn, mỗi đoạn {SEGMENT_LENGTH} ngày.")

curriculum = SegmentCurriculum(
    segments=segments,
    review_prob=REVIEW_PROB,
    advance_on_success_rate=SUCCESS_RATE_THRESHOLD,
    require_consecutive=REQUIRE_CONSECUTIVE,
    max_iterations_per_segment=MAX_ITERATIONS_PER_SEGMENT,
)

policy = GaussianMLPPolicy(n_observations=state_space, action_dim=stock_dim, log_std_init=-0.5)


def build_sampler_for_segment(segment_df):
    def make_env():
        env = StockTradingEnv(df=segment_df, **env_kwargs)
        return FixedScaleObservation(env, state_scale)

    task = Task(
        env_id="finrl-segment",
        success_fn=stock_trading_success_fn(0.0),
        max_steps=len(segment_df) - 1,
        make_env_fn=make_env,
    )
    return VectorizedGroupSampler(task, policy, device, group_size=GROUP_SIZE, use_async=False)


config = GRPOConfig(lr=3e-4, num_iterations=1, advantage_method="grpo", entropy_coef=0.001)
trainer = GRPOTrainer(policy, sampler=None, config=config, device=device)  # sampler sẽ gán mỗi iteration

# Dùng while thay vì range cố định: mỗi đoạn có thể tốn số iteration KHÁC NHAU
# tuỳ độ khó (đoạn dễ pass nhanh, đoạn khó ở lại lâu hơn tới khi chạm ngưỡng
# hoặc chạm MAX_ITERATIONS_PER_SEGMENT) -- không thể tính trước tổng số iteration
# như chế độ cố định cũ.
it = 0
while not curriculum.is_finished and it < SAFETY_MAX_TOTAL_ITERATIONS:
    segment_df = curriculum.get_segment_for_iteration()
    sampler = build_sampler_for_segment(segment_df)
    trainer.sampler = sampler

    stats = trainer.train_step()
    curriculum.report_iteration_result(stats)
    sampler.close()

    it += 1
    if it % 5 == 0:
        print(
            f"Iter {it} {curriculum.status_str()} | "
            f"reward={stats['avg_reward']:.3f} | success_rate={stats['success_rate']:.2f} | "
            f"entropy={stats['entropy']:.3f}"
        )

if it >= SAFETY_MAX_TOTAL_ITERATIONS:
    print(f"CẢNH BÁO: chạm SAFETY_MAX_TOTAL_ITERATIONS={SAFETY_MAX_TOTAL_ITERATIONS} mà curriculum "
          f"chưa xong (đang ở seg {curriculum.current_segment_index}/{len(segments)-1}). "
          f"Có thể cần tăng giới hạn này, giảm SUCCESS_RATE_THRESHOLD, hoặc dữ liệu/đoạn quá khó.")
else:
    print(f"Đã học hết toàn bộ curriculum sau {it} iteration.")

torch.save(policy.state_dict(), f"policy_finrl_curriculum_{TIC}_final.pt")