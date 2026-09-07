"""
Curriculum theo thời gian cho bài toán time-series (stock trading): thay vì
train trên toàn bộ chuỗi giá 1 lúc (nhiều regime thị trường trộn lẫn -> gradient
nhiễu), chia dữ liệu thành các đoạn (segment) liên tiếp theo thời gian, học
tuần tự từng đoạn, và thỉnh thoảng "ôn lại" đoạn cũ để tránh catastrophic
forgetting (quên đoạn đầu khi đã học sang đoạn sau).

NGUYÊN TẮC QUAN TRỌNG: mỗi GROUP (G rollout trong 1 lần GRPOTrainer.train_step())
dùng CÙNG 1 đoạn dữ liệu. Curriculum chỉ đổi đoạn GIỮA các group/iteration,
không đổi trong 1 group -- vì advantage của GRPO được chuẩn hoá tương đối giữa
các rollout trong cùng group, nên chúng cần cùng điều kiện xuất phát (cùng đoạn
giá) để phép so sánh có ý nghĩa.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd


def make_time_segments(df: pd.DataFrame, segment_length: int, stride: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Chia df (đã reset_index 0..N-1, sắp theo thời gian) thành các đoạn liên tục.
    stride=None -> các đoạn không chồng lấn (stride = segment_length).
    Mỗi đoạn trả về đã reset_index(drop=True) vì StockTradingEnv cần index 0..N-1.
    """
    stride = stride or segment_length
    segments = []
    n = len(df)
    start = 0
    while start + segment_length <= n:
        seg = df.iloc[start:start + segment_length].reset_index(drop=True)
        segments.append(seg)
        start += stride
    if not segments:
        raise ValueError(
            f"df có {n} dòng, nhỏ hơn segment_length={segment_length}. "
            "Giảm segment_length hoặc cung cấp thêm dữ liệu."
        )
    return segments


@dataclass
class SegmentCurriculum:
    """
    Quản lý việc chọn đoạn dữ liệu cho mỗi iteration.

    Cách dùng trong vòng lặp huấn luyện:
        segment_df = curriculum.get_segment_for_iteration()
        # ... build sampler cho segment_df, chạy train_step() ...
        curriculum.report_iteration_result(stats)   # stats từ GRPOTrainer.train_step()
    """
    segments: List[pd.DataFrame]
    iterations_per_segment: int = 20     # số iteration học 1 đoạn trước khi chuyển tiếp
    review_prob: float = 0.2             # xác suất 1 iteration dùng đoạn CŨ thay vì đoạn hiện tại
    advance_on_success_rate: Optional[float] = None  # nếu đặt, chỉ chuyển đoạn khi đạt ngưỡng
    # success_rate này (thay vì chuyển theo số iteration cố định)
    require_consecutive: int = 1         # số lần đạt ngưỡng LIÊN TIẾP (không đứt quãng) mới
    # cho advance -- tránh advance nhầm vì 1 lần success_rate cao ngẫu nhiên (group nhỏ, dễ nhiễu)
    max_iterations_per_segment: Optional[int] = None  # lưới an toàn: nếu dùng
    # advance_on_success_rate mà đoạn quá khó, policy có thể không bao giờ đạt
    # ngưỡng liên tiếp -> kẹt mãi ở 1 đoạn. Đặt giá trị này để ép chuyển tiếp
    # sau tối đa N iteration dù chưa đạt ngưỡng (None = không giới hạn, chỉ nên
    # dùng khi advance_on_success_rate=None).

    _current_idx: int = field(default=0, init=False)
    _iters_on_current: int = field(default=0, init=False)
    _consecutive_hits: int = field(default=0, init=False)
    _last_segment_was_review: bool = field(default=False, init=False)
    _last_review_idx: Optional[int] = field(default=None, init=False)

    def get_segment_for_iteration(self) -> pd.DataFrame:
        """Gọi TRƯỚC mỗi iteration để biết dùng đoạn nào cho group sắp thu thập."""
        if self._current_idx > 0 and random.random() < self.review_prob:
            review_idx = random.randint(0, self._current_idx - 1)
            self._last_segment_was_review = True
            self._last_review_idx = review_idx
            return self.segments[review_idx]
        self._last_segment_was_review = False
        self._last_review_idx = None
        return self.segments[self._current_idx]

    def report_iteration_result(self, stats: dict):
        """Gọi SAU mỗi iteration với stats trả về từ GRPOTrainer.train_step(),
        để curriculum quyết định khi nào chuyển sang đoạn tiếp theo.
        Lưu ý: chỉ đếm tiến độ khi iteration đó KHÔNG phải review, vì review là
        để ôn lại, không phải tiến độ học đoạn mới."""
        if self._last_segment_was_review:
            return

        self._iters_on_current += 1

        if self.advance_on_success_rate is not None:
            hit = stats["success_rate"] >= self.advance_on_success_rate
            # Phải đạt ngưỡng LIÊN TIẾP require_consecutive lần -- 1 lần đạt rồi
            # trượt lại thì reset về 0, tránh advance nhầm vì may mắn nhất thời
            # (group_size thường nhỏ (~8), success_rate 1 lần dễ nhiễu).
            self._consecutive_hits = self._consecutive_hits + 1 if hit else 0
            should_advance = self._consecutive_hits >= self.require_consecutive

            if self.max_iterations_per_segment is not None and \
                    self._iters_on_current >= self.max_iterations_per_segment:
                should_advance = True  # ép chuyển tiếp, tránh kẹt vô thời hạn ở đoạn quá khó
        else:
            should_advance = self._iters_on_current >= self.iterations_per_segment

        if should_advance and self._current_idx < len(self.segments) - 1:
            self._current_idx += 1
            self._iters_on_current = 0
            self._consecutive_hits = 0

    @property
    def current_segment_index(self) -> int:
        return self._current_idx

    @property
    def is_finished(self) -> bool:
        on_last_segment = self._current_idx >= len(self.segments) - 1
        if self.advance_on_success_rate is not None:
            done_with_last = self._consecutive_hits >= self.require_consecutive or (
                self.max_iterations_per_segment is not None
                and self._iters_on_current >= self.max_iterations_per_segment
            )
        else:
            done_with_last = self._iters_on_current >= self.iterations_per_segment
        return on_last_segment and done_with_last

    def status_str(self) -> str:
        if self._last_segment_was_review:
            return f"[review seg {self._last_review_idx}]"
        if self.advance_on_success_rate is not None:
            return (f"[seg {self._current_idx}/{len(self.segments) - 1}, "
                    f"streak {self._consecutive_hits}/{self.require_consecutive}, "
                    f"iters {self._iters_on_current}]")
        return f"[seg {self._current_idx}/{len(self.segments) - 1}, {self._iters_on_current}/{self.iterations_per_segment}]"
