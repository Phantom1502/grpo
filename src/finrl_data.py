"""
Chuẩn bị dữ liệu local CSV (chỉ có Open, High, Low, Close, Volume) thành dataframe
đúng format mà finrl.StockTradingEnv yêu cầu.

StockTradingEnv (xem finrl/meta/env_stock_trading/env_stocktrading.py) cần:
  - cột 'close' (viết thường) để tính giá trị tài sản
  - cột 'tic' (mã cổ phiếu) -- dùng để env biết đây là 1 hay nhiều mã
    (len(df.tic.unique()) == 1 -> nhánh xử lý single-stock)
  - cột 'date' -- dùng để log/track theo thời gian
  - df.index PHẢI là số nguyên 0..N-1 liên tục (env dùng self.df.loc[self.day, :])
  - các cột tech_indicator_list (nếu có) phải tồn tại sẵn trong df

Vì bạn chỉ có 5 cột OHLCV thô, hàm dưới đây tính vài indicator ĐƠN GIẢN bằng
pandas thuần (không cần finrl.FeatureEngineer / stockstats / ta-lib) để không
phải cài thêm dependency nặng. Nếu sau này bạn có nhiều mã / muốn indicator
chuẩn (MACD, RSI, Bollinger... đúng công thức FinRL dùng), nên chuyển sang
`finrl.meta.preprocessor.preprocessors.FeatureEngineer`.
"""

import numpy as np
import pandas as pd


def load_ohlcv_csv(path: str, tic: str = "STOCK") -> pd.DataFrame:
    """Đọc CSV có cột Open/High/Low/Close/Volume (không phân biệt hoa thường),
    trả về dataframe đã chuẩn hoá tên cột về lowercase + thêm cột 'tic'."""
    df = pd.read_csv(path)

    # Cột ngày có thể là index (khi to_csv với index=True) hoặc 1 cột tên Date/date.
    date_col = None
    for candidate in ("date", "Date", "datetime", "Datetime"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # Giả định cột đầu tiên là ngày (trường hợp CSV export từ yfinance có index là Date)
        date_col = df.columns[0]

    df = df.rename(columns={date_col: "date"})
    df.columns = [c.lower() if c != "date" else c for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = tic

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong CSV: {missing}")

    return df.sort_values("date").reset_index(drop=True)


def add_simple_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm vài technical indicator cơ bản bằng pandas thuần, đủ dùng để có state
    space giàu thông tin hơn OHLCV thô. Không cần cài stockstats/ta-lib.

    Trả về df đã thêm cột, và list tên các cột indicator vừa thêm (để truyền
    vào tech_indicator_list của StockTradingEnv).
    """
    df = df.copy()
    close = df["close"]

    df["sma_5"] = close.rolling(5).mean()
    df["sma_20"] = close.rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    rolling_std = close.rolling(20).std()
    df["boll_ub"] = df["sma_20"] + 2 * rolling_std
    df["boll_lb"] = df["sma_20"] - 2 * rolling_std

    indicator_cols = ["sma_5", "sma_20", "rsi_14", "macd", "boll_ub", "boll_lb"]

    # Các indicator dùng rolling window nên vài dòng đầu sẽ là NaN -> cắt bỏ.
    df = df.dropna().reset_index(drop=True)
    return df, indicator_cols


def prepare_finrl_dataframe(csv_path: str, tic: str = "STOCK"):
    """
    Pipeline đầy đủ: CSV local (OHLCV) -> dataframe sẵn sàng cho StockTradingEnv.

    Trả về (df, indicator_cols). df.index đã là 0..N-1 liên tục (bắt buộc cho env).
    """
    df = load_ohlcv_csv(csv_path, tic=tic)
    df, indicator_cols = add_simple_indicators(df)
    df.index = df.index  # đã reset_index(drop=True) trong add_simple_indicators, index đã 0..N-1
    return df, indicator_cols


def train_trade_split(df: pd.DataFrame, split_ratio: float = 0.8):
    """Chia dữ liệu theo thời gian: phần đầu để train, phần sau để trade/test.
    Re-index lại về 0..N-1 cho từng phần vì StockTradingEnv cần index liên tục."""
    n_train = int(len(df) * split_ratio)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    trade_df = df.iloc[n_train:].reset_index(drop=True)
    return train_df, trade_df


def compute_state_scale(df: pd.DataFrame, indicator_cols: list, initial_amount: float,
                         hmax: int, stock_dim: int = 1) -> "np.ndarray":
    """
    Tính vector scale CỐ ĐỊNH để chuẩn hoá state của StockTradingEnv, dùng
    thay cho gym.wrappers.NormalizeObservation (adaptive, running mean/std).

    LÝ DO CẦN SCALE CỐ ĐỊNH thay vì adaptive: khi dùng curriculum theo đoạn
    (curriculum.py), env bị tạo mới liên tục mỗi khi đổi đoạn/ôn lại đoạn cũ.
    NormalizeObservation tính running mean/std BÊN TRONG mỗi env instance, nên
    mỗi lần tạo env mới là mất hết thống kê đã tích luỹ -> chuẩn hoá không nhất
    quán giữa các đoạn, gây nhiễu học. Scale cố định (tính 1 lần từ toàn bộ
    train_df trước khi chia đoạn) đảm bảo MỌI đoạn, MỌI lần review đều dùng
    chung 1 phép chuẩn hoá -- giống hệt cách production ML tránh "data leakage
    ngược" giữa các fold nhưng vẫn giữ nhất quán thang đo.

    Chỉ hỗ trợ stock_dim=1 (single stock) khớp với finrl_data.py hiện tại.
    State layout (xem env_stocktrading._initiate_state, nhánh single-stock):
        [cash] + [close] + [shares]*stock_dim + [indicator_1, indicator_2, ...]
    """
    import numpy as np

    price_scale = float(df["close"].max())
    # Số cổ phiếu tối đa hợp lý: toàn bộ vốn ban đầu đổi hết thành cổ phiếu ở giá thấp nhất.
    shares_scale = initial_amount / max(float(df["close"].min()), 1e-6)

    indicator_scales = []
    for col in indicator_cols:
        col_abs_max = float(df[col].abs().max())
        indicator_scales.append(col_abs_max if col_abs_max > 1e-6 else 1.0)

    scale = [initial_amount, price_scale] + [shares_scale] * stock_dim + indicator_scales
    return np.array(scale, dtype=np.float32)


import gymnasium as gym


class FixedScaleObservation(gym.ObservationWrapper):
    """
    ObservationWrapper chuẩn hoá state bằng cách chia cho 1 vector scale CỐ
    ĐỊNH (tính trước bằng compute_state_scale), thay vì running mean/std.
    Kế thừa gym.ObservationWrapper (thay vì duck-type thủ công) để tương thích
    đầy đủ với gym.vector.SyncVectorEnv/AsyncVectorEnv (cần .metadata, .spec...).
    """

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def observation(self, obs):
        return obs / self.scale
