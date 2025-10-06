# backtest_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class EngineConfig:
    init_cash: float = 100_000.0
    point: float = 1.0           # minimum price increment
    tick_value: float = 1.0      # cash value per one tick per 1 contract
    finalize_trades: bool = True  # close an open trade on the last bar


def _normalize_bool(series: Optional[pd.Series], index: pd.Index) -> np.ndarray:
    if series is None:
        return np.zeros(len(index), dtype=bool)
    return series.reindex(index).fillna(False).astype(bool).to_numpy()


def _normalize_float(series: Optional[pd.Series], index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series.reindex(index), errors="coerce")


def _level_at(series: pd.Series, i: int) -> float:
    val = series.iat[i]
    if pd.isna(val):
        return np.nan
    return float(val)


def _build_position_from_signals(index: pd.Index,
                                 entries: Optional[pd.Series],
                                 exits: Optional[pd.Series],
                                 short_entries: Optional[pd.Series],
                                 short_exits: Optional[pd.Series],
                                 price: pd.Series,
                                 long_take_profit: Optional[pd.Series] = None,
                                 long_stop_loss: Optional[pd.Series] = None,
                                 short_take_profit: Optional[pd.Series] = None,
                                 short_stop_loss: Optional[pd.Series] = None
                                 ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    if price is None:
        raise ValueError("price series is required to build positions from signals")

    price_series = pd.to_numeric(price.reindex(index), errors="coerce")
    if price_series.isna().any():
        raise ValueError("Price series contains NaN after aligning with strategy signals")

    entries_arr = _normalize_bool(entries, index)
    exits_arr = _normalize_bool(exits, index)
    short_entries_arr = _normalize_bool(short_entries, index)
    short_exits_arr = _normalize_bool(short_exits, index)

    long_tp_series = _normalize_float(long_take_profit, index)
    long_sl_series = _normalize_float(long_stop_loss, index)
    short_tp_series = _normalize_float(short_take_profit, index)
    short_sl_series = _normalize_float(short_stop_loss, index)

    n = len(index)
    position_values = np.zeros(n, dtype=float)
    exit_reasons = np.empty(n, dtype=object)
    active_tp_values = np.full(n, np.nan, dtype=float)
    active_sl_values = np.full(n, np.nan, dtype=float)

    curr_pos = 0.0
    curr_tp = np.nan
    curr_sl = np.nan

    for i in range(n):
        price_t = float(price_series.iat[i])
        long_entry = entries_arr[i]
        short_entry = short_entries_arr[i]
        long_exit_signal = exits_arr[i]
        short_exit_signal = short_exits_arr[i]

        exit_reason = None
        new_pos = curr_pos
        new_tp = curr_tp
        new_sl = curr_sl

        if curr_pos > 0:
            hit_tp = np.isfinite(curr_tp) and price_t >= curr_tp
            hit_sl = np.isfinite(curr_sl) and price_t <= curr_sl
            if hit_tp:
                exit_reason = "take_profit"
                new_pos = 0.0
                new_tp = np.nan
                new_sl = np.nan
            elif hit_sl:
                exit_reason = "stop_loss"
                new_pos = 0.0
                new_tp = np.nan
                new_sl = np.nan
            elif short_entry:
                exit_reason = "reverse"
                new_pos = -1.0
                new_tp = _level_at(short_tp_series, i)
                new_sl = _level_at(short_sl_series, i)
            elif long_exit_signal:
                exit_reason = "signal_exit"
                new_pos = 0.0
                new_tp = np.nan
                new_sl = np.nan
        elif curr_pos < 0:
            hit_tp = np.isfinite(curr_tp) and price_t <= curr_tp
            hit_sl = np.isfinite(curr_sl) and price_t >= curr_sl
            if hit_tp:
                exit_reason = "take_profit"
                new_pos = 0.0
                new_tp = np.nan
                new_sl = np.nan
            elif hit_sl:
                exit_reason = "stop_loss"
                new_pos = 0.0
                new_tp = np.nan
                new_sl = np.nan
            elif long_entry:
                exit_reason = "reverse"
                new_pos = 1.0
                new_tp = _level_at(long_tp_series, i)
                new_sl = _level_at(long_sl_series, i)
            elif short_exit_signal:
                exit_reason = "signal_exit"
                new_pos = 0.0
                new_tp = np.nan
                new_sl = np.nan

        if new_pos == 0.0:
            if long_entry and not short_entry:
                new_pos = 1.0
                new_tp = _level_at(long_tp_series, i)
                new_sl = _level_at(long_sl_series, i)
            elif short_entry and not long_entry:
                new_pos = -1.0
                new_tp = _level_at(short_tp_series, i)
                new_sl = _level_at(short_sl_series, i)

        position_values[i] = new_pos
        exit_reasons[i] = exit_reason
        active_tp_values[i] = new_tp if np.isfinite(new_tp) else np.nan
        active_sl_values[i] = new_sl if np.isfinite(new_sl) else np.nan

        curr_pos = new_pos
        curr_tp = new_tp
        curr_sl = new_sl

    position_series = pd.Series(position_values, index=index, dtype=float)
    exit_reason_series = pd.Series(exit_reasons, index=index, dtype="object")
    active_tp_series = pd.Series(active_tp_values, index=index, dtype=float)
    active_sl_series = pd.Series(active_sl_values, index=index, dtype=float)

    meta = {
        "exit_reason": exit_reason_series,
        "active_take_profit": active_tp_series,
        "active_stop_loss": active_sl_series,
    }
    return position_series, meta


def _compute_equity(price: pd.Series,
                    position: pd.Series,
                    cfg: EngineConfig) -> pd.Series:
    # PnL uses integer ticks, equity updated on bar close using previous bar's position
    delta_price = price.diff().fillna(0.0)
    delta_ticks = np.rint(delta_price / cfg.point)
    bar_pnl = position.shift(1).fillna(0.0) * delta_ticks * cfg.tick_value
    equity = cfg.init_cash + bar_pnl.cumsum()
    equity.iloc[0] = cfg.init_cash
    return equity


def _compute_trades(price: pd.Series,
                    position: pd.Series,
                    cfg: EngineConfig,
                    exit_reasons: Optional[pd.Series] = None,
                    active_take_profit: Optional[pd.Series] = None,
                    active_stop_loss: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Build a trades table by detecting position changes.
    Fills at bar close.
    """
    idx = price.index
    pos = position.astype(float).reindex(idx).fillna(0.0)

    exit_reason_series = exit_reasons.reindex(idx) if exit_reasons is not None else None
    active_tp_series = active_take_profit.reindex(idx) if active_take_profit is not None else None
    active_sl_series = active_stop_loss.reindex(idx) if active_stop_loss is not None else None

    rows = []
    curr_dir = 0.0
    entry_time = None
    entry_price = None
    curr_tp = None
    curr_sl = None

    def close_trade(exit_time, exit_price, direction, e_time, e_price, take_profit_level, stop_loss_level, reason_label):
        if direction > 0:
            pnl_price = exit_price - e_price
        else:
            pnl_price = e_price - exit_price
        pnl_ticks = int(np.rint(pnl_price / cfg.point))
        pnl_cash = pnl_ticks * cfg.tick_value
        rows.append({
            "direction": "long" if direction > 0 else "short",
            "entry_time": e_time,
            "entry_price": e_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl_ticks": pnl_ticks,
            "pnl_cash": pnl_cash,
            "take_profit": take_profit_level,
            "stop_loss": stop_loss_level,
            "exit_reason": reason_label
        })

    for t in range(len(idx)):
        p = pos.iloc[t]
        price_t = float(price.iloc[t])
        time_t = idx[t]

        active_tp = None
        if active_tp_series is not None:
            val = active_tp_series.iloc[t]
            if pd.notna(val):
                active_tp = float(val)

        active_sl = None
        if active_sl_series is not None:
            val = active_sl_series.iloc[t]
            if pd.notna(val):
                active_sl = float(val)

        exit_reason = None
        if exit_reason_series is not None:
            val = exit_reason_series.iloc[t]
            exit_reason = None if pd.isna(val) else str(val)

        if curr_dir == 0.0 and p != 0.0:
            curr_dir = p
            entry_time = time_t
            entry_price = price_t
            curr_tp = active_tp
            curr_sl = active_sl
        elif curr_dir != 0.0 and p != curr_dir and p != 0.0:
            reason = exit_reason or "reverse"
            close_trade(time_t, price_t, curr_dir, entry_time, entry_price, curr_tp, curr_sl, reason)
            curr_dir = p
            entry_time = time_t
            entry_price = price_t
            curr_tp = active_tp
            curr_sl = active_sl
        elif curr_dir != 0.0 and p == 0.0:
            reason = exit_reason or "signal_exit"
            close_trade(time_t, price_t, curr_dir, entry_time, entry_price, curr_tp, curr_sl, reason)
            curr_dir = 0.0
            entry_time = None
            entry_price = None
            curr_tp = None
            curr_sl = None
        else:
            if curr_dir != 0.0:
                if active_tp is not None:
                    curr_tp = active_tp
                if active_sl is not None:
                    curr_sl = active_sl

    if cfg.finalize_trades and curr_dir != 0.0 and entry_time is not None:
        final_reason = "finalize"
        exit_price = float(price.iloc[-1])
        close_trade(idx[-1], exit_price, curr_dir, entry_time, entry_price, curr_tp, curr_sl, final_reason)

    columns = [
        "direction",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "pnl_ticks",
        "pnl_cash",
        "take_profit",
        "stop_loss",
        "exit_reason",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_backtest(price: pd.Series,
                 strategy_outputs: Dict[str, pd.Series],
                 cfg: EngineConfig) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    price: close series
    strategy_outputs:
      Either contains 'position', or contains the 4 signal Series.
    """
    if "position" in strategy_outputs:
        position = strategy_outputs["position"].astype(float).reindex(price.index).fillna(0.0)
        exit_reason = strategy_outputs.get("exit_reason")
        if exit_reason is not None:
            exit_reason = exit_reason.reindex(price.index)
        else:
            exit_reason = pd.Series([None] * len(price.index), index=price.index, dtype="object")
        active_tp = strategy_outputs.get("active_take_profit")
        if active_tp is not None:
            active_tp = pd.to_numeric(active_tp.reindex(price.index), errors="coerce")
        else:
            active_tp = pd.Series(np.nan, index=price.index, dtype=float)
        active_sl = strategy_outputs.get("active_stop_loss")
        if active_sl is not None:
            active_sl = pd.to_numeric(active_sl.reindex(price.index), errors="coerce")
        else:
            active_sl = pd.Series(np.nan, index=price.index, dtype=float)
    else:
        entries = strategy_outputs.get("entries")
        exits = strategy_outputs.get("exits")
        short_entries = strategy_outputs.get("short_entries")
        short_exits = strategy_outputs.get("short_exits")
        long_tp = strategy_outputs.get("long_take_profit")
        long_sl = strategy_outputs.get("long_stop_loss")
        short_tp = strategy_outputs.get("short_take_profit")
        short_sl = strategy_outputs.get("short_stop_loss")

        position, meta = _build_position_from_signals(
            price.index,
            entries,
            exits,
            short_entries,
            short_exits,
            price=price,
            long_take_profit=long_tp,
            long_stop_loss=long_sl,
            short_take_profit=short_tp,
            short_stop_loss=short_sl,
        )
        exit_reason = meta["exit_reason"]
        active_tp = meta["active_take_profit"]
        active_sl = meta["active_stop_loss"]

    equity = _compute_equity(price, position, cfg)
    trades = _compute_trades(price, position, cfg,
                             exit_reasons=exit_reason,
                             active_take_profit=active_tp,
                             active_stop_loss=active_sl)

    return {
        "price": price,
        "position": position,
        "equity": equity,
        "trades": trades,
        "exit_reason": exit_reason,
        "active_take_profit": active_tp,
        "active_stop_loss": active_sl,
    }


