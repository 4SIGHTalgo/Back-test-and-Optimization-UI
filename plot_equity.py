# plot_equity.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GREEN = "#00FF00"
RED = "#FF0000"
GRAY = "#808080"
CYAN = "#00B7EB"

def plot_equity_tradenum(equity: pd.Series,
                         trades: pd.DataFrame,
                         initial_balance: float,
                         title: str,
                         out_png: Path,
                         show: bool = False,
                         monte_carlo_mean: Optional[pd.Series] = None,
                         monte_carlo_lower: Optional[pd.Series] = None,
                         monte_carlo_upper: Optional[pd.Series] = None,
                         monte_carlo_confidence: Optional[float] = None,
                         monte_carlo_var: Optional[float] = None,
                         monte_carlo_cvar: Optional[float] = None,
                         monte_carlo_expected: Optional[float] = None) -> None:
    """
    Plots equity at each trade exit along an x-axis of trade numbers.
    Green above initial balance, red below.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build arrays of trade numbers and equity-at-exit
    numbers = [0]
    values  = [initial_balance]
    months  = [pd.to_datetime(equity.index[0]).strftime('%b')]
    steps   = [0]

    trades_sorted = trades.dropna(subset=["exit_time"]).sort_values("exit_time")
    for i, (_, tr) in enumerate(trades_sorted.iterrows(), 1):
        # snap exit to last equity timestamp <= exit_time
        exit_time = pd.to_datetime(tr["exit_time"])
        idx = equity.index[equity.index <= exit_time]
        if len(idx) == 0:
            continue
        val = float(equity.loc[idx[-1]])
        numbers.append(i)
        values.append(val)
        step_idx = equity.index.get_indexer([idx[-1]])
        if len(step_idx) == 0 or step_idx[0] == -1:
            step_position = len(equity) - 1
        else:
            step_position = int(step_idx[-1])
        steps.append(step_position)
        months.append(exit_time.strftime('%b'))

    if len(numbers) > 1:
        x = np.array(numbers)
        y = np.array(values)

        ax.plot(x, y, color=GREEN, linewidth=2, label="Equity", zorder=3)
        ax.fill_between(x, initial_balance, y, where=(y >= initial_balance),
                        color=GREEN, alpha=0.15, interpolate=True, zorder=1)
        ax.fill_between(x, initial_balance, y, where=(y < initial_balance),
                        color=RED, alpha=0.15, interpolate=True, zorder=1)
        ax.axhline(y=initial_balance, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5,
                   label=f"Initial: ${initial_balance:,.0f}")
        ax.scatter(x, y, color=GREEN, s=5, zorder=4, alpha=0.8)

        if monte_carlo_mean is not None and monte_carlo_lower is not None and monte_carlo_upper is not None:
            step_index = pd.Index(steps, name="step")

            def _extract(series: pd.Series) -> np.ndarray:
                ser = series.reindex(step_index)
                if ser.isna().any():
                    ser = ser.interpolate()
                    ser = ser.ffill().bfill()
                return ser.astype(float).to_numpy()

            mc_mean_vals = _extract(monte_carlo_mean)
            mc_lower_vals = _extract(monte_carlo_lower)
            mc_upper_vals = _extract(monte_carlo_upper)
            lower_band = np.minimum(mc_lower_vals, mc_upper_vals)
            upper_band = np.maximum(mc_lower_vals, mc_upper_vals)
            if monte_carlo_confidence is not None:
                envelope_label = f"MC envelope ({monte_carlo_confidence:.0%})"
            else:
                envelope_label = "MC envelope"

            ax.fill_between(x, lower_band, upper_band, color=CYAN, alpha=0.12, interpolate=True,
                            label=envelope_label, zorder=0)
            ax.plot(x, mc_mean_vals, color=CYAN, linewidth=1.8, linestyle='--', label="MC mean", zorder=2)

        ax.set_xlim(0, max(numbers) * 1.02)
        max_trades = max(numbers)
        if max_trades <= 20:
            ticks = list(range(0, max_trades + 1))
            labels = [f"{i}\n{months[i]}" if i < len(months) else f"{i}" for i in ticks]
        else:
            step = max(1, max_trades // 15)
            ticks = list(range(0, max_trades + 1, step))
            if ticks[-1] != max_trades:
                ticks.append(max_trades)
            labels = []
            for pos in ticks:
                labels.append(f"{pos}\n{months[min(pos, len(months) - 1)]}")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=9)
    else:
        ax.plot([0, 1], [initial_balance, initial_balance], color=GREEN, linewidth=2, label="Equity")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"0\n{months[0]}", "1"])

    ax.set_title(title, fontsize=14, color="white", pad=15)
    ax.set_xlabel("Trade Number", fontsize=12, color="white")
    ax.set_ylabel("Equity ($)", fontsize=12, color="white")
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#000000")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.legend(loc="lower left", framealpha=0.9, facecolor="#1a1a1a", edgecolor="#333333")

    metrics_lines = []
    if monte_carlo_var is not None:
        conf = monte_carlo_confidence if monte_carlo_confidence is not None else 0.95
        metrics_lines.append(f"VaR {conf:.0%}: {monte_carlo_var * 100.0:.2f}%")
    if monte_carlo_cvar is not None:
        conf = monte_carlo_confidence if monte_carlo_confidence is not None else 0.95
        metrics_lines.append(f"CVaR {conf:.0%}: {monte_carlo_cvar * 100.0:.2f}%")
    if monte_carlo_expected is not None:
        metrics_lines.append(f"Expected Return: {monte_carlo_expected * 100.0:.2f}%")
    if metrics_lines:
        metrics_text = "\n".join(metrics_lines)
        props_metrics = dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.85, edgecolor=CYAN)
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=9, ha="right", va="bottom", color="white", bbox=props_metrics)

    # annotation
    if len(numbers) > 1:
        final_val = values[-1]
        pnl = final_val - initial_balance
        pnl_pct = pnl / initial_balance * 100.0
        text = f"Final: ${final_val:,.0f}\nP&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%)\nTrades: {numbers[-1]}"
        props = dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.9, edgecolor=GREEN)
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10, va="top", color="white", bbox=props)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor="#000000", edgecolor="none")
    if show:
        plt.show()
    plt.close()


