# monte_carlo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    simulated_equity: pd.DataFrame
    mean_equity_path: pd.Series
    lower_equity_path: pd.Series
    upper_equity_path: pd.Series
    terminal_returns: pd.Series
    var: float
    cvar: float
    expected_return: float
    confidence_level: float


def monte_carlo_bootstrap_equity(equity: pd.Series,
                                 num_simulations: int = 1000,
                                 horizon: Optional[int] = None,
                                 seed: Optional[int] = None,
                                 confidence_level: float = 0.95) -> MonteCarloResult:
    """Run a bootstrap Monte Carlo simulation on an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve from a backtest. Index is preserved only for validation, values are used.
    num_simulations : int, default 1000
        Number of random paths to simulate via bootstrapping.
    horizon : Optional[int]
        Path length in bars. Defaults to the number of return observations available.
    seed : Optional[int]
        Seed for the random generator to make runs reproducible.
    confidence_level : float, default 0.95
        Confidence level for VaR / CVaR (e.g. 0.95 for 95 percent).

    Returns
    -------
    MonteCarloResult
        Container with simulated equity paths, summary envelopes, and terminal distribution stats.
    """
    if equity is None or len(equity) < 2:
        raise ValueError("Equity series must contain at least two observations to compute returns.")

    returns = equity.pct_change().dropna()
    if returns.empty:
        raise ValueError("Equity series must produce at least one non-zero return observation.")

    if horizon is None or horizon <= 0:
        horizon = len(returns)
    horizon = min(horizon, len(returns))

    initial_value = float(equity.iloc[0])
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(returns), size=(num_simulations, horizon))
    sampled_returns = returns.values[sample_idx]

    cumulative_growth = np.cumprod(1.0 + sampled_returns, axis=1)
    simulated_paths = cumulative_growth * initial_value

    columns = pd.Index(range(1, horizon + 1), name="step")
    simulated_df = pd.DataFrame(simulated_paths,
                                index=pd.RangeIndex(num_simulations, name="simulation"),
                                columns=columns,
                                dtype=float)
    simulated_df.insert(0, 0, initial_value)
    simulated_df.columns = pd.Index(range(0, horizon + 1), name="step")

    mean_path = simulated_df.mean(axis=0)

    lower_quantile = max(0.0, min(1.0, 1.0 - confidence_level))
    upper_quantile = max(0.0, min(1.0, confidence_level))
    lower_path = pd.Series(np.quantile(simulated_df.values, lower_quantile, axis=0),
                           index=simulated_df.columns,
                           name="lower_equity")
    upper_path = pd.Series(np.quantile(simulated_df.values, upper_quantile, axis=0),
                           index=simulated_df.columns,
                           name="upper_equity")

    terminal_equity = simulated_df.iloc[:, -1]
    terminal_returns = terminal_equity / initial_value - 1.0

    var_quantile = float(np.quantile(terminal_returns, lower_quantile))
    shortfall_mask = terminal_returns <= var_quantile
    if shortfall_mask.any():
        cvar_value = float(terminal_returns[shortfall_mask].mean())
    else:
        cvar_value = var_quantile

    expected_return_value = float(terminal_returns.mean())

    return MonteCarloResult(
        simulated_equity=simulated_df,
        mean_equity_path=mean_path,
        lower_equity_path=lower_path,
        upper_equity_path=upper_path,
        terminal_returns=terminal_returns,
        var=var_quantile,
        cvar=cvar_value,
        expected_return=expected_return_value,
        confidence_level=confidence_level,
    )
