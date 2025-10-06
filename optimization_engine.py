# optimization_engine.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest_engine import EngineConfig, run_backtest as run_engine
from monte_carlo import MonteCarloResult, monte_carlo_bootstrap_equity
from run_backtest import load_strategy


@dataclass(frozen=True)
class OptimizationProgress:
    total: int
    completed: int

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.completed / self.total))

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)


@dataclass
class OptimizationResult:
    params: Dict[str, Any]
    reward: float
    k_ratio: float
    inverse_cvar: float
    expected_return: float
    cvar: float
    equity: pd.Series
    trades: pd.DataFrame
    initial_balance: float
    monte_carlo: MonteCarloResult


@dataclass
class OptimizationOutcome:
    results: list[OptimizationResult]
    best: Optional[OptimizationResult]


ProgressCallback = Callable[[OptimizationProgress, OptimizationResult, Optional[OptimizationResult]], None]


class OptimizationEngine:
    """Grid search optimizer that evaluates strategy parameters with Monte Carlo metrics."""

    def __init__(
        self,
        price: pd.Series,
        market_data: Optional[pd.DataFrame],
        strategy_path: str,
        base_params: Optional[Dict[str, Any]] = None,
        param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        engine_config: Optional[EngineConfig] = None,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        mc_sims: int = 1000,
        mc_horizon: Optional[int] = None,
        mc_confidence: float = 0.95,
        mc_seed: Optional[int] = None,
    ) -> None:
        if price is None or len(price) == 0:
            raise ValueError("Price series is required for optimization")
        self.price = price
        self.market_data = market_data
        self.strategy_path = strategy_path
        self.base_params = dict(base_params or {})
        self.param_grid = dict(param_grid or {})
        self.engine_config = engine_config or EngineConfig()
        self.weights = tuple(float(w) for w in weights)
        self.mc_sims = int(mc_sims)
        self.mc_horizon = mc_horizon
        self.mc_confidence = float(mc_confidence)
        self.mc_seed = mc_seed

    def run(self, callback: Optional[ProgressCallback] = None) -> OptimizationOutcome:
        keys, value_lists = self._prepare_grid()
        total = 1
        for values in value_lists:
            total *= max(1, len(values))

        completed = 0
        results: list[OptimizationResult] = []
        best: Optional[OptimizationResult] = None

        iterator = product(*value_lists) if value_lists else [tuple()]
        for combo in iterator:
            overrides = dict(zip(keys, combo)) if keys else {}
            result = self._evaluate(overrides)
            results.append(result)
            completed += 1
            if best is None or result.reward > best.reward:
                best = result
            if callback is not None:
                progress = OptimizationProgress(total=total, completed=completed)
                callback(progress, result, best)
        return OptimizationOutcome(results=results, best=best)

    def _evaluate(self, overrides: Dict[str, Any]) -> OptimizationResult:
        params = {**self.base_params, **overrides}
        sanitized_params = {key: self._coerce_scalar(value) for key, value in params.items()}

        strategy = load_strategy(self.strategy_path)
        for key, value in sanitized_params.items():
            setattr(strategy, key, value)
        if hasattr(strategy, "params") and isinstance(getattr(strategy, "params"), dict):
            strategy.params.update(sanitized_params)

        outputs = strategy.generate(self.price, data=self.market_data)
        results = run_engine(self.price, outputs, self.engine_config)
        equity = results["equity"].copy()

        mc_result = monte_carlo_bootstrap_equity(
            equity,
            num_simulations=self.mc_sims,
            horizon=self.mc_horizon,
            seed=self.mc_seed,
            confidence_level=self.mc_confidence,
        )

        k_ratio = self._compute_k_ratio(equity)
        inverse_cvar = self._compute_inverse_cvar(mc_result.cvar)
        reward = self._compute_reward(k_ratio, inverse_cvar, mc_result.expected_return)

        return OptimizationResult(
            params=sanitized_params,
            reward=reward,
            k_ratio=k_ratio,
            inverse_cvar=inverse_cvar,
            expected_return=float(mc_result.expected_return),
            cvar=float(mc_result.cvar),
            equity=equity,
            trades=results["trades"],
            initial_balance=float(self.engine_config.init_cash),
            monte_carlo=mc_result,
        )

    def _prepare_grid(self) -> Tuple[list[str], list[list[Any]]]:
        keys: list[str] = []
        values: list[list[Any]] = []
        for key, raw in self.param_grid.items():
            normalized = self._normalize_values(raw)
            if not normalized:
                continue
            keys.append(key)
            values.append(normalized)
        return keys, values

    @staticmethod
    def _normalize_values(values: Sequence[Any]) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, (str, bytes, bytearray)):
            return [values]
        if isinstance(values, dict):
            iterable = values.values()
        else:
            try:
                iterable = iter(values)  # type: ignore[arg-type]
            except TypeError:
                return [values]
        normalized: list[Any] = []
        for item in iterable:
            normalized.append(OptimizationEngine._coerce_scalar(item))
        return normalized

    @staticmethod
    def _coerce_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _compute_reward(self, k_ratio: float, inverse_cvar: float, expected_return: float) -> float:
        w1, w2, w3 = self.weights
        return float(
            (w1 * self._finite(k_ratio))
            + (w2 * self._finite(inverse_cvar))
            + (w3 * self._finite(expected_return))
        )

    @staticmethod
    def _compute_k_ratio(equity: pd.Series) -> float:
        values = equity.to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 3:
            return 0.0
        x = np.arange(values.shape[0], dtype=float)[mask]
        y = values[mask]
        if y.size < 3:
            return 0.0
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0
        fitted = slope * x + intercept
        residuals = y - fitted
        dof = y.size - 2
        if dof <= 0:
            return 0.0
        se = np.sqrt(np.sum(residuals ** 2) / dof)
        se = max(se, 1e-12)
        k_ratio = slope / se * np.sqrt(y.size)
        return float(k_ratio)

    @staticmethod
    def _compute_inverse_cvar(cvar: float) -> float:
        if not np.isfinite(cvar):
            return 0.0
        risk = abs(float(cvar))
        if risk <= 1e-12:
            risk = 1e-12
        return float(1.0 / risk)

    @staticmethod
    def _finite(value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            return 0.0
        return value



