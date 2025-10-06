# run_backtest.py
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
from typing import Dict

import pandas as pd
DEFAULT_OUTPUT_DIR = (Path(__file__).resolve().parent / "outputs")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


from data_loader import CSVDateTimeLoader, CSVSingleDTLoader
from backtest_engine import EngineConfig, run_backtest
from plot_equity import plot_equity_tradenum
from monte_carlo import monte_carlo_bootstrap_equity


def load_strategy(strategy_path: str):
    """
    strategy_path format: module.submodule:ClassName
    The class must be instantiable with no arguments, and should read its params
    from module-level constants defined in the strategy file.
    """
    if ":" not in strategy_path:
        raise ValueError("strategy must be 'module:ClassName', e.g., strategies.ma_crossover:MACrossover")
    modname, classname = strategy_path.split(":", 1)
    mod = importlib.import_module(modname)
    cls = getattr(mod, classname)
    return cls()  # no params passed, strategy reads module constants


def main():
    ap = argparse.ArgumentParser(description="Modular backtest engine with tick and point sizing")
    # Data loader selection
    ap.add_argument("--loader", type=str, default="csv_datetime",
                    choices=["csv_datetime", "csv_single_dt"],
                    help="Choose CSV loader type")
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    ap.add_argument("--date-col", type=str, default="Date")
    ap.add_argument("--time-col", type=str, default="Timestamp")
    ap.add_argument("--dt-col", type=str, default="datetime")
    ap.add_argument("--close-col", type=str, default="Close")
    ap.add_argument("--tz", type=str, default=None)

    # Strategy with constants inside the file
    ap.add_argument("--strategy", type=str, required=True,
                    help="Strategy path 'module:Class', e.g., strategies.ma_crossover:MACrossover")

    # Engine
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--point", type=float, default=1.0, help="Minimum price increment")
    ap.add_argument("--tick-value", type=float, default=1.0, help="Cash per one tick")
    ap.add_argument("--finalize", action="store_true", help="Force-close any open trade on the last bar")

    # Output and display
    ap.add_argument("--outdir", type=str, default=None, help="Output directory, defaults to CSV folder")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--mc-sims", type=int, default=1000,
                    help="Number of Monte Carlo simulations to run.")
    ap.add_argument("--mc-horizon", type=int, default=None,
                    help="Length of each Monte Carlo path; defaults to the historical sample length.")
    ap.add_argument("--mc-confidence", type=float, default=0.95,
                    help="Confidence level for Monte Carlo VaR/CVaR metrics.")
    ap.add_argument("--mc-seed", type=int, default=None,
                    help="Random seed for Monte Carlo bootstrapping.")
    ap.add_argument("--skip-mc", action="store_true",
                    help="Skip Monte Carlo risk analysis.")

    args = ap.parse_args()

    # Load data
    if args.loader == "csv_datetime":
        loader = CSVDateTimeLoader(path=args.csv, date_col=args.date_col, time_col=args.time_col,
                                   close_col=args.close_col, tz=args.tz)
    else:
        loader = CSVSingleDTLoader(path=args.csv, dt_col=args.dt_col, close_col=args.close_col, tz=args.tz)
    price: pd.Series = loader.load_close()
    market_data: pd.DataFrame = loader.load_dataframe()
    outdir = Path(args.outdir) if args.outdir is not None else DEFAULT_OUTPUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    # Load strategy that embeds its own constants
    strategy = load_strategy(args.strategy)
    outputs = strategy.generate(price, data=market_data)

    # Engine run
    mc_analysis = None

    cfg = EngineConfig(init_cash=args.cash, point=args.point, tick_value=args.tick_value,
                       finalize_trades=bool(args.finalize))
    results = run_backtest(price, outputs, cfg)

    # Save outputs
    eq = results["equity"]
    pos = results["position"]
    trades = results["trades"]
    exit_reasons = results.get("exit_reason")
    active_tp = results.get("active_take_profit")
    active_sl = results.get("active_stop_loss")

    eq.to_csv(outdir / "equity.csv", header=["equity"])
    pos.to_csv(outdir / "position.csv", header=["position"])
    trades.to_csv(outdir / "trades.csv", index=False)
    risk_levels = pd.DataFrame({
        "take_profit": active_tp,
        "stop_loss": active_sl,
        "exit_reason": exit_reasons,
    })
    risk_levels.to_csv(outdir / "risk_levels.csv")

    if not args.skip_mc:
        try:
            mc_analysis = monte_carlo_bootstrap_equity(
                eq,
                num_simulations=args.mc_sims,
                horizon=args.mc_horizon,
                seed=args.mc_seed,
                confidence_level=args.mc_confidence,
            )
        except ValueError as exc:
            print(f"Monte Carlo simulation skipped: {exc}")
        else:
            mc_path_df = pd.DataFrame({
                "step": mc_analysis.mean_equity_path.index,
                "mean_equity": mc_analysis.mean_equity_path.values,
                "lower_equity": mc_analysis.lower_equity_path.values,
                "upper_equity": mc_analysis.upper_equity_path.values,
            })
            mc_path_df.to_csv(outdir / "monte_carlo_paths.csv", index=False)
            mc_analysis.terminal_returns.to_csv(outdir / "monte_carlo_terminal_returns.csv",
                                                header=["terminal_return"])
            horizon_steps = mc_analysis.simulated_equity.shape[1] - 1
            mean_terminal_equity = float(mc_analysis.mean_equity_path.iloc[-1])
            var_pct = mc_analysis.var * 100.0
            cvar_pct = mc_analysis.cvar * 100.0
            expected_pct = mc_analysis.expected_return * 100.0
            print("Monte Carlo summary:")
            print(f"  simulations: {args.mc_sims}")
            print(f"  horizon: {horizon_steps} steps")
            print(f"  mean terminal equity: ${mean_terminal_equity:,.2f}")
            print(f"  expected terminal return: {expected_pct:.2f}%")
            print(f"  VaR {args.mc_confidence:.0%}: {var_pct:.2f}%")
            print(f"  CVaR {args.mc_confidence:.0%}: {cvar_pct:.2f}%")

    # Plot equity by trade number
    plot_equity_tradenum(
        equity=eq,
        trades=trades,
        initial_balance=args.cash,
        title=f"Equity - {args.strategy}",
        out_png=outdir / "equity.png",
        show=args.show,
        monte_carlo_mean=None if mc_analysis is None else mc_analysis.mean_equity_path,
        monte_carlo_lower=None if mc_analysis is None else mc_analysis.lower_equity_path,
        monte_carlo_upper=None if mc_analysis is None else mc_analysis.upper_equity_path,
        monte_carlo_confidence=None if mc_analysis is None else mc_analysis.confidence_level,
        monte_carlo_var=None if mc_analysis is None else mc_analysis.var,
        monte_carlo_cvar=None if mc_analysis is None else mc_analysis.cvar,
        monte_carlo_expected=None if mc_analysis is None else mc_analysis.expected_return,
    )

    print(f"Saved equity to {outdir / 'equity.csv'}")
    print(f"Saved trades to {outdir / 'trades.csv'}")
    print(f"Saved position to {outdir / 'position.csv'}")
    print(f"Saved risk levels to {outdir / 'risk_levels.csv'}")
    print(f"Saved equity plot to {outdir / 'equity.png'}")


if __name__ == "__main__":
    main()

