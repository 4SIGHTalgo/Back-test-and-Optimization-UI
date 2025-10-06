# Backtest & Optimization UI

## Overview
The Backtest & Optimization UI is a desktop application built with Tkinter that helps systematic traders iterate on trading ideas quickly. The tool combines data loading, modular strategy execution, Monte Carlo risk analysis, and grid-based parameter optimization into a single workflow-driven interface.

## UI Walkthrough
The interface is organized into two primary tabs:
- **Backtest tab** – Presents strategy inputs on the left (dataset selection, capital configuration, parameter overrides) and an equity chart with performance statistics on the right so you can review results immediately after execution. In addition to a Monte Carlo simulation to provide VaR, and CVaR risk metrics.

<img width="1913" height="1023" alt="Screenshot 2025-10-06 181324" src="https://github.com/user-attachments/assets/eba412d0-a6bf-4e7c-a7bc-36d6c26f3c7e" />

- **Optimization tab** – Mirrors the configuration controls while layering in progress indicators, best-run summaries, and a table of tested parameter combinations to help compare outcomes side-by-side.

<img width="1891" height="993" alt="Screenshot 2025-10-06 181438" src="https://github.com/user-attachments/assets/049180d3-590e-4937-97dd-333e3527bad9" />
  

## Current Capabilities
- **Data management** – Load market data from CSV files using either split date/time columns or unified datetime columns. The default datasets that ship with the project are available in the [`data/`](data/) directory for quick experimentation. Each CSV follows a tabular schema such as `Date,Timestamp,Open,High,Low,Close,Volume` (e.g., `20230922,10:00:00,14841.9,14858.4,14825.1,14835.9,1755`), making it straightforward to swap in alternative market histories that respect the same column order. The [`data/util/`](data/util/) helpers (`Convert_MT5_to_CSV.py` and `tz_convert_prices.py`) streamline converting MT5 exports and shifting datasets between timezones so they align with this schema.
- **Strategy catalogue** – Bundle reusable strategies inside the [`strategies/`](strategies/) package. Each strategy derives from `Strategy` and declares its own defaults, parameters, and signal generation logic. The UI auto-discovers available strategies and lets you configure overrides without editing code.
- **Backtesting engine** – Run discrete trade simulations with configurable cash, point value, tick value, and optional forced exits on the final bar. Results include trade logs, equity curves, active risk levels, and summary metrics displayed directly in the interface.
- **Monte Carlo analysis** – Stress test any backtest with bootstrap resampling to estimate value-at-risk (VaR), conditional VaR, expected returns, and confidence bands around the equity curve.
- **Parameter optimization** – Launch grid search experiments that iterate over user-defined parameter ranges, track progress, and highlight the best-performing configuration using a composite objective.

## Validation
The backtest engine has been validated against TradingView using a moving-average crossover strategy. Matching the strategy’s optimized parameters between platforms produced near-identical equity curves and trade statistics, demonstrating parity between the two environments. The following image of the equity curve is the same equity curve seen in our optimization tab. Slight differences in metrics can be attributed to data variance.

<img width="1640" height="470" alt="Screenshot 2025-10-06 182113" src="https://github.com/user-attachments/assets/2b062271-55b8-4f89-9f4d-8553539fec01" />


## Roadmap
Future iterations will extend the platform with:
- Additional optimization techniques such as Optuna-driven Bayesian optimization, evolutionary/genetic search algorithms, and configurable reward functions tailored to different risk preferences.
- A machine learning trainer that automatically selects classification or regression models based on the user’s target (e.g., price prediction vs. regime detection). The trainer will draw from a reusable feature bank of technical indicators, allow users to specify the prediction horizon (via an easy-to-use text input for *n* steps ahead), and expose UI toggles (e.g., checkboxes) for common targets and events.
- Expanded configurability for custom events, labels, and scoring metrics so power users can adapt the workflow to proprietary datasets or hybrid discretionary/systematic processes.

## Project Structure
```
Back-test-and-Optimization-UI/
├── backtest_engine.py
├── data/
│   ├── util/
│   │   ├── Convert_MT5_to_CSV.py
│   │   └── tz_convert_prices.py
│   └── *.csv
├── monte_carlo.py
├── optimization_engine.py
├── plot_equity.py
├── requirements.txt
├── run_backtest.py
├── strategy_ui.py
└── strategies/
    ├── __init__.py
    ├── strategy_base.py
    ├── ma_crossover.py
    ├── rsi_reversal.py
    └── breakout_atr.py
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Launch the UI: `python strategy_ui.py`
3. Select a CSV dataset, choose a strategy, configure parameters, and run the backtest or optimization workflow directly from the graphical interface.

For command-line usage, the `run_backtest.py` module exposes the same engine with CLI arguments for integration into automated research pipelines.
