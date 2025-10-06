# strategy_ui.py
from __future__ import annotations
import json
import os
import threading
import importlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any, Dict, Optional, get_args, get_origin

import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from data_loader import CSVDateTimeLoader, CSVSingleDTLoader
from backtest_engine import EngineConfig, run_backtest as run_engine
from plot_equity import plot_equity_tradenum
from monte_carlo import MonteCarloResult, monte_carlo_bootstrap_equity
from run_backtest import load_strategy, DEFAULT_OUTPUT_DIR
from strategies.strategy_base import Strategy

from optimization_engine import (
    OptimizationEngine,
    OptimizationOutcome,
    OptimizationProgress,
    OptimizationResult,
)

# Colors and style
GREEN = "#00FF00"
RED = "#FF0000"
GRAY = "#808080"
CYAN = "#00B7EB"
FIGURE_BG = "#000000"
AXIS_BG = "#0a0a0a"
LEGEND_FACE = "#1a1a1a"
LEGEND_EDGE = "#333333"


class BacktestUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Backtest & Optimization UI")
        self.geometry("720x720")
        self._running = False
        self.strategy_param_vars: Dict[str, tuple[tk.Variable, Any]] = {}
        self._canvas: Optional[tk.Canvas] = None
        self._canvas_frame_id: Optional[int] = None
        self._plot_canvas: Optional[FigureCanvasTkAgg] = None
        self._figure: Optional[Figure] = None
        self._ax = None
        self._equity_line = None
        self._mc_mean_line = None
        self._mc_band = None
        self._animation_job: Optional[str] = None
        self._animate_state: Optional[Dict[str, Any]] = None
        self._pending_mc_result: Optional[Any] = None
        self._equity_points: Optional[Dict[str, Any]] = None

        self._initial_balance: Optional[float] = None
        self._initial_line = None
        self._equity_fill_above = None
        self._equity_fill_below = None
        self._equity_scatter = None
        self._equity_annotation = None
        self._metrics_text = None

        self._notebook: Optional[ttk.Notebook] = None
        self._backtest_tab: Optional[ttk.Frame] = None
        self._optimization_tab: Optional[ttk.Frame] = None

        # Optimization tab state
        self.opt_param_vars: Dict[str, tuple[tk.BooleanVar, tk.StringVar, tk.StringVar, tk.StringVar, Any]] = {}
        self.opt_params_inner: Optional[ttk.Frame] = None
        self.opt_status_text: Optional[tk.Text] = None
        self.opt_progress_var = tk.DoubleVar(value=0.0)
        self.opt_progress_text_var = tk.StringVar(value="Progress: 0 / 0")
        self.opt_best_summary_var = tk.StringVar(value="Best: none")
        self.opt_run_button: Optional[ttk.Button] = None

        self.opt_w1_var = tk.DoubleVar(value=1.0)
        self.opt_w2_var = tk.DoubleVar(value=1.0)
        self.opt_w3_var = tk.DoubleVar(value=1.0)
        self.opt_mc_sims_var = tk.IntVar(value=1000)
        self.opt_mc_horizon_var = tk.StringVar(value="")
        self.opt_mc_confidence_var = tk.DoubleVar(value=0.95)
        self.opt_mc_seed_var = tk.StringVar(value="")

        self._opt_running = False
        self._opt_thread: Optional[threading.Thread] = None
        self._opt_results: list[OptimizationResult] = []
        self._opt_best_result: Optional[OptimizationResult] = None

        # Optimization plot
        self._opt_figure: Optional[Figure] = None
        self._opt_ax = None
        self._opt_plot_canvas: Optional[FigureCanvasTkAgg] = None
        self._opt_best_line = None
        self._opt_bounds: Optional[tuple[float, float]] = None
        self._opt_iterations_total = 0

        # Track last best so we only redraw when truly better
        self._opt_last_best_reward: Optional[float] = None

        self._build_widgets()

    def _build_widgets(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        main = ttk.Frame(canvas, padding=12)
        frame_id = canvas.create_window((0, 0), window=main, anchor="nw")

        self._canvas = canvas
        self._canvas_frame_id = frame_id

        main.bind("<Configure>", self._update_scrollregion)
        canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True)
        self._notebook = notebook

        backtest_tab = ttk.Frame(notebook)
        optimization_tab = ttk.Frame(notebook)
        notebook.add(backtest_tab, text="Backtest")
        notebook.add(optimization_tab, text="Optimization")
        self._backtest_tab = backtest_tab
        self._optimization_tab = optimization_tab

        self._build_backtest_tab(backtest_tab)
        self._build_optimization_tab(optimization_tab)

    def _build_backtest_tab(self, parent: ttk.Frame) -> None:
        # Data section
        data_frame = ttk.LabelFrame(parent, text="Data")
        data_frame.pack(fill="x", padx=4, pady=4)

        self.loader_var = tk.StringVar(value="csv_datetime")
        ttk.Label(data_frame, text="Loader").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            data_frame,
            textvariable=self.loader_var,
            values=["csv_datetime", "csv_single_dt"],
            width=18,
        ).grid(row=0, column=1, sticky="w")

        self.csv_var = tk.StringVar()
        ttk.Label(data_frame, text="CSV path").grid(row=1, column=0, sticky="w")
        ttk.Entry(data_frame, textvariable=self.csv_var, width=60).grid(row=1, column=1, columnspan=2, sticky="we")
        ttk.Button(data_frame, text="Browse", command=self._browse_csv).grid(row=1, column=3, sticky="e")

        self.date_col_var = tk.StringVar(value="Date")
        self.time_col_var = tk.StringVar(value="Timestamp")
        self.dt_col_var = tk.StringVar(value="datetime")
        self.close_col_var = tk.StringVar(value="Close")
        self.tz_var = tk.StringVar(value="")

        ttk.Label(data_frame, text="Date col").grid(row=2, column=0, sticky="w")
        ttk.Entry(data_frame, textvariable=self.date_col_var, width=15).grid(row=2, column=1, sticky="w")
        ttk.Label(data_frame, text="Time col").grid(row=2, column=2, sticky="w")
        ttk.Entry(data_frame, textvariable=self.time_col_var, width=15).grid(row=2, column=3, sticky="w")

        ttk.Label(data_frame, text="DT col").grid(row=3, column=0, sticky="w")
        ttk.Entry(data_frame, textvariable=self.dt_col_var, width=15).grid(row=3, column=1, sticky="w")
        ttk.Label(data_frame, text="Close col").grid(row=3, column=2, sticky="w")
        ttk.Entry(data_frame, textvariable=self.close_col_var, width=15).grid(row=3, column=3, sticky="w")

        ttk.Label(data_frame, text="Timezone").grid(row=4, column=0, sticky="w")
        ttk.Entry(data_frame, textvariable=self.tz_var, width=20).grid(row=4, column=1, sticky="w")

        # Strategy section
        strat_frame = ttk.LabelFrame(parent, text="Strategy")
        strat_frame.pack(fill="x", padx=4, pady=4)

        self.strategy_var = tk.StringVar(value="strategies.ma_crossover:MACrossover")
        self.strategies = sorted(self._discover_strategies())
        ttk.Label(strat_frame, text="Strategy path").grid(row=0, column=0, sticky="w")
        ttk.Combobox(strat_frame, textvariable=self.strategy_var, values=self.strategies, width=50, state="readonly").grid(row=0, column=1, sticky="we")
        ttk.Button(strat_frame, text="Load defaults", command=self._load_strategy_defaults).grid(row=0, column=2, padx=4, sticky="e")

        self.kwargs_var = tk.StringVar(value="{}")
        ttk.Label(strat_frame, text="Extra kwargs (JSON)").grid(row=1, column=0, sticky="nw")
        ttk.Entry(strat_frame, textvariable=self.kwargs_var, width=60).grid(row=1, column=1, columnspan=2, sticky="we")

        self.params_frame = ttk.LabelFrame(parent, text="Strategy Parameters")
        self.params_frame.pack(fill="x", padx=4, pady=4)
        self.params_inner = ttk.Frame(self.params_frame)
        self.params_inner.pack(fill="x", padx=4, pady=4)

        # Engine section
        engine_frame = ttk.LabelFrame(parent, text="Engine")
        engine_frame.pack(fill="x", padx=4, pady=4)

        self.cash_var = tk.DoubleVar(value=100000.0)
        self.point_var = tk.DoubleVar(value=1.0)
        self.tick_value_var = tk.DoubleVar(value=1.0)
        self.finalize_var = tk.BooleanVar(value=True)

        ttk.Label(engine_frame, text="Initial cash").grid(row=0, column=0, sticky="w")
        ttk.Entry(engine_frame, textvariable=self.cash_var, width=15).grid(row=0, column=1, sticky="w")
        ttk.Label(engine_frame, text="Point").grid(row=0, column=2, sticky="w")
        ttk.Entry(engine_frame, textvariable=self.point_var, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(engine_frame, text="Tick value").grid(row=0, column=4, sticky="w")
        ttk.Entry(engine_frame, textvariable=self.tick_value_var, width=10).grid(row=0, column=5, sticky="w")
        ttk.Checkbutton(engine_frame, text="Finalize open trade", variable=self.finalize_var).grid(row=1, column=0, columnspan=2, sticky="w")

        # Monte Carlo section
        mc_frame = ttk.LabelFrame(parent, text="Monte Carlo")
        mc_frame.pack(fill="x", padx=4, pady=4)

        self.skip_mc_var = tk.BooleanVar(value=False)
        self.mc_sims_var = tk.IntVar(value=1000)
        self.mc_horizon_var = tk.StringVar(value="")
        self.mc_confidence_var = tk.DoubleVar(value=0.95)
        self.mc_seed_var = tk.StringVar(value="")

        ttk.Checkbutton(mc_frame, text="Skip Monte Carlo", variable=self.skip_mc_var).grid(row=0, column=0, sticky="w")
        ttk.Label(mc_frame, text="Simulations").grid(row=1, column=0, sticky="w")
        ttk.Entry(mc_frame, textvariable=self.mc_sims_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(mc_frame, text="Horizon").grid(row=1, column=2, sticky="w")
        ttk.Entry(mc_frame, textvariable=self.mc_horizon_var, width=10).grid(row=1, column=3, sticky="w")
        ttk.Label(mc_frame, text="Confidence").grid(row=1, column=4, sticky="w")
        ttk.Entry(mc_frame, textvariable=self.mc_confidence_var, width=10).grid(row=1, column=5, sticky="w")
        ttk.Label(mc_frame, text="Seed").grid(row=1, column=6, sticky="w")
        ttk.Entry(mc_frame, textvariable=self.mc_seed_var, width=10).grid(row=1, column=7, sticky="w")

        # Output section
        out_frame = ttk.LabelFrame(parent, text="Output")
        out_frame.pack(fill="x", padx=4, pady=4)

        self.outdir_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        ttk.Label(out_frame, text="Output directory").grid(row=0, column=0, sticky="w")
        ttk.Entry(out_frame, textvariable=self.outdir_var, width=60).grid(row=0, column=1, sticky="we")
        ttk.Button(out_frame, text="Browse", command=self._browse_outdir).grid(row=0, column=2, sticky="e")
        self.show_plot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(out_frame, text="Open plot after run", variable=self.show_plot_var).grid(row=1, column=0, sticky="w")

        # Controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=6)
        self.run_button = ttk.Button(control_frame, text="Run Backtest", command=self._start_backtest)
        self.run_button.pack(side="left")

        # Equity plot + status (resizable)
        plot_status = ttk.PanedWindow(parent, orient="vertical")
        plot_status.pack(fill="both", expand=True, padx=4, pady=4)

        plot_container = ttk.LabelFrame(plot_status, text="Equity (Live)")
        status_container = ttk.LabelFrame(plot_status, text="Status")
        plot_status.add(plot_container, weight=3)
        plot_status.add(status_container, weight=2)

        self._figure = Figure(figsize=(6, 3), dpi=100)
        self._ax = self._figure.add_subplot(111)
        self._apply_plot_style()
        self._equity_line, = self._ax.plot([], [], color=GREEN, linewidth=2, label="Equity", zorder=3)
        self._mc_mean_line, = self._ax.plot([], [], color=CYAN, linewidth=1.8, linestyle="--", label="MC mean", zorder=2)
        self._plot_canvas = FigureCanvasTkAgg(self._figure, master=plot_container)
        self._plot_canvas.draw()
        self._plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.status_text = tk.Text(status_container, height=10, state="disabled")
        self.status_text.pack(fill="both", expand=True)

    def _build_optimization_tab(self, parent: ttk.Frame) -> None:
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill="x", padx=4, pady=4)
        ttk.Label(
            info_frame,
            text="Optimization reuses the data and engine settings configured on the Backtest tab.",
        ).pack(anchor="w")

        strategy_frame = ttk.LabelFrame(parent, text="Strategy")
        strategy_frame.pack(fill="x", padx=4, pady=4)
        ttk.Label(strategy_frame, text="Strategy path").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            strategy_frame,
            textvariable=self.strategy_var,
            values=self.strategies,
            width=50,
            state="readonly",
        ).grid(row=0, column=1, sticky="we")
        ttk.Button(
            strategy_frame,
            text="Load defaults",
            command=self._load_opt_strategy_defaults,
        ).grid(row=0, column=2, padx=4, sticky="e")
        strategy_frame.columnconfigure(1, weight=1)
        ttk.Label(
            strategy_frame,
            text="Provide value lists for parameters you want to optimize. Leave blank to keep the base value.",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        params_frame = ttk.LabelFrame(parent, text="Parameter Grid")
        params_frame.pack(fill="x", padx=4, pady=4)
        header = ttk.Frame(params_frame)
        header.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(header, text="Optimize", width=10).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Parameter", width=18).grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="Start").grid(row=0, column=2, sticky="w")
        ttk.Label(header, text="End").grid(row=0, column=3, sticky="w")
        ttk.Label(header, text="Step").grid(row=0, column=4, sticky="w")
        self.opt_params_inner = ttk.Frame(params_frame)
        self.opt_params_inner.pack(fill="x", padx=4, pady=4)
        tools = ttk.Frame(params_frame)
        tools.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(tools, text="Clear values", command=self._clear_opt_params).pack(side="right")

        settings_frame = ttk.LabelFrame(parent, text="Optimization Settings")
        settings_frame.pack(fill="x", padx=4, pady=4)
        ttk.Label(settings_frame, text="K-Ratio weight (w1)").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.opt_w1_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(settings_frame, text="1/CVaR weight (w2)").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.opt_w2_var, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(settings_frame, text="Expected return weight (w3)").grid(row=0, column=4, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.opt_w3_var, width=10).grid(row=0, column=5, sticky="w")

        ttk.Label(settings_frame, text="MC simulations").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(settings_frame, textvariable=self.opt_mc_sims_var, width=10).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Label(settings_frame, text="MC horizon").grid(row=1, column=2, sticky="w", pady=(4, 0))
        ttk.Entry(settings_frame, textvariable=self.opt_mc_horizon_var, width=10).grid(row=1, column=3, sticky="w", pady=(4, 0))
        ttk.Label(settings_frame, text="MC confidence").grid(row=1, column=4, sticky="w", pady=(4, 0))
        ttk.Entry(settings_frame, textvariable=self.opt_mc_confidence_var, width=10).grid(row=1, column=5, sticky="w", pady=(4, 0))
        ttk.Label(settings_frame, text="MC seed").grid(row=1, column=6, sticky="w", pady=(4, 0))
        ttk.Entry(settings_frame, textvariable=self.opt_mc_seed_var, width=12).grid(row=1, column=7, sticky="w", pady=(4, 0))

        controls_frame = ttk.Frame(settings_frame)
        controls_frame.grid(row=2, column=0, columnspan=8, sticky="we", pady=(8, 0))
        self.opt_run_button = ttk.Button(controls_frame, text="Run Optimization", command=self._start_optimization)
        self.opt_run_button.pack(side="left")
        ttk.Label(controls_frame, textvariable=self.opt_progress_text_var).pack(side="left", padx=8)
        self.opt_progress_bar = ttk.Progressbar(controls_frame, variable=self.opt_progress_var, maximum=1.0)
        self.opt_progress_bar.pack(side="left", fill="x", expand=True, padx=(8, 0))
        settings_frame.columnconfigure(5, weight=1)
        settings_frame.columnconfigure(7, weight=1)

        best_frame = ttk.Frame(parent)
        best_frame.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(best_frame, textvariable=self.opt_best_summary_var, anchor="w").pack(fill="x")

        plot_status = ttk.PanedWindow(parent, orient="vertical")
        plot_status.pack(fill="both", expand=True, padx=4, pady=4)

        plot_container = ttk.LabelFrame(plot_status, text="Optimization Equity")
        status_container = ttk.LabelFrame(plot_status, text="Status")
        plot_status.add(plot_container, weight=3)
        plot_status.add(status_container, weight=2)

        self._opt_figure = Figure(figsize=(6, 3), dpi=100)
        self._opt_ax = self._opt_figure.add_subplot(111)
        self._style_plot(self._opt_figure, self._opt_ax, "Optimization Equity")
        self._opt_plot_canvas = FigureCanvasTkAgg(self._opt_figure, master=plot_container)
        self._opt_plot_canvas.draw()
        self._opt_plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.opt_status_text = tk.Text(status_container, height=10, state="disabled")
        self.opt_status_text.pack(fill="both", expand=True)

    # ---------- Optimization helpers ----------

    def _clear_opt_params(self) -> None:
        if self.opt_params_inner is not None:
            for widget in self.opt_params_inner.winfo_children():
                widget.destroy()
        self.opt_param_vars.clear()

    def _load_opt_strategy_defaults(self) -> None:
        strategy_path = self.strategy_var.get().strip()
        if not strategy_path:
            messagebox.showerror("Strategy", "Strategy path is required")
            return
        try:
            strategy = load_strategy(strategy_path)
        except Exception as exc:
            messagebox.showerror("Strategy", f"Could not load strategy: {exc}")
            return
        defaults = getattr(strategy, "DEFAULTS", None)
        if not isinstance(defaults, dict):
            messagebox.showinfo(
                "Strategy",
                "Strategy does not expose DEFAULTS dict, add grid entries manually.",
            )
            return
        annotations = getattr(getattr(strategy, "config", None).__class__, "__annotations__", {})
        if self.opt_params_inner is None:
            return
        self._clear_opt_params()

        for row, (key, value) in enumerate(sorted(defaults.items())):
            resolved_type = self._resolve_param_type(annotations.get(key), type(value))
            row_frame = ttk.Frame(self.opt_params_inner)
            row_frame.grid(row=row, column=0, sticky="we", pady=2)

            enabled_var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(row_frame, variable=enabled_var)
            chk.grid(row=0, column=0, padx=(0, 6))

            ttk.Label(row_frame, text=key, width=18).grid(row=0, column=1, sticky="w")

            start_var = tk.StringVar(value="" if value is None else str(value))
            end_var = tk.StringVar(value="" if value is None else str(value))
            step_default = "1" if isinstance(value, (int, np.integer)) else "0.1" if isinstance(value, (float, np.floating)) else "1"
            step_var = tk.StringVar(value=step_default)

            ttk.Entry(row_frame, textvariable=start_var, width=10).grid(row=0, column=2, padx=4)
            ttk.Entry(row_frame, textvariable=end_var, width=10).grid(row=0, column=3, padx=4)
            ttk.Entry(row_frame, textvariable=step_var, width=8).grid(row=0, column=4, padx=4)

            row_frame.columnconfigure(4, weight=1)

            is_numeric = (
                resolved_type in (int, float, np.integer, np.floating)
                or isinstance(value, (int, float, np.integer, np.floating))
            )
            if not is_numeric:
                chk.configure(state="disabled")

            self.opt_param_vars[key] = (enabled_var, start_var, end_var, step_var, resolved_type)

    def _append_opt_status(self, message: str) -> None:
        if self.opt_status_text is None:
            return
        self.opt_status_text.configure(state="normal")
        self.opt_status_text.insert("end", message + "\n")
        self.opt_status_text.configure(state="disabled")
        self.opt_status_text.see("end")

    def _append_opt_status_async(self, message: str) -> None:
        self.after(0, lambda msg=message: self._append_opt_status(msg))

    def _collect_opt_param_grid(self) -> Dict[str, list[Any]]:
        grid: Dict[str, list[Any]] = {}
        for key, (enabled_var, start_var, end_var, step_var, value_type) in self.opt_param_vars.items():
            if not enabled_var.get():
                continue
            start_text = start_var.get().strip()
            end_text = end_var.get().strip()
            step_text = step_var.get().strip()
            if not start_text or not end_text or not step_text:
                raise ValueError(f"{key}: start, end, and step must be provided")
            try:
                start_value = self._parse_param_value(start_text, value_type)
                end_value = self._parse_param_value(end_text, value_type)
                step_value = self._parse_param_value(step_text, value_type)
                values = self._generate_param_range(start_value, end_value, step_value, value_type)
            except ValueError as exc:
                raise ValueError(f"{key}: {exc}") from exc
            if not values:
                raise ValueError(f"{key}: generated range is empty")
            grid[key] = values
        return grid

    def _generate_param_range(self, start: Any, end: Any, step: Any, value_type: Any) -> list[Any]:
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("Boolean parameters are not supported for optimization ranges, enter explicit values.")
        if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)):
            step_val = int(step)
            if step_val == 0:
                raise ValueError("step must be non-zero")
            if start < end and step_val < 0:
                step_val = abs(step_val)
            if start > end and step_val > 0:
                step_val = -step_val
            values: list[int] = []
            current = int(start)
            if step_val > 0:
                while current <= int(end):
                    values.append(current)
                    current += step_val
            else:
                while current >= int(end):
                    values.append(current)
                    current += step_val
            return values

        try:
            start_float = float(start)
            end_float = float(end)
            step_float = float(step)
        except (TypeError, ValueError):
            raise ValueError("start, end, and step must be numeric values")
        if step_float == 0:
            raise ValueError("step must be non-zero")
        if start_float < end_float and step_float < 0:
            step_float = abs(step_float)
        if start_float > end_float and step_float > 0:
            step_float = -step_float
        values: list[float] = []
        current = start_float
        tolerance = abs(step_float) * 1e-9 + 1e-9
        if step_float > 0:
            while current <= end_float + tolerance:
                values.append(float(current))
                current += step_float
        else:
            while current >= end_float - tolerance:
                values.append(float(current))
                current += step_float
        return values

    def _collect_optimization_config(self) -> Dict[str, Any]:
        base_cfg = self._collect_config()
        base_cfg["skip_mc"] = False  # MC used only for metrics during optimization
        param_grid = self._collect_opt_param_grid()
        if not param_grid:
            raise ValueError("Parameter grid is empty. Add at least one parameter with values.")
        for param_name in list(param_grid.keys()):
            if param_name in base_cfg["strategy_kwargs"]:
                del base_cfg["strategy_kwargs"][param_name]
        weights = (
            float(self.opt_w1_var.get()),
            float(self.opt_w2_var.get()),
            float(self.opt_w3_var.get()),
        )
        sims = max(1, int(self.opt_mc_sims_var.get()))
        horizon_text = self.opt_mc_horizon_var.get().strip()
        horizon = int(horizon_text) if horizon_text else None
        confidence = float(self.opt_mc_confidence_var.get())
        confidence = min(max(confidence, 0.0), 1.0)
        seed_text = self.opt_mc_seed_var.get().strip()
        seed = int(seed_text) if seed_text else None
        return {
            "base": base_cfg,
            "param_grid": param_grid,
            "weights": weights,
            "mc": {
                "sims": sims,
                "horizon": horizon,
                "confidence": confidence,
                "seed": seed,
            },
        }

    def _start_optimization(self) -> None:
        if self._opt_running:
            return
        try:
            config = self._collect_optimization_config()
        except ValueError as exc:
            messagebox.showerror("Optimization", str(exc))
            return
        param_grid = config["param_grid"]
        total = 1
        for values in param_grid.values():
            total *= len(values)
        if total <= 0:
            messagebox.showerror("Optimization", "No parameter combinations to evaluate.")
            return
        self._opt_iterations_total = total
        self.opt_progress_var.set(0.0)
        self.opt_progress_text_var.set(f"Progress: 0 / {total}")
        self.opt_best_summary_var.set("Best: none")
        self._reset_opt_plot()
        self._opt_results = []
        self._opt_best_result = None
        self._opt_last_best_reward = None
        self._opt_running = True
        if self.opt_run_button is not None:
            self.opt_run_button.config(state="disabled")
        self._append_opt_status("Starting optimization...")
        thread = threading.Thread(target=self._run_optimization_safe, args=(config,), daemon=True)
        self._opt_thread = thread
        thread.start()

    def _reset_opt_plot(self) -> None:
        if self._opt_ax is None or self._opt_plot_canvas is None or self._opt_figure is None:
            return
        if self._opt_best_line is not None:
            self._remove_artist(self._opt_best_line)
            self._opt_best_line = None
        self._opt_bounds = None
        self._opt_ax.clear()
        self._style_plot(self._opt_figure, self._opt_ax, "Optimization Equity")
        leg = self._opt_ax.get_legend()
        if leg is not None:
            leg.remove()
        self._opt_plot_canvas.draw_idle()

    def _queue_opt_update(
        self,
        progress: OptimizationProgress,
        result: OptimizationResult,
        best: Optional[OptimizationResult],
    ) -> None:
        self.after(0, lambda p=progress, r=result, b=best: self._apply_opt_iteration_update(p, r, b))

    def _apply_opt_iteration_update(
        self,
        progress: OptimizationProgress,
        result: OptimizationResult,
        best: Optional[OptimizationResult],
    ) -> None:
        if not self._opt_running:
            return
        self._opt_results.append(result)
        if best is not None:
            # Record the best and draw only when it improved
            self._opt_best_result = best
            self._update_opt_plot_with_best(best)

        # Status / metrics text
        param_parts = []
        for key in self.opt_param_vars.keys():
            if key in result.params:
                param_parts.append(f"{key}={result.params[key]}")
        metrics = (
            f"[{progress.completed}/{progress.total}] "
            f"Reward={self._format_metric(result.reward)} "
            f"K={self._format_metric(result.k_ratio)} "
            f"1/CVaR={self._format_metric(result.inverse_cvar)} "
            f"ExpRet={self._format_metric(result.expected_return)}"
        )
        if param_parts:
            metrics = metrics + " | " + ", ".join(param_parts)
        self._append_opt_status(metrics)
        self._update_opt_progress(progress)
        if best is not None:
            self._update_opt_best_summary(best)
            # Keep the Backtest tab in-sync so re-runs match optimization
            self._apply_best_params_to_backtest(best)

    def _update_opt_plot_with_best(self, best: OptimizationResult) -> None:
        """Draw only the best equity curve. Replace any prior best when reward improves."""
        if self._opt_ax is None or self._opt_plot_canvas is None or best is None:
            return
        try:
            reward = float(best.reward)
        except Exception:
            reward = None

        # Redraw only if improved (with tiny epsilon for float noise)
        if self._opt_last_best_reward is not None and reward is not None:
            if reward <= self._opt_last_best_reward + 1e-12:
                return
        self._opt_last_best_reward = reward

        points = self._build_equity_points(best.equity, best.trades, best.initial_balance)
        numbers = np.asarray(points.get("numbers"), dtype=float)
        values = np.asarray(points.get("values"), dtype=float)
        if numbers.size == 0 or values.size == 0:
            return

        # Clear axis and previous best
        self._opt_ax.clear()
        self._style_plot(self._opt_figure, self._opt_ax, "Optimization Equity")
        if self._opt_best_line is not None:
            self._remove_artist(self._opt_best_line)
            self._opt_best_line = None

        # Plot the best equity curve
        self._opt_best_line, = self._opt_ax.plot(
            numbers,
            values,
            color=GREEN,
            linewidth=2.2,
            alpha=1.0,
            label="Best equity",
            zorder=5,
        )

        # Update bounds/ticks/legend
        self._update_opt_axes_bounds(numbers, values)
        self._apply_opt_ticks(points, axis=self._opt_ax, numbers=numbers)
        self._update_opt_legend()
        self._opt_plot_canvas.draw_idle()

    def _apply_best_params_to_backtest(self, best: OptimizationResult) -> None:
        """Write best params into Backtest tab widgets and clean JSON kwargs to avoid overrides."""
        if best is None or not best.params:
            return
        # Update widget rows
        for key, (var, value_type) in self.strategy_param_vars.items():
            if key not in best.params:
                continue
            val = best.params[key]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(val))
            else:
                var.set("" if val is None else str(val))

        # Remove duplicated keys from the JSON kwargs so UI values prevail
        try:
            current_json = self.kwargs_var.get() or "{}"
            kwargs = json.loads(current_json)
            if isinstance(kwargs, dict):
                changed = False
                for k in list(kwargs.keys()):
                    if k in best.params:
                        kwargs.pop(k, None)
                        changed = True
                if changed:
                    self.kwargs_var.set(json.dumps(kwargs))
        except Exception:
            # If JSON is invalid, leave it—widgets already hold correct params
            pass

    def _update_opt_axes_bounds(self, x_vals: np.ndarray, y_vals: np.ndarray) -> None:
        if self._opt_ax is None or x_vals.size == 0 or y_vals.size == 0:
            return
        min_y = float(np.min(y_vals))
        max_y = float(np.max(y_vals))
        if np.isclose(min_y, max_y):
            pad = max(1.0, abs(min_y) * 0.05 or 1.0)
            min_y -= pad
            max_y += pad
        else:
            pad = (max_y - min_y) * 0.05
            min_y -= pad
            max_y += pad
        self._opt_bounds = (min_y, max_y)
        x_min = float(x_vals[0])
        x_max = float(x_vals[-1]) if x_vals.size else x_min + 1.0
        if x_min == x_max:
            x_max = x_min + 1.0
        self._opt_ax.set_xlim(x_min, x_max)
        self._opt_ax.set_ylim(min_y, max_y)

    def _apply_opt_ticks(self, points: Dict[str, Any], axis, numbers: Optional[np.ndarray] = None) -> None:
        if axis is None:
            return
        if numbers is None:
            raw_numbers = points.get("numbers")
            numbers = np.asarray(raw_numbers, dtype=float) if raw_numbers is not None else np.asarray([], dtype=float)
        months = points.get("months", [])
        trade_count = int(points.get("trade_count", max(len(numbers) - 1, 0)))
        if trade_count <= 0:
            axis.set_xticks([])
            return
        if trade_count <= 20:
            ticks = list(range(0, trade_count + 1))
        else:
            step = max(1, trade_count // 15)
            ticks = list(range(0, trade_count + 1, step))
            if ticks[-1] != trade_count:
                ticks.append(trade_count)
        labels = [f"{pos}\n{months[min(pos, len(months) - 1)] if months else ''}" for pos in ticks]
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels, fontsize=9, color="white")

    def _update_opt_legend(self) -> None:
        if self._opt_ax is None:
            return
        handles, labels = self._opt_ax.get_legend_handles_labels()
        dedup: Dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if not label or label.startswith("_"):
                continue
            dedup[label] = handle
        legend = self._opt_ax.get_legend()
        if not dedup:
            if legend is not None:
                legend.remove()
            return
        legend = self._opt_ax.legend(
            list(dedup.values()),
            list(dedup.keys()),
            loc="upper left",
            framealpha=0.9,
            facecolor=LEGEND_FACE,
            edgecolor=LEGEND_EDGE,
        )
        for text in legend.get_texts():
            text.set_color("white")

    def _update_opt_progress(self, progress: OptimizationProgress) -> None:
        total = max(progress.total, 1)
        self.opt_progress_var.set(progress.completed / total)
        self.opt_progress_text_var.set(f"Progress: {progress.completed} / {progress.total}")

    def _update_opt_best_summary(self, best: OptimizationResult) -> None:
        summary_lines = [
            f"Best reward: {self._format_metric(best.reward)}",
            f"K-Ratio: {self._format_metric(best.k_ratio)}",
            f"1/CVaR: {self._format_metric(best.inverse_cvar)}",
            f"Exp. return: {self._format_metric(best.expected_return)}",
        ]
        points = self._build_equity_points(best.equity, best.trades, best.initial_balance)
        trade_count = int(points.get("trade_count", max(len(points.get("numbers", [])) - 1, 0)))
        summary_lines.append(f"Trades: {trade_count}")
        param_parts: list[str] = []
        for key in self.opt_param_vars.keys():
            if key in best.params:
                param_parts.append(f"{key}={best.params[key]}")
        if param_parts:
            summary_lines.append("Params: " + ", ".join(param_parts))
        self.opt_best_summary_var.set("\n".join(summary_lines))

    def _format_metric(self, value: float) -> str:
        try:
            val = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(val):
            if val > 0:
                return "inf"
            if val < 0:
                return "-inf"
            return "nan"
        return f"{val:.4f}"

    def _run_optimization_safe(self, config: Dict[str, Any]) -> None:
        try:
            self._append_opt_status_async("Running optimization...")
            outcome = self._execute_optimization(config)
        except Exception as exc:
            self._append_opt_status_async(f"Error: {exc}")
            self.after(0, lambda e=exc: self._finalize_optimization_error(e))
        else:
            self.after(0, lambda o=outcome: self._finalize_optimization_success(o))

    def _execute_optimization(self, config: Dict[str, Any]) -> OptimizationOutcome:
        base_cfg = config["base"]
        loader = self._create_loader(base_cfg)
        price = loader.load_close()
        market_data = loader.load_dataframe()
        engine_cfg = EngineConfig(
            init_cash=base_cfg["cash"],
            point=base_cfg["point"],
            tick_value=base_cfg["tick_value"],
            finalize_trades=base_cfg["finalize"],
        )
        engine = OptimizationEngine(
            price=price,
            market_data=market_data,
            strategy_path=base_cfg["strategy"],
            base_params=base_cfg["strategy_kwargs"],
            param_grid=config["param_grid"],
            engine_config=engine_cfg,
            weights=config["weights"],
            mc_sims=config["mc"]["sims"],
            mc_horizon=config["mc"]["horizon"],
            mc_confidence=config["mc"]["confidence"],
            mc_seed=config["mc"]["seed"],
        )

        def progress_callback(
            progress: OptimizationProgress,
            result: OptimizationResult,
            best: Optional[OptimizationResult],
        ) -> None:
            self._queue_opt_update(progress, result, best)

        return engine.run(callback=progress_callback)

    def _finalize_optimization_success(self, outcome: OptimizationOutcome) -> None:
        self._opt_running = False
        if self.opt_run_button is not None:
            self.opt_run_button.config(state="normal")
        self.opt_progress_var.set(1.0 if self._opt_iterations_total else 0.0)
        self.opt_progress_text_var.set(
            f"Progress: {self._opt_iterations_total} / {self._opt_iterations_total}"
            if self._opt_iterations_total
            else "Progress: 0 / 0"
        )
        self._append_opt_status("Optimization completed.")
        best = outcome.best or self._opt_best_result
        if best is not None:
            # Ensure best params are synced once more at the end
            self._apply_best_params_to_backtest(best)
            self._update_opt_best_summary(best)
            messagebox.showinfo(
                "Optimization",
                f"Optimization completed.\nBest reward: {self._format_metric(best.reward)}",
            )
        else:
            messagebox.showinfo("Optimization", "Optimization completed without valid results.")

    def _finalize_optimization_error(self, error: Exception) -> None:
        self._opt_running = False
        if self.opt_run_button is not None:
            self.opt_run_button.config(state="normal")
        self.opt_progress_var.set(0.0)
        self.opt_progress_text_var.set("Progress: 0 / 0")
        messagebox.showerror("Optimization error", str(error))

    # ---------- Canvas and UI utils ----------

    def _update_scrollregion(self, _event: tk.Event) -> None:
        if self._canvas is None:
            return
        bbox = self._canvas.bbox("all")
        if bbox is not None:
            self._canvas.configure(scrollregion=bbox)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        if self._canvas is None or self._canvas_frame_id is None:
            return
        self._canvas.itemconfigure(self._canvas_frame_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self._canvas is None:
            return
        if hasattr(event, "delta") and event.delta:
            self._canvas.yview_scroll(int(-event.delta / 120), "units")
        elif getattr(event, "num", None) in (4, 5):
            direction = -1 if event.num == 4 else 1
            self._canvas.yview_scroll(direction, "units")

    def _apply_plot_style(self) -> None:
        if self._figure is None or self._ax is None:
            return
        self._style_plot(self._figure, self._ax, "Equity (Live)")

    @staticmethod
    def _style_plot(figure: Figure, axis, title: str) -> None:
        figure.set_facecolor(FIGURE_BG)
        axis.set_facecolor(AXIS_BG)
        axis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        axis.set_title(title, fontsize=14, color="white", pad=15)
        axis.set_xlabel("Trade Number", fontsize=12, color="white")
        axis.set_ylabel("Equity ($)", fontsize=12, color="white")
        axis.tick_params(colors="white")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _=None: f"${x:,.0f}"))
        try:
            figure.tight_layout()
        except Exception:
            pass

    def _remove_artist(self, artist) -> None:
        if artist is None:
            return
        try:
            artist.remove()
        except Exception:
            pass

    def _update_legend(self) -> None:
        if self._ax is None:
            return
        handles, labels = self._ax.get_legend_handles_labels()
        dedup = {}
        for handle, label in zip(handles, labels):
            if not label or label.startswith("_"):
                continue
            dedup[label] = handle
        legend = self._ax.get_legend()
        if not dedup:
            if legend is not None:
                legend.remove()
            return
        legend = self._ax.legend(
            list(dedup.values()),
            list(dedup.keys()),
            loc="lower left",
            framealpha=0.9,
            facecolor=LEGEND_FACE,
            edgecolor=LEGEND_EDGE,
        )
        for text_item in legend.get_texts():
            text_item.set_color("white")

    # ---------- Equity animation and MC ----------

    def _build_equity_points(self, equity: pd.Series, trades: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
        points_numbers: list[int] = [0]
        points_values: list[float] = [float(initial_balance)]
        months: list[str] = []
        steps: list[int] = [0]
        if not equity.empty:
            first_month = pd.to_datetime(equity.index[0]).strftime("%b")
        else:
            first_month = ""
        months.append(first_month)
        trades_sorted = trades.dropna(subset=["exit_time"]).copy()
        trades_sorted["exit_time"] = pd.to_datetime(trades_sorted["exit_time"])
        trades_sorted = trades_sorted.sort_values("exit_time")
        for _, trade in enumerate(trades_sorted.itertuples(), 1):
            exit_time = trade.exit_time
            idx = equity.index[equity.index <= exit_time]
            if len(idx) == 0:
                continue
            exit_index = idx[-1]
            val = float(equity.loc[exit_index])
            points_numbers.append(len(points_numbers))
            points_values.append(val)
            position = equity.index.get_indexer([exit_index])
            if position.size == 0 or position[0] == -1:
                step_position = len(equity) - 1
            else:
                step_position = int(position[0])
            steps.append(step_position)
            months.append(exit_time.strftime("%b"))
        if len(points_numbers) == 1:
            points_numbers.append(1)
            points_values.append(float(initial_balance))
            steps.append(len(equity) - 1 if len(equity) else 1)
            months.append(first_month)
        numbers_array = np.asarray(points_numbers, dtype=float)
        values_array = np.asarray(points_values, dtype=float)
        step_index = pd.Index(steps, name="step")
        return {
            "numbers": numbers_array,
            "values": values_array,
            "months": months,
            "steps": steps,
            "step_index": step_index,
            "initial_balance": float(initial_balance),
            "trade_count": int(points_numbers[-1]),
        }

    def _cancel_equity_animation(self) -> None:
        if self._animation_job is not None:
            try:
                self.after_cancel(self._animation_job)
            except Exception:
                pass
            self._animation_job = None

    def _prepare_live_plot(self) -> None:
        self._cancel_equity_animation()
        self._animate_state = None
        self._pending_mc_result = None
        self._equity_points = None
        self._initial_balance = None
        for attr_name in ("_equity_fill_above", "_equity_fill_below", "_equity_scatter", "_equity_annotation"):
            self._remove_artist(getattr(self, attr_name, None))
            setattr(self, attr_name, None)
        self._remove_artist(self._initial_line)
        self._initial_line = None
        self._remove_artist(self._mc_band)
        self._mc_band = None
        if self._equity_line is not None:
            self._equity_line.set_data([], [])
            self._equity_line.set_color(GREEN)
            self._equity_line.set_linewidth(2)
            self._equity_line.set_zorder(3)
        if self._mc_mean_line is not None:
            self._mc_mean_line.set_data([], [])
            self._mc_mean_line.set_color(CYAN)
            self._mc_mean_line.set_linestyle("--")
            self._mc_mean_line.set_linewidth(1.8)
            self._mc_mean_line.set_zorder(2)
        if self._ax is not None:
            legend = self._ax.get_legend()
            if legend is not None:
                legend.remove()
            self._apply_plot_style()
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    def _animate_equity(self, points: Dict[str, Any], mc_result: Optional[MonteCarloResult]) -> None:
        if points is None or self._plot_canvas is None or self._ax is None or self._equity_line is None:
            return
        numbers = points.get("numbers")
        values = points.get("values")
        if numbers is None or values is None or len(numbers) == 0:
            return
        numbers = np.asarray(numbers, dtype=float)
        values = np.asarray(values, dtype=float)
        self._equity_points = points
        self._cancel_equity_animation()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        self._initial_balance = float(points.get("initial_balance", float(finite[0])))
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if np.isclose(y_min, y_max):
            pad = max(1.0, abs(y_min) * 0.05 or 1.0)
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.05
            y_min -= pad
            y_max += pad
        x_min = float(numbers[0])
        x_max = float(numbers[-1]) if len(numbers) > 1 else float(numbers[0] + 1)
        self._ax.set_xlim(x_min, x_max)
        self._ax.set_ylim(y_min, y_max)
        self._equity_line.set_data([numbers[0]], [values[0]])
        if self._mc_mean_line is not None:
            self._mc_mean_line.set_data([], [])
        self._remove_artist(self._mc_band)
        self._mc_band = None
        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()
        chunk = max(1, len(numbers) // 50)
        self._animate_state = {"x": numbers, "y": values, "index": 1, "chunk": chunk}
        self._pending_mc_result = mc_result
        self._update_legend()
        self._schedule_equity_step()

    def _schedule_equity_step(self) -> None:
        if self._animate_state is None or self._plot_canvas is None:
            return
        x_vals = self._animate_state["x"]
        y_vals = self._animate_state["y"]
        idx = self._animate_state.get("index", 0)
        chunk = self._animate_state.get("chunk", 1)
        next_idx = min(len(x_vals), idx + chunk)
        self._equity_line.set_data(x_vals[:next_idx], y_vals[:next_idx])
        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()
        if next_idx >= len(x_vals):
            self._animation_job = None
            self._animate_state["index"] = next_idx
            self._finalize_equity_animation()
            return
        self._animate_state["index"] = next_idx
        self._animation_job = self.after(20, self._schedule_equity_step)

    def _finalize_equity_animation(self) -> None:
        state = self._animate_state
        if state is None:
            return
        x_vals = state["x"]
        y_vals = state["y"]
        self._equity_line.set_data(x_vals, y_vals)
        self._update_equity_style(x_vals, y_vals)
        mc_result = self._pending_mc_result
        self._animate_state = None
        self._pending_mc_result = None
        if mc_result is not None:
            self._render_monte_carlo(mc_result)
        elif self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    def _render_monte_carlo(self, mc_result: MonteCarloResult) -> None:
        if self._ax is None or self._plot_canvas is None:
            return
        points = self._equity_points
        if not points:
            return
        numbers = np.asarray(points.get("numbers"), dtype=float)
        if numbers.size == 0:
            return
        step_index = points.get("step_index")

        def _extract(series: pd.Series) -> np.ndarray:
            if step_index is None:
                return series.to_numpy(dtype=float)
            ser = series.reindex(step_index)
            if ser.isna().any():
                ser = ser.interpolate().ffill().bfill()
            return ser.astype(float).to_numpy()

        mean_vals = _extract(mc_result.mean_equity_path)
        lower_vals = _extract(mc_result.lower_equity_path)
        upper_vals = _extract(mc_result.upper_equity_path)
        lower_band = np.minimum(lower_vals, upper_vals)
        upper_band = np.maximum(lower_vals, upper_vals)
        if self._mc_mean_line is not None:
            self._mc_mean_line.set_data(numbers, mean_vals)
        self._remove_artist(self._mc_band)
        label = f"MC envelope ({mc_result.confidence_level:.0%})" if mc_result.confidence_level is not None else "MC envelope"
        self._mc_band = self._ax.fill_between(numbers, lower_band, upper_band, color=CYAN, alpha=0.12, interpolate=True, label=label, zorder=0)
        y_min = min(self._ax.get_ylim()[0], float(np.nanmin(lower_band)))
        y_max = max(self._ax.get_ylim()[1], float(np.nanmax(upper_band)))
        if np.isclose(y_min, y_max):
            pad = max(1.0, abs(y_min) * 0.05 or 1.0)
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.05
            y_min -= pad
            y_max += pad
        self._ax.set_ylim(y_min, y_max)
        self._remove_artist(self._metrics_text)
        metrics_lines: list[str] = []
        conf = mc_result.confidence_level if mc_result.confidence_level is not None else 0.95
        if mc_result.var is not None:
            metrics_lines.append(f"VaR {conf:.0%}: {mc_result.var * 100.0:.2f}%")
        if mc_result.cvar is not None:
            metrics_lines.append(f"CVaR {conf:.0%}: {mc_result.cvar * 100.0:.2f}%")
        if mc_result.expected_return is not None:
            metrics_lines.append(f"Expected Return: {mc_result.expected_return * 100.0:.2f}%")
        if metrics_lines:
            props = dict(boxstyle="round", facecolor=LEGEND_FACE, alpha=0.85, edgecolor=CYAN)
            self._metrics_text = self._ax.text(0.98, 0.02, "\n".join(metrics_lines), transform=self._ax.transAxes,
                                               fontsize=9, ha="right", va="bottom", color="white", bbox=props)
        else:
            self._metrics_text = None
        self._update_legend()
        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    def _update_equity_style(self, x_vals: np.ndarray, y_vals: np.ndarray) -> None:
        if self._ax is None or self._equity_points is None:
            return
        points = self._equity_points
        numbers = points.get("numbers")
        months = points.get("months", [])
        if numbers is None or len(numbers) == 0 or len(x_vals) == 0 or len(y_vals) == 0:
            return
        for attr_name in ("_equity_fill_above", "_equity_fill_below", "_equity_scatter", "_equity_annotation"):
            self._remove_artist(getattr(self, attr_name, None))
            setattr(self, attr_name, None)
        initial = self._initial_balance if self._initial_balance is not None else float(y_vals[0])
        self._initial_balance = initial
        self._equity_fill_above = self._ax.fill_between(x_vals, initial, y_vals, where=(y_vals >= initial),
                                                        color=GREEN, alpha=0.15, interpolate=True, zorder=1)
        self._equity_fill_below = self._ax.fill_between(x_vals, initial, y_vals, where=(y_vals < initial),
                                                        color=RED, alpha=0.15, interpolate=True, zorder=1)
        self._remove_artist(self._initial_line)
        self._initial_line = self._ax.axhline(y=initial, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5,
                                              label=f"Initial: ${initial:,.0f}")
        self._equity_scatter = self._ax.scatter(x_vals, y_vals, color=GREEN, s=5, zorder=4, alpha=0.8)
        final_val = float(y_vals[-1])
        pnl = final_val - initial
        pnl_pct = (pnl / initial * 100.0) if initial else 0.0
        trades = int(points.get("trade_count", len(x_vals) - 1))
        summary = f"Final: ${final_val:,.0f}\nP&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%)\nTrades: {trades}"
        self._equity_annotation = self._ax.text(0.02, 0.98, summary, transform=self._ax.transAxes, fontsize=10,
                                                va="top", color="white",
                                                bbox=dict(boxstyle="round", facecolor=LEGEND_FACE, alpha=0.9, edgecolor=GREEN))
        if trades <= 20:
            ticks = list(range(0, trades + 1))
        else:
            step = max(1, trades // 15)
            ticks = list(range(0, trades + 1, step))
            if ticks[-1] != trades:
                ticks.append(trades)
        labels = [f"{pos}\n{months[min(pos, len(months) - 1)] if months else ''}" for pos in ticks]
        self._ax.set_xticks(ticks)
        self._ax.set_xticklabels(labels, fontsize=9, color="white")
        self._update_legend()
        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    # ---------- Backtest workflow ----------

    def _clear_strategy_params(self) -> None:
        for widget in self.params_inner.winfo_children():
            widget.destroy()
        self.strategy_param_vars.clear()

    def _resolve_param_type(self, annotation: Optional[Any], fallback: Any) -> Any:
        if annotation is None:
            return fallback
        origin = get_origin(annotation)
        if origin is None:
            return annotation
        if origin in (list, dict, tuple):
            return origin
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return args[0]
        return fallback

    def _create_param_widget(self, parent: ttk.Frame, value: Any, value_type: Any):
        if value_type is bool:
            var = tk.BooleanVar(value=bool(value))
            widget = ttk.Checkbutton(parent, variable=var)
            return var, widget
        display = "" if value is None else str(value)
        var = tk.StringVar(value=display)
        entry = ttk.Entry(parent, textvariable=var, width=40)
        return var, entry

    def _parse_param_value(self, raw: str, value_type: Any):
        text = raw.strip()
        if value_type is bool:
            if text.lower() in {"1", "true", "yes", "on"}:
                return True
            if text.lower() in {"0", "false", "no", "off"}:
                return False
            if text == "":
                return False
            raise ValueError(f"Expected boolean for strategy parameter, got '{text}'")
        if text == "":
            return None
        try:
            if (isinstance(value_type, type) and issubclass(value_type, int)) or value_type in (int, np.integer):
                return int(text)
        except Exception:
            pass
        try:
            if (isinstance(value_type, type) and issubclass(value_type, float)) or value_type in (float, np.floating):
                return float(text)
        except Exception:
            pass
        if value_type is pd.Timestamp:
            return pd.Timestamp(text)
        if value_type in (list, dict):
            return json.loads(text)
        try:
            return json.loads(text)
        except Exception:
            return text

    def _load_strategy_defaults(self) -> None:
        strategy_path = self.strategy_var.get().strip()
        if not strategy_path:
            messagebox.showerror("Strategy", "Strategy path is required")
            return
        try:
            strategy = load_strategy(strategy_path)
        except Exception as exc:
            messagebox.showerror("Strategy", f"Could not load strategy: {exc}")
            return

        defaults = getattr(strategy, "DEFAULTS", None)
        if not isinstance(defaults, dict):
            messagebox.showinfo("Strategy", "Strategy does not expose DEFAULTS dict, use JSON kwargs instead.")
            return

        annotations = getattr(getattr(strategy, "config", None).__class__, "__annotations__", {})

        self._clear_strategy_params()
        for row, (key, value) in enumerate(sorted(defaults.items())):
            resolved_type = self._resolve_param_type(annotations.get(key), type(value))
            ttk.Label(self.params_inner, text=key).grid(row=row, column=0, sticky="w")
            var, widget = self._create_param_widget(self.params_inner, value, resolved_type)
            widget.grid(row=row, column=1, sticky="we", padx=4, pady=2)
            self.params_inner.columnconfigure(1, weight=1)
            self.strategy_param_vars[key] = (var, resolved_type)

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[["CSV files", "*.csv"], ["All files", "*.*"]])
        if path:
            self.csv_var.set(path)

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.outdir_var.set(path)

    def _append_status(self, message: str) -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message + "\n")
        self.status_text.configure(state="disabled")
        self.status_text.see("end")

    def _append_status_async(self, message: str) -> None:
        self.after(0, lambda: self._append_status(message))

    def _set_running(self, running: bool) -> None:
        self._running = running
        def apply_state() -> None:
            state = "disabled" if running else "normal"
            self.run_button.config(state=state)
        self.after(0, apply_state)

    def _start_backtest(self) -> None:
        if self._running:
            return
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        self._prepare_live_plot()
        self._set_running(True)
        threading.Thread(target=self._run_backtest_safe, args=(cfg,), daemon=True).start()

    def _collect_config(self) -> Dict[str, Any]:
        csv_path = self.csv_var.get().strip()
        if not csv_path:
            raise ValueError("CSV path is required")
        try:
            extra_kwargs = json.loads(self.kwargs_var.get() or "{}")
            if not isinstance(extra_kwargs, dict):
                raise ValueError
        except ValueError:
            raise ValueError("Strategy kwargs must be a valid JSON object")

        strategy_kwargs: Dict[str, Any] = {}
        for key, (var, value_type) in self.strategy_param_vars.items():
            if isinstance(var, tk.BooleanVar):
                strategy_kwargs[key] = bool(var.get())
                continue
            raw = var.get()
            parsed = self._parse_param_value(raw, value_type)
            if parsed is not None:
                strategy_kwargs[key] = parsed

        strategy_kwargs.update(extra_kwargs)

        outdir_value = self.outdir_var.get().strip()
        outdir = Path(outdir_value) if outdir_value else DEFAULT_OUTPUT_DIR
        mc_horizon = self.mc_horizon_var.get().strip()
        mc_seed = self.mc_seed_var.get().strip()

        return {
            "loader": self.loader_var.get(),
            "csv": csv_path,
            "date_col": self.date_col_var.get().strip() or "Date",
            "time_col": self.time_col_var.get().strip() or "Timestamp",
            "dt_col": self.dt_col_var.get().strip() or "datetime",
            "close_col": self.close_col_var.get().strip() or "Close",
            "tz": self.tz_var.get().strip() or None,
            "strategy": self.strategy_var.get().strip(),
            "strategy_kwargs": strategy_kwargs,
            "cash": float(self.cash_var.get()),
            "point": float(self.point_var.get()),
            "tick_value": float(self.tick_value_var.get()),
            "finalize": bool(self.finalize_var.get()),
            "skip_mc": bool(self.skip_mc_var.get()),
            "mc_sims": int(self.mc_sims_var.get()),
            "mc_horizon": int(mc_horizon) if mc_horizon else None,
            "mc_confidence": float(self.mc_confidence_var.get()),
            "mc_seed": int(mc_seed) if mc_seed else None,
            "outdir": outdir,
            "show_plot": bool(self.show_plot_var.get()),
        }

    def _run_backtest_safe(self, cfg: Dict[str, Any]) -> None:
        try:
            self._append_status_async("Running backtest...")
            outdir, points, mc_result = self._execute_backtest(cfg)
            self.after(0, lambda pts=points, mc=mc_result: self._animate_equity(pts, mc))
            self._append_status_async("Backtest completed")
            if cfg["show_plot"]:
                self.after(0, lambda path=outdir: self._open_equity_plot(path))
            self.after(0, lambda: messagebox.showinfo("Success", "Backtest completed successfully"))
        except Exception as exc:
            self._append_status_async(f"Error: {exc}")
            self.after(0, lambda: messagebox.showerror("Backtest error", str(exc)))
        finally:
            self._set_running(False)

    def _execute_backtest(self, cfg: Dict[str, Any]) -> tuple[Path, Dict[str, Any], Optional[MonteCarloResult]]:
        loader = self._create_loader(cfg)
        price = loader.load_close()
        data = loader.load_dataframe()

        module_name, class_name = cfg["strategy"].split(":")
        module = importlib.import_module(module_name)
        strategy_class = getattr(module, class_name)
        strategy = strategy_class(**cfg["strategy_kwargs"])
        outputs = strategy.generate(price, data=data)

        engine_cfg = EngineConfig(
            init_cash=cfg["cash"],
            point=cfg["point"],
            tick_value=cfg["tick_value"],
            finalize_trades=cfg["finalize"],
        )
        results = run_engine(price, outputs, engine_cfg)

        outdir = cfg["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)

        equity = results["equity"]
        position = results["position"]
        trades = results["trades"]
        exit_reason = results.get("exit_reason")
        active_tp = results.get("active_take_profit")
        active_sl = results.get("active_stop_loss")

        equity.to_csv(outdir / "equity.csv", header=["equity"])
        position.to_csv(outdir / "position.csv", header=["position"])
        trades.to_csv(outdir / "trades.csv", index=False)
        pd.DataFrame(
            {
                "take_profit": active_tp,
                "stop_loss": active_sl,
                "exit_reason": exit_reason,
            }
        ).to_csv(outdir / "risk_levels.csv")

        mc_result: Optional[MonteCarloResult] = None
        if not cfg["skip_mc"]:
            mc_result = monte_carlo_bootstrap_equity(
                equity,
                num_simulations=cfg["mc_sims"],
                horizon=cfg["mc_horizon"],
                seed=cfg["mc_seed"],
                confidence_level=cfg["mc_confidence"],
            )
            pd.DataFrame(
                {
                    "step": mc_result.mean_equity_path.index,
                    "mean_equity": mc_result.mean_equity_path.values,
                    "lower_equity": mc_result.lower_equity_path.values,
                    "upper_equity": mc_result.upper_equity_path.values,
                }
            ).to_csv(outdir / "monte_carlo_paths.csv", index=False)
            mc_result.terminal_returns.to_csv(
                outdir / "monte_carlo_terminal_returns.csv",
                header=["terminal_return"],
            )

        plot_equity_tradenum(
            equity=equity,
            trades=trades,
            initial_balance=cfg["cash"],
            title=f"Equity - {cfg['strategy']}",
            out_png=outdir / "equity.png",
            show=False,
            monte_carlo_mean=None if mc_result is None else mc_result.mean_equity_path,
            monte_carlo_lower=None if mc_result is None else mc_result.lower_equity_path,
            monte_carlo_upper=None if mc_result is None else mc_result.upper_equity_path,
            monte_carlo_confidence=None if mc_result is None else mc_result.confidence_level,
            monte_carlo_var=None if mc_result is None else mc_result.var,
            monte_carlo_cvar=None if mc_result is None else mc_result.cvar,
            monte_carlo_expected=None if mc_result is None else mc_result.expected_return,
        )

        points = self._build_equity_points(equity, trades, cfg["cash"])
        return outdir, points, mc_result

    def _create_loader(self, cfg: Dict[str, Any]):
        if cfg["loader"] == "csv_datetime":
            return CSVDateTimeLoader(
                path=cfg["csv"],
                date_col=cfg["date_col"],
                time_col=cfg["time_col"],
                close_col=cfg["close_col"],
                tz=cfg["tz"],
            )
        return CSVSingleDTLoader(
            path=cfg["csv"],
            dt_col=cfg["dt_col"],
            close_col=cfg["close_col"],
            tz=cfg["tz"],
        )

    def _discover_strategies(self) -> list[str]:
        strategies_dir = Path(__file__).resolve().parent / "strategies"
        entries: list[str] = []
        for module_path in strategies_dir.glob("*.py"):
            name = module_path.stem
            if name.startswith("__"):
                continue
            module_name = f"strategies.{name}"
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            candidates: list[str] = []
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                    candidates.append(f"{module_name}:{attr}")
            if candidates:
                entries.extend(candidates)
        entries.sort()
        return entries

    def _open_equity_plot(self, outdir: Path) -> None:
        image_path = outdir / "equity.png"
        if not image_path.exists():
            messagebox.showwarning("Plot", f"Plot image not found at {image_path}")
            return
        try:
            os.startfile(str(image_path))  # type: ignore[attr-defined]
        except AttributeError:
            messagebox.showinfo("Plot", f"Plot saved to {image_path}")
        except Exception as exc:
            messagebox.showwarning("Plot", f"Could not open plot automatically: {exc}\nFile saved at {image_path}")


def main() -> None:
    app = BacktestUI()
    app.mainloop()


if __name__ == "__main__":
    main()
