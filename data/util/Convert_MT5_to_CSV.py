"""Utility helpers for translating MT5 export files into the app's CSV schema.

This module can be invoked as a script or imported and reused inside other
pipelines.  It expects the MT5 export to contain a ``Time`` column (usually in
``YYYY.MM.DD HH:MM`` format) alongside the OHLCV columns produced by MetaTrader 5.
The resulting CSV matches the Backtest & Optimization UI schema::

    Date,Timestamp,Open,High,Low,Close,Volume

Example::

    python data/util/Convert_MT5_to_CSV.py --input ./raw/MT5_EURUSD_M15.csv \
        --output ./data/EURUSD_M15_converted.csv

If you prefer to configure paths in code, update the ``SOURCE_CSV`` and
``OUTPUT_CSV`` constants near the bottom of the file.  By default they are left
blank so you can decide where to place your data within the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

EXPECTED_COLUMNS: Iterable[str] = (
    "Open",
    "High",
    "Low",
    "Close",
)


def convert_mt5_to_csv(source: Path, destination: Path) -> None:
    """Convert an MT5 OHLC export into the repository's CSV schema.

    Parameters
    ----------
    source:
        Path to the MT5-exported CSV file.
    destination:
        Where the formatted CSV should be written.
    """

    df = pd.read_csv(source)

    if "Time" not in df.columns:
        raise ValueError(
            "MT5 export is missing a 'Time' column. Double-check the exported file."
        )

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "MT5 export is missing required OHLC columns: " + ", ".join(missing)
        )

    timestamp = pd.to_datetime(df["Time"], errors="raise")
    df["Date"] = timestamp.dt.strftime("%Y%m%d")
    df["Timestamp"] = timestamp.dt.strftime("%H:%M:%S")

    # Prefer actual trading volume if present, otherwise fall back to tick volume.
    volume_column = None
    for candidate in ("Volume", "Real Volume", "Tick Volume", "Tick volume"):
        if candidate in df.columns:
            volume_column = candidate
            break

    if volume_column is None:
        raise ValueError(
            "MT5 export is missing a volume column (expected one of Volume, Real Volume, Tick Volume)."
        )

    formatted = df[[
        "Date",
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        volume_column,
    ]].rename(columns={volume_column: "Volume"})

    destination.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(destination, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to the MT5-exported CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination for the converted CSV (defaults to data/converted_mt5.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_csv = args.input
    destination_csv = args.output or Path("data/converted_mt5.csv")

    if source_csv is None or str(source_csv) == "":
        raise SystemExit(
            "No input CSV provided. Pass --input or update SOURCE_CSV with your MT5 export path."
        )

    convert_mt5_to_csv(source_csv, destination_csv)


# Leaving these blank ensures no hard-coded local paths leak into the repository.
SOURCE_CSV = Path("")  # Update with the path to your MT5 exported CSV file if desired.
OUTPUT_CSV = Path("data/converted_mt5.csv")

if __name__ == "__main__":
    if SOURCE_CSV:
        convert_mt5_to_csv(SOURCE_CSV, OUTPUT_CSV)
    else:
        main()
