"""Timezone conversion helper for OHLCV CSV datasets.

The script reads a CSV following the Backtest & Optimization UI schema::

    Date,Timestamp,Open,High,Low,Close,Volume

It combines the date and timestamp columns, localises them to a source timezone,
converts to the desired timezone, and then writes a CSV in the same schema with
updated date/timestamp values.

Example::

    python data/util/tz_convert_prices.py --input data/EURUSD_M15_converted.csv \
        --source-tz UTC --target-tz America/New_York \
        --output data/EURUSD_M15_converted_est.csv

You can also set the configuration via the constants at the bottom of the file.
They intentionally default to blank placeholders so no machine-specific paths are
committed to the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_timezone(
    source: Path,
    destination: Path,
    source_timezone: str,
    target_timezone: str,
) -> None:
    """Convert the timezone for a CSV formatted as the app expects."""

    df = pd.read_csv(source)

    timestamp = pd.to_datetime(df["Date"].astype(str) + " " + df["Timestamp"], utc=False)
    timestamp = timestamp.dt.tz_localize(source_timezone)
    converted = timestamp.dt.tz_convert(target_timezone)

    df["Date"] = converted.dt.strftime("%Y%m%d")
    df["Timestamp"] = converted.dt.strftime("%H:%M:%S")

    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to the CSV that needs timezone conversion")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination for the converted CSV (defaults to data/tz_converted.csv)",
    )
    parser.add_argument("--source-tz", required=True, help="Timezone of the input data (e.g. UTC)")
    parser.add_argument(
        "--target-tz",
        required=True,
        help="Timezone to convert into (e.g. America/New_York)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = args.input
    output_csv = args.output or Path("data/tz_converted.csv")

    if input_csv is None or str(input_csv) == "":
        raise SystemExit(
            "No input CSV provided. Pass --input or set SOURCE_CSV to your dataset path."
        )

    convert_timezone(input_csv, output_csv, args.source_tz, args.target_tz)


SOURCE_CSV = Path("")  # Update with the path to the CSV you want to convert.
SOURCE_TIMEZONE = ""  # e.g. "UTC"
TARGET_TIMEZONE = ""  # e.g. "America/New_York"
OUTPUT_CSV = Path("data/tz_converted.csv")

if __name__ == "__main__":
    if SOURCE_CSV and SOURCE_TIMEZONE and TARGET_TIMEZONE:
        convert_timezone(SOURCE_CSV, OUTPUT_CSV, SOURCE_TIMEZONE, TARGET_TIMEZONE)
    else:
        main()
