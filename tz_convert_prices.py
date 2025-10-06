#!/usr/bin/env python3
# Python 3.11
# pip install pandas pytz

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, time, timedelta

def parse_hhmm(s: str) -> time:
    try:
        return datetime.strptime(s.strip(), "%H:%M").time()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Expected HH:MM, got {s}") from e

def load_dataframe(csv_path: Path,
                   date_col: str | None,
                   time_col: str | None,
                   dt_col: str | None) -> tuple[pd.DataFrame, pd.DatetimeIndex, str]:
    df = pd.read_csv(csv_path, keep_default_na=False)
    if df.empty:
        raise ValueError("CSV has no rows")

    # Choose schema
    if date_col and time_col and date_col in df.columns and time_col in df.columns:
        dt_series = pd.to_datetime(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            format="%Y%m%d %H:%M:%S",
            errors="coerce"
        )
        schema = "split"
    elif dt_col and dt_col in df.columns:
        dt_series = pd.to_datetime(df[dt_col], errors="coerce")
        schema = "single"
    else:
        # Try to auto-detect a single datetime-like column
        dt_col_guess = next((c for c in df.columns
                             if c.lower() in {"datetime", "time", "timestamp"}), None)
        if dt_col_guess is None:
            raise ValueError("Could not find Date+Timestamp columns or a single datetime column.")
        dt_series = pd.to_datetime(df[dt_col_guess], errors="coerce")
        dt_col = dt_col_guess
        schema = "single"

    if dt_series.isna().any():
        bad = int(dt_series.isna().sum())
        raise ValueError(f"Failed to parse {bad} datetime rows. Check formats.")

    # Use a DatetimeIndex everywhere
    dt_index = pd.DatetimeIndex(dt_series)
    if len(dt_index) == 0:
        raise ValueError("Datetime index is empty after parsing")

    return df, dt_index, schema

def reattach_datetime(df: pd.DataFrame,
                      dt_naive: pd.DatetimeIndex,
                      schema: str,
                      date_col: str | None,
                      time_col: str | None,
                      dt_col: str | None) -> pd.DataFrame:
    if schema == "split":
        out = df.copy()
        out[date_col] = dt_naive.strftime("%Y%m%d")
        out[time_col] = dt_naive.strftime("%H:%M:%S")
        cols = list(out.columns)
        desired = [c for c in [date_col, time_col] if c in cols]
        others = [c for c in cols if c not in desired]
        out = out[desired + others]
        return out
    else:
        out = df.copy()
        c = dt_col if dt_col and dt_col in out.columns else "datetime"
        out[c] = dt_naive
        return out

def convert_with_timezones(dt: pd.DatetimeIndex, src_tz: str, dst_tz: str) -> pd.DatetimeIndex:
    aware = dt.tz_localize(src_tz, nonexistent="shift_forward", ambiguous="infer")
    converted = aware.tz_convert(dst_tz)
    return converted.tz_localize(None)

def infer_and_apply_shift(dt: pd.DatetimeIndex, expected_last_hhmm: time) -> tuple[pd.DatetimeIndex, int]:
    last = dt[-1]  # DatetimeIndex supports positional access here
    actual = last.time()
    actual_minutes = actual.hour * 60 + actual.minute
    target_minutes = expected_last_hhmm.hour * 60 + expected_last_hhmm.minute
    delta_minutes = target_minutes - actual_minutes
    if delta_minutes > 720:
        delta_minutes -= 1440
    elif delta_minutes < -720:
        delta_minutes += 1440
    return dt + timedelta(minutes=delta_minutes), int(delta_minutes)

def main():
    ap = argparse.ArgumentParser(
        description="Convert or shift a CSV price time series to a new timezone or to a target last-bar time."
    )
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument("--out", default=None, help="Path to output CSV, default: <input>.tz.csv")

    # Column schema
    ap.add_argument("--date-col", default="Date", help="Date column if split schema")
    ap.add_argument("--time-col", default="Timestamp", help="Time column if split schema")
    ap.add_argument("--dt-col", default=None, help="Single datetime column name if using single-column schema")

    # Mode A: timezone conversion
    ap.add_argument("--src-tz", default=None, help="Source timezone name, for example UTC")
    ap.add_argument("--dst-tz", default=None, help="Destination timezone name, for example America/Los_Angeles")

    # Mode B: direct shift to align last bar time
    ap.add_argument("--target-last", type=parse_hhmm, default=None,
                    help="Target HH:MM for the last bar after conversion, for example 16:45")

    args = ap.parse_args()
    in_path = Path(args.csv)
    out_path = Path(args.out) if args.out else in_path.with_suffix(in_path.suffix + ".tz.csv")

    df, dt_index, schema = load_dataframe(in_path, args.date_col, args.time_col, args.dt_col)

    if args.src_tz and args.dst_tz:
        dt_new = convert_with_timezones(dt_index, args.src_tz, args.dst_tz)
        mode_used = f"tz_convert {args.src_tz} -> {args.dst_tz}"
        delta_minutes = None
    elif args.target_last:
        dt_new, delta_minutes = infer_and_apply_shift(dt_index, args.target_last)
        sign = "+" if delta_minutes >= 0 else "-"
        mode_used = f"shift {sign}{abs(delta_minutes)} minutes to match last {args.target_last.strftime('%H:%M')}"
    else:
        raise SystemExit("Specify either --src-tz and --dst-tz, or --target-last HH:MM")

    out_df = reattach_datetime(df, pd.DatetimeIndex(dt_new), schema, args.date_col, args.time_col, args.dt_col)
    out_df.to_csv(out_path, index=False)

    old_last = dt_index[-1]
    new_last = dt_new[-1]
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Mode:   {mode_used}")
    print(f"Last bar before: {old_last}  after: {new_last}")
    if delta_minutes is not None:
        print(f"Applied shift: {delta_minutes} minutes")

if __name__ == "__main__":
    main()
