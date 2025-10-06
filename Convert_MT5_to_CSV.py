#!/usr/bin/env python3
"""
convert_to_csv_configurable.py
Configure file paths inside this script and run directly.
Outputs a CSV with columns:
Date,Timestamp,Open,High,Low,Close,Volume
"""

# ===================== USER CONFIG =====================
# Set your input file path here, for example:
# INPUT_PATH = r"C:\path\to\your\broker_export.tsv"
INPUT_PATH = r"C:\VScode\venv311\QuantFarming-R-D-\Data\US500Z25.sim_M15.csv"

# Optional explicit output path. If left empty, script writes alongside input as *_converted.csv
OUTPUT_PATH = r""
# =======================================================

import sys
import re
from pathlib import Path
import pandas as pd

RE_WHITESPACE = r"[ \t+]+"

def _clean_col(col: str) -> str:
    c = col.strip()
    c = c.strip("<>")
    c = c.replace("-", "_")
    c = re.sub(r"\s+", "_", c)
    return c.upper()

def _read_input(path: Path) -> pd.DataFrame:
    # Try tab, then generic whitespace
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(path, sep=r"[ \t]+", engine="python")
    # If still single column, retry without header
    if df.columns.size == 1:
        df = pd.read_csv(path, sep=r"[ \t]+", engine="python", header=None)
    df.columns = [_clean_col(str(c)) for c in df.columns]
    return df

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_price_cols = ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE"]
    for rc in required_price_cols:
        if rc not in df.columns:
            aliases = {
                "DATE": ["DAY", "TRADEDATE"],
                "TIME": ["TIMESTAMP", "TRADETIME"],
                "OPEN": ["O"],
                "HIGH": ["H"],
                "LOW": ["L"],
                "CLOSE": ["C", "LAST"]
            }.get(rc, [])
            for a in aliases:
                if a in df.columns:
                    df[rc] = df[a]
                    break
            else:
                raise ValueError(f"Missing required column: {rc}")
    # Volume preference
    vol_col = None
    if "VOL" in df.columns and (pd.to_numeric(df["VOL"], errors="coerce").fillna(0) > 0).any():
        vol_col = "VOL"
    elif "TICKVOL" in df.columns:
        vol_col = "TICKVOL"
    elif "VOLUME" in df.columns:
        vol_col = "VOLUME"
    else:
        for c in df.columns:
            if "VOL" in c:
                vol_col = c
                break
    if vol_col is None:
        df["VOLUME_OUT"] = 0
    else:
        df["VOLUME_OUT"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).astype(int)
    return df

def _fmt_date(d: pd.Series) -> pd.Series:
    s = d.astype(str).str.strip()
    s = s.str.replace(r"\D", "", regex=True)
    needs_parse = s.str.len() != 8
    if needs_parse.any():
        parsed = pd.to_datetime(d, errors="coerce", dayfirst=False, infer_datetime_format=True)
        s.loc[needs_parse] = parsed.dt.strftime("%Y%m%d")
    return s

def _fmt_time(t: pd.Series) -> pd.Series:
    s = t.astype(str).str.strip()
    def fix_one(x: str) -> str:
        x = x.replace(".", ":").replace("-", ":").replace("/", ":")
        parts = re.split(r"[:\s]", x)
        parts = [p for p in parts if p != ""]
        if len(parts) == 1 and parts[0].isdigit():
            p = parts[0]
            if len(p) == 6:
                return f"{p[0:2]}:{p[2:4]}:{p[4:6]}"
            if len(p) == 4:
                return f"{p[0:2]}:{p[2:4]}:00"
        while len(parts) < 3:
            parts.append("00")
        hh = parts[0].zfill(2)
        mm = parts[1].zfill(2)
        ss = parts[2].zfill(2)
        return f"{hh}:{mm}:{ss}"
    return s.map(fix_one)

def convert(src_path: Path, out_path: Path | None) -> Path:
    df = _read_input(src_path)
    df = _ensure_columns(df)
    out = pd.DataFrame({
        "Date": _fmt_date(df["DATE"]),
        "Timestamp": _fmt_time(df["TIME"]),
        "Open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "High": pd.to_numeric(df["HIGH"], errors="coerce"),
        "Low": pd.to_numeric(df["LOW"], errors="coerce"),
        "Close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "Volume": df["VOLUME_OUT"].astype(int)
    })
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if out_path is None or str(out_path).strip() == "":
        out_path = src_path.with_suffix("").with_name(src_path.stem + "_converted.csv")
    out.to_csv(out_path, index=False)
    return out_path

def main():
    if not INPUT_PATH:
        print("Please set INPUT_PATH at the top of this script to your source file path.")
        sys.exit(2)
    src = Path(INPUT_PATH)
    if not src.exists():
        print(f"Input file not found: {src}")
        sys.exit(2)
    out = Path(OUTPUT_PATH) if OUTPUT_PATH else None
    outp = convert(src, out)
    print(f"Saved: {outp}")

if __name__ == "__main__":
    main()
