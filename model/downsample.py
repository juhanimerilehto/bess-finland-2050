"""Downsample a high-frequency Fingrid CSV to fixed intervals (time-weighted mean).

Usage, from the repo root:
    python model/downsample.py 371 --minutes 15
writes data/raw/fingrid_371_15min.csv (startTime,endTime,value). The raw file is kept.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=int)
    ap.add_argument("--minutes", type=int, default=15)
    a = ap.parse_args()
    step = a.minutes * 60
    acc = defaultdict(lambda: [0.0, 0.0])          # value*seconds, seconds
    src = RAW / f"fingrid_{a.dataset}.csv"
    with open(src, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                t0 = datetime.fromisoformat(r["startTime"].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(r["endTime"].replace("Z", "+00:00"))
                v = float(r["value"])
            except (TypeError, ValueError, KeyError):
                continue
            dur = (t1 - t0).total_seconds()
            ep = t0.timestamp()
            slot = ep - ep % step
            acc[slot][0] += v * dur
            acc[slot][1] += dur
    out = RAW / f"fingrid_{a.dataset}_{a.minutes}min.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("startTime,endTime,value\n")
        for slot in sorted(acc):
            vs, sec = acc[slot]
            t0 = datetime.fromtimestamp(slot, timezone.utc)
            t1 = t0 + timedelta(seconds=step)
            f.write(f"{t0:%Y-%m-%dT%H:%M:%S}.000Z,{t1:%Y-%m-%dT%H:%M:%S}.000Z,{vs / sec:.4f}\n")
    print(f"{src.name}: {sum(1 for _ in acc)} slots -> {out.name}")


if __name__ == "__main__":
    main()
