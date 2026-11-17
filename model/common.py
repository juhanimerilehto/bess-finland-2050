"""Shared helpers: parameter loading, checksums, interpolation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "model" / "params.json"
OUT = ROOT / "outputs"
BASE_YEAR = 2025


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_prefix(path: Path, n: int = 16) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


def pick(entry, case: str = "central"):
    """Return the value of a parameter entry for a case (low/central/high)."""
    if isinstance(entry, dict):
        if case in entry:
            return entry[case]
        if "value" in entry:
            return entry["value"]
        if case == "central" and "low" in entry and "high" in entry:
            return 0.5 * (entry["low"] + entry["high"])
    return entry


def loglinear(year: float, points: dict[int, float]) -> float:
    """Piecewise log-linear interpolation of a positive index through {year: value}.

    Constant before the first and after the last point.
    """
    ys = sorted(points)
    if year <= ys[0]:
        return points[ys[0]]
    if year >= ys[-1]:
        return points[ys[-1]]
    for a, b in zip(ys, ys[1:]):
        if a <= year <= b:
            w = (year - a) / (b - a)
            return math.exp((1 - w) * math.log(points[a]) + w * math.log(points[b]))
    raise ValueError(year)


def crf(rate: float, years: float) -> float:
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(_fmt(x) for x in r) + "\n")


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)
