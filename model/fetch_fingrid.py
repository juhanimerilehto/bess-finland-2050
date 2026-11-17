"""Download Fingrid open data used by the model (run locally; needs an API key).

Usage (PowerShell, conda env active), from the repo root:
    python model/fetch_fingrid.py --start 2023-01-01 --end 2026-11-15

The API key (https://developer-data.fingrid.fi) is read from a .env file in the
repo root (or the current directory), line FINGRID_API_KEY=<key>, or from the
environment variable of the same name.

Writes one CSV per dataset to data/raw/fingrid_<id>.csv plus a manifest with
SHA-256 checksums. Licence: CC BY 4.0 (Fingrid). Rate limit: 1 request / 2 s.
NOTE: written without network access to data.fingrid.fi; check the endpoint and
parameter names against the developer portal on first run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://data.fingrid.fi/api/datasets/{id}/data"

DATASETS = {
    424: "Total installed BESS capacity (MW), from 11.3.2026",
    398: "BESS discharging power, real-time (MW, 3 min)",
    399: "BESS charging power, real-time (MW, 3 min)",
    124: "Electricity consumption in Finland (15 min)",
    371: "Electric boiler consumption sum, real-time",
    316: "FCR-N hourly procured volume",
    317: "FCR-N hourly price",
    315: "FCR-D up hourly procured volume",
    318: "FCR-D up hourly price",
    281: "FCR-D down hourly procured volume",
    283: "FCR-D down hourly price",
    1: "aFRR up procured capacity",
    2: "aFRR down procured capacity",
    52: "aFRR up marginal capacity price",
    51: "aFRR down marginal capacity price",
    327: "mFRR capacity market up volume (hourly)",
    329: "mFRR capacity market up price (hourly)",
    244: "mFRR up-regulation energy price (15 min)",
    106: "mFRR down-regulation energy price (15 min)",
    375: "Activated mFRR balancing regulation upward sum (15 min, from 4 Mar 2025)",
    376: "Activated mFRR balancing regulation downward sum (15 min, from 4 Mar 2025)",
    34: "Ordered up-regulations, balancing energy market FI (1 h, to 4 Mar 2025)",
    33: "Ordered down-regulations, balancing energy market FI (1 h, to 4 Mar 2025)",
    349: "aFRR energy activation volume, up (15 min, from 12 Jun 2024)",
    350: "aFRR energy activation volume, down (15 min, from 12 Jun 2024)",
    347: "aFRR energy price VWAP, up (15 min)",
    348: "aFRR energy price VWAP, down (15 min)",
}


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end) into calendar-month windows (ISO dates)."""
    y, m = int(start[:4]), int(start[5:7])
    out, cur = [], start
    while cur < end:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        nxt = min(f"{y:04d}-{m:02d}-01", end)
        out.append((cur, nxt))
        cur = nxt
    return out


class RateLimited(RuntimeError):
    pass


def _retry_after(e: urllib.error.HTTPError, default: float) -> float:
    try:
        return max(float(e.headers.get("Retry-After", "")), 1.0)
    except (TypeError, ValueError):
        return default


def _get(ds: int, start: str, end: str, key: str, page: int, page_size: int,
         retries: int = 10) -> dict:
    """One API call with back-off. Honours Retry-After; waits grow to 10 min on 429."""
    q = urllib.parse.urlencode({
        "startTime": f"{start}T00:00:00Z", "endTime": f"{end}T00:00:00Z",
        "format": "json", "page": page, "pageSize": page_size,
        "sortBy": "startTime", "sortOrder": "asc"})
    req = urllib.request.Request(BASE.format(id=ds) + "?" + q, headers={"x-api-key": key})
    for attempt in range(retries):
        wait = min(15 * 2 ** attempt, 600)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = _retry_after(e, wait) if e.code == 429 else wait
                print(f"      HTTP {e.code}; waiting {wait:.0f} s", flush=True)
                time.sleep(wait)
                continue
            if e.code == 429:
                raise RateLimited("Fingrid rate limit persists; the daily quota may be used up. "
                                  "Re-run later: completed months are cached.") from e
            raise
        except (TimeoutError, ConnectionError, urllib.error.URLError) as e:
            if attempt < retries - 1:
                print(f"      {type(e).__name__}; waiting {wait:.0f} s", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


CACHE = RAW / "cache_fingrid"
# First month with data (saves requests on empty ranges)
DATA_START = {398: "2025-07-01", 399: "2025-07-01", 424: "2026-03-01",
              375: "2025-03-01", 376: "2025-03-01", 349: "2024-06-01", 350: "2024-06-01",
              347: "2024-06-01", 348: "2024-06-01"}
PACE_S = 6.0          # pause between calls; Fingrid rate-limits per key


def _cache_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        next(f)
        return [dict(zip(("startTime", "endTime", "value"), line.rstrip("\n").split(","))) for line in f]


def _cache_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write("startTime,endTime,value\n")
        for r in rows:
            f.write(f"{r.get('startTime')},{r.get('endTime')},{r.get('value')}\n")
    tmp.replace(path)


def fetch(ds: int, start: str, end: str, key: str, page_size: int = 20000,
          pace: float = PACE_S) -> list[dict]:
    """Whole range in one query, paged at the maximum page size.

    Pages are requested until a short or empty page (pagination metadata is not
    relied on). Each full page is cached (data/raw/cache_fingrid/) so a re-run
    resumes; the final short page is always re-requested. Calls are paced.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    start = max(start, DATA_START.get(ds, start))
    rows, page = [], 1
    while True:
        cpath = CACHE / f"{ds}_{start}_{end}_p{page:03d}_{page_size}.csv"
        if cpath.is_file():
            data = _cache_rows(cpath)
            src = "cached"
        else:
            data = _get(ds, start, end, key, page, page_size).get("data", [])
            if len(data) == page_size:          # only full pages are final
                _cache_write(cpath, data)
            src = "downloaded"
            time.sleep(pace)
        rows += data
        print(f"    page {page}: {len(data)} rows ({src})", flush=True)
        if len(data) < page_size:
            break
        page += 1
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: str(r.get("startTime", ""))):
        t = r.get("startTime")
        if t not in seen:
            seen.add(t)
            out.append(r)
    _report_gaps(ds, out, end)
    return out


def _report_gaps(ds: int, rows: list[dict], end: str, tolerance_h: float = 6.0) -> None:
    """Warn about holes longer than tolerance_h and about data ending early
    (the first version of this script was silently truncated at the end)."""
    from datetime import datetime, timezone
    ts = []
    for r in rows:
        try:
            ts.append(datetime.fromisoformat(str(r["startTime"]).replace("Z", "+00:00")))
        except (KeyError, ValueError):
            continue
    gaps = [(a, b) for a, b in zip(ts, ts[1:]) if (b - a).total_seconds() > tolerance_h * 3600]
    if ts:
        print(f"    span {ts[0]:%Y-%m-%d %H:%M} .. {ts[-1]:%Y-%m-%d %H:%M} UTC, {len(gaps)} gap(s) > {tolerance_h:g} h")
    for a, b in gaps[:5]:
        print(f"      gap {a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M}")
    if ts:
        lag_h = (datetime.fromisoformat(end).replace(tzinfo=timezone.utc) - ts[-1]).total_seconds() / 3600
        if lag_h > 24:
            print(f"    WARNING: data ends {lag_h / 24:.1f} days before the requested end ({end}); "
                  f"possible truncation or a discontinued dataset")


KEY_NAMES = ("FINGRID_API_KEY", "FINGRID_KEY", "FINGRID_APIKEY", "API_KEY")


def load_dotenv(paths: list[Path]) -> Path | None:
    """Minimal .env reader (no dependency). Does not override variables already set.

    Handles UTF-8 BOM (Windows Notepad), 'export ' prefixes, quotes, comments and
    spaces around '='.
    """
    for path in paths:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            q = value[:1]
            if q in ("\"", "'") and q in value[1:]:
                value = value[1:value.index(q, 1)]          # quoted: drop anything after
            elif " #" in value:
                value = value.split(" #", 1)[0].strip()     # unquoted inline comment
            os.environ.setdefault(name, value)
        return path
    return None


def get_key() -> str:
    env_file = load_dotenv([ROOT / ".env", Path.cwd() / ".env"])
    for name in KEY_NAMES:
        if os.environ.get(name):
            return os.environ[name]
    where = f"{env_file}" if env_file else f"no .env found at {ROOT / '.env'} or {Path.cwd() / '.env'}"
    raise SystemExit(f"No API key found ({where}). Add a line FINGRID_API_KEY=<key> to the .env "
                     f"in the repo root, or set the environment variable.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-11-15")
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--pace", type=float, default=PACE_S, help="seconds between API calls")
    ap.add_argument("--page-size", type=int, default=20000)
    a = ap.parse_args()
    key = get_key()
    RAW.mkdir(parents=True, exist_ok=True)
    mpath = RAW / "fingrid_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
    manifest = {int(k): v for k, v in manifest.items()}
    for ds, name in DATASETS.items():
        if a.only and ds not in a.only:
            continue
        print(f"[{ds}] {name} ...", flush=True)
        rows = fetch(ds, a.start, a.end, key, page_size=a.page_size, pace=a.pace)
        out = RAW / f"fingrid_{ds}.csv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write("startTime,endTime,value\n")
            for r in rows:
                f.write(f"{r.get('startTime')},{r.get('endTime')},{r.get('value')}\n")
        manifest[ds] = {"name": name, "rows": len(rows), "start": a.start, "end": a.end,
                        "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        print(f"    {len(rows)} rows")
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")   # after each dataset


if __name__ == "__main__":
    main()
