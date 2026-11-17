"""Download Finnish day-ahead and imbalance prices from the ENTSO-E Transparency
Platform (run locally; needs a security token).

Usage (PowerShell, conda env active), from the repo root:
    python model/fetch_entsoe.py --start 2023-01-01 --end 2026-11-15

The token is read from the repo-root .env (ENTSOE_API_KEY=<token>; also accepted:
ENTSOE_TOKEN, ENTSOE_SECURITY_TOKEN, ENTSOE_KEY, SECURITY_TOKEN) or the environment.

Outputs (UTC timestamps, EUR/MWh):
    data/raw/entsoe_da_FI.csv    day-ahead prices (A44); PT60M until 30 Sep 2025, PT15M after
    data/raw/entsoe_imb_FI.csv   imbalance prices (A85)
    data/raw/entsoe_manifest.json
Curve type A03 (repeated values omitted) is expanded to every interval.
NOTE: written without network access to the ENTSO-E API; endpoint and parameter
names follow the Transparency Platform RESTful API guide.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fetch_fingrid import ROOT, load_dotenv, month_windows

RAW = ROOT / "data" / "raw"
API = "https://web-api.tp.entsoe.eu/api"
FI = "10YFI-1--------U"
KEY_NAMES = ("ENTSOE_API_KEY", "ENTSOE_TOKEN", "ENTSOE_SECURITY_TOKEN", "ENTSOE_KEY", "SECURITY_TOKEN")
RES = {"PT15M": 15, "PT30M": 30, "PT60M": 60, "PT1H": 60}


def get_token() -> str:
    env_file = load_dotenv([ROOT / ".env", Path.cwd() / ".env"])
    for n in KEY_NAMES:
        if os.environ.get(n):
            return os.environ[n]
    raise SystemExit(f"No ENTSO-E token found (.env: {env_file}). Add ENTSOE_API_KEY=<token> to the repo-root .env.")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _findall(el, name):
    return [c for c in el.iter() if _local(c.tag) == name]


def parse_prices(xml_bytes: bytes, price_tag: str = "price.amount") -> list[tuple[str, int, float, str]]:
    """Return (utc_start_iso, resolution_min, price, category) for every interval.

    Handles both A01 (all points) and A03 (points omitted when unchanged) curves.
    Raises LookupError for an Acknowledgement document (e.g. no data).
    """
    root = ET.fromstring(xml_bytes)
    if _local(root.tag) == "Acknowledgement_MarketDocument":
        txt = " ".join((e.text or "") for e in _findall(root, "text"))
        raise LookupError(txt.strip() or "acknowledgement")
    out = []
    for ts in _findall(root, "TimeSeries"):
        for period in [c for c in ts if _local(c.tag) == "Period"]:
            ti = _find(period, "timeInterval")
            start = datetime.strptime(_find(ti, "start").text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            end = datetime.strptime(_find(ti, "end").text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            res = RES[_find(period, "resolution").text]
            n = int((end - start).total_seconds() // 60 // res)
            pts = {}
            for pt in [c for c in period if _local(c.tag) == "Point"]:
                pos = int(_find(pt, "position").text)
                val = _find(pt, price_tag)
                cat = _find(pt, "imbalance_Price.category")
                if val is not None:
                    pts[pos] = (float(val.text), cat.text if cat is not None else "")
            last = None
            for pos in range(1, n + 1):
                if pos in pts:
                    last = pts[pos]
                if last is None:
                    continue
                t = start + timedelta(minutes=res * (pos - 1))
                out.append((t.strftime("%Y-%m-%dT%H:%M:%SZ"), res, last[0], last[1]))
    return out


def parse_response(body: bytes, price_tag: str) -> list[tuple[str, int, float, str]]:
    """Parse plain XML or a ZIP archive of XML documents (imbalance prices, A85)."""
    if body[:2] == b"PK":
        rows, acks = [], []
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            for name in sorted(zf.namelist()):
                if not name.lower().endswith(".xml"):
                    continue
                try:
                    rows += parse_prices(zf.read(name), price_tag)
                except LookupError as e:
                    acks.append(str(e))
        if not rows and acks:
            raise LookupError("; ".join(acks))
        return rows
    return parse_prices(body, price_tag)


def request(params: dict, token: str, retries: int = 8) -> bytes:
    """GET with retries on rate limits, server errors, timeouts and dropped connections."""
    q = urllib.parse.urlencode({"securityToken": token, **params})
    for attempt in range(retries):
        wait = min(5 * 2 ** attempt, 120)
        try:
            with urllib.request.urlopen(API + "?" + q, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read()
            if e.code == 400 and b"Acknowledgement" in body:
                return body                      # 'no matching data' comes back as 400 + acknowledgement
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                print(f"      HTTP {e.code}, retry in {wait} s", flush=True)
                time.sleep(wait)
                continue
            raise
        except (TimeoutError, ConnectionError, urllib.error.URLError, OSError) as e:
            if attempt < retries - 1:
                print(f"      {type(e).__name__}: {e}; retry in {wait} s", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def _stamp(d: str) -> str:
    return d.replace("-", "") + "0000"


CACHE = RAW / "cache_entsoe"


def _cache_read(path: Path) -> list[tuple]:
    out = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            t, res, v, cat = line.rstrip("\n").split(",")
            out.append((t, int(res), float(v), cat))
    return out


def _cache_write(path: Path, rows: list[tuple]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write("utc_start,resolution_min,price_eur_mwh,category\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    tmp.replace(path)                            # atomic: a crash never leaves a half-written month


def fetch(kind: str, start: str, end: str, token: str, refresh_last: bool = True) -> list[tuple]:
    """Month by month, cached in data/raw/cache_entsoe/ so that a re-run resumes.

    Completed months are read from the cache; the last (possibly partial) month is
    re-downloaded unless refresh_last is False.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    windows = month_windows(start, end)
    for i, (a, b) in enumerate(windows):
        cpath = CACHE / f"{kind}_{a}_{b}.csv"
        if cpath.is_file() and not (refresh_last and i == len(windows) - 1):
            got = _cache_read(cpath)
            rows += got
            print(f"    {a}..{b}: {len(got)} points (cached)", flush=True)
            continue
        if kind == "da":
            params = {"documentType": "A44", "in_Domain": FI, "out_Domain": FI,
                      "periodStart": _stamp(a), "periodEnd": _stamp(b)}
            tag = "price.amount"
        else:
            params = {"documentType": "A85", "controlArea_Domain": FI,
                      "periodStart": _stamp(a), "periodEnd": _stamp(b)}
            tag = "imbalance_Price.amount"
        try:
            got = parse_response(request(params, token), tag)
        except LookupError as e:
            print(f"    {a}: no data ({e})")
            got = []
        _cache_write(cpath, got)
        rows += got
        print(f"    {a}..{b}: {len(got)} points", flush=True)
        time.sleep(0.3)
    # de-duplicate: keep the finest resolution for a timestamp, first category order
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (r[0], r[1], r[3])):
        key = (r[0], r[3])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-11-15")
    ap.add_argument("--only", choices=["da", "imb"], nargs="*")
    a = ap.parse_args()
    token = get_token()
    RAW.mkdir(parents=True, exist_ok=True)
    mpath = RAW / "entsoe_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
    for kind, fname in (("da", "entsoe_da_FI.csv"), ("imb", "entsoe_imb_FI.csv")):
        if a.only and kind not in a.only:
            continue
        print(f"[{kind}] {fname}")
        rows = fetch(kind, a.start, a.end, token)
        out = RAW / fname
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write("utc_start,resolution_min,price_eur_mwh,category\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
        manifest[kind] = {"file": fname, "rows": len(rows), "start": a.start, "end": a.end,
                          "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        print(f"    {len(rows)} rows written")
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")   # after each series
    for kind, fname in (("da", "entsoe_da_FI.csv"), ("imb", "entsoe_imb_FI.csv")):
        f = RAW / fname
        if kind not in manifest and f.is_file():                              # backfill a missing entry
            manifest[kind] = {"file": fname, "rows": sum(1 for _ in open(f, encoding="utf-8")) - 1,
                              "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "note": "backfilled"}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
