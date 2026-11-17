"""Run every model step in order and record input checksums.

Two levels of reproduction:
  * full: with the market data in data/raw (Fingrid open data and ENTSO-E day-ahead
    prices, see data/README.md), every step runs from the raw series;
  * from derived data: without data/raw, the market-evidence steps are skipped and the
    forecast, tables and figures are rebuilt from the derived series in outputs/.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import OUT, PARAMS_PATH, ROOT, sha256_prefix  # noqa: E402

import cost  # noqa: E402
import reserves  # noqa: E402
import retained  # noqa: E402
import spreads  # noqa: E402
import fleet_ops  # noqa: E402
import balancing  # noqa: E402


def check_raw() -> dict:
    """Compare data/raw against the download manifests; report what was used."""
    raw = ROOT / "data" / "raw"
    out = {"present": [], "missing": [], "changed": []}
    for mf in ("fingrid_manifest.json", "entsoe_manifest.json"):
        f = raw / mf
        if not f.is_file():
            continue
        for key, rec in json.loads(f.read_text(encoding="utf-8")).items():
            if not isinstance(rec, dict) or "sha256" not in rec:
                continue
            name = rec.get("file") or f"fingrid_{key}.csv"
            path = raw / name
            if not path.is_file():
                out["missing"].append(name)
                continue
            full = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            (out["present"] if full == rec.get("sha256") else out["changed"]).append(name)
    if out["changed"]:
        print("WARNING: raw files differ from the archived snapshot:", ", ".join(out["changed"]))
    return {k: sorted(v) for k, v in out.items()}


def main() -> None:
    retained.main()
    cost.main()
    if (ROOT / "data" / "raw" / "fingrid_317.csv").is_file():
        reserves.main()
    if (ROOT / "data" / "raw" / "entsoe_da_FI.csv").is_file():
        spreads.main()
    da = (ROOT / "data" / "raw" / "entsoe_da_FI.csv").is_file()
    if da and (ROOT / "data" / "raw" / "fingrid_244.csv").is_file():
        fleet_ops.main()
    if da and (ROOT / "data" / "raw" / "fingrid_375.csv").is_file():
        balancing.main()
    runpy.run_path(str(ROOT / "figures" / "make_figures.py"), run_name="__main__")
    if (OUT / "da_arbitrage_annual.csv").is_file():
        import f_equilibrium
        import ensemble
        f_equilibrium.main()          # M3 paths + Sobol
        ensemble.main()               # M1-M5 + pool + electricity use
        import stress
        stress.main()                 # one-at-a-time stress cases (M3) + seed stability
        import make_numbers
        make_numbers.main()           # paper/generated/*.tex
        runpy.run_path(str(ROOT / "figures" / "forecast_figures.py"), run_name="__main__")
        runpy.run_path(str(ROOT / "figures" / "comparison_figures.py"), run_name="__main__")
    manifest = {
        "params_sha256": sha256_prefix(PARAMS_PATH),
        "raw_data": check_raw(),
        **{f"{m}_sha256": sha256_prefix(HERE / m) for m in
           ("cost.py", "f_equilibrium.py", "f_methods.py", "ensemble.py", "stress.py", "make_numbers.py")},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("run_all complete:", manifest)


if __name__ == "__main__":
    main()
