"""Reserve-market evidence from Fingrid open data (hourly markets, 2023-2026).

For each product: procured volume (MW) x capacity price (EUR/MW,h) per hour gives
the hourly-market procurement cost; the sum of prices over a year is the
capacity revenue a 1 MW resource would have earned if accepted every hour
(an upper bound for one product; products cannot all be stacked on one MW).
"""
from __future__ import annotations

import csv
from collections import defaultdict

from common import OUT, ROOT, load_params, write_csv

RAW = ROOT / "data" / "raw"

PRODUCTS = {
    # name: (volume dataset, price dataset)
    "FCR-N": (316, 317),
    "FCR-D up": (315, 318),
    "FCR-D down": (281, 283),
    "aFRR up": (1, 52),
    "aFRR down": (2, 51),
    "mFRR up (hourly CM)": (327, 329),
}


def load_hourly(ds: int) -> dict[str, float]:
    out = {}
    with open(RAW / f"fingrid_{ds}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[r["startTime"][:13]] = float(r["value"])   # key: YYYY-MM-DDTHH
            except (TypeError, ValueError):
                continue
    return out


def product_tables() -> tuple[list[list], list[list]]:
    annual, monthly = [], []
    for name, (vds, pds) in PRODUCTS.items():
        vol, price = load_hourly(vds), load_hourly(pds)
        acc = defaultdict(lambda: [0.0, 0.0, 0, 0.0])      # cost, price_sum, hours, vol_sum
        macc = defaultdict(lambda: [0.0, 0])
        for t, p in price.items():
            v = max(vol.get(t, 0.0), 0.0)
            y, m = t[:4], t[:7]
            a = acc[y]
            a[0] += v * p
            a[1] += p
            a[2] += 1
            a[3] += v
            macc[m][0] += p
            macc[m][1] += 1
        for y in sorted(acc):
            cost, psum, h, vsum = acc[y]
            annual.append([name, y, h, cost / 1e6, psum / h, cost / vsum if vsum else 0.0,
                           vsum / h, psum / h * 8760 / 1e3])
        for m in sorted(macc):
            psum, h = macc[m]
            monthly.append([name, m, psum / h])
    return annual, monthly


def storage_capacity() -> list[list]:
    cap = load_hourly(424)
    by_month = defaultdict(list)
    for t, v in cap.items():
        by_month[t[:7]].append(v)
    return [[m, max(v)] for m, v in sorted(by_month.items())]


def main() -> dict:
    p = load_params()
    annual, monthly = product_tables()
    write_csv(OUT / "reserves_annual.csv",
              ["product", "year", "hours", "hourly_market_cost_meur", "mean_price_eur_mw_h",
               "volume_weighted_price", "mean_volume_mw", "revenue_1mw_all_hours_keur_yr"], annual)
    write_csv(OUT / "reserves_monthly_price.csv", ["product", "month", "mean_price_eur_mw_h"], monthly)
    cap = storage_capacity()
    write_csv(OUT / "storage_capacity_monthly.csv", ["month", "max_installed_mw"], cap)

    totals = defaultdict(float)
    for r in annual:
        totals[r[1]] += r[3]
    pool25 = p["reserve_pool"]["procurement_cost_2025_meur"]["value"]
    # Hourly markets are a subset of Fingrid's total (no FFR, yearly FCR, weekly mFRR, aFRR energy)
    assert totals["2025"] <= pool25 * 1.05, (totals["2025"], pool25)
    return {"annual": annual, "totals": dict(totals), "capacity": cap, "pool_2025": pool25}


if __name__ == "__main__":
    o = main()
    print("Hourly-market capacity procurement cost (MEUR):", {k: round(v, 1) for k, v in o["totals"].items()})
    print(f"Fingrid total reserve procurement 2025: {o['pool_2025']} MEUR")
    for r in o["annual"]:
        print(f"{r[0]:<20} {r[1]}  cost {r[3]:6.1f} MEUR  mean price {r[4]:7.2f}  "
              f"mean vol {r[6]:6.1f} MW  1 MW all hours {r[7]:6.1f} kEUR/yr")
    print("Installed storage (dataset 424):", o["capacity"][0], "->", o["capacity"][-1])
