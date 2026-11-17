"""Day-ahead arbitrage value by duration (perfect-foresight daily LP), 2023-2026.

For each Helsinki delivery day and duration d (h), a 1 MW / d MWh battery chooses
charge c_t and discharge x_t (MW) at the market resolution (60 min to 30 Sep 2025,
15 min after) to maximise
    sum_t p_t (x_t - c_t) dt  -  k_deg * sum_t x_t dt
s.t. e_t = e_{t-1} + sqrt(eta) c_t dt - x_t dt / sqrt(eta),  0 <= e_t <= DoD * d,
     0 <= c_t, x_t <= 1,  e_0 = e_T = 0  (each day stands alone).
Perfect foresight on day-ahead prices is an upper bound for DA-only arbitrage; it
excludes intraday, balancing and reserve revenue and any price impact of the
battery itself. Also reported: the simple top-bottom spread TBd (mean of the d
highest minus the d lowest hourly-equivalent prices per day).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import cost
from common import OUT, ROOT, load_params, pick, write_csv

RAW = ROOT / "data" / "raw"
HEL = ZoneInfo("Europe/Helsinki")
DURATIONS = (1, 2, 4, 8)


def load_da() -> dict[str, list[tuple[float, float]]]:
    """Return {local delivery date: [(dt_hours, price), ...]} in time order."""
    days = defaultdict(list)
    with open(RAW / "entsoe_da_FI.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = datetime.strptime(r["utc_start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            loc = t.astimezone(HEL)
            days[loc.date().isoformat()].append((t, int(r["resolution_min"]) / 60, float(r["price_eur_mwh"])))
    out = {}
    for d, rows in days.items():
        rows.sort()
        out[d] = [(dt, p) for _, dt, p in rows]
    return out


def daily_lp(prices: np.ndarray, dt: float, dur_h: float, rte: float, dod: float, k_deg: float):
    """Solve one day. Returns (margin EUR, discharged MWh, charged MWh) per MW."""
    n = len(prices)
    s = np.sqrt(rte)
    # variables: c (n), x (n), e (n)
    nv = 3 * n
    cvec = np.concatenate([prices * dt, -(prices - k_deg) * dt, np.zeros(n)])  # minimise -profit
    A = lil_matrix((n + 1, nv))
    b = np.zeros(n + 1)
    for t in range(n):
        A[t, 2 * n + t] = 1.0
        if t > 0:
            A[t, 2 * n + t - 1] = -1.0
        A[t, t] = -s * dt
        A[t, n + t] = dt / s
    A[n, 3 * n - 1] = 1.0  # e_T = 0
    bounds = [(0, 1)] * (2 * n) + [(0, dod * dur_h)] * n
    res = linprog(cvec, A_eq=A.tocsr(), b_eq=b, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    c, x = res.x[:n], res.x[n:2 * n]
    margin = float(np.dot(prices, x - c) * dt - k_deg * x.sum() * dt)
    return margin, float(x.sum() * dt), float(c.sum() * dt)


def top_bottom(prices: np.ndarray, dt: float, d: float, rte: float = 1.0) -> float:
    """Mean of the top-d hours minus mean of the bottom-d hours divided by RTE."""
    k = max(1, int(round(d / dt)))
    s = np.sort(prices)
    return float(s[-k:].mean() - s[:k].mean() / rte)


def main() -> dict:
    p = load_params()
    o = p["operation"]
    rte, dod = pick(o["rte"], "central"), pick(o["dod"], "central")
    k_deg = cost.cycle_cost(p, 2026)["degradation"]
    days = load_da()
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0, 0.0])   # margin, MWh out, MWh in, days, TB, TB net
    daily_rows = []
    for day in sorted(days):
        y = day[:4]
        if y < "2023" or day > "2026-11-15":
            continue
        dts = {dt for dt, _ in days[day]}
        if len(dts) != 1:   # mixed-resolution day (should not occur); skip safely
            continue
        dt = dts.pop()
        pr = np.array([v for _, v in days[day]])
        if len(pr) * dt < 23 - 1e-9:
            continue        # incomplete day
        for d in DURATIONS:
            m, out, inn = daily_lp(pr, dt, d, rte, dod, k_deg)
            tb = top_bottom(pr, dt, d)
            tbn = top_bottom(pr, dt, d, rte)
            a = acc[(y, d)]
            a[0] += m; a[1] += out; a[2] += inn; a[3] += 1; a[4] += tb; a[5] += tbn
            daily_rows.append([day, d, m, out, tb])
    rows = []
    for (y, d), (m, out, inn, n, tb, tbn) in sorted(acc.items()):
        scale = 365 / n
        ann_cost = cost.annualised_cost_per_kw(p, 2026, d)["total"]
        gross = (m + k_deg * out) * scale / 1000   # revenue minus charging cost; wear is in capital cost
        rows.append([y, d, n, m * scale / 1000, gross, out * scale, (m / out) if out else 0.0,
                     tb / n, tbn / n, ann_cost, gross / ann_cost])
    write_csv(OUT / "da_arbitrage_annual.csv",
              ["year", "duration_h", "days", "margin_net_of_wear_keur_per_mw_yr", "margin_keur_per_mw_yr",
               "discharged_mwh_per_mw_yr", "margin_per_mwh_discharged", "mean_daily_TBd_eur_mwh",
               "mean_daily_TBd_net_eur_mwh", "annualised_cost_2026_keur_per_mw_yr", "margin_to_cost_ratio"], rows)
    write_csv(OUT / "da_arbitrage_daily.csv", ["date", "duration_h", "margin_eur_per_mw", "discharged_mwh", "TBd"], daily_rows)
    return {"rows": rows, "k_deg": k_deg, "rte": rte, "dod": dod}


if __name__ == "__main__":
    o = main()
    print(f"RTE {o['rte']}, DoD {o['dod']}, degradation {o['k_deg']:.1f} EUR/MWh (2026 central)")
    print("year d  days  margin kEUR/MW/yr  MWh/MW/yr  EUR/MWh  TBd  cost_2026  margin/cost")
    for r in o["rows"]:
        print(f"{r[0]} {r[1]}h {r[2]:4d} net {r[3]:7.1f} gross {r[4]:7.1f} {r[5]:7.0f} TB {r[7]:6.1f} TBnet {r[8]:6.1f} cost {r[9]:7.1f} {r[10]:6.2f}")
