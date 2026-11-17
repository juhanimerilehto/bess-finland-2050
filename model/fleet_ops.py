"""Observed operation of the Finnish battery fleet and balancing-energy value.

(a) Fleet operation from Fingrid real-time BESS charging (399) and discharging (398)
    power, 3-min: monthly charged and discharged energy, the discharge/charge ratio,
    NOTE: the ratio exceeds 1 in most months (up to 1.8), which is physically
    impossible for a storage fleet; the real-time charging series evidently does not
    cover all charging (Fingrid: operations-control data, may freeze). The series
    give the scale of activity, not a measured round-trip efficiency.
    equivalent full cycles against installed MW (424) and an assumed duration, and
    charging as a share of national consumption (124).
(b) Balancing-energy premium: mFRR up (244) and down (106) energy prices against the
    day-ahead price (ENTSO-E) per 15 min. The upper-bound value of 1 MW offered in
    both directions is sum(max(up - DA, 0) + max(DA - down, 0)) * dt, i.e. as if
    activated in every interval with a premium, ignoring state of charge and bid
    acceptance. It bounds what balancing energy can add to the arbitrage margin.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from common import OUT, ROOT, write_csv

RAW = ROOT / "data" / "raw"


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_fingrid(ds: int) -> list[tuple[datetime, datetime, float]]:
    out = []
    with open(RAW / f"fingrid_{ds}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.append((_ts(r["startTime"]), _ts(r["endTime"]), float(r["value"])))
            except (TypeError, ValueError):
                continue
    return out


def energy_by_month(rows) -> dict[str, float]:
    """Integrate a power series (MW) to MWh per month (interval = end - start)."""
    acc = defaultdict(float)
    for a, b, v in rows:
        acc[a.strftime("%Y-%m")] += max(v, 0.0) * (b - a).total_seconds() / 3600
    return acc


def coverage_by_month(rows) -> dict[str, float]:
    acc = defaultdict(float)
    for a, b, _ in rows:
        acc[a.strftime("%Y-%m")] += (b - a).total_seconds() / 3600
    return acc


def fleet_ops(duration_h: float = 1.9) -> list[list]:
    dis, chg = load_fingrid(398), load_fingrid(399)
    cons = load_fingrid(124)
    cap = load_fingrid(424)
    e_dis, e_chg = energy_by_month(dis), energy_by_month(chg)
    cov_d, cov_c = coverage_by_month(dis), coverage_by_month(chg)
    # consumption 124 is MWh/h per interval -> energy = value * hours
    e_cons = defaultdict(float)
    for a, b, v in cons:
        e_cons[a.strftime("%Y-%m")] += v * (b - a).total_seconds() / 3600
    cap_m = defaultdict(list)
    for a, _, v in cap:
        cap_m[a.strftime("%Y-%m")].append(v)
    rows = []
    for m in sorted(set(e_dis) & set(e_chg)):
        y, mo = int(m[:4]), int(m[5:7])
        nxt = datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc)
        hours = (nxt - datetime(y, mo, 1, tzinfo=timezone.utc)).total_seconds() / 3600
        cov = min(cov_d[m], cov_c[m]) / hours
        mw = max(cap_m[m]) if cap_m.get(m) else float("nan")
        cycles = e_dis[m] / (mw * duration_h * 0.9) if cap_m.get(m) else float("nan")
        rows.append([m, round(cov, 3), e_chg[m] / 1000, e_dis[m] / 1000,
                     e_dis[m] / e_chg[m] if e_chg[m] else float("nan"),
                     mw, cycles, 100 * e_chg[m] / e_cons[m] if e_cons.get(m) else float("nan")])
    return rows


def load_da_15min() -> dict[datetime, float]:
    """Day-ahead price per 15-min slot (hourly prices repeated before Oct 2025)."""
    da = {}
    with open(RAW / "entsoe_da_FI.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = datetime.strptime(r["utc_start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            res = int(r["resolution_min"])
            for k in range(res // 15):
                da[t + timedelta(minutes=15 * k)] = float(r["price_eur_mwh"])
    return da


def _to_15min(rows) -> dict[datetime, float]:
    """Expand a price series to 15-min slots (hourly values repeated)."""
    out = {}
    for a, b, v in rows:
        n = max(1, int((b - a).total_seconds() // 900))
        for k in range(n):
            out[a + timedelta(minutes=15 * k)] = v
    return out


def balancing_premium() -> list[list]:
    da = load_da_15min()
    up, dn = _to_15min(load_fingrid(244)), _to_15min(load_fingrid(106))
    acc = defaultdict(lambda: [0.0, 0.0, 0, 0, 0, 0.0])
    for t, p in da.items():
        if t not in up or t not in dn:
            continue
        y = t.strftime("%Y")
        a = acc[y]
        pu, pd = max(up[t] - p, 0.0), max(p - dn[t], 0.0)
        a[0] += pu * 0.25
        a[1] += pd * 0.25
        a[2] += 1
        a[3] += pu > 0.01
        a[4] += pd > 0.01
        a[5] += abs(up[t] - p) + abs(p - dn[t])
    rows = []
    for y in sorted(acc):
        su, sd, n, nu, nd, absum = acc[y]
        scale = 35040 / n
        rows.append([y, n, su * scale / 1000, sd * scale / 1000, (su + sd) * scale / 1000,
                     100 * nu / n, 100 * nd / n])
    return rows


def main() -> dict:
    ops = fleet_ops()
    write_csv(OUT / "fleet_ops_monthly.csv",
              ["month", "data_coverage", "charged_gwh", "discharged_gwh", "discharge_to_charge",
               "installed_mw_max", "equiv_full_cycles_at_1.9h", "charging_share_of_consumption_pct"], ops)
    bal = balancing_premium()
    write_csv(OUT / "balancing_premium_annual.csv",
              ["year", "intervals", "up_premium_keur_per_mw_yr", "down_premium_keur_per_mw_yr",
               "upper_bound_keur_per_mw_yr", "pct_intervals_up_premium", "pct_intervals_down_premium"], bal)
    return {"ops": ops, "bal": bal}


if __name__ == "__main__":
    o = main()
    print("month  cov  charged_GWh  discharged_GWh  dis/chg  MW  cycles  %cons")
    for r in o["ops"]:
        print(f"{r[0]} {r[1]:.2f} {r[2]:8.1f} {r[3]:8.1f} {r[4]:6.3f} {r[5]:7.0f} {r[6]:6.1f} {r[7]:5.2f}")
    print("\nyear intervals  up kEUR/MW/yr  down  upper bound  %up  %down")
    for r in o["bal"]:
        print(f"{r[0]} {r[1]:6d} {r[2]:8.1f} {r[3]:8.1f} {r[4]:8.1f} {r[5]:5.1f} {r[6]:5.1f}")
