"""Balancing-energy pool: activated volume x premium over day-ahead (FI, 2023-2026).

mFRR: up/down activated volumes 375/376 (MW, max power of the quarter hour, from
4 Mar 2025; energy approximated as MW x 0.25 h, an upper approximation) and the
hourly ordered regulation 33/34 (MWh/h) before that; prices 244/106 (EUR/MWh).
aFRR: activation volumes 349/350 (MW per 15-min step from 12 Jun 2024; early rows
carry 1-h end times but are stepped every 15 min, so each row is weighted 0.25 h)
and VWAP energy prices 347/348.

Premium over day-ahead is the value a flexible resource earns by selling up-
regulation above, or buying down-regulation below, the day-ahead price it could
otherwise trade at. The pool is the total across all providers (batteries compete
with hydro, boilers and others for it).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from common import OUT, write_csv
from fleet_ops import _to_15min, load_da_15min, load_fingrid

EAM_START = datetime(2025, 3, 4, 12, 0, tzinfo=timezone.utc)


def _step_series(ds: int) -> dict[datetime, float]:
    """Value per row start (no expansion); used for 15-min stepped volume series."""
    return {a: v for a, _, v in load_fingrid(ds)}


def mfrr_volumes() -> tuple[dict, dict]:
    """Up and down activated energy (MWh) per 15-min slot, both positive."""
    up, dn = defaultdict(float), defaultdict(float)
    for ds, target, sign in ((34, up, 1), (33, dn, -1)):          # hourly MWh, before EAM
        for a, b, v in load_fingrid(ds):
            if a >= EAM_START:
                continue
            e = max(sign * v, 0.0)
            n = max(1, int((b - a).total_seconds() // 900))
            for k in range(n):
                target[a.replace() if k == 0 else a + (b - a) * k / n] += e / n
    for ds, target, sign in ((375, up, 1), (376, dn, -1)):        # MW per quarter, from EAM
        for a, _, v in load_fingrid(ds):
            if a >= EAM_START:
                target[a] += max(sign * v, 0.0) * 0.25
    return up, dn


def afrr_volumes() -> tuple[dict, dict]:
    up = {a: max(v, 0.0) * 0.25 for a, v in _step_series(349).items()}
    dn = {a: max(-v, 0.0) * 0.25 for a, v in _step_series(350).items()}
    return up, dn


def pool(vol_up, vol_dn, p_up, p_dn, da) -> dict[str, list[float]]:
    """Annual [MWh up, MWh down, EUR up premium, EUR down premium, slots priced]."""
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for t in set(vol_up) | set(vol_dn):
        if t not in da:
            continue
        y = t.strftime("%Y")
        a = acc[y]
        eu, ed = vol_up.get(t, 0.0), vol_dn.get(t, 0.0)
        if eu > 0 and t in p_up:
            a[0] += eu
            a[2] += eu * (p_up[t] - da[t])
        if ed > 0 and t in p_dn:
            a[1] += ed
            a[3] += ed * (da[t] - p_dn[t])
        a[4] += 1
    return acc


def main() -> dict:
    da = load_da_15min()
    mu, md = mfrr_volumes()
    m = pool(mu, md, _to_15min(load_fingrid(244)), _to_15min(load_fingrid(106)), da)
    au, ad = afrr_volumes()
    a = pool(au, ad, _step_series(347), _step_series(348), da)
    rows = []
    for name, acc in (("mFRR", m), ("aFRR", a)):
        for y in sorted(acc):
            uv, dv, ue, de, n = acc[y]
            rows.append([name, y, uv / 1000, dv / 1000, ue / 1e6, de / 1e6, (ue + de) / 1e6,
                         ue / uv if uv else 0.0, de / dv if dv else 0.0])
    write_csv(OUT / "balancing_pool_annual.csv",
              ["product", "year", "up_gwh", "down_gwh", "up_premium_meur", "down_premium_meur",
               "pool_meur", "avg_up_premium_eur_mwh", "avg_down_premium_eur_mwh"], rows)
    fleet = supportable_fleet() if (OUT / "da_arbitrage_annual.csv").is_file() else []
    return {"rows": rows, "fleet": fleet}




# ------------------------------------------------- first-order free-entry check
def supportable_fleet(capture: float = 0.85, shares=(0.5, 1.0), ref_year: str = "2026",
                      ref_days: int = 318) -> list[list]:
    """GW of batteries the measured market could support with the market held fixed.

    Per-MW revenue = capture x DA arbitrage margin (perfect foresight, own MW) +
    share x (reserve capacity pool + balancing energy pool) / fleet. Setting it equal
    to the annualised cost gives fleet = share x pool / (cost - capture x margin).
    Ignores further cannibalisation of spreads and pools by the fleet itself, stacking
    conflicts between products, and changes in VRE / demand: an upper-leaning check,
    not a projection.
    """
    import csv

    import cost
    from common import load_params

    p = load_params()
    out = OUT
    f = 365 / ref_days
    rp = sum(float(r["hourly_market_cost_meur"]) for r in csv.DictReader(open(out / "reserves_annual.csv"))
             if r["year"] == ref_year)
    bp = sum(float(r["pool_meur"]) for r in csv.DictReader(open(out / "balancing_pool_annual.csv"))
             if r["year"] == ref_year)
    pool = (rp + bp) * f
    da = {r["duration_h"]: float(r["margin_keur_per_mw_yr"])
          for r in csv.DictReader(open(out / "da_arbitrage_annual.csv")) if r["year"] == ref_year}
    rows = []
    for y in (2026, 2028, 2030, 2035, 2040):
        for d in (1, 2, 4, 8):
            for case in ("low", "central", "high"):
                a = cost.annualised_cost_per_kw(p, y, d, case)["total"]
                gap = a - capture * da[str(d)]
                row = [y, d, case, a, capture * da[str(d)], pool]
                row += [(s * pool / gap if gap > 0 else float("inf")) for s in shares]
                rows.append(row)
    write_csv(out / "supportable_fleet.csv",
              ["cost_year", "duration_h", "cost_case", "annual_cost_keur_mw", "da_margin_captured_keur_mw",
               "pool_meur_" + ref_year] + [f"fleet_gw_share_{s:g}" for s in shares], rows)
    return rows


if __name__ == "__main__":
    for r in main()["rows"]:
        print(f"{r[0]} {r[1]}  up {r[2]:6.1f} GWh  down {r[3]:6.1f} GWh  premium up {r[4]:6.1f} "
              f"down {r[5]:6.1f}  pool {r[6]:6.1f} MEUR  avg {r[7]:6.1f} / {r[8]:6.1f} EUR/MWh")
