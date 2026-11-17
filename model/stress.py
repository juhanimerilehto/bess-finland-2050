"""Stress cases on the M3 central path (one change at a time) and seed stability of the pool."""
from __future__ import annotations

import copy

import numpy as np

import f_equilibrium as m3
from common import OUT, load_params, write_csv

YEARS = list(m3.YEARS)


def _case(p, A, m, label, **chg):
    s = m3.central_sample(p)
    for k, v in chg.items():
        s[k] = np.array([float(v)])
    r = m3.simulate(p, s, A, m)
    row = [label]
    for y in (2030, 2035, 2040, 2050):
        i = YEARS.index(y)
        row += [float(r["gw"][0, i]), float(r["gwh"][0, i])]
    row.append(float(r["dstar"][0, YEARS.index(2050)]))
    first4 = next((y for y, d in zip(YEARS, r["dstar"][0]) if d >= 4), None)
    row.append(first4 if first4 else "")
    return row


def _chem(p, mult, perf):
    q = copy.deepcopy(p)
    q["learning"]["method_C_component"]["cell_cost_multiplier_after_2030"]["value"] = mult
    if perf:
        o = q["operation"]
        o["aux_kw_per_mwh"]["central"] *= 0.6
        o["cycle_life"]["path"] = {k: [x * 1.5 for x in v] for k, v in o["cycle_life"]["path"].items()}
        o["rte"]["gain_pp"] = {"2025": 0, "2035": 4, "2050": 6}
    return q


def stress_cases(p):
    f = p["forecast"]["m3_equilibrium"]
    A = m3.cost_table(p, f["durations_h"])
    m = m3.da_margins()
    rows = [
        _case(p, A, m, "Central"),
        _case(p, A, m, "Slow global learning (high cost path)", cost_u=1),
        _case(p, A, m, "Fast global learning (low cost path)", cost_u=-1),
        _case(p, A, m, "Sticky European premium", premium_u=1),
        _case(p, A, m, "Fast premium convergence", premium_u=-1),
        _case(p, A, m, "Slow learning and sticky premium", cost_u=1, premium_u=1),
        _case(p, A, m, "Fast learning and premium convergence", cost_u=-1, premium_u=-1),
        _case(p, A, m, "Cheap capital (WACC 5%; 25-year life)", finance_u=-1),
        _case(p, A, m, "Dear capital (WACC 10%; 15-year life)", finance_u=1),
        _case(p, A, m, "Low domestic project costs (grid connection 100 EUR/kW)", project_u=-1),
        _case(p, A, m, "High domestic project costs (grid connection 250 EUR/kW)", project_u=1),
        _case(p, A, m, "No shrinkage of the flexibility pool", eps_pool_sat=0.0),
        _case(p, A, m, "Strong shrinkage of the flexibility pool (0.9)", eps_pool_sat=f["eps_pool_sat"][2]),
        _case(p, A, m, "Strong cannibalisation (eps_own 1.2)", eps_own=f["eps_own"][2]),
        _case(p, A, m, "Weak cannibalisation (eps_own 0.3)", eps_own=f["eps_own"][0]),
        _case(p, A, m, "Reserve pool collapse (share 0.3; no growth)", pool_share=f["pool_share"][0], eps_pool=0),
        _case(p, A, m, "Data-centre-led demand (200 TWh in 2050; VRE x1.6)",
              cons_2035=f["consumption_twh"]["2035"][2], cons_2050=f["consumption_twh"]["2050"][2],
              vre_mult=f["vre_share_mult_2050"][2]),
        _case(p, A, m, "No pumped hydro", phes=0),
        _case(p, A, m, "Both pumped-hydro projects (1.05 GW)", phes=1.05),
        _case(p, A, m, "Strong duration tilt (0.3)", duration_tilt=f["duration_tilt"][2]),
        _case(p, A, m, "Low capture (0.75)", capture=f["capture"][0]),
    ]
    qe = copy.deepcopy(p)
    qe["forecast"]["m3_equilibrium"]["cannibal_basis"] = "energy"
    rows.append(_case(qe, A, m, "Arbitrage cannibalised by energy (GWh) rather than power"))
    for label, mult, perf in (("Post-lithium chemistry: cells -50% from 2030 and better performance", 0.5, True),
                              ("Bound: cells free of charge from 2030", 0.0, False)):
        q = _chem(p, mult, perf)
        rows.append(_case(q, m3.cost_table(q, f["durations_h"]), m, label))
    hdr = ["case"]
    for y in (2030, 2035, 2040, 2050):
        hdr += [f"gw_{y}", f"gwh_{y}"]
    hdr += ["new_build_h_2050", "first_year_4h"]
    write_csv(OUT / "stress_cases.csv", hdr, rows)
    return rows


def seed_stability(p, seeds=(20261116, 1, 2, 3, 4), n=4000):
    import ensemble
    rows = []
    for sd in seeds:
        q = copy.deepcopy(p)
        q["forecast"]["mc"]["seed"] = sd
        o = ensemble.main(q, n=n, write=False)
        for r in o["states"]:
            rows.append([sd, r[0], r[2], r[3], r[4], r[6]])
    write_csv(OUT / "seed_stability.csv", ["seed", "year", "gw_low", "gw_base", "gw_high", "gwh_base"], rows)
    out = []
    for y in sorted({r[1] for r in rows}):
        b = np.array([r[3] for r in rows if r[1] == y])
        h = np.array([r[4] for r in rows if r[1] == y])
        out.append([y, float((b.max() - b.min()) / b.mean() * 100), float((h.max() - h.min()) / h.mean() * 100)])
    write_csv(OUT / "seed_stability_summary.csv", ["year", "base_range_pct", "high_range_pct"], out)
    return out


def pool_sensitivity(p, n=4000):
    """Pooled states under alternative pooling and M4 peer normalisation."""
    import ensemble
    variants = [("Central pool", {}),
                ("M4 peers scaled by electricity consumption", {"peer": 1.0}),
                ("M4 peers half by population and half by consumption", {"peer": 0.5}),
                ("Equal weights for M2 and M5 after 2040", {"w": {}})]
    rows = []
    for label, v in variants:
        q = copy.deepcopy(p)
        if "peer" in v:
            q["forecast"]["m4_topdown"]["peer_share_per_twh"] = v["peer"]
        if "w" in v:
            q["forecast"]["ensemble"]["weights_from_2040"] = v["w"]
        o = ensemble.main(q, n=n, write=False)
        st = {r[0]: r for r in o["states"]}
        m4 = {r[0]: r for r in o["by_method"] if r[1] == "M4"}
        rows.append([label] + [st[y][k] for y in (2035, 2050) for k in (2, 3, 4)] + [m4[2050][3]])
    write_csv(OUT / "pool_sensitivity.csv", ["variant", "low_2035", "base_2035", "high_2035",
                                             "low_2050", "base_2050", "high_2050", "m4_p50_2050"], rows)
    return rows


def main():
    p = load_params()
    r = stress_cases(p)
    s = seed_stability(p)
    pool_sensitivity(p)
    return r, s


if __name__ == "__main__":
    r, s = main()
    for row in r:
        print(f"{row[0]:<58} 2035 {row[3]:5.2f} GW {row[4]:5.1f} GWh | 2050 {row[7]:5.2f} GW {row[8]:5.1f} GWh | d {row[9]:.0f} h, 4h from {row[10]}")
    for row in s:
        print(f"{row[0]} base range {row[1]:.1f}%  high range {row[2]:.1f}%")
