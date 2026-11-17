"""Write every model-derived number and table used in the paper to paper/generated/.

numbers.tex defines \\val{key} lookups; tab_*.tex are complete tabular bodies. The
manuscript contains no hand-typed model output, so text cannot drift from the run.
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np

import cost
import f_equilibrium as m3
import f_methods
from common import OUT, PARAMS_PATH, ROOT, load_params, sha256_prefix

GEN = ROOT / "paper" / "generated"
CK = [2030, 2035, 2040, 2045, 2050]
VALS: dict[str, str] = {}


def rd(name):
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def put(key, v, nd=1):
    if isinstance(v, str):
        VALS[key] = v
    else:
        s = f"{v:,.{nd}f}".replace(",", "\\,")
        VALS[key] = s.replace("-", "$-$") if s.startswith("-") else s


def f1(v, nd=1):
    s = f"{float(v):,.{nd}f}".replace(",", "\\,")
    return s.replace("-", "$-$") if s.startswith("-") else s


def write(name, body):
    (GEN / name).write_text(body, encoding="utf-8")


# ------------------------------------------------------------------ forecast
def forecast_numbers(p):
    st = rd("ensemble_states.csv")
    lines = []
    s0 = p["forecast"]["start"]
    put("start-gw", s0["stock_gw"], 2)
    put("start-gwh", s0["stock_gwh"], 2)
    put("start-h", s0["stock_gwh"] / s0["stock_gw"], 1)
    lines.append(f"2026 (observed) & -- & & \\textbf{{{f1(s0['stock_gw'], 2)}}} & & & \\textbf{{{f1(s0['stock_gwh'], 1)}}} & & {f1(s0['stock_gwh'] / s0['stock_gw'])} \\\\")
    for r in st:
        y = r["year"]
        for k in ("gw", "gwh", "dur"):
            for c in ("low", "base", "high"):
                put(f"st-{k}-{c}-{y}", float(r[f"{k}_{c}"]), 1)
        put(f"vin-p50-{y}", float(r["vincent_gw_p50"]), 1)
        meth = r["methods"].replace("+", ", ")
        lines.append(f"{y} & {meth} & {f1(r['gw_low'])} & \\textbf{{{f1(r['gw_base'])}}} & {f1(r['gw_high'])} & "
                     f"{f1(r['gwh_low'])} & \\textbf{{{f1(r['gwh_base'])}}} & {f1(r['gwh_high'])} & {f1(r['dur_base'])} \\\\")
    write("tab_states.tex", "\n".join(lines) + "\n")

    bm = rd("ensemble_by_method.csv")
    for r in bm:
        for k in ("gw", "gwh"):
            for q in ("p10", "p50", "p90"):
                put(f"m-{r['method']}-{k}-{q}-{r['year']}", float(r[f"{k}_{q}"]), 1)
    rows = []
    for m in ("M1", "M2", "M3", "M4", "M5"):
        cells = []
        for y in CK:
            r = next((x for x in bm if x["method"] == m and int(x["year"]) == y), None)
            cells.append(f"{f1(r['gw_p50'])} ({f1(r['gw_p10'])}--{f1(r['gw_p90'])})" if r else "--")
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    for m in ("M1", "M2", "M3", "M4", "M5"):
        cells = []
        for y in CK:
            r = next((x for x in bm if x["method"] == m and int(x["year"]) == y), None)
            cells.append(f"{f1(r['gwh_p50'])} ({f1(r['gwh_p10'])}--{f1(r['gwh_p90'])})" if r else "--")
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    write("tab_methods.tex", "\n".join(rows) + "\n")

    loo = rd("ensemble_leave_one_out.csv")
    mx = {}
    for r in loo:
        put(f"loo-{r['left_out']}-{r['year']}", float(r["gw_p50_shift_pct"]), 0)
        y = int(r["year"])
        if y >= 2035:
            mx[y] = max(mx.get(y, 0), abs(float(r["gw_p50_shift_pct"])))
    put("loo-max-late", max(mx.values()), 0)
    rows = []
    for y in CK:
        cells = []
        for m in ("M1", "M2", "M3", "M4", "M5"):
            r = next((x for x in loo if x["left_out"] == m and int(x["year"]) == y), None)
            cells.append(f"{float(r['gw_p50_shift_pct']):+.0f}".replace("-", "$-$") if r else "--")
        stt = next(x for x in st if int(x["year"]) == y)
        rows.append(f"{y} & {f1(stt['gw_base'])} & " + " & ".join(cells) + f" & {f1(stt['vincent_gw_p50'])} \\\\")
    write("tab_loo.tex", "\n".join(rows) + "\n")

    use = rd("ensemble_electricity_use.csv")
    rows = []
    for r in use:
        y = r["year"]
        for k in ("throughput_twh", "loss_twh", "aux_twh", "net_twh", "share_pct", "minutes_of_load"):
            for q in ("p10", "p50", "p90"):
                put(f"use-{k.split('_')[0]}-{q}-{y}", float(r[f"{k}_{q}"]), 0 if k == "minutes_of_load" else 2)
        if int(y) in CK:
            rows.append(f"{y} & {f1(r['throughput_twh_p50'])} & {f1(r['loss_twh_p50'], 2)} & {f1(r['aux_twh_p50'], 2)} & "
                        f"{f1(r['net_twh_p50'], 2)} ({f1(r['net_twh_p10'], 2)}--{f1(r['net_twh_p90'], 2)}) & "
                        f"{f1(r['share_pct_p50'], 2)} & {f1(r['minutes_of_load_p50'], 0)} ({f1(r['minutes_of_load_p10'], 0)}--{f1(r['minutes_of_load_p90'], 0)}) \\\\")
    write("tab_use.tex", "\n".join(rows) + "\n")
    aux_share = float(use[-1]["aux_twh_p50"]) / (float(use[-1]["aux_twh_p50"]) + float(use[-1]["loss_twh_p50"])) * 100
    put("aux-share-2050", aux_share, 0)
    put("min-load-now", s0["stock_gwh"] / (p["forecast"]["m3_equilibrium"]["consumption_twh"]["2026"] * 1000 / 8760) * 60, 0)

    # M1 / M2 diagnostics
    n = p["forecast"]["mc"]["n"]
    seed = p["forecast"]["mc"]["seed"]
    m1 = f_methods.apply_floor(p, *f_methods.m1_pipeline(p, n, seed))
    i28 = list(m3.YEARS).index(2028)
    g2 = f_methods.m2_diffusion(p, n, seed, m1[0][:, i28])[2]
    q = np.percentile(g2["dt_10_90"], [10, 50, 90])
    put("m2-dt-p10", q[0], 0)
    put("m2-dt-p50", q[1], 1)
    put("m2-dt-p90", q[2], 0)
    fg, _ = f_methods.committed_floor(p)
    for y in (2028, 2035, 2040):
        put(f"floor-{y}", float(fg[list(m3.YEARS).index(y)]), 1)


def m3_numbers(p):
    sob = rd("m3_sobol.csv")
    if sob and "ST_se" in sob[0]:
        put("sobol-se-max", max(max(float(r["S1_se"]), float(r["ST_se"])) for r in sob), 3)
    for r in sob:
        put(f"sob-{r['parameter']}-{r['year']}-{r['metric']}-S1", float(r["S1"]), 2)
        put(f"sob-{r['parameter']}-{r['year']}-{r['metric']}-ST", float(r["ST"]), 2)
    order = sorted({r["parameter"] for r in sob},
                   key=lambda k: -max(float(r["ST"]) for r in sob if r["parameter"] == k))
    labels = {"cost_u": "Technology learning (cells, DC BOS, PCS)", "premium_u": "European premium convergence", "capture": "Arbitrage capture",
              "eps_own": "Own-fleet cannibalisation $\\varepsilon_{\\mathrm{own}}$",
              "eps_flex": "Flexibility-need elasticity $\\varepsilon_{\\mathrm{flex}}$",
              "eps_pool": "Pool growth elasticity $\\varepsilon_{\\mathrm{pool}}$", "pool_share": "Battery share of pool",
              "cons_2035": "Consumption 2035", "cons_2050": "Consumption 2050", "vre_mult": "VRE-share multiplier",
              "build_limit": "Build-rate limit", "phes": "Pumped hydro", "grid_forming": "Grid-forming units",
              "duration_tilt": "Duration tilt $\\tau$", "finance_u": "Financing (WACC, economic life)",
              "project_u": "Domestic project costs (grid connection)",
              "eps_pool_sat": "Pool shrinkage with fleet $\\varepsilon_{\\mathrm{sat}}$"}
    rows = []
    for k in order:
        g = {(int(r["year"]), r["metric"]): r for r in sob if r["parameter"] == k}
        cells = []
        for y in (2035, 2050):
            for mt in ("gw", "gwh"):
                r = g[(y, mt)]
                cells.append(f"{max(float(r['S1']), 0):.2f} & {max(float(r['ST']), 0):.2f}")
        rows.append(f"{labels.get(k, k)} & " + " & ".join(cells) + " \\\\")
    write("tab_sobol.tex", "\n".join(rows) + "\n")

    # duration choice shares
    f = p["forecast"]["m3_equilibrium"]
    A = m3.cost_table(p, f["durations_h"])
    mg = m3.da_margins()
    rng = np.random.default_rng(p["forecast"]["mc"]["seed"])
    s = m3.sample_from_unit(p, rng.random((p["forecast"]["mc"]["n"], len(m3.PARAM_NAMES))))
    r = m3.simulate(p, s, A, mg)
    for y in (2030, 2035, 2040, 2045, 2050):
        i = list(m3.YEARS).index(y)
        d = r["dstar"][:, i]
        for dd in (1, 2, 4, 8):
            put(f"m3d{dd}-{y}", float((d == dd).mean() * 100), 0)
    for k, v in mg.items():
        if k != "q":
            put(f"dam-{k}", v, 1)
    kap = p["forecast"]["m3_equilibrium"]["capture"][1]
    put("dam-inc-24", mg[4] - mg[2], 1)
    put("dam-inc-24-cap", (mg[4] - mg[2]) * kap, 1)
    put("capture-central", kap, 2)
    put("ann-inc-24-2026", cost.annualised_cost_per_kw(p, 2026, 4)["total"]
        - cost.annualised_cost_per_kw(p, 2026, 2)["total"], 1)
    c = m3.simulate(p, m3.central_sample(p), A, mg)
    first4 = next(y for y, d in zip(m3.YEARS, c["dstar"][0]) if d >= 4)
    put("m3c-first4h", str(int(first4)))
    for y in (2030, 2035, 2040, 2050):
        i = list(m3.YEARS).index(y)
        put(f"m3c-gw-{y}", float(c["gw"][0, i]), 1)
        put(f"m3c-gwh-{y}", float(c["gwh"][0, i]), 1)
        put(f"m3c-nstar-{y}", float(c["nstar"][0, i]), 1)
    put("m3c-nstar-2026", float(c["nstar"][0, 0]), 2)

    stress = rd("stress_cases.csv")
    rows = []
    for r in stress:
        lab = r["case"].replace("%", "\\%").replace("-50", "$-$50").replace("eps_own", "$\\varepsilon_{\\mathrm{own}}$").replace("x1.6", "$\\times$1.6")
        rows.append(f"{lab} & {f1(r['gw_2035'])} & {f1(r['gwh_2035'])} & {f1(r['gw_2050'])} & {f1(r['gwh_2050'])} & "
                    f"{int(float(r['new_build_h_2050']))} & {r['first_year_4h'] or '--'} \\\\")
    write("tab_stress.tex", "\n".join(rows) + "\n")
    import re
    for r in stress:
        sl = re.sub(r"[^a-z0-9]+", "", r["case"].lower().split("(")[0])
        for k in ("gw_2035", "gwh_2035", "gw_2050", "gwh_2050"):
            put(f"stress-{sl}-{k.replace('_', '-')}", float(r[k]), 1)
        put(f"stress-{sl}-first4h", r["first_year_4h"] or "--")
    ps = rd("pool_sensitivity.csv") if (OUT / "pool_sensitivity.csv").is_file() else []
    prow = []
    for i, r in enumerate(ps):
        prow.append(f"{r['variant']} & " + " & ".join(f1(r[k]) for k in ("low_2035", "base_2035", "high_2035", "low_2050", "base_2050", "high_2050", "m4_p50_2050")) + " \\\\")
        for k in ("base_2035", "base_2050", "high_2050", "m4_p50_2050"):
            put(f"poolsens{i}-{k.replace('_', '-')}", float(r[k]), 1)
    write("tab_poolsens.tex", "\n".join(prow) + "\n")
    tbn = [float(r["mean_daily_TBd_net_eur_mwh"]) for r in rd("da_arbitrage_annual.csv") if r["duration_h"] == "4"]
    put("tbnet4-min", min(tbn), 0)
    put("tbnet4-max", max(tbn), 0)
    tbg = [float(r["mean_daily_TBd_eur_mwh"]) for r in rd("da_arbitrage_annual.csv") if r["duration_h"] == "4"]
    put("tb4-min", min(tbg), 0)
    put("tb4-max", max(tbg), 0)
    ss = rd("seed_stability_summary.csv")
    put("seed-base-max", max(float(r["base_range_pct"]) for r in ss), 1)
    put("seed-high-max", max(float(r["high_range_pct"]) for r in ss), 1)


# ---------------------------------------------------------------------- cost
def cost_numbers(p):
    a = cost.anchor_2025(p)
    put("cE", a["c_e"], 1)
    put("cP", a["c_p"], 1)
    rows = []
    for y in (2026, 2030, 2035, 2040, 2045, 2050):
        cells = [f1(cost.capex_per_kw(p, y, d)["per_kwh"], 0) for d in (1, 2, 4, 8)]
        lo = cost.capex_per_kw(p, y, 4, "low")["per_kwh"]
        hi = cost.capex_per_kw(p, y, 4, "high")["per_kwh"]
        ann = [f1(cost.annualised_cost_per_kw(p, y, d)["total"], 1) for d in (1, 2, 4, 8)]
        lc = cost.lcos(p, y, 4, 250)["lcos_excl_charging"]
        rows.append(f"{y} & " + " & ".join(cells) + f" & {f1(lo, 0)}--{f1(hi, 0)} & " + " & ".join(ann) + f" & {f1(lc, 0)} \\\\")
        for d in (1, 2, 4, 8):
            put(f"capex-{d}-{y}", cost.capex_per_kw(p, y, d)["per_kwh"], 0)
            put(f"ann-{d}-{y}", cost.annualised_cost_per_kw(p, y, d)["total"], 1)
        put(f"capex-4-low-{y}", lo, 0)
        put(f"capex-4-high-{y}", hi, 0)
        put(f"lcos-4-{y}", lc, 0)
        put(f"lcos-2-{y}", cost.lcos(p, y, 2, 250)["lcos_excl_charging"], 0)
        put(f"cyc-{y}", cost.cycle_cost(p, y)["total"], 1)
    put("charge-adder", cost.lcos(p, 2030, 4, 250)["charging_per_mwh"], 0)
    write("tab_cost.tex", "\n".join(rows) + "\n")
    for y in (2025, 2035, 2050):
        put(f"idxA-{y}", cost.equipment_index(p, y, "central", "A"), 2)
        put(f"idxB-{y}", cost.equipment_index(p, y, "central", "B"), 2)
        put(f"mhr-{y}", cost.marginal_hour_cost(p, y)["ratio"], 2)
    cl = {k: cost.cycle_cost(p, 2025, k)["total"] for k in ("low", "central", "high")}
    put("cyc-2025-low", cl["low"], 1)
    put("cyc-2025-high", cl["high"], 1)
    put("cyc-2025", cl["central"], 1)
    for r in cost.validate_projects(p):
        k = "mert" if str(r[0]).startswith("Mertaniemi") else "terv"
        put(f"val-{k}-model", float(r[1]), 0)
        put(f"val-{k}-obs", float(r[2]), 0)
        put(f"val-{k}-ratio", float(r[3]), 2)
    be = rd("breakeven_spread.csv")
    for r in be:
        if r["case"] == "central":
            put(f"be-{r['duration_h']}-{r['year']}", float(r["required_net_spread_eur_mwh"]), 0)


# -------------------------------------------------------------------- market
def market_numbers(p):
    res = rd("reserves_annual.csv")
    bal = rd("balancing_pool_annual.csv")
    da = rd("da_arbitrage_annual.csv")
    yrs = ["2023", "2024", "2025", "2026"]
    resp = {y: sum(float(r["hourly_market_cost_meur"]) for r in res if r["year"] == y) for y in yrs}
    # 2026 to date -> annualise using hours column
    h26 = max(float(r["hours"]) for r in res if r["year"] == "2026")
    resp_ann = dict(resp)
    resp_ann["2026"] = resp["2026"] * 8760 / h26
    put("res-hours-2026", h26 / 24, 0)
    for y in yrs:
        put(f"respool-{y}", resp_ann[y], 1)
    prods = sorted({r["product"] for r in bal})
    bp = {(r["product"], r["year"]): float(r["pool_meur"]) * (8760 / h26 if r["year"] == "2026" else 1) for r in bal}
    rows = []
    for y in yrs:
        mf = bp.get(("mFRR", y))
        af = bp.get(("aFRR", y))
        tot = resp_ann[y] + (mf or 0) + (af or 0)
        put(f"mfrrpool-{y}", mf or 0, 1)
        put(f"afrrpool-{y}", af or 0, 1)
        put(f"totpool-{y}", tot, 0)
        dm = {int(r["duration_h"]): r for r in da if r["year"] == y}
        cells = [f1(dm[d]["margin_keur_per_mw_yr"]) for d in (1, 2, 4, 8)]
        rat = [float(dm[d]["margin_to_cost_ratio"]) for d in (1, 2, 4, 8)]
        for d in (1, 2, 4, 8):
            put(f"da-{d}-{y}", float(dm[d]["margin_keur_per_mw_yr"]), 1)
            put(f"daratio-{d}-{y}", float(dm[d]["margin_to_cost_ratio"]) * 100, 0)
        lab = y + ("$^{a}$" if y == "2026" else "")
        rows.append(f"{lab} & {f1(resp_ann[y])} & {f1(mf) if mf else '--'} & {f1(af) if af else '--'} & {f1(tot, 0)} & "
                    + " & ".join(cells) + f" & {min(rat) * 100:.0f}--{max(rat) * 100:.0f} \\\\")
    write("tab_market.tex", "\n".join(rows) + "\n")
    allr = [float(r["margin_to_cost_ratio"]) for r in da]
    put("daratio-min", min(allr) * 100, 0)
    put("daratio-max", max(allr) * 100, 0)
    sf = rd("supportable_fleet.csv")
    for r in sf:
        if r["cost_case"] == "central":
            put(f"sf-{r['duration_h']}-{r['cost_year']}-half", float(r["fleet_gw_share_0.5"]), 1)
            put(f"sf-{r['duration_h']}-{r['cost_year']}-full", float(r["fleet_gw_share_1"]), 1)
    cap = rd("storage_capacity_monthly.csv")
    put("cap-first", float(cap[0]["max_installed_mw"]), 0)
    put("cap-first-month", cap[0]["month"])


def comparison_numbers(p):
    ext = json.load(open(ROOT / "data" / "inputs" / "external_comparisons.json", encoding="utf-8"))
    rows = []
    for r in ext["regions_now"]:
        kw = r["gw"] / r["pop_m"]
        rows.append(f"{r['region']} & {r['when']} & {f1(r['gw'], 2 if r['gw'] < 10 else 0)} & {f1(kw, 2)} & {r['duration_h']} \\\\")
        put(f"pc-{r['region'].split(' ')[0].lower()}", kw, 2)
    rows.append("\\midrule")
    st = {int(r["year"]): r for r in rd("ensemble_states.csv")}
    s50 = st[2050]
    pop = 5.6
    rows.append(f"Finland, this study (Base; P10--P90) & 2050 & {f1(s50['gw_base'])} & {f1(float(s50['gw_base']) / pop, 2)} "
                f"({f1(float(s50['gw_low']) / pop, 2)}--{f1(float(s50['gw_high']) / pop, 2)}) & {f1(s50['dur_base'])} \\\\")
    put("pc-fi-2050", float(s50["gw_base"]) / pop, 2)
    put("pc-fi-2050-low", float(s50["gw_low"]) / pop, 2)
    put("pc-fi-2050-high", float(s50["gw_high"]) / pop, 2)
    m3 = {int(r["year"]): r for r in rd("ensemble_by_method.csv") if r["method"] == "M3"}
    put("pc-fi-2050-m3", float(m3[2050]["gw_p50"]) / pop, 2)
    for r in ext["regions_2050"]:
        if "gw_range" in r:
            lo, hi = r["gw_range"]
            rows.append(f"{r['region']} & 2050 & {f1(lo)}--{f1(hi)} & {f1(lo / r['pop_m'], 2)}--{f1(hi / r['pop_m'], 2)} & -- \\\\")
        else:
            kw = r["gw"] / r["pop_m"]
            rows.append(f"{r['region']} & 2050 & {f1(r['gw'], 0 if r['gw'] > 100 else 1)} & {f1(kw, 2)} & "
                        f"{'$\\sim$6' if 'Denmark' in r['region'] else ('3.6 (2036)' if 'BNEF' in r['region'] else '--')} \\\\")
            put(f"pc2050-{r['region'].split(',')[0].split(' ')[0].lower()}-{'bnef' if 'BNEF' in r['region'] else 'x'}", kw, 2)
    write("tab_regions.tex", "\n".join(rows) + "\n")

    # cost comparison values
    for y in (2030, 2035, 2040, 2050):
        for c in ("low", "central", "high"):
            put(f"idxC-{c}-{y}", cost.equipment_index(p, y, c, "C"), 2)
        for k in ("low", "high"):
            put(f"idxC-prem{k}-{y}", cost.equipment_index(p, y, "central", "C", premium=k), 2)
        put(f"idxA-{y}", cost.equipment_index(p, y, "central", "A"), 2)
        put(f"idxB-{y}", cost.equipment_index(p, y, "central", "B"), 2)
        comp = cost.components(p, y)
        for k in ("cells", "dc_bos", "premium", "c_e", "c_p"):
            put(f"comp-{k.replace('_', '')}-{y}", comp[k], 0)
    crow = []
    for y in (2025, 2030, 2035, 2040, 2050):
        c = cost.components(p, y)
        t4 = c["c_e"] + c["c_p"] / 4
        il = [cost.equipment_index(p, y, k) for k in ("low", "central", "high")]
        ai = [cost.capex_per_kw(p, y, 4, k)["per_kwh"] for k in ("low", "central", "high")]
        crow.append(f"{y} & {f1(c['cells'], 0)} & {f1(c['freight'], 0)} & {f1(c['dc_bos'], 0)} & {f1(c['premium'], 0)} & "
                    f"{f1(c['c_e'], 0)} & {f1(c['c_p'], 0)} & {f1(t4, 0)} & {il[1]:.2f} ({il[0]:.2f}--{il[2]:.2f}) & "
                    f"{f1(ai[1], 0)} ({f1(ai[0], 0)}--{f1(ai[2], 0)}) \\\\")
    write("tab_components.tex", "\n".join(crow) + "\n")
    a = cost.anchor_2025(p)
    c25 = cost.components(p, 2025)
    for k in ("cells", "dc_bos", "premium", "freight"):
        put(f"comp-{k.replace('_', '')}-2025", c25[k], 1)
    put("prem-share-2025", c25["premium"] / a["c_e"] * 100, 0)
    for y in (2026, 2027):
        put(f"idxC-central-{y}", cost.equipment_index(p, y, "central", "C"), 2)
    put("aux-2025", cost.aux_kw_per_mwh(p, 2025), 1)
    put("aux-2050", cost.aux_kw_per_mwh(p, 2050), 1)

    # specification table: simpler specification (single index, bundled cost axis) vs current
    bfile = OUT / "baseline_simple_spec" / "numbers.tex"
    if bfile.is_file():
        import re as _re
        old = {m.group(1): m.group(2) for m in _re.finditer(r"val@([^}]+)\}\{([^}]*)\}", bfile.read_text(encoding="utf-8"))}
        ov = lambda k: float(old[k].replace("\\,", "").replace("$-$", "-"))
        n_m = {(r["method"], int(r["year"])): r for r in rd("ensemble_by_method.csv")}
        n_m3 = {int(r["year"]): r for r in rd("m3_equilibrium_paths.csv")}
        rows = []
        def line(label, o, n, nd=1):
            ch = (n / o - 1) * 100
            chs = f"{ch:+.0f}".replace("-", "$-$")
            rows.append(f"{label} & {f1(o, nd)} & {f1(n, nd)} & {chs}\\,\\% \\\\")
        for y in (2035, 2050):
            line(f"4\\,h installed cost {y} (EUR/kWh)", ov(f"capex-4-{y}"), cost.capex_per_kw(p, y, 4)["per_kwh"], 0)
            line(f"4\\,h annualised cost {y} (kEUR/MW/yr)", ov(f"ann-4-{y}"), cost.annualised_cost_per_kw(p, y, 4)["total"])
        for y in (2035, 2050):
            line(f"Equilibrium, central path {y} (GW)", ov(f"m3c-gw-{y}"), float(n_m3[y]["central_gw"]))
            line(f"Equilibrium, median {y} (GW)", ov(f"m-M3-gw-p50-{y}"), float(n_m[("M3", y)]["gw_p50"]))
        for y in (2035, 2050):
            for k, lab in (("low", "Low"), ("base", "Base"), ("high", "High")):
                line(f"Pooled {lab} {y} (GW)", ov(f"st-gw-{k}-{y}"), float(st[y][f"gw_{k}"]))
            line(f"Pooled Base {y} (GWh)", ov(f"st-gwh-base-{y}"), float(st[y]["gwh_base"]))
        write("tab_revision.tex", "\n".join(rows) + "\n")
        for k in ("m-M3-gw-p50-2035", "m3c-gw-2035", "st-gw-base-2050", "st-gw-low-2050", "st-gw-base-2035"):
            put(f"old-{k}", ov(k), 1)


def meta_numbers(p):
    put("params-sha", sha256_prefix(PARAMS_PATH))
    for m in ("f_equilibrium", "f_methods", "ensemble", "cost"):
        put(f"sha-{m.replace('_', '')}", sha256_prefix(ROOT / "model" / f"{m}.py"))
    put("version", p["_meta"]["version"])
    put("doi", p["_meta"]["doi"])
    put("ndraws", f"{p['forecast']['mc']['n']:,}".replace(",", "\\,"))
    put("sobolbase", f"{p['forecast']['mc']['sobol_base']:,}".replace(",", "\\,"))
    put("seed", str(p["forecast"]["mc"]["seed"]))
    npar = len(m3.PARAM_NAMES)
    put("m3-npar", str(npar))
    put("sobol-evals", f"{p['forecast']['mc']['sobol_base'] * (npar + 2):,}".replace(",", "\\,"))


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    p = load_params()
    forecast_numbers(p)
    m3_numbers(p)
    cost_numbers(p)
    market_numbers(p)
    comparison_numbers(p)
    meta_numbers(p)
    lines = ["% generated by model/make_numbers.py -- do not edit",
             "\\makeatletter",
             "\\newcommand{\\val}[1]{\\@ifundefined{val@#1}{\\textbf{??#1??}\\PackageWarning{numbers}{undefined value #1}}{\\@nameuse{val@#1}}}"]
    for k, v in sorted(VALS.items()):
        lines.append(f"\\@namedef{{val@{k}}}{{{v}}}")
    lines.append("\\makeatother")
    write("numbers.tex", "\n".join(lines) + "\n")
    print(f"{len(VALS)} values, tables written to {GEN}")


if __name__ == "__main__":
    main()
