"""Cost and technology module.

Capital cost is split into an energy-proportional part (EUR/kWh) and a
power-proportional part (EUR/kW), each multiplied by an equipment cost index
I_eq(t). Installation/BoP is a share of equipment; grid connection is a
power-side cost that does not follow battery learning. This split is what makes
the economic duration lengthen as batteries get cheaper.

Three learning methods give the equipment cost:
  C  component model (default): cells learn on global Li-ion doublings with a
     materials floor; DC balance of system and power conversion learn on
     stationary doublings; the European premium over Chinese equipment follows
     its own convergence path; 2026-27 follow the observed price rebound.
  A  exogenous index through published forecasts (BNEF 2035, NREL ratios to 2050)
  B  single-index experience curve for cells: I = (1 - LR) ** doublings(t)
A and B are kept as comparison paths.
"""
from __future__ import annotations

from common import BASE_YEAR, OUT, crf, load_params, loglinear, pick, write_csv

CASES = ("low", "central", "high")  # low = low cost (fast learning), high = high cost


# ---------------------------------------------------------------- 2025 anchor
def anchor_2025(p: dict) -> dict:
    """Derive Europe 2025 energy and power cost components (EUR) from BNEF.

    Global 2 h and 4 h turnkey prices give c_E + c_P/2 = p2, c_E + c_P/4 = p4.
    Europe is the global split scaled by the Europe/global average ratio.
    """
    a = p["cost_anchor_2025"]
    fx = p["fx"]["usd_per_eur_2025"]["value"]
    p2, p4 = a["global_turnkey_2h_usd_per_kwh"]["value"], a["global_turnkey_4h_usd_per_kwh"]["value"]
    c_p_usd = 4 * (p2 - p4)             # USD/kW
    c_e_usd = p4 - c_p_usd / 4          # USD/kWh
    avg = a["global_turnkey_avg_usd_per_kwh"]["value"]
    implied_avg_h = c_p_usd / (avg - c_e_usd)
    prem = a["europe_turnkey_avg_usd_per_kwh"]["value"] / avg
    eu4_2025_usd = prem * (c_e_usd + c_p_usd / 4)
    return {
        "c_e": prem * c_e_usd / fx,     # EUR/kWh
        "c_p": prem * c_p_usd / fx,     # EUR/kW
        "europe_premium": prem,
        "global_avg_duration_h": implied_avg_h,
        "europe_4h_2025_usd": eu4_2025_usd,
        "europe_4h_2035_index": a["europe_turnkey_4h_2035_usd_per_kwh"]["value"] / eu4_2025_usd,
    }


# ------------------------------------------------------ component model (C)
_K = {"low": 0, "central": 1, "high": 2}


def _pts(d: dict) -> dict:
    return {int(k): v for k, v in d.items() if not str(k).startswith("_") and k != "class"}


def components(p: dict, year: float, case: str = "central", premium: str | None = None,
               near_term: bool = True) -> dict:
    """European equipment components (EUR2025) for a vintage built in `year`.

    case     learning case for cells, DC BOS and power conversion (low = low cost)
    premium  convergence case for the European premium (defaults to `case`)
    """
    C = p["learning"]["method_C_component"]
    a = anchor_2025(p)
    premium = premium or case
    y = max(float(year), BASE_YEAR)
    cell0 = C["cell_2025_eur_per_kwh"]["value"]
    frt = C["freight_duty_eur_per_kwh"]["value"]
    bos0 = C["dc_bos_2025_eur_per_kwh"]["value"]
    prem0 = a["c_e"] - cell0 - frt - bos0
    d_all = _lin(y, _pts(C["doublings_all_liion"][case]))
    d_st = _lin(y, _pts(C["doublings_stationary"]))
    lr_c = C["cell_learning_rate"][case]
    lr_b = C["bos_pcs_learning_rate"][case]
    cells = max(C["cell_floor_eur_per_kwh"][case], cell0 * (1 - lr_c) ** d_all)
    if case == "high":
        cells *= 1 + C["cell_cycle_premium_high_2030"]["value"] * max(0.0, 1 - abs(y - 2030) / 5)
    m = C.get("cell_cost_multiplier_after_2030", {}).get("value", 1.0)   # stress cases only
    if m != 1.0:
        cells *= 1 + (m - 1) * min(max((y - 2028) / 2, 0.0), 1.0)
    bos = bos0 * (1 - lr_b) ** d_st
    pcs = a["c_p"] * (1 - lr_b) ** d_st
    r1 = C["premium_decline_to_2035"][premium]
    r2 = C["premium_decline_after_2035"][premium]
    prem = prem0 * (1 - r1) ** (min(y, 2035) - BASE_YEAR) * (1 - r2) ** max(y - 2035, 0)
    out = {"cells": cells, "freight": frt, "dc_bos": bos, "premium": prem,
           "c_e": cells + frt + bos + prem, "c_p": pcs}
    if near_term and y < 2030:
        # observed 2026 rebound and 2027 end of the Chinese export rebate, then
        # linear return to the learning path by 2030
        # The rebound is a hardware (traded-equipment) price movement: the index target
        # is defined on the `case` path and met by rescaling cells, freight, DC BOS and
        # PCS only, so the European premium keeps its own convergence path (and its
        # spread across premium cases) in 2026-2029.
        nt = {int(k): v[case] for k, v in C["near_term_index"].items() if k.isdigit()}
        e4 = lambda o: o["c_e"] + o["c_p"] / 4
        hw = lambda o: o["cells"] + o["freight"] + o["dc_bos"] + o["c_p"] / 4
        base = components(p, BASE_YEAR, case, case, near_term=False)
        c30 = components(p, 2030, case, case, near_term=False)
        ref = components(p, y, case, case, near_term=False)
        pts = {2025: 1.0, **nt, 2030: e4(c30) / e4(base)}
        target = _lin(y, pts)
        scale = (target * e4(base) - ref["premium"]) / hw(ref)
        for k in ("cells", "freight", "dc_bos", "c_p"):
            out[k] *= scale
        out["c_e"] = out["cells"] + out["freight"] + out["dc_bos"] + out["premium"]
    return out


def augmentation_share(p: dict, year: float, case: str = "central") -> float:
    A = p["learning"]["method_C_component"]["augmentation_share"]
    return _lin(max(year, BASE_YEAR), {int(k): v[case] for k, v in A.items() if k.isdigit()})


# ------------------------------------------------------------ learning index
def equipment_index(p: dict, year: float, case: str = "central", method: str = "C",
                    premium: str | None = None) -> float:
    """Equipment cost of a 4 h system relative to the 2025 central anchor."""
    if method == "C":
        a = anchor_2025(p)
        c = components(p, year, case, premium)
        return (c["c_e"] + c["c_p"] / 4) / (a["c_e"] + a["c_p"] / 4)
    if method == "A":
        L = p["learning"]["method_A_exogenous"]
        pts = {BASE_YEAR: 1.0, 2035: L["index_2035"][case], 2050: L["index_2050"][case]}
        if case == "central":
            pts[2035] = anchor_2025(p)["europe_4h_2035_index"]  # derived, not typed
        return loglinear(year, pts)
    if method == "B":
        L = p["learning"]["method_B_experience"]
        lr_case = {"low": "high", "central": "central", "high": "low"}[case]  # low cost <- high LR
        lr = L["learning_rate"][lr_case]
        d_pts = {BASE_YEAR: 0.0,
                 2035: L["doublings_2025_2035"][lr_case],
                 2050: L["doublings_2025_2050"][lr_case]}
        d = _lin(year, d_pts)
        return (1 - lr) ** d
    raise ValueError(method)


def _lin(x: float, pts: dict) -> float:
    ys = sorted(pts)
    if x <= ys[0]:
        return pts[ys[0]]
    if x >= ys[-1]:
        return pts[ys[-1]]
    for a, b in zip(ys, ys[1:]):
        if a <= x <= b:
            return pts[a] + (pts[b] - pts[a]) * (x - a) / (b - a)


def back_index_2024(p: dict) -> float:
    """Equipment index for 2024 relative to 2025 from the BNEF Europe y/y decline."""
    return 1.0 / (1.0 - p["cost_anchor_2025"]["europe_yoy_decline_2025"]["value"])


def eq_index(p, year, case="central", method="C", premium=None):
    if year < BASE_YEAR:
        return back_index_2024(p) ** (BASE_YEAR - year)
    return equipment_index(p, year, case, method, premium)


# ------------------------------------------------------------------ capex
def capex_per_kw(p: dict, year: float, duration_h: float, case: str = "central",
                 method: str = "C", include_connection: bool = True, premium: str | None = None,
                 project: str | None = None) -> dict:
    """Installed capital cost per kW of power for a system of given duration.

    Method C: equipment = c_E(t) d (1 + augmentation) + c_P(t); installation is an
    absolute cost (b x 2025 equipment) that decays slowly; grid connection is flat.
    Methods A/B: equipment = (c_E d + c_P) I(t); installation = b x equipment.

    Uncertainty axes (each 'low' | 'central' | 'high', low = low cost):
      case     technology: component learning, materials floor, near-term path, augmentation
      premium  convergence of the European premium (defaults to case)
      project  domestic project costs: grid connection, installation decay (defaults to case)
    """
    a = anchor_2025(p)
    cs = p["cost_structure"]
    premium = premium or case
    project = project or case
    bop = pick(cs["bop_install_share"], "central")
    conn0 = pick(cs["grid_connection_eur_per_kw"], project)
    drift = pick(cs["connection_cost_drift_per_yr"], project)
    if method == "C":
        if year < BASE_YEAR:
            ieq = back_index_2024(p) ** (BASE_YEAR - year)
            ce, cp, aug = a["c_e"] * ieq, a["c_p"] * ieq, augmentation_share(p, BASE_YEAR, case)
            install = bop * (ce * duration_h + cp)
        else:
            c = components(p, year, case, premium)
            ce, cp, aug = c["c_e"], c["c_p"], augmentation_share(p, year, case)
            ieq = (ce + cp / 4) / (a["c_e"] + a["c_p"] / 4)
            g = p["learning"]["method_C_component"]["install_decay_per_yr"][project]
            install = bop * (a["c_e"] * duration_h + a["c_p"]) * (1 + g) ** (year - BASE_YEAR)
        equip = ce * duration_h * (1 + aug) + cp
    else:
        ieq = eq_index(p, year, case, method)
        equip = (a["c_e"] * duration_h + a["c_p"]) * ieq
        install = bop * equip
    conn = conn0 * (1 + drift) ** (year - BASE_YEAR) if include_connection else 0.0
    total = equip + install + conn
    return {"equipment": equip, "install": install, "connection": conn, "total": total,
            "per_kwh": total / duration_h, "index": ieq}


# ------------------------------------------------- time-varying performance
_FLIP = {"low": "high", "central": "central", "high": "low"}   # low cost <- good performance


def rte_at(p: dict, year: float, case: str = "central") -> float:
    """AC round-trip efficiency at grid connection for a vintage built in `year`."""
    r = p["operation"]["rte"]
    g = r.get("gain_pp", {"2025": 0})
    pts = {int(k): v for k, v in g.items()}
    ys = sorted(pts)
    if year <= ys[0]:
        gain = pts[ys[0]]
    elif year >= ys[-1]:
        gain = pts[ys[-1]]
    else:
        for a, b in zip(ys, ys[1:]):
            if a <= year <= b:
                gain = pts[a] + (pts[b] - pts[a]) * (year - a) / (b - a)
    return min(0.95, pick(r, _FLIP[case]) + gain / 100)


def cycle_life_at(p: dict, year: float, case: str = "central") -> float:
    c = p["operation"]["cycle_life"]
    path = c.get("path")
    idx = {"low": 0, "central": 1, "high": 2}[_FLIP[case]]
    if not path:
        return pick(c, _FLIP[case])
    pts = {int(k): v[idx] for k, v in path.items()}
    return loglinear(year, pts)


def aux_kw_per_mwh(p: dict, year: float = BASE_YEAR, case: str = "central") -> float:
    o = p["operation"]["aux_kw_per_mwh"]
    f = o.get("path_factor")
    fac = _lin(max(year, BASE_YEAR), {int(k): v for k, v in f.items()}) if f else 1.0
    return pick(o, case) * fac


def aux_cost_per_kw(p: dict, duration_h: float, case: str = "central",
                    price: float | None = None, year: float = BASE_YEAR) -> float:
    """Annual auxiliary electricity cost (EUR/kW-yr): aux kW/MWh x d x 8760 h x price."""
    o = p["operation"]
    aux = aux_kw_per_mwh(p, year, case)
    pr = price if price is not None else pick(o["charge_price_eur_per_mwh"], "central")
    return aux * duration_h * 8.76 * pr / 1000


# ---------------------------------------------------------------- cycle cost
def cycle_cost(p: dict, year: float, case: str = "central", method: str = "C",
               charge_price: float | None = None, premium: str | None = None) -> dict:
    """Marginal cost of one discharged MWh (EUR/MWh): degradation + efficiency loss.

    c = C_repl(t) / (N_cycles * DoD)  +  (1/RTE - 1) * p_charge
    """
    o = p["operation"]
    if method == "C" and year >= BASE_YEAR:
        ridx = components(p, year, case, premium)["c_e"] / anchor_2025(p)["c_e"]
    else:
        ridx = eq_index(p, year, case, method)
    repl = pick(o["replacement_cost_eur_per_kwh_2025"], case) * ridx
    life = cycle_life_at(p, year, case)
    dod = pick(o["dod"], _FLIP[case])
    rte = rte_at(p, year, case)
    pc = charge_price if charge_price is not None else pick(o["charge_price_eur_per_mwh"], case)
    degr = repl * 1000 / (life * dod)
    loss = (1 / rte - 1) * pc
    return {"degradation": degr, "efficiency_loss": loss, "total": degr + loss}


# ---------------------------------------------------------------------- LCOS
def annualised_cost_per_kw(p: dict, year: float, duration_h: float, case: str = "central",
                           method: str = "C", premium: str | None = None, finance: str | None = None,
                           project: str | None = None) -> dict:
    """Annualised cost of ownership (EUR/kW/yr).

    finance  cost of capital and economic life (defaults to case)
    project  grid connection, installation decay, fixed O&M share, auxiliary load (defaults to case)
    """
    cs = p["cost_structure"]
    finance = finance or case
    project = project or case
    c = capex_per_kw(p, year, duration_h, case, method, premium=premium, project=project)
    r = pick(cs["wacc"], finance)
    life = pick(cs["economic_life_yr"], _FLIP[finance])
    fom = pick(cs["fixed_om_share"], project) * (c["equipment"] + c["install"])
    fee = cs["tso_storage_fee_eur_per_kw_yr"]["value"]
    cap = c["total"] * crf(r, life)
    aux = aux_cost_per_kw(p, duration_h, project, year=year) if "aux_kw_per_mwh" in p["operation"] else 0.0
    return {"capital": cap, "fixed_om": fom, "tso_fee": fee, "aux": aux, "total": cap + fom + fee + aux}


def lcos(p: dict, year: float, duration_h: float, cycles_per_year: float,
         case: str = "central", method: str = "C", charge_price: float | None = None) -> dict:
    """Levelised cost per discharged MWh, with and without charging energy."""
    o = p["operation"]
    ann = annualised_cost_per_kw(p, year, duration_h, case, method)["total"]
    dod = pick(o["dod"], case if case == "central" else {"low": "high", "high": "low"}[case])
    rte = rte_at(p, year, case)
    pc = charge_price if charge_price is not None else pick(o["charge_price_eur_per_mwh"], case)
    mwh_out_per_kw = duration_h * dod * cycles_per_year / 1000   # MWh per kW per year
    fixed_part = ann / mwh_out_per_kw
    charging = pc / rte
    return {"fixed_per_mwh": fixed_part, "charging_per_mwh": charging,
            "lcos_excl_charging": fixed_part, "lcos_incl_charging": fixed_part + charging}


# --------------------------------------------------------- duration economics
def marginal_hour_cost(p: dict, year: float, case: str = "central", method: str = "C") -> dict:
    """Cost of one extra hour of storage per kW vs the power-side cost per kW.

    The ratio falls over time because battery learning reduces the energy part
    while grid connection does not learn: this is the duration pressure.
    """
    one_h = capex_per_kw(p, year, 1.0, case, method)
    two_h = capex_per_kw(p, year, 2.0, case, method)
    extra_hour = two_h["total"] - one_h["total"]
    power_side = one_h["total"] - extra_hour
    return {"extra_hour_eur_per_kw": extra_hour, "power_side_eur_per_kw": power_side,
            "ratio": extra_hour / power_side}


# ------------------------------------------------------- break-even spreads
def breakeven_spread(p: dict, year: float, duration_h: float, case: str = "central",
                     method: str = "C", cycles_per_year: float = 365) -> dict:
    """Captured net spread (EUR/MWh discharged) at which arbitrage alone covers cost.

    Net spread = average discharge price - average charge price / RTE over the cycled
    hours. Cell wear is not charged again here: at Finnish utilisation calendar life
    binds, and capacity fade is covered by the augmentation allowance inside the
    annualised capital cost. The marginal degradation cost is reported separately.
    """
    ann = annualised_cost_per_kw(p, year, duration_h, case, method)["total"]
    dod = pick(p["operation"]["dod"], "central")
    mwh = duration_h * dod * cycles_per_year / 1000
    degr = cycle_cost(p, year, case, method)["degradation"]
    return {"annual_cost_eur_per_kw": ann, "fixed_per_mwh": ann / mwh,
            "degradation_per_mwh": degr, "required_net_spread": ann / mwh}


# --------------------------------------------------------------- validation
def validate_projects(p: dict) -> list[list]:
    rows = []
    v = p["validation_projects"]
    m = v["mertaniemi"]
    d = m["mwh"] / m["mw"]
    obs_kw = m["meur"] * 1e6 / (m["mw"] * 1e3)
    mod = capex_per_kw(p, m["contract_year"], d, include_connection=False)
    rows.append(["Mertaniemi 2024, %.2f h, EUR/kW (excl. connection)" % d, mod["total"], obs_kw, mod["total"] / obs_kw])
    t = v["tervola_varevaara"]
    d = t["mwh"] / t["mw"]
    obs_kw = t["meur_min"] * 1e6 / (t["mw"] * 1e3)
    mod = capex_per_kw(p, t["contract_year"], d, include_connection=False)
    rows.append(["Tervola 2026, %.2f h, EUR/kW (excl. connection; obs. is a floor incl. services)" % d,
                 mod["total"], obs_kw, mod["total"] / obs_kw])
    return rows


# ---------------------------------------------------------------------- run
YEARS = list(range(2025, 2051))
DURATIONS = (1, 2, 4, 8)


def main() -> dict:
    p = load_params()
    a = anchor_2025(p)
    out = {"anchor": a}

    rows = []
    for y in YEARS:
        for m in ("C", "A", "B"):
            for c in CASES:
                ieq = eq_index(p, y, c, m)
                row = [y, m, c, ieq]
                for d in DURATIONS:
                    k = capex_per_kw(p, y, d, c, m)
                    row += [k["total"], k["per_kwh"]]
                cc = cycle_cost(p, y, c, m)
                mh = marginal_hour_cost(p, y, c, m)
                row += [cc["degradation"], cc["efficiency_loss"], cc["total"], mh["ratio"]]
                rows.append(row)
    hdr = ["year", "method", "case", "equipment_index"]
    for d in DURATIONS:
        hdr += [f"capex_{d}h_eur_per_kw", f"capex_{d}h_eur_per_kwh"]
    hdr += ["cycle_degradation_eur_mwh", "cycle_loss_eur_mwh", "cycle_cost_eur_mwh", "extra_hour_to_power_cost"]
    write_csv(OUT / "cost_paths.csv", hdr, rows)

    lrows = []
    for y in (2025, 2030, 2035, 2040, 2050):
        for d in DURATIONS:
            for cyc in (250, 365):
                l = lcos(p, y, d, cyc)
                lrows.append([y, d, cyc, l["lcos_excl_charging"], l["lcos_incl_charging"]])
    write_csv(OUT / "lcos_central.csv", ["year", "duration_h", "cycles_per_year", "lcos_excl_charging", "lcos_incl_charging"], lrows)

    brows = []
    for y in (2026, 2030, 2035, 2040, 2045, 2050):
        for d in DURATIONS:
            for c in CASES:
                b = breakeven_spread(p, y, d, c)
                brows.append([y, d, c, b["annual_cost_eur_per_kw"], b["required_net_spread"]])
    write_csv(OUT / "breakeven_spread.csv",
              ["year", "duration_h", "case", "annual_cost_eur_per_kw_yr", "required_net_spread_eur_mwh"], brows)

    vrows = validate_projects(p)
    write_csv(OUT / "cost_validation.csv", ["case", "model", "observed", "ratio"], vrows)
    out["validation"] = vrows

    # Retained check: blog's 15-30 EUR/MWh cycle cost at current equipment prices
    cc25 = {c: cycle_cost(p, 2025, c)["total"] for c in CASES}
    lo_b, hi_b = p["retained_blog"]["cycle_cost_eur_per_mwh"]
    assert lo_b <= cc25["central"] <= hi_b, cc25
    assert cc25["low"] <= lo_b and cc25["high"] >= hi_b, cc25
    out["cycle_cost_2025"] = cc25
    return out


if __name__ == "__main__":
    o = main()
    a = o["anchor"]
    print(f"Europe 2025 anchor: c_E = {a['c_e']:.1f} EUR/kWh, c_P = {a['c_p']:.1f} EUR/kW "
          f"(premium {a['europe_premium']:.3f}, global mix {a['global_avg_duration_h']:.2f} h, "
          f"2035 index {a['europe_4h_2035_index']:.3f})")
    print("Cycle cost 2025 (EUR/MWh):", {k: round(v, 1) for k, v in o["cycle_cost_2025"].items()})
    for r in o["validation"]:
        print(f"{r[0]}: model {r[1]:.0f}, observed {r[2]:.0f}, ratio {r[3]:.2f}")
