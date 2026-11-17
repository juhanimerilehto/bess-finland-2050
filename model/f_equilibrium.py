"""M3 - Economic free-entry equilibrium for Finnish BESS, 2026-2050.

Each year, investors add batteries until a new MW no longer earns its annualised
cost. For duration d and fleet N (GW), revenue per MW (kEUR/MW/yr) is

    R_d(N, t) = c * m_d * F(t)^ev * (N_eff / N0)^(-eo)  +  s * P0 * F(t)^ep / N

  m_d   : measured day-ahead arbitrage margin, mean 2024-2026 (spreads.py), at the
          2026 fleet N0 (perfect foresight); c = capture factor
  F(t)  : flexibility-need index = consumption growth x VRE-share multiplier
  eo    : own-fleet cannibalisation elasticity; N_eff includes pumped hydro and
          a weight on grid-forming batteries
  P0    : 2026 reserve-capacity + balancing-energy pool (MEUR), growing as F^ep;
          s = share of the pool captured by batteries (rest: hydro, boilers, demand)

The desired fleet N*_d solves R_d(N*) = A_d(t) (annualised cost, cost.py). The
market builds the duration with the largest N*_d. Stock follows vintages with
Weibull survival, a build-rate limit and a connection cap; it never falls faster
than retirement. Vectorised over Monte Carlo samples.
"""
from __future__ import annotations

import csv
import math

import numpy as np

import cost
from common import OUT, load_params

YEARS = np.arange(2026, 2051)


# ------------------------------------------------------------------ sampling
def tri(rng, lmh, n):
    lo, mo, hi = lmh
    if hi <= lo:
        return np.full(n, float(mo))
    return rng.triangular(lo, mo, hi, n)


def tri_from_u(u, lmh):
    """Inverse CDF of a triangular distribution (for Sobol samples in [0,1])."""
    a, c, b = lmh
    if b <= a:
        return np.full_like(u, float(c))
    fc = (c - a) / (b - a)
    return np.where(u < fc, a + np.sqrt(u * (b - a) * (c - a)), b - np.sqrt((1 - u) * (b - a) * (b - c)))


PARAM_NAMES = ["cost_u", "premium_u", "finance_u", "project_u", "capture", "eps_own", "eps_flex", "eps_pool",
               "eps_pool_sat", "pool_share", "cons_2035", "cons_2050", "vre_mult", "build_limit", "phes",
               "grid_forming", "duration_tilt"]


def sample_from_unit(p: dict, U: np.ndarray) -> dict:
    """Map an (n, k) array of uniforms to parameter arrays (order = PARAM_NAMES)."""
    f = p["forecast"]["m3_equilibrium"]
    col = {k: U[:, i] for i, k in enumerate(PARAM_NAMES)}
    ph = f["phes"]
    cum = np.cumsum(ph["probs"])
    phes_idx = np.searchsorted(cum, col["phes"], side="right").clip(0, len(ph["options_gw"]) - 1)
    return {
        "cost_u": 2 * col["cost_u"] - 1,                       # -1 low cost .. +1 high cost
        "premium_u": 2 * col["premium_u"] - 1,                  # -1 fast .. +1 sticky European premium
        "finance_u": 2 * col["finance_u"] - 1,                  # -1 cheap capital, long life .. +1 dear, short
        "project_u": 2 * col["project_u"] - 1,                  # -1 cheap grid connection/local costs .. +1 dear
        "eps_pool_sat": tri_from_u(col["eps_pool_sat"], f["eps_pool_sat"]),
        "capture": tri_from_u(col["capture"], f["capture"]),
        "eps_own": tri_from_u(col["eps_own"], f["eps_own"]),
        "eps_flex": tri_from_u(col["eps_flex"], f["eps_flex"]),
        "eps_pool": tri_from_u(col["eps_pool"], f["eps_pool"]),
        "pool_share": tri_from_u(col["pool_share"], f["pool_share"]),
        "cons_2035": tri_from_u(col["cons_2035"], f["consumption_twh"]["2035"]),
        "cons_2050": tri_from_u(col["cons_2050"], f["consumption_twh"]["2050"]),
        "vre_mult": tri_from_u(col["vre_mult"], f["vre_share_mult_2050"]),
        "build_limit": tri_from_u(col["build_limit"], f["build_limit_gw_yr"]),
        "phes": np.array(ph["options_gw"])[phes_idx],
        "grid_forming": tri_from_u(col["grid_forming"], f["grid_forming_2050_gw"]),
        "duration_tilt": tri_from_u(col["duration_tilt"], f["duration_tilt"]),
    }


def central_sample(p: dict) -> dict:
    f = p["forecast"]["m3_equilibrium"]
    one = lambda v: np.array([float(v)])
    return {"cost_u": one(0), "premium_u": one(0), "finance_u": one(0), "project_u": one(0),
            "eps_pool_sat": one(f["eps_pool_sat"][1]), "capture": one(f["capture"][1]),
            "eps_own": one(f["eps_own"][1]), "eps_flex": one(f["eps_flex"][1]), "eps_pool": one(f["eps_pool"][1]),
            "pool_share": one(f["pool_share"][1]), "cons_2035": one(f["consumption_twh"]["2035"][1]),
            "cons_2050": one(f["consumption_twh"]["2050"][1]), "vre_mult": one(f["vre_share_mult_2050"][1]),
            "build_limit": one(f["build_limit_gw_yr"][1]), "phes": one(0.55),
            "grid_forming": one(f["grid_forming_2050_gw"][1]), "duration_tilt": one(f["duration_tilt"][1])}


# ------------------------------------------------------------- precomputation
def da_margins(p: dict | None = None) -> dict:
    """Mean 2024-26 day-ahead margin by duration (kEUR/MW/yr, revenue minus charging cost,
    perfect foresight) and the discharged energy behind it (MWh/MW/yr, key 'q')."""
    rows = list(csv.DictReader(open(OUT / "da_arbitrage_annual.csv", encoding="utf-8")))
    out, q = {}, {}
    for d in (1, 2, 4, 8):
        sel = [r for r in rows if r["duration_h"] == str(d) and r["year"] in ("2024", "2025", "2026")]
        out[d] = sum(float(r["margin_keur_per_mw_yr"]) for r in sel) / len(sel)
        q[d] = sum(float(r["discharged_mwh_per_mw_yr"]) for r in sel) / len(sel)
    out["q"] = q
    return out


def efficiency_gain(p: dict, year: int, d: int, margins: dict) -> float:
    """Margin gain (kEUR/MW/yr) of a vintage built in `year` from round-trip efficiency
    above the level at which margins were measured: q x p_ch x (1/eta_ref - 1/eta(t)).
    First order: dispatch is held fixed, so this is a lower bound on the gain."""
    o = p["operation"]
    eta_ref = cost.rte_at(p, 2026, "central")
    eta = cost.rte_at(p, year, "central")
    pc = o["charge_price_eur_per_mwh"]["central"]
    return margins["q"][d] * pc * (1 / eta_ref - 1 / eta) / 1000


CASES = ("low", "central", "high")


def cost_table(p: dict, durations) -> np.ndarray:
    """A[tech, premium, finance, project, year, d]: annualised cost (kEUR/MW/yr) of a new
    system under the component cost model, for the low/central/high case of each of four
    independent cost dimensions (0 = low cost)."""
    A = np.zeros((3, 3, 3, 3, len(YEARS), len(durations)))
    for ti, tc in enumerate(CASES):
        for pi, pc in enumerate(CASES):
            for fi, fc in enumerate(CASES):
                for ji, jc in enumerate(CASES):
                    for yi, y in enumerate(YEARS):
                        for di, d in enumerate(durations):
                            A[ti, pi, fi, ji, yi, di] = cost.annualised_cost_per_kw(
                                p, int(y), d, tc, "C", premium=pc, finance=fc, project=jc)["total"]
    return A


def _logi(lo, ce, hi, u):
    return np.where(u < 0, np.exp(np.log(ce) + (-u) * (np.log(lo) - np.log(ce))),
                    np.exp(np.log(ce) + u * (np.log(hi) - np.log(ce))))


def interp_cost(A, s: dict, yi, di):
    """Log-interpolate between low/central/high along each cost dimension (u in [-1, 1])."""
    def rec(sub, axes):
        if not axes:
            return sub
        u = s[axes[0]]
        return _logi(rec(sub[0], axes[1:]), rec(sub[1], axes[1:]), rec(sub[2], axes[1:]), u)
    return rec(A[..., yi, di], ["cost_u", "premium_u", "finance_u", "project_u"])


def flex_index(s: dict, year: int, c0: float) -> np.ndarray:
    """F(t): consumption growth (piecewise linear 2026-2035-2050) x VRE-share multiplier."""
    if year <= 2035:
        cons = c0 + (s["cons_2035"] - c0) * (year - 2026) / 9
    else:
        cons = s["cons_2035"] + (s["cons_2050"] - s["cons_2035"]) * (year - 2035) / 15
    w = 1 + (s["vre_mult"] - 1) * (year - 2026) / 24
    return cons / c0 * w


# -------------------------------------------------------------------- solver
def desired_fleet(rev_arb, pool_term, n_extra, a_cost, n0, eo, esat=0.0, lo=0.0, hi=80.0, iters=48):
    """Solve rev_arb*((N+n_extra)/n0)^(-eo) + pool_term*((N+n_extra)/n0)^(-esat)/(N+n_extra) = a_cost
    for N (vectorised). The flexibility pool is shared with competing capacity n_extra and
    shrinks with the size of the flexible fleet at elasticity esat.

    Revenue is strictly decreasing in N. Where revenue at N=hi exceeds cost the
    solution is hi; where revenue at a tiny N is below cost the solution is 0.
    """
    lo_a = np.full_like(a_cost, 1e-3)
    hi_a = np.full_like(a_cost, hi)

    def f(N):
        tot = N + n_extra
        return rev_arb * (tot / n0) ** (-eo) + pool_term * (tot / n0) ** (-esat) / tot - a_cost

    flo, fhi = f(lo_a), f(hi_a)
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        fm = f(mid)
        pos = fm > 0
        lo_a = np.where(pos, mid, lo_a)
        hi_a = np.where(pos, hi_a, mid)
    N = 0.5 * (lo_a + hi_a)
    N = np.where(fhi > 0, hi, N)
    N = np.where(flo <= 0, 0.0, N)
    return N


def simulate(p: dict, s: dict, A=None, margins=None, keep_paths=True) -> dict:
    f = p["forecast"]["m3_equilibrium"]
    st = p["forecast"]["start"]
    durs = f["durations_h"]
    A = cost_table(p, durs) if A is None else A
    m = da_margins() if margins is None else margins
    n = len(s["capture"])
    n0 = st["stock_gw"]
    c0 = f["consumption_twh"]["2026"]
    shape = f["weibull_shape"]
    L0, L1 = f["life_mean_yr"]["2025"], f["life_mean_yr"]["2050"]
    cap = f["cap_gw"]

    # vintages: initial fleet as one 2025 vintage at 1.9 h
    vint_mw = [np.full(n, n0)]
    vint_d = [np.full(n, st["stock_gwh"] / n0)]
    vint_y = [2025]

    wmin = f.get("warranty_min_life_yr", 0.0)

    def survival(age, vy):
        """Weibull survival after a warranted minimum life: no retirement before wmin
        years; thereafter mean remaining life L - wmin."""
        L = L0 + (L1 - L0) * min(max((vy - 2025) / 25, 0), 1)
        lam = (L - wmin) / math.gamma(1 + 1 / shape)
        return math.exp(-((max(age - wmin, 0) / lam) ** shape))

    out_gw, out_gwh, out_dstar, out_nstar = [], [], [], []
    for yi, y in enumerate(YEARS):
        y = int(y)
        F = flex_index(s, y, c0)
        phes = np.where(((s["phes"] >= 1.0) & (y >= f["phes"]["online"]["1.05"])) |
                        ((s["phes"] >= 0.5) & (s["phes"] < 1.0) & (y >= f["phes"]["online"]["0.55"])) |
                        ((s["phes"] >= 1.0) & (y >= f["phes"]["online"]["0.55"]) & (y < f["phes"]["online"]["1.05"])),
                        np.where(s["phes"] >= 1.0, np.where(y >= f["phes"]["online"]["1.05"], s["phes"], 0.55), s["phes"]), 0.0)
        gf = s["grid_forming"] * min(max((y - 2026) / 24, 0), 1)
        n_extra = phes + f["grid_forming_cannibal_weight"] * gf
        pool_term = s["pool_share"] * p["forecast"]["m3_equilibrium"]["pool_meur_2026"] * F ** s["eps_pool"]
        best_n = np.zeros(n)
        best_d = np.full(n, float(durs[0]))
        elig = f.get("pool_eligibility_by_duration", {})
        d0 = st["stock_gwh"] / n0
        energy_basis = f.get("cannibal_basis", "power") == "energy"
        for di, d in enumerate(durs):
            a = interp_cost(A, s, yi, di)
            md = m[d] + efficiency_gain(p, y, d, m)
            rev_arb = s["capture"] * md * F ** (s["eps_flex"] * (1 + s["duration_tilt"] * math.log2(d / 2)))
            if energy_basis:   # arbitrage cannibalised by shifted energy rather than power
                rev_arb = rev_arb * (d / d0) ** (-s["eps_own"])
            nstar = desired_fleet(rev_arb, pool_term * elig.get(str(d), 1.0), n_extra, a, n0, s["eps_own"],
                                  s["eps_pool_sat"])
            better = nstar > best_n
            best_n = np.where(better, nstar, best_n)
            best_d = np.where(better, float(d), best_d)
        best_n = np.minimum(best_n, cap)
        # surviving stock
        surv_mw = np.zeros(n)
        surv_mwh = np.zeros(n)
        for mw, dd, vy in zip(vint_mw, vint_d, vint_y):
            sv = survival(y - vy, vy)
            surv_mw += mw * sv
            surv_mwh += mw * sv * dd
        com = f.get("committed", {})
        com_add = com.get("gw", 0.0) * com.get("cod", {}).get(str(y), 0.0)
        if com_add > 0:                                     # decisions already taken
            vint_mw.append(np.full(n, com_add))
            vint_d.append(np.full(n, com["duration_h"]))
            vint_y.append(y)
            surv_mw = surv_mw + com_add
            surv_mwh = surv_mwh + com_add * com["duration_h"]
        add = np.clip(best_n - surv_mw, 0, s["build_limit"]) if y > 2026 else np.zeros(n)
        add = np.minimum(add, np.maximum(cap - surv_mw, 0))
        vint_mw.append(add)
        vint_d.append(best_d)
        vint_y.append(y)
        # co-located grid-forming units (VJV2024 weak-grid rule) are physical BESS outside
        # the market equilibrium: added to the stock at gf_duration_h, and counted above as
        # competitors at grid_forming_cannibal_weight
        out_gw.append(surv_mw + add + gf)
        out_gwh.append(surv_mwh + add * best_d + gf * f.get("grid_forming_duration_h", 1.0))
        out_dstar.append(best_d)
        out_nstar.append(best_n)
    res = {"gw": np.array(out_gw).T, "gwh": np.array(out_gwh).T,
           "dstar": np.array(out_dstar).T, "nstar": np.array(out_nstar).T,
           "grid_forming": np.outer(s["grid_forming"], np.clip((YEARS - 2026) / 24, 0, 1)),
           "phes_2050": s["phes"]}
    return res


# ---------------------------------------------------------------------- Sobol
def sobol(p: dict, nbase: int, seed: int, A, margins, years=(2035, 2050)):
    """Saltelli (2010) first-order and Jansen (1999) total-effect indices."""
    k = len(PARAM_NAMES)
    rng = np.random.default_rng(seed + 1)
    UA, UB = rng.random((nbase, k)), rng.random((nbase, k))
    yidx = [int(np.where(YEARS == y)[0][0]) for y in years]

    def run(U):
        r = simulate(p, sample_from_unit(p, U), A, margins)
        return {y: (r["gw"][:, i], r["gwh"][:, i]) for y, i in zip(years, yidx)}

    fA, fB = run(UA), run(UB)
    out = []
    for j, name in enumerate(PARAM_NAMES):
        UAB = UA.copy()
        UAB[:, j] = UB[:, j]
        fAB = run(UAB)
        for y in years:
            for oi, metric in enumerate(("gw", "gwh")):
                a, b, ab = fA[y][oi], fB[y][oi], fAB[y][oi]

                def est(ix):
                    aa, bb, abab = a[ix], b[ix], ab[ix]
                    var = np.var(np.concatenate([aa, bb]))
                    if var <= 0:
                        return 0.0, 0.0
                    return np.mean(bb * (abab - aa)) / var, 0.5 * np.mean((aa - abab) ** 2) / var
                s1, st = est(np.arange(nbase))
                bs = np.array([est(rng.integers(0, nbase, nbase)) for _ in range(100)])
                out.append([name, y, metric, s1, st, float(bs[:, 0].std()), float(bs[:, 1].std())])
    return out


# ----------------------------------------------------------------------- main
def main(p: dict | None = None, n: int | None = None, do_sobol: bool = True, write: bool = True) -> dict:
    p = p or load_params()
    mc = p["forecast"]["mc"]
    n = n or mc["n"]
    durs = p["forecast"]["m3_equilibrium"]["durations_h"]
    A = cost_table(p, durs)
    m = da_margins()
    rng = np.random.default_rng(mc["seed"])
    s = sample_from_unit(p, rng.random((n, len(PARAM_NAMES))))
    r = simulate(p, s, A, m)
    c = simulate(p, central_sample(p), A, m)
    rows = []
    for yi, y in enumerate(YEARS):
        gw, gwh, dd = r["gw"][:, yi], r["gwh"][:, yi], r["dstar"][:, yi]
        rows.append([int(y)] + [float(np.percentile(gw, q)) for q in (5, 10, 25, 50, 75, 90, 95)]
                    + [float(np.percentile(gwh, q)) for q in (10, 50, 90)]
                    + [float(c["gw"][0, yi]), float(c["gwh"][0, yi]), float(c["dstar"][0, yi])])
    from common import write_csv
    if not write:
        return {"rows": rows, "sobol": [], "central": c, "samples": r}
    write_csv(OUT / "m3_equilibrium_paths.csv",
              ["year", "gw_p5", "gw_p10", "gw_p25", "gw_p50", "gw_p75", "gw_p90", "gw_p95",
               "gwh_p10", "gwh_p50", "gwh_p90", "central_gw", "central_gwh", "central_new_duration_h"], rows)
    np.savez_compressed(OUT / "m3_samples.npz", gw=r["gw"], gwh=r["gwh"], years=YEARS)
    sob = sobol(p, mc["sobol_base"], mc["seed"], A, m) if do_sobol else []
    if sob:
        write_csv(OUT / "m3_sobol.csv", ["parameter", "year", "metric", "S1", "ST", "S1_se", "ST_se"], sob)
    return {"rows": rows, "sobol": sob, "central": c, "samples": r}


if __name__ == "__main__":
    import time
    t0 = time.time()
    o = main()
    for r in o["rows"]:
        if r[0] in (2026, 2027, 2028, 2030, 2035, 2040, 2045, 2050):
            print(f"{r[0]}  GW p10 {r[2]:5.1f}  p50 {r[4]:5.1f}  p90 {r[6]:5.1f} | GWh p50 {r[9]:6.1f} | "
                  f"central {r[11]:5.1f} GW {r[12]:6.1f} GWh d_new {r[13]:.0f} h")
    print(f"({time.time() - t0:.0f} s)")
    for row in o["sobol"]:
        if row[2] == "gw" and row[4] > 0.05:
            print(f"Sobol {row[1]} {row[0]:<13} S1 {row[3]:.2f} ST {row[4]:.2f}")
