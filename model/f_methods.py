"""Forecast methods M1, M2, M4, M5 (M3 is f_equilibrium.py).

Every method returns Monte Carlo samples of fleet power (GW) and energy (GWh) on the
annual grid YEARS (end-of-year stock), shape (n, len(YEARS)); NaN where a method is
not informative. All draws come from one seeded generator per method.

  M1  pipeline       project queue x stage-realisation probability x COD spread (to 2032)
  M2  diffusion      logistic stock; present growth g and saturation K sampled
  M4  top-down       (a) global fleet x Finnish share; (b) Nordic per-capita peers in 2050
  M5  envelope       retained blog states (2027-28) and Fingrid states (2030-2040)
"""
from __future__ import annotations

import numpy as np

from common import OUT, load_params, write_csv
from f_equilibrium import YEARS, tri_from_u

T0 = 2026.87  # date of the starting stock (Fingrid 424, Nov 2026)


def _tri(rng, lmh, n):
    return tri_from_u(rng.random(n), lmh)


def _start(p):
    s = p["forecast"]["start"]
    return s["stock_gw"], s["stock_gwh"]


# ------------------------------------------------------------- duration model
def gwh_from_gw(p: dict, gw: np.ndarray, d2050: np.ndarray) -> np.ndarray:
    """Stock-flow energy for a GW path (top-down methods). Net additions after the
    committed tranche take the new-build duration d_new(t) (linear 2.8 h in 2027 ->
    d2050); net declines scale energy proportionally."""
    g0, e0 = _start(p)
    c = p["forecast"]["m3_equilibrium"]["committed"]
    gwh = np.empty_like(gw)
    prev_gw = np.full(gw.shape[0], g0)
    prev_e = np.full(gw.shape[0], e0)
    committed_left = np.full(gw.shape[0], c["gw"])
    for i, y in enumerate(YEARS):
        if y <= 2026:
            gwh[:, i] = e0 * gw[:, i] / g0
            prev_gw, prev_e = gw[:, i], gwh[:, i]
            continue
        dnew = 2.8 + (d2050 - 2.8) * np.clip((y - 2027) / (2050 - 2027), 0, 1)
        add = gw[:, i] - prev_gw
        pos = np.maximum(add, 0)
        from_comm = np.minimum(pos, committed_left)
        committed_left = committed_left - from_comm
        e = prev_e + from_comm * c["duration_h"] + (pos - from_comm) * dnew
        e = np.where(add < 0, prev_e * gw[:, i] / np.maximum(prev_gw, 1e-9), e)
        gwh[:, i] = e
        prev_gw, prev_e = gw[:, i], e
    return gwh


def committed_floor(p: dict):
    """Physical floor (GW, GWh) per year: today's fleet plus the FID tranche (at the
    low realisation probability), with Weibull survival (M3 lifetime parameters).
    No method may sit below stock that already exists or is under construction."""
    import math
    f = p["forecast"]["m3_equilibrium"]
    g0, e0 = _start(p)
    c = f["committed"]
    st = p["forecast"]["m1_pipeline"]["stages"]["fid_remaining"]
    k = f["weibull_shape"]
    wmin = f.get("warranty_min_life_yr", 0.0)
    lam = (f["life_mean_yr"]["2025"] - wmin) / math.gamma(1 + 1 / k)
    surv = lambda age: np.exp(-((np.maximum(age - wmin, 0) / lam) ** k))
    t = YEARS + 1.0
    built = 2025.5  # mean commissioning of today's fleet
    cg = c["gw"] * st["p"][0]
    gw = g0 * surv(t - built) + cg * np.clip(t - 2028.0, 0, 1) * surv(t - 2028.0)
    gwh = e0 * surv(t - built) + cg * c["duration_h"] * np.clip(t - 2028.0, 0, 1) * surv(t - 2028.0)
    return gw, gwh


def apply_floor(p, gw, gwh):
    fg, fe = committed_floor(p)
    low = gw < fg[None, :]
    gw = np.where(low, fg[None, :], gw)
    gwh = np.where(low, np.maximum(gwh, fe[None, :]), gwh)
    return gw, gwh


# ------------------------------------------------------------------------- M1
def m1_pipeline(p: dict, n: int, seed: int):
    """Stage-gate Monte Carlo. Each stage's realised share = sampled probability x a
    lumpiness factor (Beta, mean 1, reflects few large projects); COD year drawn per
    sample from stage weights. Valid to 2032 (queue horizon); NaN after."""
    rng = np.random.default_rng(seed + 1)
    g0, e0 = _start(p)
    gw = np.full((n, len(YEARS)), g0)
    gwh = np.full((n, len(YEARS)), e0)
    for name, st in p["forecast"]["m1_pipeline"]["stages"].items():
        prob = _tri(rng, st["p"], n)
        lump = rng.beta(8, 8, n) * 2  # mean 1, sd ~0.24
        real = np.clip(prob * lump, 0, 1) * st["gw"]
        cods = np.array(st["cod_years"])
        w = np.array(st["cod_weights"], dtype=float)
        # spread each sample's realised capacity over COD years with jittered weights
        jw = rng.dirichlet(w * 20, n)
        for k, y in enumerate(cods):
            idx = YEARS >= y
            gw[:, idx] += (real * jw[:, k])[:, None]
            gwh[:, idx] += (real * jw[:, k] * st["duration_h"])[:, None]
    gw[:, YEARS > 2032] = np.nan
    gwh[:, YEARS > 2032] = np.nan
    return gw, gwh


# ------------------------------------------------------------------------- M2
def m2_diffusion(p: dict, n: int, seed: int, n2028: np.ndarray | None = None):
    """Logistic through today's stock and the realised committed stock at end-2028
    (from M1, sample-wise); K log-uniform."""
    rng = np.random.default_rng(seed + 2)
    f = p["forecast"]["m2_diffusion"]
    g0, _ = _start(p)
    if n2028 is None:
        n2028 = m1_pipeline(p, n, seed)[0][:, list(YEARS).index(2028)]
    lo, hi = f["k_gw_loguniform"]
    K = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    K = np.maximum(K, n2028 * 1.1)
    t1, t2 = T0, 2029.0
    logit = lambda x: np.log(x / (1 - x))
    r = (logit(n2028 / K) - logit(g0 / K)) / (t2 - t1)
    t = YEARS.astype(float) + 1.0
    z = logit(g0 / K)[:, None] + r[:, None] * (t[None, :] - t1)
    gw = K[:, None] / (1 + np.exp(-z))
    gw[:, YEARS == 2026] = g0
    d2050 = _tri(rng, p["forecast"]["duration_model"]["d_new_2050_h"], n)
    return gw, gwh_from_gw(p, gw, d2050), {"K": K, "r": r, "dt_10_90": np.log(81) / r}


# ------------------------------------------------------------------------- M4
def _loglin_rows(xk: list[float], vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Row-wise log-linear interpolation; vals shape (n, len(xk))."""
    lv = np.log(vals)
    xk = np.asarray(xk, float)
    pos = np.clip(np.searchsorted(xk, x, side="right") - 1, 0, len(xk) - 2)
    w = np.clip((x - xk[pos]) / (xk[pos + 1] - xk[pos]), 0, 1)
    return np.exp(lv[:, pos] * (1 - w) + lv[:, pos + 1] * w)


def m4_topdown(p: dict, n: int, seed: int):
    """Half the samples from (a) global x share, half from (b) per-capita peers."""
    rng = np.random.default_rng(seed + 4)
    f = p["forecast"]["m4_topdown"]
    g0, _ = _start(p)
    na = n // 2
    nb = n - na
    # (a) global fleet: one quantile per sample, comonotone across checkpoints
    u = rng.random(na)
    ks = sorted(int(k) for k in f["global_gw"] if not k.startswith("_"))
    glob = np.column_stack([tri_from_u(u, f["global_gw"][str(k)]) for k in ks])
    glob0 = g0 / f["fi_share_now"]
    xk = [T0] + ks
    G = _loglin_rows(xk, np.column_stack([np.full(na, glob0), glob]), YEARS + 1.0)
    sh50 = _tri(rng, f["fi_share_2050"], na)
    S = _loglin_rows([T0, 2050.0], np.column_stack([np.full(na, f["fi_share_now"]), sh50]), YEARS + 1.0)
    gw_a = G * S
    # (b) Nordic peers: 2050 point, log-linear from today; each draw picks a peer and a
    # normalisation (per person or per TWh of electricity consumption)
    names = [k for k in f["peer_per_capita_2050_kw"] if not k.startswith("_")]
    pick_peer = rng.integers(0, len(names), nb)
    per_twh = rng.random(nb) < f.get("peer_share_per_twh", 0.5)
    g50 = np.empty(nb)
    for j in range(nb):
        nm = names[pick_peer[j]]
        kw = tri_from_u(rng.random(1), f["peer_per_capita_2050_kw"][nm])[0]
        if per_twh[j]:
            pk = f["peer_now"][nm]
            g50[j] = kw * pk["pop_m"] * f["fi_consumption_twh_now"] / pk["consumption_twh"]
        else:
            g50[j] = kw * f["fi_population_m"]
    gw_b = _loglin_rows([T0, 2051.0], np.column_stack([np.full(nb, g0), g50]), YEARS + 1.0)
    gw = np.vstack([gw_a, gw_b])
    gw[:, YEARS == 2026] = g0
    perm = rng.permutation(n)
    gw = gw[perm]
    d2050 = _tri(rng, p["forecast"]["duration_model"]["d_new_2050_h"], n)
    sub = np.array(["a"] * na + [("b_twh" if t else "b_cap") for t in per_twh])[perm]
    return gw, gwh_from_gw(p, gw, d2050), {"sub": sub}


# ------------------------------------------------------------------------- M5
def m5_envelope(p: dict, n: int, seed: int):
    """Equal-weight mixture over the published states (low/mid/high, one index per
    sample, comonotone across years: a scenario is internally consistent), +-10%
    jitter. Energy: today's fleet and the committed tranche at their durations,
    scenario additions at 4-8 h (Fingrid Q3/2026)."""
    rng = np.random.default_rng(seed + 5)
    f = p["forecast"]["m5_envelope"]["points"]
    g0, e0 = _start(p)
    c = p["forecast"]["m3_equilibrium"]["committed"]
    dlo, dhi = p["forecast"]["duration_model"]["m5_scenario_additions_h"]
    k = rng.integers(0, 3, n)
    jit = rng.uniform(0.9, 1.1, n)
    dsc = rng.uniform(dlo, dhi, n)
    gw = np.full((n, len(YEARS)), np.nan)
    gwh = np.full((n, len(YEARS)), np.nan)
    gw[:, YEARS == 2026] = g0
    gwh[:, YEARS == 2026] = e0
    for key, v in f.items():
        y = int(key)
        x = np.array(v, float)[k] * jit
        gw[:, YEARS == y] = x[:, None]
        comm = np.clip(x - g0, 0, c["gw"])
        extra = np.maximum(x - g0 - c["gw"], 0)
        dd = c["duration_h"] if y <= 2028 else dsc
        gwh[:, YEARS == y] = (e0 + comm * c["duration_h"] + extra * dd)[:, None]
    last = p["forecast"]["m5_envelope"].get("extend_flat_after")
    if last:
        i = list(YEARS).index(int(last))
        gw[:, i + 1:] = gw[:, [i]]
        gwh[:, i + 1:] = gwh[:, [i]]
    return gw, gwh


# ----------------------------------------------------------------------- run
def run_all_methods(p: dict | None = None, n: int | None = None, write: bool = True) -> dict:
    import f_equilibrium as m3
    p = p or load_params()
    mc = p["forecast"]["mc"]
    n = n or mc["n"]
    seed = mc["seed"]
    out = {}
    out["M1"] = apply_floor(p, *m1_pipeline(p, n, seed))
    g2 = m2_diffusion(p, n, seed, out["M1"][0][:, list(YEARS).index(2028)])
    out["M2"] = apply_floor(p, *g2[:2])
    r3 = m3.main(p, n=n, do_sobol=False, write=write)["samples"]
    out["M3"] = (r3["gw"], r3["gwh"])
    out["M4"] = apply_floor(p, *m4_topdown(p, n, seed)[:2])
    out["M5"] = apply_floor(p, *m5_envelope(p, n, seed))
    rows = []
    for m, (gw, gwh) in out.items():
        for i, y in enumerate(YEARS):
            if np.all(np.isnan(gw[:, i])):
                continue
            q = np.nanpercentile(gw[:, i], [10, 50, 90])
            e = np.nanpercentile(gwh[:, i], [10, 50, 90])
            rows.append([m, int(y), *map(float, q), *map(float, e)])
    if not write:
        return out
    write_csv(OUT / "method_paths.csv",
              ["method", "year", "gw_p10", "gw_p50", "gw_p90", "gwh_p10", "gwh_p50", "gwh_p90"], rows)
    return out


if __name__ == "__main__":
    o = run_all_methods()
    for m, (gw, gwh) in o.items():
        for y in (2027, 2028, 2030, 2035, 2040, 2045, 2050):
            i = int(np.where(YEARS == y)[0][0])
            if np.all(np.isnan(gw[:, i])):
                continue
            q = np.nanpercentile(gw[:, i], [10, 50, 90])
            e = np.nanpercentile(gwh[:, i], [10, 50, 90])
            print(f"{m} {y}  GW {q[0]:5.1f} {q[1]:5.1f} {q[2]:5.1f} | GWh {e[0]:6.1f} {e[1]:6.1f} {e[2]:6.1f}")
