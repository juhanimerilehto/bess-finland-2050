"""Ensemble: linear opinion pool of M1-M5 at each checkpoint, planning states,
energy/duration, electricity use of the fleet, and pooling-rule robustness.

States: Low/Base/High = P10/P50/P90 of the pool. Each method enters with equal weight
(same number of draws) only at checkpoints where it carries its own information
(params forecast.ensemble.methods_by_checkpoint). GW and GWh stay paired per draw.

Electricity use (TWh/yr): conversion losses E_out (1/eta - 1) with E_out = GWh x DoD x
cycles, plus auxiliaries aux(kW/MWh) x GWh x 8.76e-3. Robustness: leave-one-method-out
P50 and quantile averaging (Vincentisation) as an alternative pool.
"""
from __future__ import annotations

import numpy as np

import f_methods
from common import OUT, load_params, write_csv
from f_equilibrium import YEARS, tri_from_u

NEAR_TERM = {"2027": ["M1", "M3", "M5"], "2028": ["M1", "M3", "M5"]}


def _idx(y):
    return int(np.where(YEARS == y)[0][0])


WEIGHTS: dict = {}


def weights_for(y: int, methods: list[str]) -> dict:
    w = {m: 1.0 for m in methods}
    if y >= 2040:
        for m, v in WEIGHTS.items():
            if m in w:
                w[m] = v
    return w


def pool(samples: dict, methods: list[str], y: int):
    """Linear opinion pool: each method contributes draws in proportion to its weight
    (the first n x w draws; draws are independent and identically distributed)."""
    i = _idx(y)
    w = weights_for(y, methods)
    gws, gwhs = [], []
    for m in methods:
        g, h = samples[m][0][:, i], samples[m][1][:, i]
        k = int(round(len(g) * w[m]))
        gws.append(g[:k])
        gwhs.append(h[:k])
    gw, gwh = np.concatenate(gws), np.concatenate(gwhs)
    ok = ~np.isnan(gw)
    return gw[ok], gwh[ok]


def fleet_rte(p: dict, y: int, u: np.ndarray) -> np.ndarray:
    """Fleet-average AC round-trip efficiency: vintage RTE with ~8-year mean age."""
    r = p["operation"]["rte"]
    g = {int(k): v for k, v in r["gain_pp"].items()}
    ys = sorted(g)
    gain = float(np.interp(y - 8, ys, [g[k] for k in ys]))
    return np.minimum(0.95, tri_from_u(u, [r["low"], r["central"], r["high"]]) + gain / 100)


def consumption_twh(p: dict, y: int, u: np.ndarray) -> np.ndarray:
    c = p["forecast"]["m3_equilibrium"]["consumption_twh"]
    c26 = c["2026"]
    c35 = tri_from_u(u, c["2035"])
    c50 = tri_from_u(u, c["2050"])
    if y <= 2035:
        return c26 + (c35 - c26) * (y - 2026) / 9
    return c35 + (c50 - c35) * (y - 2035) / 15


def electricity_use(p: dict, gwh: np.ndarray, y: int, rng) -> dict:
    n = len(gwh)
    op = p["forecast"]["operation_projection"]
    cyc = tri_from_u(rng.random(n), op["cycles_per_year"])
    eta = fleet_rte(p, y, rng.random(n))
    a = p["operation"]["aux_kw_per_mwh"]
    aux = tri_from_u(rng.random(n), [a["low"], a["central"], a["high"]])
    e_out = gwh * op["dod"] * cyc / 1000
    loss = e_out * (1 / eta - 1)
    auxt = aux * gwh * 8.76e-3
    cons = consumption_twh(p, y, rng.random(n))
    minutes = gwh / (cons * 1000 / 8760) * 60
    return {"throughput_twh": e_out, "loss_twh": loss, "aux_twh": auxt, "net_twh": loss + auxt,
            "share_pct": (loss + auxt) / cons * 100, "minutes_of_load": minutes}


def q(x, qs=(10, 50, 90)):
    return [float(v) for v in np.percentile(x, qs)]


def main(p: dict | None = None, n: int | None = None, write: bool = True) -> dict:
    p = p or load_params()
    e = p["forecast"]["ensemble"]
    WEIGHTS.clear()
    WEIGHTS.update(e.get("weights_from_2040", {}))
    samples = f_methods.run_all_methods(p, n, write=write)
    rng = np.random.default_rng(p["forecast"]["mc"]["seed"] + 99)
    plan = {**NEAR_TERM, **e["methods_by_checkpoint"]}
    states, loo, use_rows, by_method = [], [], [], []
    pooled = {}
    for ys, methods in plan.items():
        y = int(ys)
        gw, gwh = pool(samples, methods, y)
        pooled[y] = (gw, gwh)
        dur = gwh / gw
        lo, ba, hi = q(gw)
        elo, eba, ehi = q(gwh)
        dlo, dba, dhi = q(dur)
        # quantile averaging (Vincentisation)
        wv = weights_for(y, methods)
        vq = np.average([q(samples[m][0][:, _idx(y)]) for m in methods], axis=0, weights=[wv[m] for m in methods])
        states.append([y, "+".join(methods), lo, ba, hi, elo, eba, ehi, dlo, dba, dhi, *map(float, vq)])
        for m in methods:
            g_m, h_m = samples[m][0][:, _idx(y)], samples[m][1][:, _idx(y)]
            by_method.append([y, m, *q(g_m), *q(h_m)])
            if len(methods) > 1:
                rest = [k for k in methods if k != m]
                g2, h2 = pool(samples, rest, y)
                loo.append([y, m, float(np.median(g2)), float(np.median(h2)),
                            float(np.median(g2) / ba - 1) * 100])
        u = electricity_use(p, gwh, y, rng)
        use_rows.append([y] + sum((q(u[k]) for k in ("throughput_twh", "loss_twh", "aux_twh",
                                                      "net_twh", "share_pct", "minutes_of_load")), []))
    if not write:
        return {"states": states, "loo": loo, "use": use_rows, "by_method": by_method,
                "samples": samples, "pooled": pooled}
    write_csv(OUT / "ensemble_states.csv",
              ["year", "methods", "gw_low", "gw_base", "gw_high", "gwh_low", "gwh_base", "gwh_high",
               "dur_low", "dur_base", "dur_high", "vincent_gw_p10", "vincent_gw_p50", "vincent_gw_p90"], states)
    write_csv(OUT / "ensemble_by_method.csv",
              ["year", "method", "gw_p10", "gw_p50", "gw_p90", "gwh_p10", "gwh_p50", "gwh_p90"], by_method)
    write_csv(OUT / "ensemble_leave_one_out.csv",
              ["year", "left_out", "gw_p50", "gwh_p50", "gw_p50_shift_pct"], loo)
    hdr = ["year"]
    for k in ("throughput_twh", "loss_twh", "aux_twh", "net_twh", "share_pct", "minutes_of_load"):
        hdr += [f"{k}_p10", f"{k}_p50", f"{k}_p90"]
    write_csv(OUT / "ensemble_electricity_use.csv", hdr, use_rows)
    np.savez_compressed(OUT / "ensemble_pooled.npz",
                        **{f"gw_{y}": v[0] for y, v in pooled.items()},
                        **{f"gwh_{y}": v[1] for y, v in pooled.items()})
    return {"states": states, "loo": loo, "use": use_rows, "by_method": by_method,
            "samples": samples, "pooled": pooled}


if __name__ == "__main__":
    o = main()
    print("year methods              GW low/base/high    GWh low/base/high     h base  Vincent p50")
    for r in o["states"]:
        print(f"{r[0]} {r[1]:<16} {r[2]:5.1f} {r[3]:5.1f} {r[4]:5.1f}   {r[5]:6.1f} {r[6]:6.1f} {r[7]:6.1f}   {r[9]:4.1f}   {r[12]:5.1f}")
    print("\nLeave-one-out P50 shift (%)")
    for r in o["loo"]:
        print(f"{r[0]} -{r[1]}: {r[4]:+5.0f}%")
    print("\nElectricity use: net TWh p10/50/90, share %, minutes of load p50")
    for r in o["use"]:
        print(f"{r[0]}  net {r[10]:.2f} {r[11]:.2f} {r[12]:.2f} TWh | {r[13]:.2f}-{r[15]:.2f} % | throughput {r[2]:.1f} TWh | {r[17]:.0f} min")
