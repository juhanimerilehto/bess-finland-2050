"""Figures for the multi-method forecast (reads outputs/*.csv written by ensemble.py)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from common import OUT, load_params  # noqa: E402

FIG = ROOT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ENS = "#2a78d6"
METH = {"M1": "#eb6834", "M2": "#1f9d6b", "M3": "#0d366b", "M4": "#8a5cd1", "M5": "#b07d10"}
NAMES = {"M1": "M1 pipeline", "M2": "M2 diffusion", "M3": "M3 equilibrium", "M4": "M4 top-down",
         "M5": "M5 official states"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "lines.linewidth": 2,
})


def read(name):
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png")
    plt.close(fig)


def fig_forecast(p):
    st = read("ensemble_states.csv")
    bm = read("ensemble_by_method.csv")
    s0 = p["forecast"]["start"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, unit, key in ((axes[0], "GW", "gw"), (axes[1], "GWh", "gwh")):
        ys = [2026] + [int(r["year"]) for r in st]
        lo = [s0[f"stock_{key}"]] + [float(r[f"{key}_low"]) for r in st]
        ba = [s0[f"stock_{key}"]] + [float(r[f"{key}_base"]) for r in st]
        hi = [s0[f"stock_{key}"]] + [float(r[f"{key}_high"]) for r in st]
        ax.fill_between(ys, lo, hi, color=ENS, alpha=0.15, linewidth=0, label="Pool P10–P90")
        ax.plot(ys, ba, color=ENS, label="Pool P50 (Base)")
        for x, b, h in zip(ys[-1:], ba[-1:], hi[-1:]):
            ax.annotate(f"{b:.0f}", (x, b), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
            ax.annotate(f"{h:.0f}", (x, h), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK2)
        off = {"M1": -0.6, "M2": -0.3, "M3": 0.0, "M4": 0.3, "M5": 0.6}
        for m, col in METH.items():
            rows = [r for r in bm if r["method"] == m]
            ax.errorbar([int(r["year"]) + off[m] for r in rows], [float(r[f"{key}_p50"]) for r in rows],
                        yerr=[[float(r[f"{key}_p50"]) - float(r[f"{key}_p10"]) for r in rows],
                              [float(r[f"{key}_p90"]) - float(r[f"{key}_p50"]) for r in rows]],
                        fmt="o", ms=3, color=col, elinewidth=0.8, capsize=0, label=NAMES[m])
        ax.set_xlim(2025.5, 2052.5)
        ax.set_ylim(0, None)
        ax.set_ylabel(f"Installed storage ({unit})")
    axes[0].set_title("Power", loc="left", fontsize=9)
    axes[1].set_title("Energy", loc="left", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_ylim(3, 250)
    axes[1].set_yticks([3, 10, 30, 100, 200])
    axes[1].set_yticklabels(["3", "10", "30", "100", "200"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig_forecast.pdf")
    fig.savefig(FIG / "fig_forecast.png")
    plt.close(fig)


SOBOL_LABELS = {"eps_own": "Own-fleet cannibalisation", "eps_flex": "Flexibility-need elasticity",
                "finance_u": "Financing (WACC, life)", "project_u": "Domestic project costs",
                "premium_u": "European premium", "cost_u": "Technology learning",
                "duration_tilt": "Duration tilt", "cons_2035": "Consumption 2035",
                "cons_2050": "Consumption 2050", "vre_mult": "VRE multiplier",
                "build_limit": "Build-rate limit", "capture": "Arbitrage capture",
                "phes": "Pumped hydro", "grid_forming": "Grid-forming units",
                "pool_share": "Battery pool share", "eps_pool_sat": "Pool shrinkage",
                "eps_pool": "Pool growth"}


def fig_sobol(p):
    rows = read("m3_sobol.csv")
    params = sorted({r["parameter"] for r in rows},
                    key=lambda k: -max(float(r["ST"]) for r in rows if r["parameter"] == k))[:8]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, metric in zip(axes, ("gw", "gwh")):
        for j, (y, col) in enumerate(((2035, "#86b6ef"), (2050, "#1c5cab"))):
            v = {r["parameter"]: float(r["ST"]) for r in rows if r["metric"] == metric and int(r["year"]) == y}
            ax.barh([i + (j - 0.5) * 0.38 for i in range(len(params))], [max(v.get(k, 0), 0) for k in params],
                    height=0.36, color=col, label=str(y))
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([SOBOL_LABELS.get(k, k) for k in params])
        if metric == "gw":
            ax.invert_yaxis()  # shared y: invert once
        ax.set_xlim(0, 1)
        ax.set_xlabel("Total-effect Sobol index $S_T$")
        ax.set_title(f"M3 fleet {'power' if metric == 'gw' else 'energy'}", loc="left", fontsize=9)
    axes[0].legend(loc="lower right", fontsize=8)
    save(fig, "fig_sobol")


def fig_use(p):
    u = read("ensemble_electricity_use.csv")
    ys = [int(r["year"]) for r in u]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    loss = [float(r["loss_twh_p50"]) for r in u]
    aux = [float(r["aux_twh_p50"]) for r in u]
    a1.bar(ys, loss, width=1.4, color="#1c5cab", label="Conversion losses (P50)")
    a1.bar(ys, aux, width=1.4, bottom=loss, color="#86b6ef", label="Auxiliaries (P50)")
    a1.errorbar(ys, [float(r["net_twh_p50"]) for r in u],
                yerr=[[float(r["net_twh_p50"]) - float(r["net_twh_p10"]) for r in u],
                      [float(r["net_twh_p90"]) - float(r["net_twh_p50"]) for r in u]],
                fmt="none", ecolor=INK2, elinewidth=0.8, label="Net use P10–P90")
    a1.set_ylabel("Electricity use of the fleet (TWh/yr)")
    a1.legend(fontsize=7.5, loc="upper left")
    a2.plot(ys, [float(r["minutes_of_load_p50"]) for r in u], color=ENS, marker="o", ms=3)
    a2.fill_between(ys, [float(r["minutes_of_load_p10"]) for r in u], [float(r["minutes_of_load_p90"]) for r in u],
                    color=ENS, alpha=0.15, linewidth=0)
    a2.set_ylabel("Stored energy, minutes of average load")
    a2.set_ylim(0, None)
    for a in (a1, a2):
        a.set_xticks([2030, 2035, 2040, 2045, 2050])
    save(fig, "fig_electricity_use")


def main():
    p = load_params()
    fig_forecast(p)
    fig_sobol(p)
    fig_use(p)


if __name__ == "__main__":
    main()
