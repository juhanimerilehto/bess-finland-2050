"""Comparison figures: cost paths against other projections and observed prices, and
Finnish storage forecasts by publication vintage against observed and pooled stock."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import cost  # noqa: E402
from common import OUT, load_params  # noqa: E402

FIG = ROOT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREYS = ["#b9b8b4", "#8f8e8a", "#52514e"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "lines.linewidth": 1.8,
})
EXT = json.load(open(ROOT / "data" / "inputs" / "external_comparisons.json", encoding="utf-8"))
YRS = list(range(2025, 2051))


def _xy(points):
    ks = sorted(points, key=float)
    return [float(k) for k in ks], [points[k] for k in ks]


def fig_cost_compare(p):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    # left: levels, EUR/kWh for 4 h
    lo = [cost.capex_per_kw(p, y, 4, "low")["per_kwh"] for y in YRS]
    ce = [cost.capex_per_kw(p, y, 4, "central")["per_kwh"] for y in YRS]
    hi = [cost.capex_per_kw(p, y, 4, "high")["per_kwh"] for y in YRS]
    a1.fill_between(YRS, lo, hi, color=BLUE, alpha=0.13, linewidth=0)
    a1.plot(YRS, ce, color=BLUE, label="This study, Finland 4 h all-in")
    proj = EXT["cost_projections_eur_per_kwh_4h"]
    styles = {"NREL 2025 update, mid (US, all-in)": (GREYS[2], "-"),
              "NREL 2023 update, mid (US, all-in)": (GREYS[1], ":"),
              "DEA 2018/19 via PyPSA (excl. grid)": (GREYS[0], "--"),
              "DEA May 2025 (small unit, excl. grid)": (GREYS[1], "-"),
              "BNEF Europe turnkey (excl. EPC, grid)": (ORANGE, "-"),
              "BNEF China turnkey (excl. EPC, grid)": (ORANGE, ":")}
    for k, (col, ls) in styles.items():
        x, y = _xy(proj[k]["points"])
        a1.plot(x, y, color=col, ls=ls, lw=1.3, marker="o", ms=2.5, label=k.replace(" update", ""))
    obs = EXT["observed_eur_per_kwh"]
    a1.scatter([o["year"] for o in obs], [o["value"] for o in obs], s=14, color=INK, zorder=5, marker="D",
               label="Observed contracts and benchmarks, 2024–26")
    a1.set_xlim(2023.5, 2051)
    a1.set_ylim(0, 430)
    a1.set_ylabel("EUR per kWh, 4 h system (EUR2025)")
    a1.set_title("Cost levels by source and scope", loc="left", fontsize=9)
    a1.legend(fontsize=6.3, loc="upper right")
    # right: indices
    il = [cost.equipment_index(p, y, "low") for y in YRS]
    ic = [cost.equipment_index(p, y, "central") for y in YRS]
    ih = [cost.equipment_index(p, y, "high") for y in YRS]
    a2.fill_between(YRS, il, ih, color=BLUE, alpha=0.13, linewidth=0)
    a2.plot(YRS, ic, color=BLUE, label="This study: component model (C)")
    a2.plot(YRS, [cost.equipment_index(p, y, "central", "A") for y in YRS], color=BLUE, ls="--", lw=1.2,
            label="Published forecasts (A: BNEF to 2035, NREL mid)")
    a2.plot(YRS, [cost.equipment_index(p, y, "central", "B") for y in YRS], color=ORANGE, lw=1.3,
            label="Cells only, Wright's law (B, 18 %)")
    for k, col, ls in (("NREL 2025 low", GREYS[0], "-"), ("NREL 2025 mid", GREYS[2], "-"),
                       ("NREL 2025 high", GREYS[0], "-"), ("DEA 2018/19 via PyPSA", GREYS[0], "--"),
                       ("DEA May 2025", GREYS[1], "-.")):
        x, y = _xy(EXT["index_benchmarks"][k])
        a2.plot(x, y, color=col, ls=ls, lw=1.1, label=k if k != "NREL 2025 high" else None)
    a2.scatter([2035, 2035], [0.61, 0.56], s=16, color=INK, marker="D", zorder=5, label="BNEF 2035, Europe / China")
    a2.set_ylim(0, 1.15)
    a2.set_xlim(2024.5, 2051)
    a2.set_ylabel("Cost index (2025 = 1)")
    a2.set_title("Learning paths", loc="left", fontsize=9)
    a2.legend(fontsize=6.3, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_cost_compare.pdf")
    fig.savefig(FIG / "fig_cost_compare.png")
    plt.close(fig)


def fig_vintages(p):
    st = list(csv.DictReader(open(OUT / "ensemble_states.csv", encoding="utf-8")))
    cap = list(csv.DictReader(open(OUT / "storage_capacity_monthly.csv", encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    s0 = p["forecast"]["start"]
    ys = [2026.87] + [int(r["year"]) for r in st]
    ax.fill_between(ys, [s0["stock_gw"]] + [float(r["gw_low"]) for r in st],
                    [s0["stock_gw"]] + [float(r["gw_high"]) for r in st], color=BLUE, alpha=0.13, linewidth=0,
                    label="This study, P10–P90 (Nov 2026)")
    ax.plot(ys, [s0["stock_gw"]] + [float(r["gw_base"]) for r in st], color=BLUE, label="This study, Base")
    mx = [2026 + (int(r["month"][5:]) - 0.5) / 12 for r in cap]
    ax.plot(mx, [float(r["max_installed_mw"]) / 1000 for r in cap], color=INK, lw=2.4, label="Observed (Fingrid 424)")
    vint = EXT["finland_deployment_vintages"]
    def col(pub):
        return GREYS[0] if pub < 2025 else (GREYS[1] if pub < 2026.5 else ORANGE)
    seen = set()
    for v in vint:
        lo, hi = v["gw"]
        c = col(v["published"])
        lab = None
        key = "old" if v["published"] < 2025 else ("mid" if v["published"] < 2026.5 else "new")
        if key not in seen:
            lab = {"old": "Published before 2025", "mid": "Published 2025 – mid-2026",
                   "new": "Published Aug–Oct 2026"}[key]
            seen.add(key)
        x = v["year"] + 0.15 * (hash(v["source"]) % 5 - 2)
        if hi > lo:
            ax.plot([x, x], [lo, hi], color=c, lw=4, solid_capstyle="butt", label=lab)
        else:
            ax.scatter([x], [lo], color=c, s=28, marker="s", zorder=4, label=lab)
    ax.annotate("Fingrid vision 2040:\n2 / 7 / 11 / 31 GW", (2040.4, 24), fontsize=7, color=INK2)
    ax.annotate("ERAA 2023,\nall Nordics", (2030.3, 1.0), fontsize=7, color=INK2)
    ax.annotate("Expert\nelicitation", (2035.3, 0.3), fontsize=7, color=INK2)
    ax.set_xlim(2025.5, 2051)
    ax.set_ylim(0, 33)
    ax.set_ylabel("Battery storage power in Finland (GW)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_vintages.pdf")
    fig.savefig(FIG / "fig_vintages.png")
    plt.close(fig)


def main():
    p = load_params()
    fig_cost_compare(p)
    fig_vintages(p)


if __name__ == "__main__":
    main()
