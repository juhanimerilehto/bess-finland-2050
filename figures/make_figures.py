"""Figures for the cost and technology module (paper figures, static)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import cost  # noqa: E402
from common import load_params  # noqa: E402

FIG = ROOT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CAT = {"A": "#2a78d6", "B": "#eb6834"}
DUR = {1: "#86b6ef", 2: "#3987e5", 4: "#1c5cab", 8: "#0d366b"}  # ordinal blue ramp

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "lines.linewidth": 2,
})

YEARS = list(range(2025, 2051))


def fig_learning(p):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    labels = {"A": "A: published forecasts (BNEF to 2035, NREL ratios)", "B": "B: experience curve"}
    for m in ("A", "B"):
        lo = [cost.equipment_index(p, y, "low", m) for y in YEARS]
        ce = [cost.equipment_index(p, y, "central", m) for y in YEARS]
        hi = [cost.equipment_index(p, y, "high", m) for y in YEARS]
        ax.fill_between(YEARS, lo, hi, color=CAT[m], alpha=0.12, linewidth=0)
        ax.plot(YEARS, ce, color=CAT[m], label=labels[m])
        ax.annotate(f"{ce[-1]:.2f}", (2050, ce[-1]), xytext=(4, 5 if m == "B" else -5), textcoords="offset points",
                    va="center", color=INK, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(2025, 2052)
    ax.set_ylabel("Equipment cost index (2025 = 1)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Battery equipment cost, central paths with low–high bands", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cost_learning.pdf")
    fig.savefig(FIG / "fig_cost_learning.png")
    plt.close(fig)


def fig_capex_cycle(p):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ends = {}
    for d, col in DUR.items():
        v = [cost.capex_per_kw(p, y, d, include_connection=False)["per_kwh"] for y in YEARS]
        ax1.plot(YEARS, v, color=col, label=f"{d} h")
        ends[d] = v[-1]
    # end labels: keep at least `gap` EUR/kWh apart so they never overlap
    gap, placed = 16.0, {}
    for d in sorted(ends, key=lambda k: ends[k]):
        y = ends[d]
        if placed:
            y = max(y, max(placed.values()) + gap)
        placed[d] = y
    for d, y in placed.items():
        ax1.annotate(f"{d} h  {ends[d]:.0f}", (2050, ends[d]), xytext=(2051.0, y), textcoords="data",
                     va="center", color=INK, fontsize=8, annotation_clip=False)
    m = p["validation_projects"]["mertaniemi"]
    obs = m["meur"] * 1e6 / (m["mwh"] * 1e3)
    mod = cost.capex_per_kw(p, m["contract_year"], m["mwh"] / m["mw"], include_connection=False)["per_kwh"]
    ax1.scatter([m["contract_year"]], [obs], s=36, color=INK, zorder=5, edgecolor="white", linewidth=1.5)
    ax1.scatter([m["contract_year"]], [mod], s=30, color="white", zorder=5, edgecolor=INK, linewidth=1.2)
    ax1.annotate("Mertaniemi 2024 (1.1 h):\nobserved (filled), model (open)", (2024, obs), xytext=(6, -26),
                 textcoords="offset points", fontsize=7, color=INK2)
    ax1.set_xlim(2023.5, 2055)
    ax1.set_ylim(0, None)
    ax1.set_ylabel("Installed cost, EUR per kWh (central)")
    ax1.set_title("Installed cost by duration (excl. grid connection)", loc="left", fontsize=9, color=INK)

    yrs = (2025, 2035, 2050)
    deg = [cost.cycle_cost(p, y)["degradation"] for y in yrs]
    los = [cost.cycle_cost(p, y)["efficiency_loss"] for y in yrs]
    x = range(len(yrs))
    ax2.axhspan(*p["retained_blog"]["cycle_cost_eur_per_mwh"], color="#f0efec", zorder=0)
    ax2.text(-0.42, 29.3, "Market review: 15–30", ha="left", va="top", fontsize=7, color=INK2)
    ax2.bar(x, deg, width=0.55, color=CAT["A"], label="Degradation")
    ax2.bar(x, los, width=0.55, bottom=[d + 0.3 for d in deg], color=CAT["B"], label="Efficiency loss")
    for i, (d, l) in enumerate(zip(deg, los)):
        ax2.text(i, d + l + 1.0, f"{d + l:.0f}", ha="center", fontsize=8, color=INK)
    ax2.set_xticks(list(x), [str(y) for y in yrs])
    ax2.set_ylim(0, 40)
    ax2.set_ylabel("EUR per discharged MWh")
    ax2.set_title("Marginal cycle cost (central)", loc="left", fontsize=9, color=INK)
    ax2.legend(loc="upper right", fontsize=7, ncol=2, columnspacing=0.8, handlelength=1.2)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_capex_cycle.pdf")
    fig.savefig(FIG / "fig_capex_cycle.png")
    plt.close(fig)


def fig_reserves(p):
    import csv
    from datetime import date
    out = ROOT / "outputs"
    if not (out / "reserves_monthly_price.csv").is_file():
        return
    series = {}
    for r in csv.DictReader(open(out / "reserves_monthly_price.csv", encoding="utf-8")):
        series.setdefault(r["product"], []).append((r["month"], float(r["mean_price_eur_mw_h"])))
    show = {"FCR-N": "#2a78d6", "FCR-D up": "#eb6834", "aFRR up": "#1baf7a", "mFRR up (hourly CM)": "#6250d6"}
    label = {"mFRR up (hourly CM)": "mFRR up"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})

    def d(m):
        return date(int(m[:4]), int(m[5:7]), 15)

    ends = []
    for name, col in show.items():
        xs = [d(m) for m, _ in series[name]]
        ys = [v for _, v in series[name]]
        ax1.plot(xs, ys, color=col, label=label.get(name, name), linewidth=1.6)
        ends.append((ys[-1], label.get(name, name)))
    ax1.set_ylabel("Mean capacity price, EUR/MW,h (monthly)")
    ax1.set_ylim(0, None)
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.set_title("Finnish hourly reserve markets, 2023–2026", loc="left", fontsize=9, color=INK)

    cap = [(r["month"], float(r["max_installed_mw"]))
           for r in csv.DictReader(open(out / "storage_capacity_monthly.csv", encoding="utf-8"))]
    ax2.plot([d(m) for m, _ in cap], [v / 1000 for _, v in cap], color="#2a78d6", label="Installed, Fingrid dataset 424")
    b = p["retained_blog"]
    pts = [(date(2026, 8, 15), b["operating_aug2026_mw"] / 1000, "Market review: operating, Aug 2026"),
           (date(2027, 12, 31), b["text_fid_stock_mw"]["2027"] / 1000, "Operating + FID, end-2027"),
           (date(2028, 12, 31), b["text_fid_stock_mw"]["2028"] / 1000, "end-2028")]
    ax2.scatter([x for x, _, _ in pts], [y for _, y, _ in pts], s=30, color=INK, zorder=5,
                edgecolor="white", linewidth=1.2)
    offs = [((6, -12), "left"), ((-4, 7), "right"), ((-4, 7), "right")]
    for (x, y, t), (xy, ha) in zip(pts, offs):
        ax2.annotate(t, (x, y), xytext=xy, textcoords="offset points", fontsize=7, color=INK2, ha=ha)
    ax2.set_ylim(0, 4.5)
    ax2.set_ylabel("Battery storage power, GW")
    ax2.set_title("Installed storage and committed capacity", loc="left", fontsize=9, color=INK)
    ax2.legend(loc="lower right", fontsize=7)
    for ax in (ax1, ax2):
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(date(2026, 1, 1), date(2029, 3, 1))
    fig.tight_layout()
    fig.savefig(FIG / "fig_reserves.pdf")
    fig.savefig(FIG / "fig_reserves.png")
    plt.close(fig)


def fig_arbitrage(p):
    import csv
    f = ROOT / "outputs" / "da_arbitrage_annual.csv"
    if not f.is_file():
        return
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    years = ["2023", "2024", "2025", "2026"]
    ycol = {"2023": "#86b6ef", "2024": "#3987e5", "2025": "#1c5cab", "2026": "#0d366b"}
    durs = [1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    w = 0.18
    for i, y in enumerate(years):
        vals = [float(next(r["margin_keur_per_mw_yr"] for r in rows if r["year"] == y and r["duration_h"] == str(d)))
                for d in durs]
        xs = [k + (i - 1.5) * (w + 0.02) for k in range(len(durs))]
        ax.bar(xs, vals, width=w, color=ycol[y], label=y + (" (to 14 Nov, annualised)" if y == "2026" else ""))
    cost_years = {2026: ("#0b0b0b", "-"), 2035: ("#52514e", "--"), 2050: ("#8a8984", ":")}
    for cy, (col, ls) in cost_years.items():
        for k, d in enumerate(durs):
            c = cost.annualised_cost_per_kw(p, cy, d)["total"]
            ax.hlines(c, k - 0.45, k + 0.45, color=col, linestyle=ls, linewidth=1.6,
                      label=f"Annualised cost {cy} (central)" if k == 0 else None)
    ax.set_xticks(range(len(durs)), [f"{d} h" for d in durs])
    ax.set_ylabel("kEUR per MW per year")
    ax.set_ylim(0, 190)
    ax.set_title("Day-ahead arbitrage margin net of charging cost (perfect foresight) vs cost of ownership", loc="left", fontsize=9, color=INK)
    h, l = ax.get_legend_handles_labels()
    order = [i for i, t in enumerate(l) if t[:4].isdigit()] + [i for i, t in enumerate(l) if not t[:4].isdigit()]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", fontsize=7, ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_arbitrage.pdf")
    fig.savefig(FIG / "fig_arbitrage.png")
    plt.close(fig)


if __name__ == "__main__":
    p = load_params()
    fig_learning(p)
    fig_capex_cycle(p)
    fig_reserves(p)
    fig_arbitrage(p)
    print("figures written")
