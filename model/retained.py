"""Reproduce the retained states of the Gasum blog (Merilehto 2026a) and the
Fingrid Q3/2026 storage states it references. Every published number is
asserted; the script stops if any differs.
"""
from __future__ import annotations

from common import OUT, load_params, write_csv

HOURS_PER_YEAR = 8760


def blog_states(p: dict) -> dict:
    b = p["retained_blog"]
    states = {"2026-08": {"operating": b["operating_aug2026_mw"], "total": b["operating_aug2026_mw"]}}
    for year, parts in b["pipeline"].items():
        states[year] = dict(parts)
        states[year]["total"] = sum(parts.values())
        states[year]["operating_plus_fid"] = parts["operating"] + parts["fid"]
    return states


def check_blog(p: dict) -> list[list]:
    b = p["retained_blog"]
    s = blog_states(p)
    rows = []

    # Chart totals (bar heights) and text statements
    published_totals = {"2027": 3940, "2028": 4440}
    for y, tot in published_totals.items():
        assert s[y]["total"] == tot, (y, s[y]["total"], tot)
        rows.append([f"chart total {y} (MW)", s[y]["total"], tot])
    for y, v in b["text_fid_stock_mw"].items():
        assert s[y]["operating_plus_fid"] == v, (y, s[y]["operating_plus_fid"], v)
        rows.append([f"text: operating+FID end-{y} (MW)", s[y]["operating_plus_fid"], v])

    # Duration of the August 2026 stock and 'about twenty minutes' of average load
    dur_h = b["operating_aug2026_mwh"] / b["operating_aug2026_mw"]
    avg_load_mw = b["consumption_2025_twh"] * 1e6 / HOURS_PER_YEAR
    minutes = b["operating_aug2026_mwh"] / avg_load_mw * 60
    assert 1.5 <= dur_h <= 2.0, dur_h
    assert abs(minutes - b["minutes_of_average_load"]) <= 2.0, minutes
    rows.append(["Aug-2026 stock duration (h)", dur_h, "~1.9"])
    rows.append(["Aug-2026 energy as minutes of average load", minutes, b["minutes_of_average_load"]])

    # 2028 stock vs Fingrid Moderate 2030
    fm = b["fingrid_states_mw"]["2030"]["moderate"]
    ratio = s["2028"]["operating_plus_fid"] / fm
    assert 0.95 <= ratio <= 1.0, ratio  # 'already at the Moderate 2030 level, about 4 GW'
    rows.append(["operating+FID 2028 / Fingrid Moderate 2030", ratio, "~1 ('about 4 GW')"])
    return rows


def check_fingrid_duration(p: dict) -> list[list]:
    """Fingrid Q3/2026: scenario energy-to-power 4-8 MWh/MW; High 2035 'almost 50 GWh'."""
    f = p["fingrid_q3_2026"]
    high_2035_mw = p["retained_blog"]["fingrid_states_mw"]["2035"]["high"]
    lo, hi = f["scenario_mwh_per_mw"]
    gwh_lo, gwh_hi = high_2035_mw * lo / 1000, high_2035_mw * hi / 1000
    implied_h = f["high_2035_gwh_approx"] * 1000 / high_2035_mw
    assert lo <= implied_h <= hi, implied_h
    rows = [["Fingrid High 2035 energy range at 4-8 h (GWh)", f"{gwh_lo:.0f}-{gwh_hi:.0f}", "-"],
            ["Implied average duration for ~50 GWh (h)", implied_h, "within 4-8"]]
    for y in ("2030", "2035"):
        for case, mw in p["retained_blog"]["fingrid_states_mw"][y].items():
            rows.append([f"Fingrid {case} {y}: GWh at 4 h / 8 h", f"{mw*lo/1000:.0f} / {mw*hi/1000:.0f}", "-"])
    return rows


def reserve_pool_bound(p: dict) -> list[list]:
    """GW of batteries the 2025 reserve procurement pool could pay at observed revenues."""
    r = p["reserve_pool"]
    pool = r["procurement_cost_2025_meur"]["value"] * 1e6
    rows = []
    for label, key in (("2026", "bess_revenue_keur_per_mw_yr_2026"), ("2023-24", "bess_revenue_keur_per_mw_yr_2023_24")):
        lo, hi = r[key]["low"] * 1e3, r[key]["high"] * 1e3
        rows.append([f"reserve pool / revenue per MW at {label} levels (GW)", f"{pool/hi/1000:.2f}-{pool/lo/1000:.2f}", "-"])
    return rows


def main() -> list[list]:
    p = load_params()
    rows = check_blog(p) + check_fingrid_duration(p) + reserve_pool_bound(p)
    write_csv(OUT / "retained_checks.csv", ["quantity", "reproduced", "published_or_note"], rows)
    return rows


if __name__ == "__main__":
    for r in main():
        print(" | ".join(str(x) if not isinstance(x, float) else f"{x:.3f}" for x in r))
    print("Retained states reproduced; all assertions passed.")
