import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cost  # noqa: E402
import retained  # noqa: E402
from common import load_params  # noqa: E402


class TestRetained(unittest.TestCase):
    def setUp(self):
        self.p = load_params()

    def test_blog_totals_and_text(self):
        s = retained.blog_states(self.p)
        self.assertEqual(s["2027"]["total"], 3940)
        self.assertEqual(s["2028"]["total"], 4440)
        self.assertEqual(s["2027"]["operating_plus_fid"], 3600)
        self.assertEqual(s["2028"]["operating_plus_fid"], 3900)

    def test_all_retained_assertions(self):
        retained.check_blog(self.p)
        retained.check_fingrid_duration(self.p)


class TestCost(unittest.TestCase):
    def setUp(self):
        self.p = load_params()
        self.a = cost.anchor_2025(self.p)
        self.fx = self.p["fx"]["usd_per_eur_2025"]["value"]

    def test_anchor_reproduces_bnef_global_2h_4h(self):
        prem = self.a["europe_premium"]
        c_e_usd = self.a["c_e"] * self.fx / prem
        c_p_usd = self.a["c_p"] * self.fx / prem
        self.assertAlmostEqual(c_e_usd + c_p_usd / 2, 124, places=9)
        self.assertAlmostEqual(c_e_usd + c_p_usd / 4, 110, places=9)

    def test_method_A_hits_bnef_europe_4h_2035(self):
        idx = cost.equipment_index(self.p, 2035, "central", "A")
        usd = (self.a["c_e"] + self.a["c_p"] / 4) * idx * self.fx
        self.assertAlmostEqual(usd, 101, places=6)

    def test_component_model_reproduces_bnef_europe_2035(self):
        c = cost.components(self.p, 2035)
        usd = (c["c_e"] + c["c_p"] / 4) * self.fx
        self.assertAlmostEqual(usd, 101, delta=101 * 0.03)

    def test_component_anchor_2025_exact(self):
        c = cost.components(self.p, 2025)
        self.assertAlmostEqual(c["c_e"], self.a["c_e"], places=9)
        self.assertAlmostEqual(c["c_p"], self.a["c_p"], places=9)

    def test_index_monotone_and_ordered(self):
        for m in ("C", "A"):
            prev = 10.0
            for y in range(2027, 2051):
                i = cost.equipment_index(self.p, y, "central", m)
                self.assertLessEqual(i, prev + 1e-12)
                prev = i
            for y in (2026, 2030, 2040, 2050):
                lo, ce, hi = (cost.equipment_index(self.p, y, c, m) for c in ("low", "central", "high"))
                self.assertLessEqual(lo, ce)
                self.assertLessEqual(ce, hi)

    def test_component_and_published_paths_agree_central(self):
        for y in (2030, 2035, 2040, 2050):
            a = cost.equipment_index(self.p, y, "central", "A")
            c = cost.equipment_index(self.p, y, "central", "C")
            self.assertLess(abs(c / a - 1), 0.15, (y, a, c))

    def test_premium_axis_orders_cost(self):
        # includes the near-term rebound years, where the premium spread must survive
        for y in (2026, 2027, 2028, 2030, 2035, 2050):
            v = [cost.equipment_index(self.p, y, "central", "C", premium=k) for k in ("low", "central", "high")]
            self.assertTrue(v[0] < v[1] < v[2], (y, v))

    def test_cost_dimensions_are_separate(self):
        """Financing and domestic project costs order annualised cost; technology learning does not
        move grid connection; each axis orders cost low < central < high."""
        y, d = 2035, 4
        a = [cost.annualised_cost_per_kw(self.p, y, d, "central", premium="central", finance=f,
                                         project="central")["total"] for f in ("low", "central", "high")]
        self.assertTrue(a[0] < a[1] < a[2], a)
        g = [cost.capex_per_kw(self.p, y, d, c, premium="central", project="central")["connection"]
             for c in ("low", "central", "high")]
        self.assertEqual(len(set(round(v, 6) for v in g)), 1, g)
        pr = [cost.annualised_cost_per_kw(self.p, y, d, "central", premium="central", finance="central",
                                          project=j)["total"] for j in ("low", "central", "high")]
        self.assertTrue(pr[0] < pr[1] < pr[2], pr)

    def test_cycle_cost_brackets_blog(self):
        lo_b, hi_b = self.p["retained_blog"]["cycle_cost_eur_per_mwh"]
        c = {k: cost.cycle_cost(self.p, 2025, k)["total"] for k in ("low", "central", "high")}
        self.assertTrue(lo_b <= c["central"] <= hi_b)
        self.assertTrue(c["low"] <= lo_b and c["high"] >= hi_b)

    def test_validation_projects_within_25pct(self):
        for r in cost.validate_projects(self.p):
            self.assertTrue(0.75 <= r[3] <= 1.25, r)

    def test_duration_pressure_rises(self):
        r25 = cost.marginal_hour_cost(self.p, 2025)["ratio"]
        r50 = cost.marginal_hour_cost(self.p, 2050)["ratio"]
        self.assertLess(r50, r25)




class TestTechCards(unittest.TestCase):
    def test_cards_load_and_have_role(self):
        import json
        from common import ROOT
        d = json.load(open(ROOT / "model" / "tech_cards.json", encoding="utf-8"))
        for k, v in d.items():
            if k == "_meta":
                continue
            self.assertIn("role", v, k) if k != "grid_forming" else self.assertIn("model_use", v)

    def test_proposed_rte_brackets_pnnl_and_nrel(self):
        import json
        from common import ROOT
        c = json.load(open(ROOT / "model" / "tech_cards.json", encoding="utf-8"))["li_ion_lfp"]["rte_ac"]
        pm = c["proposed_model"]
        self.assertLessEqual(pm["low"], c["pnnl_transformer_2021"] + 1e-9)
        self.assertTrue(pm["low"] <= c["nrel_2025"] <= pm["high"])


class TestForecast(unittest.TestCase):
    """Asserted reproductions and consistency checks for M1-M5 and the pool."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import ensemble
        import f_methods
        cls.np = np
        cls.p = load_params()
        cls.fm = f_methods
        cls.o = ensemble.main(cls.p, n=600, write=False)
        cls.st = {r[0]: r for r in cls.o["states"]}
        cls.Y = list(f_methods.YEARS)

    def test_m1_reproduces_blog_fid_stock_2028(self):
        gw = self.o["samples"]["M1"][0][:, self.Y.index(2028)]
        self.assertAlmostEqual(float(self.np.median(gw)), 3.9, delta=0.15)

    def test_m2_passes_through_anchors(self):
        gw = self.o["samples"]["M2"][0]
        m1 = self.o["samples"]["M1"][0]
        i = self.Y.index(2028)
        self.assertLess(float(self.np.max(self.np.abs(gw[:, i] - m1[:, i]))), 0.05)

    def test_m5_medians_hit_published_states(self):
        gw = self.o["samples"]["M5"][0]
        for y, v in self.p["forecast"]["m5_envelope"]["points"].items():
            self.assertAlmostEqual(float(self.np.median(gw[:, self.Y.index(int(y))])), v[1], delta=0.06 * v[1])

    def test_no_method_below_committed_floor(self):
        fg, _ = self.fm.committed_floor(self.p)
        for y, row in self.st.items():
            for m in row[1].split("+"):
                if m == "M3":
                    continue  # M3 carries its own committed vintages and retirements
                g = self.o["samples"][m][0][:, self.Y.index(y)]
                self.assertTrue((g >= fg[self.Y.index(y)] - 1e-9).all(), (m, y))

    def test_base_within_fingrid_states(self):
        f = self.p["retained_blog"]["fingrid_states_mw"]
        for y in (2030, 2035):
            lo, hi = f[str(y)]["moderate"] / 1000, f[str(y)]["high"] / 1000
            self.assertTrue(lo <= self.st[y][3] <= hi, (y, self.st[y][3]))

    def test_base_monotone_and_states_ordered(self):
        ys = sorted(y for y in self.st if y >= 2028)
        for a, b in zip(ys, ys[1:]):
            self.assertLessEqual(self.st[a][3], self.st[b][3] + 1e-9)
        for r in self.st.values():
            self.assertTrue(r[2] <= r[3] <= r[4] and r[5] <= r[6] <= r[7])

    def test_energy_at_least_one_hour(self):
        for r in self.st.values():
            self.assertGreaterEqual(r[8], 1.0)

    def test_m3_calibrated_to_2026_fleet(self):
        import f_equilibrium as m3
        c = m3.simulate(self.p, m3.central_sample(self.p))
        nstar = float(c["nstar"][0, 0]) if "nstar" in c else float(c["gw"][0, 0])
        self.assertAlmostEqual(nstar / self.p["forecast"]["start"]["stock_gw"], 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
