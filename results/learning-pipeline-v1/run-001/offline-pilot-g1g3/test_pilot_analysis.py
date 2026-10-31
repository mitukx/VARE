"""Unit tests for pilot_analysis.py (stdlib unittest + numpy). Run:
    python3 -m unittest test_pilot_analysis -v   (from this directory; synthetic data only)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import pilot_analysis as pa  # noqa: E402
import make_synthetic_graded as syn  # noqa: E402

SMALL = dict(pa.PREREG, n_seeds=20, n_boot=200)


def write(rows):
    d = tempfile.mkdtemp()
    p = Path(d) / "graded.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


class Labels(unittest.TestCase):
    def test_all_passed(self):
        self.assertEqual(pa.all_passed({"a": "PASSED", "b": "PASSED"}), 1)
        self.assertEqual(pa.all_passed({"a": "PASSED", "b": "XFAIL"}), 0)
        self.assertEqual(pa.all_passed({"a": "MISSING"}), 0)
        self.assertEqual(pa.all_passed({}), 0)  # unapplied edit: no tests ran

    def test_build_labels_match_logged_verdicts(self):
        rows = syn.make(n_tasks=20, seed=3)
        d = pa.build(rows)
        self.assertEqual(len(d["t"]), 80)
        for r in d["recs"]:
            self.assertEqual(r["t"], r["reward"])
            self.assertEqual(bool(r["c"]), r["cheap_passed"])
            self.assertEqual(r["c"], r["c_derived"])
        self.assertTrue(np.all(d["c_nat"] >= d["t"]))  # F2P-only cannot reject a trusted pass here

    def test_rejects_heldout_and_bad_schema(self):
        rows = syn.make(n_tasks=2, seed=1)
        rows[0]["split"] = "heldout"
        with self.assertRaises(pa.InputError):
            pa.load_rows(write(rows))
        rows = syn.make(n_tasks=2, seed=1)
        rows[1]["schema"] = "other"
        with self.assertRaises(pa.InputError):
            pa.load_rows(write(rows))

    def test_rejects_mixed_checkpoint_group(self):
        rows = syn.make(n_tasks=2, seed=1)
        rows[1]["checkpoint_step"] = 5
        with self.assertRaises(pa.InputError):
            pa.build(rows)


class Advantages(unittest.TestCase):
    def test_constant_group_zero_and_normalized(self):
        g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        r = np.array([1, 1, 1, 1, 1, 0, 0, 0], float)
        a = pa.advantages(r, g, 2)
        self.assertTrue(np.allclose(a[:4], 0))
        self.assertAlmostEqual(float(a[4:].mean()), 0, places=9)
        self.assertAlmostEqual(float(a[4:].std()), 1, places=6)

    def test_mask_excludes(self):
        g = np.zeros(4, int)
        r = np.array([1, 0, 1, 0], float)
        a = pa.advantages(r, g, 1, mask=np.array([1, 1, 0, 0], bool))
        self.assertEqual(a[2], 0)
        self.assertEqual(a[3], 0)
        self.assertGreater(a[0], 0)

    def test_sensitivity_positive_everywhere_for_flips(self):
        g = np.array([0, 0, 0, 0])
        s = pa.sensitivity(np.zeros(4), g, 1)
        self.assertTrue(np.all(s > 0))


class Estimators(unittest.TestCase):
    def test_ht_unbiased(self):
        rng = np.random.default_rng(0)
        c, t, p = np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]), 0.2
        draws = [pa.ht(c, t, (rng.random(3) < p).astype(float), p) for _ in range(40000)]
        self.assertTrue(np.allclose(np.mean(draws, axis=0), t, atol=0.05))

    def test_backward_unbiased(self):
        rho0, rho1 = 0.2, 0.15
        r1, r0 = pa.backward(np.array([1.0]), rho0, rho1)[0], pa.backward(np.array([0.0]), rho0, rho1)[0]
        self.assertAlmostEqual((1 - rho1) * r1 + rho1 * r0, 1.0, places=9)
        self.assertAlmostEqual(rho0 * r1 + (1 - rho0) * r0, 0.0, places=9)

    def test_phase2_budget_and_floor(self):
        rng = np.random.default_rng(1)
        score = rng.random(300)
        pi2 = pa.phase2_probs(score, 30.0, 0.05)
        p = 1 - 0.95 * (1 - pi2)
        self.assertAlmostEqual(float(p.sum()), 30.0, places=3)
        self.assertTrue(np.all(p >= 0.05 - 1e-12))
        self.assertTrue(np.all(pi2 <= 1))

    def test_loo_excludes_own_group(self):
        g = np.array([0, 0, 1, 1])
        c = np.array([1.0, 1.0, 1.0, 1.0])
        t = np.array([0.0, 0.0, 1.0, 1.0])
        u = pa.loo_error_rate(c, t, np.ones(4), g, 2)
        self.assertAlmostEqual(u[0], (0 + 1) / (2 + 2))  # sees only group 1 (no errors)
        self.assertAlmostEqual(u[2], (2 + 1) / (2 + 2))  # sees only group 0 (two errors)

    def test_adaptive_ht_approximately_unbiased(self):
        rows = syn.make(n_tasks=30, seed=5)
        d = pa.build(rows)
        rng = np.random.default_rng(2)
        acc = []
        for _ in range(400):
            out = pa.run_methods(d["c_nat"], d["t"], d["gidx"], d["n_groups"], d["kappa"], rng, 0.10, 0.05)
            acc.append(out["B6_adaptive_ht"][0])
        bias = np.abs(np.mean(acc, axis=0) - d["t"]).mean()
        self.assertLess(bias, 0.15)


class PrimaryMetric(unittest.TestCase):
    def test_shrinkage_cannot_win(self):
        rng = np.random.default_rng(4)
        a_star = rng.normal(size=50)
        noisy = a_star + rng.normal(scale=1.5, size=50)  # MSE(noisy) > MSE(0) regime seen on synthetic data
        self.assertEqual(pa.cosine(np.zeros(50), a_star), 0.0)
        self.assertAlmostEqual(pa.cosine(0.1 * noisy, a_star), pa.cosine(noisy, a_star), places=12)
        # MSE prefers the shrunk / zero estimate; cos does not.
        self.assertLess(np.mean((0.1 * noisy - a_star) ** 2), np.mean((noisy - a_star) ** 2))
        self.assertLess(np.mean(a_star ** 2), np.mean((noisy - a_star) ** 2))
        self.assertGreater(pa.cosine(noisy, a_star), pa.cosine(np.zeros(50), a_star))

    def test_setting_cos_matches_kernel_without_noise(self):
        d = pa.build(syn.make(n_tasks=20, seed=9))
        pr = dict(SMALL, n_seeds=3, n_boot=50)
        r = pa.analyse_setting(d, (0.0, 0.0), pr, np.random.default_rng(0))
        self.assertAlmostEqual(r["cos"]["B2_cheap_only"], 1.0, places=6)  # c == t exactly
        self.assertAlmostEqual(r["cos"]["B4_uniform_ht"], 1.0, places=6)
        self.assertLess(r["cos"]["B3_audit_only"], 1.0)


class RFTAndGroups(unittest.TestCase):
    def test_purity_and_cap(self):
        recs = [dict(instance_id="a", sample_index=i, sha=f"s{i}") for i in range(4)]
        t = np.array([1.0, 0.0, 1.0, 1.0])
        pur, k = pa.accepted_purity(np.ones(4), t, recs, 2)
        self.assertEqual(k, 2)
        self.assertAlmostEqual(pur, 0.5)
        recs[1]["sha"] = "s0"  # duplicate output skipped
        pur, k = pa.accepted_purity(np.ones(4), t, recs, 2)
        self.assertEqual(k, 2)
        self.assertAlmostEqual(pur, 1.0)

    def test_informative_count(self):
        g = np.array([0, 0, 1, 1, 2, 2])
        t = np.array([0, 1, 0, 0, 1, 1], float)
        c = np.array([0, 1, 1, 0, 1, 1], float)
        self.assertEqual(pa.informative_groups(t, c, g, 3), 2)


class EndToEnd(unittest.TestCase):
    def test_abort_when_few_informative(self):
        rows = syn.make(n_tasks=20, seed=7, p_mixed=0.0, p_fp_given_wrong=0.0)
        res = pa.analyse(write(rows), SMALL)
        self.assertTrue(res["summary"]["abort"])
        self.assertEqual(res["verdict"]["G1"], "inconclusive")
        self.assertEqual(res["settings"], {})

    def test_full_run_schema(self):
        p = write(syn.make(seed=11))
        res = pa.analyse(p, SMALL)
        s = res["summary"]
        self.assertEqual(s["n_rollouts"], 312)
        self.assertEqual(s["n_groups"], 78)
        self.assertFalse(s["abort"])
        self.assertEqual(s["reward_vs_trusted_tests_mismatch"], 0)
        self.assertEqual(set(res["settings"]), {"syn_0.1_0.1", "syn_0.3_0.1", "syn_0.1_0.3", "nat"})
        for r in res["settings"].values():
            lo, hi = r["dcos_G1_B4_minus_B2"]["ci"]
            for m, v in r["cos"].items():
                self.assertTrue(-1.0 <= v <= 1.0, m)
            self.assertIn("mse_over_zero_baseline", r["secondary"])
            self.assertLessEqual(lo, hi)
            self.assertEqual(r["rft_false_accept_rate_mean"]["B3_audit_only"] in (0.0, None), True)
        self.assertIn(res["verdict"]["G1"], {"supported", "rejected", "inconclusive"})
        self.assertIn(res["verdict"]["G3"], {"supported", "rejected", "inconclusive"})
        md = pa.to_markdown(res)
        self.assertIn("G1", md)
        self.assertEqual(res["prereg"]["version"], "g1g3-offline-pilot-v2")
        self.assertEqual(s["prereg_sha256"], pa.PREREG_SHA256)

    def test_cli_writes_outputs(self):
        p = write(syn.make(n_tasks=30, seed=13))
        out = Path(tempfile.mkdtemp())
        rc = pa.main(["--graded", str(p), "--out-dir", str(out), "--n-seeds", "10", "--n-boot", "100"])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "pilot_result.json").exists())
        self.assertTrue((out / "pilot_verdict.md").exists())
        self.assertTrue(json.loads((out / "pilot_result.json").read_text())["summary"]["prereg_overridden"])


if __name__ == "__main__":
    unittest.main()
