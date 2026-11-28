"""Regression checks for decisions that are hard to distinguish on one real episode."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robot_data_studio.quality.action_state.action_state import analyze, usable_mask
from robot_data_studio.quality.action_state.calibrate import propose
from robot_data_studio.quality.action_state.profiles import joint, preset
from robot_data_studio.quality.action_state.run_operator import run_parquet


def test_profile(**extra):
    c = joint("j", "a", "s", 0, evidence="verified position command")
    c.update(extra)
    return {"version": 1, "dataset": "fixture", "channels": [c]}


test_profile.__test__ = False


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(300) / 30.0
        self.a = 0.7 * np.sin(3 * self.t) + 0.1 * np.sin(9 * self.t)
        # Real feedback never equals a shifted command exactly; add encoder-level noise.
        noise = 1e-4 * np.random.default_rng(1).standard_normal(300)
        self.s = np.r_[np.repeat(self.a[0], 3), self.a[:-3]] + noise

    def run_pair(self, a, s, p=None, **kwargs):
        return analyze(
            "ep",
            self.t,
            {"j": {"action": a, "state": s}},
            p or test_profile(),
            **kwargs,
        )["channels"]["j"]

    def test_best_lag_positive_for_response_and_mask_preserves_good_tracking(self):
        r = self.run_pair(self.a, self.s)
        self.assertAlmostEqual(r["alignment"]["best_lag_s"], 0.1)
        self.assertEqual(r["command_rows"], 300)

    def test_missing_excitation_is_uninformative_not_a_drop(self):
        r = self.run_pair(np.ones(300), np.ones(300))
        self.assertIsNone(r["alignment"]["best_lag_s"])
        self.assertEqual(r["command_rows"], 300)

    def test_copy_and_next_proxy(self):
        r = self.run_pair(self.a, self.a)
        self.assertEqual((r["command_rows"], r["reason"]), (0, "copy_candidate"))
        a = np.r_[self.s[1:], self.s[-1]]
        r = self.run_pair(a, self.s, test_profile(role="next_state"))
        self.assertEqual(r["alignment"]["signature"], "next_state")
        self.assertEqual(r["usable_proxy_windows"][0]["rows"], [0, 299])
        a[50] += 0.2
        r = self.run_pair(a, self.s, test_profile(role="next_state"))
        self.assertIn([50, 51], [w["rows"] for w in r["proxy_mask_windows"]])

    def test_bad_channel_does_not_disable_sibling(self):
        a = np.linspace(-1, 2, 300)
        good = self.s
        p = test_profile()
        p["channels"].append(dict(p["channels"][0], name="other"))
        report = analyze(
            "ep",
            self.t,
            {
                "j": {"action": a, "state": np.zeros(300)},
                "other": {"action": self.a, "state": good},
            },
            p,
        )
        self.assertTrue(
            any(
                e["code"] == "unresponsive_feedback"
                for e in report["channels"]["j"]["events"]
            )
        )
        self.assertEqual(report["channels"]["other"]["command_rows"], 300)

    def test_missing_sample_invalid_values_and_upstream_mask(self):
        a = self.a.copy()
        a[3] = np.nan
        v = np.ones(300, bool)
        v[7] = False
        k = np.ones(300, bool)
        k[22:44] = False
        q = analyze(
            "ep",
            self.t,
            {"j": dict(action=a, state=self.s, state_valid=v)},
            test_profile(),
            keep=k,
        )
        c = q["channels"]["j"]
        self.assertEqual(
            [w["rows"] for w in c["command_mask_windows"]], [[3, 4], [7, 8]]
        )
        self.assertEqual(c["command_rows"], 276)
        self.assertEqual(
            analyze("e", self.t, {}, test_profile())["command_disposition"],
            "no_command",
        )

    def test_interior_step_and_hold_timeout_survives_until_feedback_returns(self):
        t = np.arange(900) / 30
        a = np.zeros(900)
        s = a.copy()
        a[330:] = 0.6
        s[450:] = 0.6
        r = analyze("e", t, {"j": dict(action=a, state=s)}, test_profile(edge="off"))[
            "channels"
        ]["j"]
        rows = [e["rows"] for e in r["events"] if e["code"] == "unresponsive_feedback"]
        self.assertTrue(any(lo <= 335 and hi == 450 for lo, hi in rows))
        s[335:] = 0.6
        self.assertFalse(
            analyze("e", t, {"j": dict(action=a, state=s)}, test_profile(edge="off"))[
                "channels"
            ]["j"]["events"]
        )

    def test_clock_gaps_and_no_new_edge_at_keep_boundaries(self):
        t = self.t.copy()
        t[100:] += 8
        a = np.zeros(300)
        s = np.zeros(300)
        a[100:] = 1
        r = analyze("ep", t, {"j": dict(action=a, state=s)}, test_profile(edge="mask"))[
            "channels"
        ]["j"]
        self.assertFalse(r["events"])
        self.assertLess(r["command_keep_windows"][0]["end_s"], 4)
        with self.assertRaises(ValueError):
            analyze("e", t[::-1], {}, test_profile())

    def test_tail_truncates_at_true_end_and_excludes_unrelated_channel(self):
        a = np.zeros(300)
        s = np.zeros(300)
        a[-8:] = 1
        r = self.run_pair(a, s, test_profile(edge="mask"))
        e = r["events"][0]
        self.assertEqual(e["code"], "handover_step")
        self.assertEqual(e["rows"][1], 300)
        self.assertFalse(e["evidence"]["caught"])
        self.assertAlmostEqual(e["end_s"], 10)
        rr = self.run_pair(a, s, test_profile(edge="review"))
        self.assertEqual(rr["command_rows"], 300)

    def test_internal_step_needs_edge_evidence(self):
        t = np.arange(900) / 30
        a = np.zeros(900)
        s = a.copy()
        a[450:] = 1
        s[460:] = 1
        r = analyze("e", t, {"j": dict(action=a, state=s)}, test_profile(edge="mask"))[
            "channels"
        ]["j"]
        self.assertEqual(r["command_rows"], 900)

    def test_phase_shift_across_new_keep_or_missing_feedback_never_links(self):
        p = test_profile(role="next_state")
        a = np.r_[self.s[1:], 0]
        k = np.ones(300, bool)
        k[50:60] = False
        r = self.run_pair(a, self.s, p, keep=k)
        self.assertEqual(
            [v["rows"] for v in r["usable_proxy_windows"]], [[0, 49], [60, 299]]
        )

    def grip(self):
        c = copy.deepcopy(preset("lingyu")["channels"][-2])
        c["name"] = "j"
        return dict(version=1, dataset="grip", channels=[c])

    def test_gripper_wrong_anchor_direction_and_missing_evidence(self):
        p = self.grip()
        p["channels"][0]["anchors"]["action"].reverse()
        r = self.run_pair(self.a, self.s, p)
        self.assertEqual((r["command_rows"], r["reason"]), (0, "mapping_unverified"))
        del p["channels"][0]["anchors"]["evidence"]
        self.assertEqual(self.run_pair(self.a, self.s, p)["command_rows"], 0)

    def test_gripper_stuck_against_object_and_alternating_opposite_response(self):
        p = self.grip()
        target = np.where((np.arange(300) // 60) % 2 == 0, 2.0, -1.6)
        feedback = np.full(300, 0.3)
        self.assertEqual(self.run_pair(target, feedback, p)["command_rows"], 300)
        inverted = np.where(target > 0, -0.148, 1.168)
        r = self.run_pair(target, inverted, p)
        self.assertEqual(r["reason"], "gripper_direction_conflict")
        r = self.run_pair(np.zeros(300), inverted, p)
        self.assertEqual(r["reason"], "constant_gripper_command_with_feedback_cycles")

    def test_bad_aperture_scale_detected_before_clipping(self):
        p = self.grip()
        r = self.run_pair(np.full(300, 2000.0), np.ones(300), p)
        self.assertEqual(r["reason"], "gripper_scale_conflict")

    def test_consumable_native_mask(self):
        k = np.ones(300, bool)
        k[3:7] = False
        r = analyze(
            "ep",
            self.t,
            {"j": dict(action=self.a, state=self.s)},
            test_profile(),
            keep=k,
        )
        np.testing.assert_array_equal(usable_mask(r, ["j"])[:, 0], k)
        self.assertFalse(np.any(usable_mask(r, ["j"], supervision="state_proxy")))
        with self.assertRaises(ValueError):
            usable_mask(r, ["j", "j"])

    def test_units_and_unknown_semantics(self):
        p = test_profile()
        p["channels"][0]["action"]["unit"] = "deg"
        self.assertEqual(
            self.run_pair(np.rad2deg(self.a), self.s, p)["command_rows"], 300
        )
        p["channels"][0]["action"]["unit"] = "m/s"
        self.assertEqual(
            self.run_pair(self.a, self.s, p)["reason"], "mapping_unverified"
        )
        self.assertEqual(
            self.run_pair(self.a, self.s, test_profile(role="unknown"))["command_rows"],
            0,
        )

    def test_calibration_is_versioned_and_rejects_mixed_references(self):
        r = analyze(
            "1", self.t, {"j": dict(action=self.a, state=self.s)}, test_profile()
        )
        rr = [dict(r, episode_id=str(i)) for i in range(4)]
        q = propose(rr, 3)
        self.assertEqual(q["channels"]["j"]["support"], 4)
        with self.assertRaises(ValueError):
            propose([*rr, r], 3)
        with self.assertRaises(ValueError):
            propose([*rr, dict(r, dataset="other")], 3)

    def test_parquet_adapter_and_file_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep.parquet"
            p = test_profile()
            pq.write_table(
                pa.table(
                    dict(
                        timestamp=[0.0, 0.1, 0.0, 0.1],
                        episode_index=[0, 0, 1, 1],
                        a=[[0.0], [0.1], [0.2], [0.3]],
                        s=[[0.0], [0.05], [0.15], [0.2]],
                    )
                ),
                path,
            )
            r = run_parquet(p, path, "1", episode_index=1)
            self.assertEqual(r["source_rows"], 2)
            json.dumps(r, allow_nan=False)
            with self.assertRaises(ValueError):
                run_parquet(p, path, "one", single_episode=True)

    def test_semantic_gate_withholds_contradicted_targets(self):
        rng = np.random.default_rng(0)
        cases = {
            "semantic_same_state_noisy": self.s + rng.normal(0, 2e-4, 300),
            "semantic_future_state": np.r_[self.s[5:], np.repeat(self.s[-1], 5)],
            "semantic_action_lags_state": np.r_[np.repeat(self.a[0], 10), self.a[:-10]]
            + rng.normal(0, 1e-4, 300),
            "semantic_sign_flip": -self.a,
            "semantic_unit_mismatch": self.a * 180 / np.pi,
        }
        for reason, a in cases.items():
            r = self.run_pair(a, self.s)
            self.assertEqual((r["route"], r["reason"], r["command_rows"]), ("review", reason, 0), reason)
        r = self.run_pair(self.a, self.s)
        self.assertEqual((r["route"], r["semantic"]["verdict"]), ("command", "command"))

    def test_accepted_semantic_exception_needs_evidence(self):
        a = self.a * 180 / np.pi
        r = self.run_pair(a, self.s, test_profile(accept_semantic={"verdicts": ["unit_mismatch"]}))
        self.assertEqual(r["reason"], "semantic_unit_mismatch")
        p = test_profile(accept_semantic={"verdicts": ["unit_mismatch"], "evidence": "vendor spec"})
        self.assertEqual(self.run_pair(a, self.s, p)["route"], "command")

    def test_probe_only_copy_is_withheld_from_both_training_masks(self):
        t = np.arange(600) / 30
        s = np.sin(1.2 * t) + 0.23 * np.sin(4.1 * t)
        for lag, verdict in [(0, "same_state"), (1, "next_state")]:
            a = np.roll(s, -lag)
            a[::33] += 0.02
            report = analyze("ep", t, {"j": dict(action=a, state=s)}, test_profile())
            ch = report["channels"]["j"]
            self.assertEqual(ch["alignment"]["signature"], "undetermined")
            self.assertEqual(ch["semantic"]["verdict"], verdict)
            self.assertEqual(ch["route"], "review")
            self.assertFalse(usable_mask(report, ["j"]).any())
            self.assertFalse(usable_mask(report, ["j"], supervision="state_proxy").any())

    def test_release_scoped_agibot_arm_rechecks_each_native_row(self):
        arm = joint("left", "action", "state", 0, evidence="cache target")
        other = joint("head", "action", "state", 78, evidence="cache target")
        a = np.full(300, 0.2)
        s = a.copy()
        s[12] += 0.1
        valid = np.ones(300, bool)
        valid[50] = False
        p = dict(version=1, dataset="agibotworld-beta", channels=[arm, other],
                 cache_source=dict(release="agibotworld-beta/v3-subtask-20261116",
                                   robot="agibot_g1", layout="robot100"))
        original = copy.deepcopy(p)
        signals = {name: dict(action=a, state=s, state_valid=valid) for name in ("left", "head")}
        report = analyze("ep", self.t, signals, p)
        self.assertEqual(p, original)
        self.assertEqual(report["channels"]["left"]["route"], "state_proxy")
        self.assertFalse(usable_mask(report, ["left"]).any())
        expected = valid.copy()
        expected[12] = False
        np.testing.assert_array_equal(usable_mask(report, ["left"], supervision="state_proxy")[:, 0], expected)
        np.testing.assert_array_equal(usable_mask(report, ["head"])[:, 0], valid)
        signals["left"]["state"] = s + 0.3
        mismatch = analyze("ep", self.t, signals, p)
        self.assertEqual(mismatch["channels"]["left"]["route"], "review")
        self.assertFalse(usable_mask(mismatch, ["left"], supervision="state_proxy").any())

    def test_published_cache_preset_via_parquet_entrypoint(self):
        p = preset("agibotworld-beta-robot100-v3-20261116")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agibot-arm-cache.parquet"
            values = np.full((3, 100), 0.2)
            pq.write_table(pa.table({
                "timestamp": [0.0, 0.033, 0.067],
                "action": values.tolist(), "state": values.tolist(),
            }), path)
            report = run_parquet(p, path, "cache-fixture", single_episode=True)
        names = [str(k) for k in range(14)]
        self.assertEqual(list(report["channels"]), names)
        self.assertFalse(usable_mask(report, names).any())
        self.assertTrue(usable_mask(report, names, supervision="state_proxy").all())

    def test_unverified_cache_scopes_and_mismatched_slots_do_not_inherit(self):
        source = dict(release="agibotworld-beta/v3-subtask-20261116", robot="agibot_g1", layout="robot100")
        ch = joint("j", "action", "state", 13, evidence="verified independent target")
        for dataset, info, channel in [
            ("another", source, ch),
            ("agibotworld-beta", {}, ch),
            ("agibotworld-beta", dict(source, release="agibotworld-beta/other"), ch),
            ("agibotworld-beta", dict(source, robot="another"), ch),
            ("agibotworld-beta", dict(source, layout="raw"), ch),
            ("agibotworld-beta", source, dict(ch, state=dict(ch["state"], component=12))),
            ("agibotworld-beta", source, dict(ch, action=dict(ch["action"], column="raw"))),
        ]:
            p = dict(version=1, dataset=dataset, cache_source=info, channels=[channel])
            self.assertEqual(self.run_pair(np.ones(300), np.ones(300), p)["command_rows"], 300)

    def test_declared_proxy_needs_most_rows_to_agree(self):
        a = self.s + 0.01
        a[:5] = self.s[:5]
        r = self.run_pair(a, self.s, test_profile(role="same_state"))
        self.assertEqual((r["route"], r["reason"]), ("review", "declared_proxy_disagrees"))
        self.assertLess(r["proxy_match_fraction"], 0.05)
        self.assertEqual(r["usable_proxy_windows"], [])


if __name__ == "__main__":
    unittest.main()
