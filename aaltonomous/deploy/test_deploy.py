"""Deploy tests, mock backend, real-time (takes ~45 s). Docker test only with AALTO_TEST_IMAGE=<image> set. Run: python -m unittest aaltonomous/deploy/test_deploy.py"""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from aaltonomous.deploy import protocol as P
from aaltonomous.deploy.policy_server import HoldPolicy, Policy, serve
from aaltonomous.deploy.replay import load_obs, replay
from aaltonomous.deploy import runner as R
from aaltonomous.deploy.runner import run, split_action


def start(policy, port):
    ready = threading.Event()
    threading.Thread(target=serve, args=(policy, port, "127.0.0.1", ready), daemon=True).start()
    assert ready.wait(5)
    return ready


class ChunkPolicy(Policy):
    name = "chunk"

    def __init__(self, k=10):
        self.k, self.calls = k, []

    def act(self, obs):
        self.calls.append(time.monotonic())
        a = np.tile(obs["arm_q"].astype(float), (self.k, 1))
        a[:, 3] += np.linspace(0.0, 0.2, self.k)
        return {"arm_q_target": a, "hand_q_target": np.zeros((self.k, 14))}


class Silent(Policy):
    name = "silent"

    def act(self, obs):
        time.sleep(10)


class Slow(HoldPolicy):  # a VLM-style policy: 1.2 s per call, answers with a 10-tick chunk
    name = "slow"

    def act(self, obs):
        time.sleep(1.2)
        a = super().act(obs)
        return {k: np.repeat(np.asarray(v)[None], 10, 0) for k, v in a.items()}


class SlowTau(Slow):  # slow chunks with a feed-forward torque (one value beyond the clamp)
    name = "slow_tau"
    TAU = np.r_[[-8.5, -2.3, -0.1, -3.3, 0.0, -1.2, -0.2], [-8.5, 2.3, 0.1, -3.3, 0.0, -1.2, 40.0]]

    def act(self, obs):
        return {**super().act(obs), "arm_tau_ff": self.TAU}


class Failing(Policy):  # every call raises -> error replies
    name = "failing"

    def act(self, obs):
        raise RuntimeError("boom")


class Bad(Policy):
    name = "bad"

    def act(self, obs):
        return {"arm_q_target": np.full(14, np.nan), "hand_q_target": np.zeros(5)}


class Wild(Policy):  # valid shape, far outside limits and velocity
    name = "wild"

    def act(self, obs):
        return {"arm_q_target": np.full(14, 50.0), "hand_q_target": np.full(14, -50.0)}


class T(unittest.TestCase):
    def go(self, policy, port, episodes=1, time_limit=3.0, **kw):
        srv = start(policy, port)
        try:
            with tempfile.TemporaryDirectory() as d:
                s, run_dir = run("mock", f"ws://127.0.0.1:{port}", "t", episodes, out=d, time_limit=time_limit, **kw)
                files = sorted(p.name for p in run_dir.iterdir())
                log = np.load(run_dir / "episode_000.npz") if (run_dir / "episode_000.npz").exists() else None
                log = {k: log[k] for k in log.files} if log is not None else None
            return s, files, log
        finally:
            srv.server.shutdown()

    def test_protocol_roundtrip(self):
        o = {"a": np.arange(6, dtype=np.uint16).reshape(2, 3), "b": "x", "c": np.float32(2)}
        r = P.unpack(P.pack({"obs": o}))["obs"]
        np.testing.assert_array_equal(r["a"], o["a"])
        self.assertEqual(r["a"].dtype, np.uint16)
        self.assertEqual(r["b"], "x")

    def test_end_to_end_hold(self):
        s, files, log = self.go(HoldPolicy(), 8101, episodes=2, time_limit=1.5, video=True)
        self.assertEqual(len(s), 2)
        self.assertEqual(s[0]["end_reason"], "time_limit")
        self.assertEqual(s[0]["safety"]["clipped_steps"], 0)
        self.assertGreater(s[0]["effective_hz"], 25)
        for f in ("episode_000.npz", "episode_000.jsonl", "episode_000.mp4", "episode_001.npz", "summary.jsonl"):
            self.assertIn(f, files)
        self.assertEqual(log["arm_q_cmd"].shape[1], 14)

    def test_malformed_action_rejected(self):
        s, _, log = self.go(Bad(), 8102, time_limit=2.0)
        self.assertGreater(s[0]["safety"]["clipped_steps"], 0)
        self.assertTrue(np.isfinite(log["arm_q_cmd"]).all() and np.isfinite(log["hand_q_cmd"]).all())
        self.assertTrue(np.abs(log["arm_q_cmd"]).max() < 0.01)  # held, never moved

    def test_wild_action_clipped_and_aborted(self):
        s, _, log = self.go(Wild(), 8103, time_limit=5.0)
        self.assertEqual(s[0]["end_reason"], "aborted")
        steps = np.abs(np.diff(log["arm_q_cmd"], axis=0)).max()
        self.assertLessEqual(steps, 1.0 / 30 + 1e-6)

    def test_slow_policy_held_not_stopped(self):
        # spec §1.1: the runner holds while a slow policy computes; the watchdog must not end the episode
        s, _, log = self.go(Slow(), 8104, time_limit=4.0)
        self.assertEqual(s[0]["end_reason"], "time_limit")
        self.assertEqual(s[0]["safety"]["clipped_steps"], 0)
        self.assertNotIn("watchdog_hold", s[0]["safety"]["counts"])
        self.assertNotIn("watchdog_stop", s[0]["safety"]["counts"])
        self.assertGreater(s[0]["wait_ticks"], 60)
        self.assertGreater(s[0]["latency_ms_max"], 1100)
        self.assertLess(s[0]["command_hz"], 15)                    # waits don't count as commands
        self.assertTrue(log["policy_wait"].any() and (log["hold"] >= log["policy_wait"]).all())
        self.assertLess(np.abs(np.diff(log["arm_q_cmd"], axis=0)).max(), 1e-9)

    def test_stuck_policy_latency_cap(self):
        t0 = time.monotonic()
        s, _, _ = self.go(Silent(), 8108, episodes=2, time_limit=10.0, max_latency=1.0)
        self.assertEqual([x["end_reason"] for x in s], ["policy_timeout", "policy_timeout"])  # 2nd: reconnected
        self.assertLess(time.monotonic() - t0, 8.0)
        self.assertNotIn("watchdog_stop", s[0]["safety"]["counts"])

    def test_error_policy_watchdog(self):
        t0 = time.monotonic()
        s, _, _ = self.go(Failing(), 8109, time_limit=10.0)
        self.assertEqual(s[0]["end_reason"], "watchdog")
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertEqual(s[0]["safety"]["counts"].get("watchdog_stop"), 1)

    def test_cli_env_kwargs(self):
        got = {}
        orig = R.run
        R.run = lambda *a, **k: (got.update(args=a, kw=k), ([], Path(".")))[1]
        try:
            R.main(["--backend", "sim", "--team", "t"])
            self.assertTrue(got["args"][13]["head_rgbd"])
            self.assertEqual(got["args"][13]["img_server_ip"], "127.0.0.1")
            R.main(["--backend", "real", "--team", "t", "--no-head-rgbd", "--task", "insert_shapes"])
            self.assertFalse(got["args"][13]["head_rgbd"])
            self.assertEqual(got["args"][13]["img_server_ip"], "192.168.123.164")
        finally:
            R.run = orig

    @unittest.skipUnless(os.environ.get("AALTO_TEST_IMAGE"), "set AALTO_TEST_IMAGE=<submission image> (docker)")
    def test_container_mock(self):
        with tempfile.TemporaryDirectory() as d:
            s, run_dir = run("mock", None, "t", 2, out=d, time_limit=2.0, container=os.environ["AALTO_TEST_IMAGE"],
                             host_port=8110, gpus=os.environ.get("AALTO_TEST_GPUS") or None, do_export=True)
            meta = __import__("json").loads((run_dir / "run.json").read_text())
            self.assertEqual([x["end_reason"] for x in s], ["time_limit"] * 2)
            self.assertLess(meta["startup_s"], 120)
            self.assertTrue((run_dir / "container.log").exists() and run_dir.with_suffix(".tar.gz").exists())
        self.assertNotIn("aalto-sub-", __import__("subprocess").run(["docker", "ps", "-a"], capture_output=True,
                                                                    text=True).stdout)

    def test_chunk_timing(self):
        pol = ChunkPolicy(k=10)
        s, _, log = self.go(pol, 8105, time_limit=2.0)
        t = log["t"]
        # a chunk of 10 = 10 ticks at 30 Hz = 0.333 s between calls
        gaps = np.diff(pol.calls)[1:]
        self.assertGreater(len(gaps), 3)
        self.assertTrue(np.all(np.abs(gaps - 10 / 30) < 0.08), gaps)
        self.assertGreater(s[0]["effective_hz"], 28)
        self.assertLess(s[0]["hold_ticks"], 0.2 * s[0]["ticks"])

    def test_arm_tau_ff_kept_on_waits_dropped_at_end(self):
        from aaltonomous.interface.env import AaltoEnv
        env = AaltoEnv(backend="mock", time_limit=3.0, realtime=True)
        s, _, log = self.go(SlowTau(), 8108, time_limit=3.0, env=env)
        want = SlowTau.TAU.copy()
        want[13] = 2.5                                         # wrist yaw clamp
        tau, wait = log["arm_tau_cmd"], log["policy_wait"]
        first = np.flatnonzero(np.abs(tau).sum(1) > 0)[0]     # before the first reply: runner holds with 0
        self.assertTrue(np.all(tau[:first] == 0) and wait[:first].all())
        np.testing.assert_array_equal(tau[first:], np.tile(want, (len(tau) - first, 1)))  # policy ticks AND waits
        self.assertTrue(wait[first:].any())
        self.assertEqual(s[0]["end_reason"], "time_limit")
        self.assertEqual(s[0]["safety"]["clipped_steps"], 0)
        np.testing.assert_array_equal(env.tau_cmd, np.zeros(14))  # released at the end of the episode
        np.testing.assert_array_equal(env.backend.arm_tau, np.zeros(14))
        env.close()

    def test_split_action(self):
        a = split_action({"arm_q_target": np.zeros((4, 14)), "hand_q_target": np.zeros(14), "done": True})
        self.assertEqual(len(a), 4)
        self.assertTrue(a[-1]["done"] and "done" not in a[0])
        self.assertEqual(len(split_action({"arm_q_target": np.zeros((3, 14)), "hand_q_target": np.zeros((2, 14))})), 1)
        tau = np.arange(14.0)
        a = split_action({"arm_q_target": np.zeros((4, 14)), "hand_q_target": np.zeros(14), "arm_tau_ff": tau})
        self.assertTrue(len(a) == 4 and all(np.array_equal(t["arm_tau_ff"], tau) for t in a))  # (14,) repeated
        a = split_action({"arm_q_target": np.zeros((4, 14)), "hand_q_target": np.zeros(14),
                          "arm_tau_ff": np.outer(np.arange(4), tau)})
        self.assertTrue(all(np.array_equal(t["arm_tau_ff"], i * tau) for i, t in enumerate(a)))   # row per tick
        self.assertEqual(len(split_action({"arm_q_target": np.zeros((4, 14)), "hand_q_target": np.zeros(14),
                                           "arm_tau_ff": np.zeros((3, 14))})), 1)               # rows mismatch

    def test_replay(self):
        start(HoldPolicy(), 8106)
        r = replay("ws://127.0.0.1:8106", load_obs(None, 60, True))
        self.assertTrue(r["pass"], r)
        start(Bad(), 8107)
        self.assertFalse(replay("ws://127.0.0.1:8107", load_obs(None, 5, True))["pass"])


if __name__ == "__main__":
    unittest.main()
