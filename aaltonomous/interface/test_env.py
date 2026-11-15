"""AaltoEnv tests on the mock backend (WP1 offline part).
Run: python -m unittest aaltonomous/interface/test_env.py"""
import time
import unittest

import numpy as np

from aaltonomous.interface.env import AaltoEnv
from aaltonomous.interface.safety import N, SafetyConfig
from aaltonomous.interface.tasks import BINS, COLORS, make_instruction, sample_layout

KEYS = {"head_rgb": ((480, 640, 3), np.uint8), "head_depth": ((480, 640), np.uint16),
        "left_wrist_rgb": ((480, 640, 3), np.uint8), "right_wrist_rgb": ((480, 640, 3), np.uint8),
        "arm_q": ((N,), np.float32), "arm_dq": ((N,), np.float32), "hand_q": ((N,), np.float32)}
Z = np.zeros(N)


class T(unittest.TestCase):
    def test_obs_keys(self):
        obs = AaltoEnv().reset()
        for k, (shape, dt) in KEYS.items():
            self.assertEqual((obs[k].shape, obs[k].dtype), (shape, dt), k)
        self.assertIsInstance(obs["instruction"], str)
        self.assertEqual(obs["t"], 0.0)

    def test_open_hand_is_not_a_clip(self):
        env = AaltoEnv()
        env.reset()
        for _ in range(60):
            _, info = env.step({"arm_q_target": Z, "hand_q_target": Z})
        self.assertEqual(info["clipped_steps"], 0)

    def test_scripted_reach_and_chunk(self):
        # Elbow sweep 0 -> 1 rad passes the palms through the table top -> no counter keep-out in this test.
        env = AaltoEnv(safety=SafetyConfig(counter_keepout_min=None))
        env.reset()
        goal = np.zeros(N); goal[3] = goal[10] = 1.0  # elbows
        traj = np.linspace(0, 1, 60)[:, None] * goal  # 2 s chunk, within 1 rad/s
        obs, info = env.step({"arm_q_target": traj, "hand_q_target": np.zeros((60, N))})
        self.assertEqual(info["ticks"], 60)
        self.assertAlmostEqual(info["t"], 2.0, places=6)
        self.assertEqual(info["clipped_steps"], 0)
        self.assertLess(np.abs(obs["arm_q"] - goal).max(), 0.05)

    def test_done_and_time_limit(self):
        env = AaltoEnv(time_limit=0.5)
        env.reset()
        _, info = env.step({"arm_q_target": Z, "hand_q_target": Z, "done": True})
        self.assertEqual(info["end_reason"], "done")
        with self.assertRaises(RuntimeError):
            env.step({"arm_q_target": Z, "hand_q_target": Z})
        env.reset()
        info = {}
        for _ in range(20):
            _, info = env.step({"arm_q_target": Z, "hand_q_target": Z})
            if info["episode_over"]:
                break
        self.assertEqual(info["end_reason"], "time_limit")

    def test_bad_policy_aborts(self):
        env = AaltoEnv()
        env.reset()
        for _ in range(100):
            _, info = env.step({"arm_q_target": np.full(N, 5.0), "hand_q_target": Z})
            if info["episode_over"]:
                break
        self.assertEqual(info["end_reason"], "aborted")
        self.assertIn("arm_position", info["safety_summary"]["counts"])

    def test_watchdog_via_hold(self):
        env = AaltoEnv()
        env.reset()
        env.step({"arm_q_target": Z, "hand_q_target": Z})
        modes = [env.hold() for _ in range(40)]  # 1.33 s with no action
        self.assertIn("stale", modes)
        self.assertEqual(modes[-1], "safe_stop")
        _, info = env.step({"arm_q_target": Z, "hand_q_target": Z})
        self.assertEqual(info["end_reason"], "safe_stop")

    def test_realtime_rate(self):
        env = AaltoEnv(realtime=True)
        env.reset()
        t0 = time.monotonic()
        env.step({"arm_q_target": np.zeros((15, N)), "hand_q_target": np.zeros((15, N))})
        self.assertAlmostEqual(time.monotonic() - t0, 0.5, delta=0.05)

    def test_sim_clock_pacing(self):
        """With a backend clock (sim), ticks and t follow SIM time: a sim at 0.5x real time -> 2x wall per tick."""
        class HalfSpeedClock:          # sim time = 0.5 * wall
            def __init__(self):
                self.w0 = time.monotonic()

            def now(self):
                return 100.0 + 0.5 * (time.monotonic() - self.w0)

            def wait_until(self, t):
                while self.now() < t:
                    time.sleep(0.001)
        env = AaltoEnv(realtime=True)
        env.clock = env.backend.clock = HalfSpeedClock()
        env.reset()
        t0 = time.monotonic()
        _, info = env.step({"arm_q_target": np.zeros((6, N)), "hand_q_target": np.zeros((6, N))})
        self.assertAlmostEqual(time.monotonic() - t0, 0.4, delta=0.05)   # 6 ticks = 0.2 s sim = 0.4 s wall
        self.assertAlmostEqual(info["t"], 0.2, delta=0.02)
        self.assertEqual(info["safety_mode"], "ok")

    def test_tasks(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            cubes = sample_layout(rng)
            self.assertTrue(3 <= len(cubes) <= 6)
            self.assertEqual({c["color"] for c in cubes}, set(COLORS))
            text, mapping = make_instruction("sort_language", rng)
            self.assertEqual(set(mapping), set(COLORS))
            self.assertEqual(set(mapping.values()), set(BINS))
            self.assertNotIn("{", text)
        a, b = AaltoEnv(seed=3), AaltoEnv(seed=3)
        a.reset(); b.reset()
        self.assertEqual(a.layout, b.layout)

class TestArmTauFF(unittest.TestCase):
    """Action key arm_tau_ff: per-tick feed-forward torque next to the arm targets (bounded in AaltoEnv)."""
    TAU = np.r_[[-8.5, -2.3, -0.1, -3.3, 0.0, -1.2, -0.2], [-8.5, 2.3, 0.1, -3.3, 0.0, -1.2, 0.2]]

    def setUp(self):
        self.env = AaltoEnv()
        self.env.reset()
        self.sent = []  # (arm_q, arm_tau) of every backend command
        cmd = self.env.backend.command

        def rec(arm_q, hand_q, arm_tau=None):
            self.sent.append((np.array(arm_q, float), np.array(arm_tau, float)))
            cmd(arm_q, hand_q, arm_tau)
        self.env.backend.command = rec

    def chunk(self, k=5, **kw):
        a = np.zeros((k, N))
        a[:, 3] = np.linspace(0.01, 0.05, k)
        return {"arm_q_target": a, "hand_q_target": np.zeros((k, N)), **kw}

    def test_absent_is_zero_and_q_unchanged(self):
        self.env.step(self.chunk())
        without = list(self.sent)
        self.sent.clear()
        self.setUp()                                       # fresh env, same start
        self.env.step(self.chunk(arm_tau_ff=self.TAU))
        self.assertTrue(all(not t.any() for _, t in without))
        np.testing.assert_array_equal(np.array([q for q, _ in self.sent]), np.array([q for q, _ in without]))
        np.testing.assert_array_equal(self.env.backend.arm_tau, self.TAU)

    def test_broadcast_and_rows(self):
        self.env.step(self.chunk(arm_tau_ff=self.TAU))
        self.assertEqual(len(self.sent), 5)
        for _, t in self.sent:
            np.testing.assert_array_equal(t, self.TAU)
        self.sent.clear()
        rows = np.outer(np.arange(1, 6), np.full(N, 0.1))
        self.env.step(self.chunk(arm_tau_ff=rows))
        np.testing.assert_allclose(np.array([t for _, t in self.sent]), rows)

    def test_clamp_and_nonfinite(self):
        tau = self.TAU.copy()
        tau[0], tau[2], tau[5], tau[13] = 100.0, np.nan, -9.0, np.inf
        _, info = self.env.step({"arm_q_target": Z, "hand_q_target": Z, "arm_tau_ff": tau})
        want = self.TAU.copy()
        want[0], want[2], want[5], want[13] = 15.0, 0.0, -2.5, 0.0
        np.testing.assert_array_equal(self.sent[-1][1], want)
        self.assertIn("arm_tau_ff_clamp", info["clip_reasons"])
        self.assertIn("invalid:arm_tau_ff:nonfinite", info["clip_reasons"])
        self.assertEqual(info["clipped_steps"], 0)   # the filter's clip / abort count stays q-only
        self.assertTrue(np.all(self.env.tau_max <= self.env.filter.spec.arm_effort))

    def test_shape_error(self):
        self.env.step({"arm_q_target": Z, "hand_q_target": Z, "arm_tau_ff": self.TAU})
        q = np.full(N, 0.0)
        q[3] = 0.02
        _, info = self.env.step({"arm_q_target": q, "hand_q_target": Z, "arm_tau_ff": np.ones(13)})
        np.testing.assert_array_equal(self.sent[-1][0], q)             # the target is still taken
        np.testing.assert_array_equal(self.sent[-1][1], self.TAU)      # the torque holds its last value
        self.assertIn("invalid:arm_tau_ff:shape(13,)", info["clip_reasons"])
        _, info = self.env.step({"arm_q_target": q, "hand_q_target": Z, "arm_tau_ff": "x"})
        self.assertIn("invalid:arm_tau_ff:shape()", info["clip_reasons"])
        n = len(self.sent)
        _, info = self.env.step(self.chunk(arm_tau_ff=np.zeros((3, N))))  # rows != chunk: whole reply rejected
        self.assertEqual(len(self.sent), n + 1)
        np.testing.assert_array_equal(self.sent[-1][0], q)
        np.testing.assert_array_equal(self.sent[-1][1], self.TAU)
        self.assertIn("invalid:arm_q_target:shape(5, 14)", info["clip_reasons"])

    def test_holds_keep_it_safe_stop_and_release_drop_it(self):
        self.env.step({"arm_q_target": Z, "hand_q_target": Z, "arm_tau_ff": self.TAU})
        modes = [self.env.hold() for _ in range(40)]       # 1.33 s with no action: stale, then safe stop
        taus = [t for _, t in self.sent[-40:]]
        for m, t in zip(modes, taus):
            np.testing.assert_array_equal(t, self.TAU if m in ("ok", "stale") else Z)
        self.assertIn("stale", modes)
        self.assertEqual(modes[-1], "safe_stop")
        self.env.reset()
        np.testing.assert_array_equal(self.sent[-1][1], Z)               # every episode starts at 0
        self.env.step({"arm_q_target": Z, "hand_q_target": Z, "arm_tau_ff": self.TAU})
        n = len(self.sent)
        self.env.release_tau()                                             # episode end
        np.testing.assert_array_equal(self.sent[-1][0], self.env.filter.arm_cmd)
        np.testing.assert_array_equal(self.sent[-1][1], Z)
        self.env.release_tau()
        self.assertEqual(len(self.sent), n + 1)                            # nothing more to send

    def test_abort_drops_it(self):
        for _ in range(100):
            _, info = self.env.step({"arm_q_target": np.full(N, 5.0), "hand_q_target": Z, "arm_tau_ff": self.TAU})
            if info["episode_over"]:
                break
        self.assertEqual(info["end_reason"], "aborted")
        np.testing.assert_array_equal(self.sent[-1][1], Z)
        np.testing.assert_array_equal(self.sent[-2][1], self.TAU)


class TestDDSBackendSafety(unittest.TestCase):
    def test_arm_tau_reaches_ctrl_dual_arm(self):
        import threading
        from aaltonomous.interface.backends import DDSBackend

        class Arm:
            def ctrl_dual_arm(self, q, tau): self.q, self.tau = q, tau

        b = DDSBackend.__new__(DDSBackend)
        b.arm, b._lock = Arm(), threading.Lock()
        q, tau = np.linspace(-1.0, 1.0, N), np.linspace(-8.0, 8.0, N)
        b.command(q, Z)
        np.testing.assert_array_equal(b.arm.tau, Z)   # no torque given: 0, as before
        b.command(q, Z, tau)
        np.testing.assert_array_equal(b.arm.q, q)
        np.testing.assert_array_equal(b.arm.tau, tau)

    def test_arm_holds_measured_pose_until_armed(self):
        import numpy as np
        from aaltonomous.interface.backends import make_safe_arm_controller

        class Fake:
            measured = np.full(14, 0.3)

            def __init__(self):
                self.q_target = np.zeros(14)  # what upstream does first

            def get_current_dual_arm_q(self):
                return self.measured

        a = make_safe_arm_controller(Fake)()
        np.testing.assert_allclose(a.q_target, 0.3)   # not zero
        a.q_target = np.zeros(14)                       # ignored while unarmed
        np.testing.assert_allclose(a.q_target, 0.3)
        a.arm()
        np.testing.assert_allclose(a.q_target, 0.3)
        a.q_target = np.full(14, 0.5)
        np.testing.assert_allclose(a.q_target, 0.5)

    def test_debug_mode_failure_refuses(self):
        from aaltonomous.interface.backends import enter_debug_mode

        class Sw:
            def __init__(self, r): self.r = r
            def Enter_Debug_Mode(self): return self.r

        enter_debug_mode(Sw((0, {"name": ""})))
        for bad in ((None, None), (0, {"name": "ai"}), (3, {"name": ""})):
            with self.assertRaises(RuntimeError):
                enter_debug_mode(Sw(bad))


if __name__ == "__main__":
    unittest.main()
