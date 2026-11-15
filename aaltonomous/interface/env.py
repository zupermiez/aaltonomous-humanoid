"""AaltoEnv: the frozen challenge interface (spec §4). Same API and keys for mock, sim and real.

    env = AaltoEnv(backend="mock" | "sim" | "real", task="sort_fixed" | "sort_language", seed=0)
    obs = env.reset()                 # real: prints a layout card and waits for staff to confirm
    obs, info = env.step(action)      # action: arm_q_target (14,) or (k,14), hand_q_target likewise, done,
                                      # optional arm_tau_ff (Nm, same rows as arm_q_target or one (14,) row)

Every target passes through SafetyFilter. A (k,14) chunk runs k ticks at 30 Hz and returns the last obs.
step() blocks to keep 30 Hz (mock with realtime=False uses a virtual clock instead, for tests).
Clock: real = wall time; sim = the SIM's clock when the sim publishes one (backend.clock, aalto_task/hooks.py),
so 30 Hz is 30 Hz of physics time even when the sim runs below real time (it does: RTF ~0.5-0.65 with cameras).
obs/info "t", time_limit and the safety filter's watchdog then all count sim seconds.
If the caller is late (policy computing), the robot holds the last target; the filter's watchdog sees the gap.
arm_tau_ff is a feed-forward torque passed to the arm motors next to the target (e.g. gravity compensation). It is
not part of the safety filter; the env bounds it (_arm_tau, ARM_TAU_MAX). Holds keep the last one; safe stop,
abort and episode end (release_tau) drop it to 0. Absent = 0, as before.
"""
from __future__ import annotations

import time

import numpy as np

from .backends import make_backend
from .safety import N, JointSpec, SafetyConfig, SafetyFilter
from .tasks import make_instruction, sample_layout
from ..scene.scene import Scene

HZ = 30.0
# |arm_tau_ff| per joint (Nm), also capped at the URDF effort: shoulder pitch/roll/yaw + elbow 15, wrist roll/pitch/yaw
# 2.5. The largest gravity load anywhere in the joint range (g1_body29_hand14.urdf) is 8.7 / 1.2 Nm.
ARM_TAU_MAX = np.array([15.0, 15.0, 15.0, 15.0, 2.5, 2.5, 2.5] * 2)


def _tau_rows(tau, k):
    """arm_tau_ff of a k-tick chunk -> k rows ((14,) is repeated), or None if its rows don't match the chunk."""
    try:
        tau = np.asarray(tau, dtype=float)
    except (TypeError, ValueError):
        return None
    if tau.ndim == 2 and len(tau) == k:
        return list(tau)
    return [tau] * k if tau.ndim == 1 else None


class AaltoEnv:
    def __init__(self, backend="mock", task="sort_fixed", seed=0, time_limit=120.0, safety: SafetyConfig = None,
                 realtime=None, instruction=None, **backend_kwargs):
        self.task = task
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)
        self.scene = Scene.load()
        self.fixed_instruction = instruction
        self.backend = make_backend(backend, **backend_kwargs)
        self.filter = SafetyFilter(JointSpec(), safety or SafetyConfig(dt=1.0 / HZ))
        self.dt = self.filter.cfg.dt
        self.realtime = (backend != "mock") if realtime is None else realtime
        self.clock = getattr(self.backend, "clock", None) if self.realtime else None   # sim clock or None (wall)
        self.tau_max = np.minimum(ARM_TAU_MAX, self.filter.spec.arm_effort)
        self.tau_cmd = np.zeros(N)  # arm feed-forward torque the backend is holding
        self._t0 = time.monotonic()
        self.t = 0.0
        self.episode_over = True

    # ------------------------------------------------------------- clock
    def _now(self):
        if self.clock is not None:
            return self.clock.now() - self._t0
        return time.monotonic() - self._t0 if self.realtime else self._vt

    def _wait_tick(self):
        if self.clock is not None:
            self._next += self.dt
            self.clock.wait_until(self._t0 + self._next)
        elif self.realtime:
            self._next += self.dt
            time.sleep(max(0.0, self._next - (time.monotonic() - self._t0)))
        else:
            self._vt += self.dt

    # ------------------------------------------------------------- API
    def reset(self):
        if self.fixed_instruction is not None:
            self.instruction, self.mapping = self.fixed_instruction, None
        else:
            self.instruction, self.mapping = make_instruction(self.task, self.rng)
        self.layout = sample_layout(self.rng, **self.scene.layout_kwargs())
        self.backend.reset_scene(self.layout, self.instruction)
        arm_q, _, hand_q = self.backend.joints()
        self._t0, self._vt = (self.clock.now() if self.clock is not None else time.monotonic()), 0.0
        self._next = 0.0
        self.filter.reset(arm_q, hand_q, 0.0)
        self.tau_cmd = np.zeros(N)
        self.backend.command(arm_q, hand_q, self.tau_cmd)
        self.episode_over = False
        self.t = 0.0
        return self._obs()

    def step(self, action):
        if self.episode_over:
            raise RuntimeError("episode is over; call reset()")
        arm = np.asarray(action.get("arm_q_target", np.full(N, np.nan)), float) if isinstance(action, dict) else None
        hand = np.asarray(action.get("hand_q_target", np.full(N, np.nan)), float) if isinstance(action, dict) else None
        tau = action.get("arm_tau_ff") if isinstance(action, dict) else None
        taus = None if tau is None or arm is None or arm.ndim != 2 else _tau_rows(tau, len(arm))
        if (arm is not None and arm.ndim == 2 and hand is not None and hand.ndim == 2 and len(arm) == len(hand)
                and (tau is None or taus is not None)):
            ticks = [{"arm_q_target": a, "hand_q_target": h} for a, h in zip(arm, hand)]
            for tk, tt in zip(ticks, taus or ()):
                tk["arm_tau_ff"] = tt
        else:
            ticks = [action]  # single action, or malformed -> the filter rejects it and holds

        clip_reasons = []
        for a in ticks:
            t = self._now()
            if t - self.filter.last_action_t > self.filter.cfg.hold_after_s:
                self.filter.tick(t)  # late caller: let the watchdog see the gap
            arm_q, _, hand_q = self.backend.joints()
            res = self.filter.filter(a, t, arm_q=arm_q, hand_q=hand_q)
            clip_reasons += res.reasons
            self._send(res, self._arm_tau(a, t, clip_reasons) if res.mode == "ok" else None)
            self._wait_tick()
            self.backend.advance(self.dt)
            if res.mode in ("aborted", "safe_stop"):
                break

        self.t = self._now()
        done_flag = bool(action.get("done", False)) if isinstance(action, dict) else False
        reason = ("aborted" if res.mode == "aborted" else "safe_stop" if res.mode == "safe_stop"
                  else "done" if done_flag else "time_limit" if self.t >= self.time_limit else None)
        self.episode_over = reason is not None
        info = {"t": self.t, "ticks": len(ticks), "safety_mode": res.mode, "clipped": bool(clip_reasons),
                "clip_reasons": sorted(set(clip_reasons)), "clipped_steps": res.clipped_steps,
                "episode_over": self.episode_over, "end_reason": reason, "success": None}
        if self.episode_over:
            info["safety_summary"] = self.filter.summary()
        return self._obs(), info

    def hold(self):
        """Run one tick with no new action (runner calls this while the policy computes)."""
        res = self.filter.tick(self._now())
        self._send(res, self.tau_cmd)  # keeps the last feed-forward while stale; 0 in safe stop / abort
        self._wait_tick()
        self.backend.advance(self.dt)
        return res.mode

    def release_tau(self):
        """Episode end: drop the feed-forward torque to 0, the arms keep holding the last target."""
        if self.tau_cmd.any():
            self.tau_cmd = np.zeros(N)
            self.backend.command(self.filter.arm_cmd.copy(), self.filter.hand_cmd.copy(), self.tau_cmd)

    def close(self):
        self.release_tau()
        self.backend.close()

    # ------------------------------------------------------------- arm feed-forward torque
    def _send(self, res, tau):
        """Command one tick. tau (Nm) only while the filter is ok / stale; 0 (also for None) in safe stop / abort."""
        self.tau_cmd = tau if tau is not None and res.mode in ("ok", "stale") else np.zeros(N)
        self.backend.command(res.arm_q_cmd, res.hand_q_cmd, self.tau_cmd)

    def _arm_tau(self, action, t, reasons):
        """This tick's arm_tau_ff: absent -> 0; wrong shape -> the last one is held (logged, like a bad target, but
        not a clipped step: the filter and its abort count stay q-only); NaN/inf entries -> 0; clamped to tau_max."""
        v = action.get("arm_tau_ff") if isinstance(action, dict) else self.tau_cmd
        if v is None:
            return np.zeros(N)
        try:
            v = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            v = np.asarray(None)
        new = []
        if v.shape != (N,):
            new.append(f"invalid:arm_tau_ff:shape{tuple(v.shape)}")
            v = self.tau_cmd
        else:
            if not np.all(np.isfinite(v)):
                new.append("invalid:arm_tau_ff:nonfinite")
                v = np.where(np.isfinite(v), v, 0.0)
            out = np.clip(v, -self.tau_max, self.tau_max)
            if np.any(out != v):
                new.append("arm_tau_ff_clamp")
            v = out
        if new:
            reasons += new
            self.filter._log(t, new, {})
        return v

    # ------------------------------------------------------------- obs
    def _obs(self):
        arm_q, arm_dq, hand_q = self.backend.joints()
        obs = self.backend.images()
        obs.update(arm_q=arm_q.astype(np.float32), arm_dq=arm_dq.astype(np.float32),
                   hand_q=hand_q.astype(np.float32), instruction=self.instruction, t=float(self.t))
        return obs
