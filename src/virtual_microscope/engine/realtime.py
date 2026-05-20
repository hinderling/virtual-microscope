"""RealtimeEngine — Background thread that drives autonomous simulation dynamics.

Runs sim.step(dt) on a background thread at a configurable tick rate.
snap_frame() is wrapped with a lock so rendering and stepping never overlap.

Usage:
    engine = RealtimeEngine(sim, time_scale=1.0, tick_hz=10)
    engine.patch_snap_frame()
    engine.start()
    # ... sim.snap_frame() is now thread-safe ...
    engine.stop()
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RealtimeEngine:
    """Background dynamics engine for any sim with a step(dt) method.

    Args:
        sim: Simulation object with step(dt) or step_autonomous(dt) method.
        time_scale: Wall-clock seconds to sim-seconds multiplier.
            1.0 = real-time, 5.0 = 5x faster.
        tick_hz: Background ticks per second (controls step granularity).
        max_dt: Maximum sim-time per tick (prevents large jumps when
            snap_frame blocks the engine thread). Default 1.0s sim-time.
    """

    def __init__(self, sim, time_scale: float = 1.0, tick_hz: int = 10,
                 max_dt: float = 1.0, idle_timeout: float = 0.0,
                 bridge=None):
        self._sim = sim
        self._time_scale = time_scale
        self._tick_hz = tick_hz
        self._max_dt = max_dt
        self._idle_timeout = idle_timeout  # seconds; 0 = disabled
        self._bridge = bridge  # SimulationBridge ref (for SLM processor access)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._paused = False
        self._last_snap_time = 0.0  # set by touch() on each snap

    @property
    def lock(self) -> threading.Lock:
        """Expose lock for external coordination."""
        return self._lock

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float):
        self._time_scale = max(0.0, value)

    def pause(self):
        """Pause dynamics (thread keeps running but skips step calls)."""
        self._paused = True

    def resume(self):
        """Resume dynamics after pause."""
        self._paused = False
        self._last_snap_time = time.monotonic()

    def touch(self):
        """Record activity (called on each snap to reset idle timer).

        Also wakes the engine from idle-paused state so that simulations
        resume automatically when the user starts snapping again.
        """
        self._last_snap_time = time.monotonic()
        if self._paused:
            self._paused = False

    def start(self):
        """Start the background dynamics thread."""
        self._sim.auto_step = False  # engine owns the clock
        # Clear fixed_dt so the engine's wall-clock dt is respected
        if hasattr(self._sim, 'fixed_dt'):
            self._sim.fixed_dt = 0.0
        self._stop.clear()
        self._paused = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="RealtimeEngine")
        self._thread.start()

    def stop(self):
        """Stop the background dynamics thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def patch_snap_frame(self):
        """Wrap sim.snap_frame() with the engine lock for thread safety."""
        original = self._sim.snap_frame
        lock = self._lock

        def locked_snap(*args, **kwargs):
            with lock:
                return original(*args, **kwargs)

        self._sim.snap_frame = locked_snap

    def _run(self):
        """Background loop: call step(dt) at tick_hz."""
        last = time.monotonic()
        # Prefer step_autonomous (skips SLM effects) if available
        step_fn = (self._sim.step_autonomous
                   if hasattr(self._sim, 'step_autonomous')
                   else self._sim.step)

        interval = 1.0 / self._tick_hz
        while not self._stop.wait(interval):
            if self._paused:
                last = time.monotonic()  # reset clock so no jump on resume
                continue
            # Auto-pause after idle timeout (no snaps for N seconds)
            if (self._idle_timeout > 0 and self._last_snap_time > 0
                    and time.monotonic() - self._last_snap_time > self._idle_timeout):
                logger.info("RealtimeEngine idle timeout — auto-pausing")
                self._paused = True
                self._last_snap_time = 0.0  # reset so next snap triggers resume
                last = time.monotonic()
                continue
            now = time.monotonic()
            dt = (now - last) * self._time_scale
            last = now
            # Cap dt to prevent huge jumps when snap_frame blocks the thread
            if self._max_dt > 0:
                dt = min(dt, self._max_dt)
            with self._lock:
                try:
                    # Continuously re-apply SLM mask so stimulation persists
                    # between agent snaps (otherwise friction damps velocity away)
                    if (self._bridge is not None
                            and self._bridge._current_slm_mask is not None
                            and hasattr(self._sim, '_handle_mask')):
                        self._sim._handle_mask(self._bridge._current_slm_mask)
                    step_fn(dt)
                    # Tick SLM processor so stimulation decays
                    if (self._bridge is not None
                            and self._bridge._slm_processor is not None):
                        self._bridge._slm_processor.tick(dt)
                except Exception:
                    logger.exception("RealtimeEngine step error")
