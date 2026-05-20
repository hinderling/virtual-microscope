import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Callable, Optional

from pymmcore_plus import PropertyType
import numpy as np
from numpy.typing import DTypeLike
from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
from pymmcore_plus.experimental.unicore import pymm_property
import virtual_microscope.engine.simulation_bridge as bridge_module

class SimCameraDevice(CameraDevice):
    """
    pymmcore_camera_sim.py

    A virtual camera device for pymmcore that generates images using the microscope_sim.py simulation.
    """
    _exposure: float = 50.0
    _brightness: float = 1.0
    _gain: float = 1.0
    _binning: int = 2 # default 2x2
    _mask: Optional[np.ndarray] = None
    _led_channel: str = None
    _filter_wheel_channel: str = None
    _last_sim_time: float = 0.0
    # Region of interest as (x, y, width, height); None == full sensor.
    _roi: Optional[tuple[int, int, int, int]] = None

    def __init__(self) -> None:

        super().__init__()
        self.bridge = bridge_module.GLOBAL_BRIDGE
        self._mask = None
        # change limits of binning
        self.set_property_limits("Binning", (0, 20))

    def _get_bridge(self):
        """Always use the current global bridge (may be swapped at runtime)."""
        return bridge_module.GLOBAL_BRIDGE

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = float(exposure)
        self.core.events.exposureChanged.emit(self.get_label(), float(exposure))

    def _sensor_hw(self) -> tuple[int, int]:
        """Full (height, width) of the simulated sensor, ignoring any ROI."""
        bridge = self._get_bridge()
        if bridge is None:
            return 512, 512  # Default fallback dimensions
        sim = bridge._sim
        return sim.viewport_height, sim.viewport_width

    def _is_rgb(self) -> bool:
        bridge = self._get_bridge()
        return bool(bridge is not None and getattr(bridge._sim, 'rgb_mode', False))

    def shape(self) -> tuple[int, ...]:
        """Frame shape — ROI-cropped (h, w[, 3]) when an ROI is active."""
        if self._roi is not None:
            _, _, w, h = self._roi
        else:
            h, w = self._sensor_hw()
        return (h, w, 3) if self._is_rgb() else (h, w)

    def dtype(self) -> DTypeLike:
        bridge = self._get_bridge()
        if bridge is not None:
            return getattr(bridge._sim, 'pixel_dtype', np.uint8)
        return np.uint8

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self._mask = mask

    # Minimum interval between frames (seconds) to avoid starving the Qt main
    # thread.  The rendering holds the GIL, so without a gap the UI freezes.
    _MIN_FRAME_INTERVAL: float = 0.08  # ~12 FPS max

    def start_sequence(
        self,
        n: int | None,
        get_buffer: Callable[[Sequence[int], DTypeLike], np.ndarray],
    ) -> Iterator[Mapping]:

        count = 0
        while n is None or count < n:
            t0 = time.perf_counter()
            bridge = self._get_bridge()
            self._mask = bridge.get_slm_mask() # type: ignore
            surf = bridge.snap(brightness=self._brightness, exposure=self._exposure,
                               gain=self._gain)  # type: ignore

            # Digital ROI: crop the rendered full-frame to the requested
            # region. The sim always renders its full viewport; the ROI is
            # purely a readout crop, so this works for every backend.
            if self._roi is not None:
                x, y, w, h = self._roi
                surf = surf[y:y + h, x:x + w]

            # Record simulation time for the SimTime property
            sim = bridge._sim if bridge else None
            if sim is not None:
                self._last_sim_time = getattr(sim, '_time', 0.0)

            buf = get_buffer(surf.shape, self.dtype())
            buf[:] = surf

            yield {
                "data": buf,
                "timestamp": time.time()
                }
            count += 1

            # Throttle: ensure minimum interval so the main thread gets GIL time
            elapsed = time.perf_counter() - t0
            remaining = self._MIN_FRAME_INTERVAL - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # define property brightness
    @pymm_property(
        limits=(0.0,100.0),
        sequence_max_length=100,
        name="brightness",
        property_type=PropertyType.Float
    )
    def brightness(self) -> float:
        """
        Get the brightness of the virtual camera.
        """
        return self._brightness

    # setter methods
    @brightness.setter
    def brightness(self, value: float) -> None:
        """
        Send the values to the virtual hardware to update the brightness.
        """
        self._brightness = value

    @brightness.sequence_loader
    def _load_brightness_sequence(self, sequence: Sequence[float]) -> None:
        pass

    @brightness.sequence_starter
    def _start_brightness_sequence(self) -> None:
        pass

    @pymm_property(
        limits=(0.0, 1e9),
        name="SimTime",
        property_type=PropertyType.Float,
    )
    def sim_time(self) -> float:
        """Simulation time (seconds) at the moment of the last snap.

        Returns the sim's internal clock (``_time`` attribute) captured
        when ``start_sequence`` yields a frame.  For backends without an
        explicit clock, returns 0.0.
        """
        return self._last_sim_time

    @sim_time.setter
    def sim_time(self, value: float) -> None:
        pass  # read-only; ignore writes

    @pymm_property(
        limits=(1.0, 32.0),
        sequence_max_length=100,
        name="Gain",
        property_type=PropertyType.Float,
    )
    def gain(self) -> float:
        """Analog gain multiplier (1.0 = no amplification, 32.0 = max).

        Higher gain amplifies the signal but also amplifies noise,
        reducing the effective dynamic range. Use higher gain for dim
        samples when increasing exposure is not possible (e.g. fast
        timelapse, motion blur concerns, photobleaching-sensitive).
        """
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = max(1.0, min(32.0, value))

    def get_binning(self) -> int:
        """
        Return the current binning of the virtual camera.
        """
        return self._binning

    def set_binning(self, binning: int) -> None:
        """
        Set the current binning of the virtual camera.
        """
        self._binning = binning

    # ROI support ------------------------------------------------------
    #
    # The sim always renders its full viewport; ROI is applied as a
    # digital crop of the finished frame in start_sequence(). This gives
    # every backend real ROI support for free. Implementing set_roi also
    # fixes a teardown bug: pymmcore-plus's MDAEngine snapshots the ROI at
    # sequence start and restores it via setROI() in teardown_sequence().
    # The unicore CameraDevice base raises NotImplementedError from
    # set_roi, and that exception propagates out of MDARunner._finish_run
    # *before* it emits sequenceFinished -- silently stranding listeners
    # (e.g. napari-micromanager's _mda_running flag stays True, freezing
    # the snap preview after a run).

    def get_roi(self) -> tuple[int, int, int, int]:
        """Return the current ROI as (x, y, width, height)."""
        if self._roi is not None:
            return self._roi
        h, w = self._sensor_hw()
        return (0, 0, w, h)

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set a rectangular ROI; frames are cropped to it on readout."""
        x, y, width, height = int(x), int(y), int(width), int(height)
        sh, sw = self._sensor_hw()
        if x == 0 and y == 0 and width == sw and height == sh:
            self._roi = None  # full frame -> no crop
            return
        if (width <= 0 or height <= 0 or x < 0 or y < 0
                or x + width > sw or y + height > sh):
            raise ValueError(
                f"ROI ({x}, {y}, {width}, {height}) is out of bounds for a "
                f"{sw}x{sh} sensor."
            )
        self._roi = (x, y, width, height)

    def clear_roi(self) -> None:
        """Reset the ROI to the full sensor frame."""
        self._roi = None
