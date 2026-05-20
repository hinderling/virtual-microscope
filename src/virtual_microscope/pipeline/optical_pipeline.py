"""
Optical Pipeline — Standalone image degradation module.

Takes a "perfect" rendered image and applies realistic optical effects:
  - PSF convolution (Gaussian or depth-dependent)
  - Poisson-Gaussian noise (shot + read noise)
  - Dark current
  - Vignetting (radial corner darkening)
  - Illumination unevenness
  - Photobleaching (cumulative, stateful)

Used by all simulation backends. Each backend renders a clean ground-truth
image, then passes it through OpticalPipeline for realistic degradation.

Usage:
    pipeline = OpticalPipeline(
        psf_sigma=1.5,
        noise={"photon_scale": 4.0, "read_std": 3.0},
        vignette=0.15,
    )
    degraded = pipeline.apply(perfect_image)
"""

import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from virtual_microscope.pipeline.debris_overlay import add_debris


class OpticalPipeline:
    """Configurable optical degradation pipeline."""

    def __init__(
        self,
        psf_sigma: float = 0.0,
        noise: dict | None = None,
        vignette: float = 0.0,
        illumination_unevenness: float = 0.0,
        photobleach_rate: float = 0.0,
        focus_drift_rate: float = 0.0,
        debris: dict | None = None,
        camera: dict | None = None,
        chromatic: dict | None = None,
        rng_seed: int | None = None,
    ):
        """
        Args:
            psf_sigma: Gaussian PSF standard deviation in pixels. 0 = no blur.
            noise: Dict with keys:
                - photon_scale: Poisson noise scaling (higher = less noise). 0 = skip.
                - read_std: Gaussian read noise std dev. 0 = skip.
                - dark_current: Mean dark current per pixel per 100ms exposure. 0 = skip.
                - banding_std: Row banding noise std dev (sCMOS artifact). 0 = skip.
                    Typical: 1.0 (subtle), 3.0 (visible stripes).
            vignette: Vignetting strength (0 = none, 0.15 = typical, 0.3 = strong).
            illumination_unevenness: Low-freq intensity variation (0 = none, 0.05 = subtle).
            photobleach_rate: Fractional signal loss per exposure (0 = none, 0.001 = slow).
            focus_drift_rate: Additional PSF sigma per frame (0 = none, 0.2 = moderate drift).
            debris: Dict with debris overlay config (None = no debris).
            camera: Dict with camera sensor model (None = default uint8):
                - bit_depth: Output bit depth (8, 12, or 16). Default 8.
                - gain: Analog gain multiplier. Default 1.0.
                - offset: Black level offset in ADU. Default 0.
                - saturation: Maximum ADU value (auto-set from bit_depth if not given).
                - hot_pixels: Fraction of sensor pixels with elevated dark current
                    (0 = none, 0.0005 = typical, 0.002 = degraded sensor).
                    Hot pixels have 50-255 ADU value, fixed positions across frames.
            chromatic: Dict with chromatic aberration config (None = none):
                - objective_type: 'achromat', 'fluorite', or 'apochromat'. Default 'fluorite'.
                - reference_wavelength: Wavelength with zero shift (nm). Default 550.
                Lateral shift coefficients (fraction of radial distance per 100nm):
                    achromat=0.017, fluorite=0.008, apochromat=0.002
                Axial shift coefficients (µm focal shift per 100nm):
                    achromat=2.0, fluorite=0.8, apochromat=0.3
            rng_seed: Random seed for reproducible noise.
        """
        self.psf_sigma = psf_sigma
        self.focus_drift_rate = focus_drift_rate
        self.noise_cfg = noise or {}
        self.vignette_strength = vignette
        self.illumination_unevenness = illumination_unevenness
        self.photobleach_rate = photobleach_rate
        self.debris_cfg = debris
        self.camera_cfg = camera or {}
        self.chromatic_cfg = chromatic or {}

        self.rng = np.random.default_rng(rng_seed)
        self._rng_seed = rng_seed

        # Hot pixel map (lazy-initialized on first apply, fixed across frames)
        self._hot_pixel_map = None

        # Photobleaching state: cumulative exposure per pixel
        self._bleach_map = None
        self._total_exposure = 0.0

        # Focus drift state: cumulative defocus across frames
        self._focus_drift_sigma = 0.0  # additional blur from drift
        self._n_frames_applied = 0

        # Cached vignette falloff (lazy-init, invalidated on dimension change)
        self._vignette_cache = None  # (h, w, falloff_array)


    # ---- Public API ----

    def apply(self, img: np.ndarray, exposure_ms: float = 50.0,
              wavelength_nm: float = 0) -> np.ndarray:
        """Apply full optical pipeline to image (stateless — no photobleaching).

        Args:
            img: Input image (uint8, 2D or 3D).
            exposure_ms: Exposure time in ms (affects noise level).
            wavelength_nm: Emission wavelength in nm (0 = skip chromatic effects).
                Common values: DAPI=460, GFP=510, RFP=580, Cy5=670.

        Returns:
            Degraded image (uint8, same shape).
        """
        result = img.copy()
        psf_sigma = self._chromatic_psf_sigma(wavelength_nm)
        result = self._apply_psf(result, sigma_override=psf_sigma)
        result = self._apply_chromatic_lateral(result, wavelength_nm)
        result = self._apply_illumination(result)
        result = self._apply_vignette(result)
        result = self._apply_noise(result, exposure_ms)
        result = self._apply_camera(result)
        result = self._apply_debris(result)
        return result

    def apply_with_bleach(self, img: np.ndarray, exposure_ms: float = 50.0,
                          wavelength_nm: float = 0) -> np.ndarray:
        """Apply pipeline with cumulative effects (photobleaching + focus drift).

        Each call may reduce signal (bleaching) and increase blur (focus drift).
        Call reset_bleach() to start fresh.
        """
        result = img.copy()

        # Apply photobleaching before other effects
        if self.photobleach_rate > 0:
            result = self._apply_photobleach(result, exposure_ms)

        # Apply PSF with cumulative focus drift + chromatic scaling
        base_sigma = self._chromatic_psf_sigma(wavelength_nm)
        if self.focus_drift_rate > 0:
            effective_sigma = base_sigma + self._focus_drift_sigma
            result = self._apply_psf(result, sigma_override=effective_sigma)
            self._focus_drift_sigma += self.focus_drift_rate
            self._n_frames_applied += 1
        else:
            result = self._apply_psf(result, sigma_override=base_sigma)

        result = self._apply_chromatic_lateral(result, wavelength_nm)
        result = self._apply_illumination(result)
        result = self._apply_vignette(result)
        result = self._apply_noise(result, exposure_ms)
        result = self._apply_camera(result)
        result = self._apply_debris(result)
        return result

    def reset_bleach(self):
        """Reset photobleaching and focus drift state."""
        self._bleach_map = None
        self._focus_drift_sigma = 0.0
        self._n_frames_applied = 0
        self._total_exposure = 0.0

    def bleach_region(self, cx: float, cy: float, radius: float,
                      strength: float = 5.0, img_shape: tuple = (512, 512)):
        """Instantly bleach a circular region (FRAP-style).

        Uses a Gaussian intensity profile matching a real laser beam.
        The peak bleach is at the center; at ``radius`` the bleach is
        ~60 % of ``strength``.

        Parameters
        ----------
        cx, cy : float — center of bleach region in pixels
        radius : float — 1/e² radius of the bleach spot
        strength : float — peak bleach intensity at center.
            Signal attenuation = exp(-strength), so 2.0 → ~87 % loss.
        img_shape : tuple — (height, width) of the image
        """
        h, w = img_shape[:2]
        if self._bleach_map is None:
            self._bleach_map = np.zeros((h, w), dtype=np.float64)

        yy, xx = np.mgrid[:h, :w]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        sigma = radius * 0.7  # Gaussian sigma so ~60 % at r=radius
        bleach = strength * np.exp(-dist_sq / (2.0 * sigma ** 2))
        self._bleach_map = np.maximum(self._bleach_map, bleach)
        self._bleach_sigma0 = sigma  # store for diffusion scaling

    def apply_recovery(self, rate: float = 0.15, permanent_fraction: float = 0.1):
        """Simulate fluorescence recovery via spatial diffusion.

        Each call convolves the recoverable bleach map with a Gaussian
        kernel, simulating lateral diffusion of unbleached fluorophores
        into the bleached zone.  Edges recover first (closest to unbleached
        neighbours), center recovers last — matching real FRAP kinetics.

        The kernel sigma is scaled to the bleach spot size so that
        ``rate`` has a consistent meaning regardless of resolution:
        bleach-map centre halves in roughly ``1 / rate`` frames.

        Parameters
        ----------
        rate : float — diffusion speed per call (0–1).
            Kernel sigma per step = sqrt(rate) * bleach_sigma0.
        permanent_fraction : float — fraction that never recovers (0-1).
            Represents permanently destroyed fluorophores.
        """
        if self._bleach_map is None:
            return
        from scipy.ndimage import gaussian_filter

        # Store initial bleach for permanent fraction calculation
        if not hasattr(self, '_initial_bleach_map'):
            self._initial_bleach_map = self._bleach_map.copy()

        floor = self._initial_bleach_map * permanent_fraction
        recoverable = np.maximum(0, self._bleach_map - floor)

        # Kernel sigma scaled to bleach spot so t½ ≈ 1/rate frames
        sigma0 = getattr(self, '_bleach_sigma0', 50.0)
        sigma = max(1.0, np.sqrt(rate) * sigma0)
        diffused = gaussian_filter(recoverable, sigma=sigma)
        self._bleach_map = floor + diffused

    # ---- Chromatic aberration helpers ----

    # Lateral shift: fraction of radial distance per 100nm wavelength deviation
    _LATERAL_COEFF = {"achromat": 0.017, "fluorite": 0.008, "apochromat": 0.002}
    # Axial shift: µm focal offset per 100nm wavelength deviation
    _AXIAL_COEFF = {"achromat": 2.0, "fluorite": 0.8, "apochromat": 0.3}
    # DOF per objective (µm) — used to convert axial shift to defocus blur
    _DOF = {10: 6.0, 20: 4.0, 40: 1.5}

    def _chromatic_psf_sigma(self, wavelength_nm: float) -> float:
        """Compute effective PSF sigma accounting for chromatic aberration.

        PSF width scales linearly with wavelength (FWHM = 0.51λ/NA).
        Also adds defocus blur from axial chromatic shift.
        Returns the base psf_sigma if no chromatic config or wavelength=0.
        """
        if not self.chromatic_cfg or wavelength_nm <= 0:
            return self.psf_sigma

        ref_wl = self.chromatic_cfg.get("reference_wavelength", 550)
        obj_type = self.chromatic_cfg.get("objective_type", "fluorite")

        # PSF width scales with wavelength ratio
        wl_ratio = wavelength_nm / ref_wl
        scaled_sigma = self.psf_sigma * wl_ratio

        # Axial chromatic shift → additional defocus blur
        axial_coeff = self._AXIAL_COEFF.get(obj_type, 0.8)
        wl_diff = (wavelength_nm - ref_wl) / 100.0
        axial_shift_um = axial_coeff * wl_diff  # µm

        # Convert axial shift to defocus blur: sigma_defocus ~ |shift| / DOF * blur_scale
        obj_mag = self.chromatic_cfg.get("magnification", 10)
        dof = self._DOF.get(obj_mag, 6.0)
        defocus_sigma = abs(axial_shift_um) / dof * 2.0  # 2px blur per DOF unit

        return scaled_sigma + defocus_sigma

    def _apply_chromatic_lateral(self, img: np.ndarray, wavelength_nm: float) -> np.ndarray:
        """Apply lateral chromatic aberration (radial shift from center).

        Longer wavelengths shift outward, shorter wavelengths shift inward.
        Effect is zero at image center and increases toward corners.
        """
        if not self.chromatic_cfg or wavelength_nm <= 0:
            return img

        ref_wl = self.chromatic_cfg.get("reference_wavelength", 550)
        obj_type = self.chromatic_cfg.get("objective_type", "fluorite")
        lateral_coeff = self._LATERAL_COEFF.get(obj_type, 0.008)

        wl_diff = (wavelength_nm - ref_wl) / 100.0

        # Max shift at corners for this wavelength
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        corner_r = np.sqrt(cx**2 + cy**2)
        max_shift = corner_r * lateral_coeff * abs(wl_diff)

        # Skip if shift is sub-pixel everywhere
        if max_shift < 0.2:
            return img

        # Build coordinate map with radial shift
        y_grid, x_grid = np.mgrid[:h, :w].astype(np.float32)
        dx = x_grid - cx
        dy = y_grid - cy
        r = np.sqrt(dx**2 + dy**2)

        shift_mag = r * lateral_coeff * wl_diff
        # Direction: radial (outward for positive wl_diff)
        safe_r = np.where(r > 0.1, r, 1.0)
        shift_x = shift_mag * (dx / safe_r)
        shift_y = shift_mag * (dy / safe_r)

        # Source coordinates (where to sample from)
        src_y = y_grid - shift_y
        src_x = x_grid - shift_x

        if img.ndim == 2:
            return map_coordinates(
                img.astype(np.float32), [src_y, src_x],
                order=1, mode='nearest'
            ).clip(0, 255).astype(np.uint8)
        else:
            result = np.empty_like(img)
            for c in range(img.shape[2]):
                result[:, :, c] = map_coordinates(
                    img[:, :, c].astype(np.float32), [src_y, src_x],
                    order=1, mode='nearest'
                ).clip(0, 255).astype(np.uint8)
            return result

    # ---- Pipeline stages ----

    def _apply_psf(self, img: np.ndarray, sigma_override: float | None = None) -> np.ndarray:
        """Apply PSF convolution (Gaussian blur)."""
        sigma = sigma_override if sigma_override is not None else self.psf_sigma
        if sigma <= 0:
            return img
        # Kernel size must be odd and large enough for the sigma
        ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd
        ksize = max(3, ksize)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def _apply_noise(self, img: np.ndarray, exposure_ms: float = 50.0) -> np.ndarray:
        """Apply Poisson-Gaussian noise model with optional banding.

        - Poisson (shot) noise: signal-dependent, dominates in bright regions.
            Scales with exposure: longer exposure collects more photons,
            improving SNR (signal grows as N, noise as sqrt(N)).
        - Gaussian (read) noise: signal-independent, dominates in dark regions.
            Fixed regardless of exposure (readout electronics artifact).
        - Dark current: exposure-dependent baseline.
        - Row banding: per-row offset noise (sCMOS artifact).
        """
        photon_scale = self.noise_cfg.get("photon_scale", 0)
        read_std = self.noise_cfg.get("read_std", 0)
        dark_current = self.noise_cfg.get("dark_current", 0)
        banding_std = self.noise_cfg.get("banding_std", 0)

        if photon_scale <= 0 and read_std <= 0 and dark_current <= 0 and banding_std <= 0:
            return img

        f = img.astype(np.float32)

        exposure_ms = float(exposure_ms)

        # Dark current: adds baseline proportional to exposure
        if dark_current > 0:
            dc = dark_current * (exposure_ms / 100.0)
            f += dc

        # Poisson shot noise — scales with exposure.
        # More exposure = more photons collected = higher effective photon_scale
        # = less relative noise.  Reference exposure is 50ms.
        if photon_scale > 0:
            exposure_factor = max(exposure_ms / 50.0, 0.1)
            effective_scale = photon_scale * exposure_factor
            photons = np.maximum(f * effective_scale, 0)
            noisy = self.rng.poisson(photons).astype(np.float32) / effective_scale
        else:
            noisy = f

        # Gaussian read noise (same pattern across all channels)
        if read_std > 0:
            h = noisy.shape[0]
            w = noisy.shape[1]
            read = self.rng.normal(0, read_std, (h, w)).astype(np.float32)
            if noisy.ndim == 3:
                for c in range(noisy.shape[2]):
                    noisy[:, :, c] += read
            else:
                noisy += read

        # Row banding noise (sCMOS artifact: each row has a random offset)
        if banding_std > 0:
            h = noisy.shape[0]
            row_offsets = self.rng.normal(0, banding_std, h).astype(np.float32)
            if noisy.ndim == 3:
                noisy += row_offsets[:, np.newaxis, np.newaxis]
            else:
                noisy += row_offsets[:, np.newaxis]

        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_vignette(self, img: np.ndarray) -> np.ndarray:
        """Apply radial vignetting (darker corners)."""
        if self.vignette_strength <= 0:
            return img

        h, w = img.shape[:2]

        # Reuse cached falloff if dimensions match
        if self._vignette_cache is not None:
            ch, cw, falloff = self._vignette_cache
            if ch == h and cw == w:
                pass  # reuse falloff
            else:
                falloff = self._compute_vignette_falloff(h, w)
                self._vignette_cache = (h, w, falloff)
        else:
            falloff = self._compute_vignette_falloff(h, w)
            self._vignette_cache = (h, w, falloff)

        f = img.astype(np.float32)
        if f.ndim == 3:
            for c in range(f.shape[2]):
                f[:, :, c] *= falloff
        else:
            f *= falloff
        return np.clip(f, 0, 255).astype(np.uint8)

    def _compute_vignette_falloff(self, h: int, w: int) -> np.ndarray:
        """Compute radial vignette falloff array."""
        cy, cx = h / 2, w / 2
        Y, X = np.ogrid[:h, :w]
        max_dist = np.sqrt(cx**2 + cy**2)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2) / max_dist
        return (1.0 - self.vignette_strength * dist**2).astype(np.float32)

    def _apply_illumination(self, img: np.ndarray) -> np.ndarray:
        """Apply illumination unevenness (low-frequency intensity variation)."""
        if self.illumination_unevenness <= 0:
            return img

        h, w = img.shape[:2]
        # Generate smooth random illumination pattern
        # Use a very low-res random field, then upscale
        small_h, small_w = max(2, h // 64), max(2, w // 64)
        pattern = self.rng.normal(1.0, self.illumination_unevenness, (small_h, small_w)).astype(np.float32)
        pattern = cv2.resize(pattern, (w, h), interpolation=cv2.INTER_CUBIC)
        pattern = np.clip(pattern, 0.5, 1.5)

        f = img.astype(np.float32)
        if f.ndim == 3:
            for c in range(f.shape[2]):
                f[:, :, c] *= pattern
        else:
            f *= pattern
        return np.clip(f, 0, 255).astype(np.uint8)

    def _apply_photobleach(self, img: np.ndarray, exposure_ms: float) -> np.ndarray:
        """Apply cumulative photobleaching.

        Signal decreases exponentially with cumulative exposure.
        Only bright regions bleach (proportional to their intensity).
        """
        h, w = img.shape[:2]

        # Initialize bleach map on first call
        if self._bleach_map is None:
            self._bleach_map = np.zeros((h, w), dtype=np.float64)

        # Accumulate exposure (brighter regions bleach faster)
        f = img.astype(np.float64)
        if f.ndim == 3:
            exposure_contribution = f.mean(axis=2) / 255.0
        else:
            exposure_contribution = f / 255.0
        self._bleach_map += exposure_contribution * exposure_ms * self.photobleach_rate
        self._total_exposure += exposure_ms

        # Apply bleaching: signal = original * exp(-bleach_map)
        attenuation = np.exp(-self._bleach_map).astype(np.float32)

        result = img.astype(np.float32)
        if result.ndim == 3:
            for c in range(result.shape[2]):
                result[:, :, c] *= attenuation
        else:
            result *= attenuation
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_camera(self, img: np.ndarray) -> np.ndarray:
        """Apply camera sensor model: gain, offset, bit-depth, and hot pixels.

        Simulates the analog-to-digital conversion of a real camera sensor.
        At bit_depth=8 (default) with no hot pixels, this is a no-op.
        """
        if not self.camera_cfg:
            return img

        gain = self.camera_cfg.get("gain", 1.0)
        offset = self.camera_cfg.get("offset", 0)
        bit_depth = self.camera_cfg.get("bit_depth", 8)
        max_adu = self.camera_cfg.get("saturation", (1 << bit_depth) - 1)
        hot_fraction = self.camera_cfg.get("hot_pixels", 0)

        no_transform = (gain == 1.0 and offset == 0 and bit_depth == 8)

        if no_transform and hot_fraction <= 0:
            return img

        f = img.astype(np.float32)

        # Apply analog gain and offset
        if not no_transform:
            f = f * gain + offset

            # Quantize to bit_depth precision
            if bit_depth > 8:
                scale = max_adu / 255.0
                f = f * scale
                f = np.round(f)
                f = np.clip(f, 0, max_adu)
                f = f * (255.0 / max_adu)
            else:
                f = np.clip(f, 0, 255)

        # Apply hot pixels (persistent bright spots from elevated dark current)
        if hot_fraction > 0:
            f = self._apply_hot_pixels(f, hot_fraction)

        return np.clip(f, 0, 255).astype(np.uint8)

    def _apply_hot_pixels(self, img: np.ndarray, fraction: float) -> np.ndarray:
        """Add hot pixels — fixed bright spots from elevated dark current.

        Hot pixels are at fixed sensor positions (generated once, reused).
        Each hot pixel has a random intensity between 50 and 255 ADU.
        """
        h, w = img.shape[:2]

        # Generate hot pixel map once (fixed across frames)
        if self._hot_pixel_map is None:
            hp_rng = np.random.default_rng(
                self._rng_seed + 5555 if self._rng_seed else 5555)
            n_hot = max(1, int(h * w * fraction))
            # Random positions
            ys = hp_rng.integers(0, h, n_hot)
            xs = hp_rng.integers(0, w, n_hot)
            # Random intensities (50-255)
            vals = hp_rng.uniform(50, 255, n_hot).astype(np.float32)
            # Store as sparse map
            self._hot_pixel_map = np.zeros((h, w), dtype=np.float32)
            self._hot_pixel_map[ys, xs] = vals

        # Apply: take maximum of original and hot pixel value
        if img.ndim == 2:
            return np.maximum(img, self._hot_pixel_map)
        else:
            result = img.copy()
            for c in range(img.shape[2]):
                result[:, :, c] = np.maximum(img[:, :, c], self._hot_pixel_map)
            return result

    def _apply_debris(self, img: np.ndarray) -> np.ndarray:
        """Apply debris overlay if configured."""
        if self.debris_cfg is None:
            return img
        cfg = self.debris_cfg
        return add_debris(
            img,
            density=cfg.get("density", 0.0002),
            bright_fraction=cfg.get("bright_fraction", 0.6),
            size_range=tuple(cfg.get("size_range", [2, 12])),
            defocus_fraction=cfg.get("defocus_fraction", 0.3),
            rng_seed=self._rng_seed + 9000 if self._rng_seed else None,
        )

    # ---- Convenience constructors ----

    @classmethod
    def fluorescence(cls, rng_seed=None, **overrides):
        """Preset for typical fluorescence imaging."""
        defaults = dict(
            psf_sigma=1.0,
            noise={"photon_scale": 3.0, "read_std": 2.5},
            vignette=0.0,
            rng_seed=rng_seed,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def brightfield(cls, rng_seed=None, **overrides):
        """Preset for typical brightfield/phase-contrast imaging."""
        defaults = dict(
            psf_sigma=0.5,
            noise={"photon_scale": 5.0, "read_std": 3.0},
            vignette=0.10,
            rng_seed=rng_seed,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def noisy_fluorescence(cls, rng_seed=None, **overrides):
        """Preset for low-SNR fluorescence (dim markers, short exposure)."""
        defaults = dict(
            psf_sigma=1.5,
            noise={"photon_scale": 1.5, "read_std": 4.0, "dark_current": 0.5},
            vignette=0.05,
            illumination_unevenness=0.03,
            rng_seed=rng_seed,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def timelapse(cls, rng_seed=None, **overrides):
        """Preset for time-lapse with photobleaching."""
        defaults = dict(
            psf_sigma=1.0,
            noise={"photon_scale": 3.0, "read_std": 2.5},
            photobleach_rate=0.0005,
            rng_seed=rng_seed,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def chromatic_fluorescence(cls, objective_type="fluorite",
                                magnification=20, rng_seed=None, **overrides):
        """Preset for fluorescence with chromatic aberration.

        Use wavelength_nm in apply() to get channel-specific effects:
            pipeline.apply(img, wavelength_nm=460)   # DAPI
            pipeline.apply(img, wavelength_nm=510)   # GFP
            pipeline.apply(img, wavelength_nm=580)   # RFP
        """
        defaults = dict(
            psf_sigma=1.0,
            noise={"photon_scale": 3.0, "read_std": 2.5},
            chromatic={
                "objective_type": objective_type,
                "reference_wavelength": 550,
                "magnification": magnification,
            },
            rng_seed=rng_seed,
        )
        defaults.update(overrides)
        return cls(**defaults)
