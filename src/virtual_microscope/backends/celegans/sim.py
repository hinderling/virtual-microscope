"""
CelegansSim — C. elegans crawling organism backend.

A deformable worm body with sinusoidal locomotion that moves through the
world. Forces stage tracking: the worm leaves the FOV if the agent doesn't
follow it.

Usage via SimulationBridge:
    sim = CelegansSim(world_size=2048, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class CelegansSim(SimBase):
    """C. elegans (nematode) simulation compatible with SimulationBridge.

    Renders a single worm on a large world (default 2048x2048).
    The worm crawls using sinusoidal body waves and moves through the world.
    The agent must track it by moving the stage.

    Channels:
      - mode 0: Brightfield — worm body visible as dark/transparent shape
      - mode 1: GFP-pharynx — bright spot in head region (nucleus channel)
      - mode 2: mCherry-body — dim body wall fluorescence (membrane channel)
    """

    continuous = True
    _default_temperature = 20.0

    def __init__(
        self,
        world_size: int = 2048,
        viewport_width: int = 512,
        viewport_height: int = 512,
        worm_length: float = 250.0,
        worm_width: float = 18.0,
        n_segments: int = 50,
        speed: float = 40.0,
        wave_frequency: float = 0.6,
        wave_amplitude: float = 35.0,
        wavelength: float = 200.0,
        seed: int = 42,
        fixed_dt: float = 0.0,
    ):
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=1, fixed_dt=fixed_dt,
            auto_step=True, snaps_per_step=1,
            mode_map={
                ("Electra1(402/454)", "CYAN"): 0,      # DIC brightfield
                ("TagGFP2(483/506)", "GREEN"): 1,      # GFP-pharynx
                ("mScarlet3(569/582)", "ORANGE"): 2,   # mCherry-body
            },
        )

        self._noise_rng = np.random.default_rng(seed + 9999)

        # Worm body parameters
        self.worm_length = worm_length
        self.worm_width = worm_width
        self.n_segments = n_segments
        self.speed = speed
        self.wave_frequency = wave_frequency
        self.wave_amplitude = wave_amplitude
        self.wavelength = wavelength

        # Optical pipelines per channel
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.8, noise={"photon_scale": 8.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 100),
            1: OpticalPipeline(
                psf_sigma=1.2, noise={"photon_scale": 4.0, "read_std": 3.0},
                vignette=0.12, rng_seed=seed + 101),
            2: OpticalPipeline(
                psf_sigma=1.2, noise={"photon_scale": 4.0, "read_std": 3.0},
                vignette=0.12, rng_seed=seed + 102),
        }

        # Worm state: position and heading
        self._head_pos = np.array([
            world_size / 2.0,
            world_size / 2.0,
        ], dtype=np.float64)
        self._heading = self.rng.uniform(0, 2 * np.pi)
        self._phase = 0.0

        # Turn dynamics — worm occasionally changes direction
        self._turn_rate = 0.0  # current angular velocity (rad/s)
        self._turn_timer = 0.0
        self._next_turn_time = self.rng.uniform(3.0, 8.0)
        self._next_turn_rate = 0.0

        # Precompute width profile (tapered at ends)
        self._width_profile = self._compute_width_profile()

        # Compute initial body
        self._body_points = self._compute_body()

        # Background texture (agar surface)
        self._bg_texture = self._generate_background()

        # Drug response state
        self._drug_active = False
        self._drug_name = None
        self._drug_effect = 0.0  # 0=none, 1=full effect
        self._drug_washing_out = False
        # Store original locomotion parameters for recovery
        self._base_speed = self.speed
        self._base_wave_amplitude = self.wave_amplitude
        self._base_wave_frequency = self.wave_frequency

        # Drug profiles: each defines how locomotion parameters change
        self._drug_profiles = {
            "levamisole": {
                "speed_mult": 0.0,       # complete paralysis
                "amplitude_mult": 0.15,  # hypercontraction → rigid posture
                "frequency_mult": 0.0,   # wave stops
                "onset_rate": 0.20,      # fast onset (cholinergic agonist)
                "washout_rate": 0.08,
            },
            "aldicarb": {
                "speed_mult": 0.05,      # near-complete paralysis
                "amplitude_mult": 0.10,  # very reduced movement
                "frequency_mult": 0.05,
                "onset_rate": 0.05,      # slow onset (AChE inhibitor)
                "washout_rate": 0.03,
            },
            "ivermectin": {
                "speed_mult": 0.0,       # flaccid paralysis
                "amplitude_mult": 0.0,   # no body bends
                "frequency_mult": 0.0,
                "onset_rate": 0.10,      # moderate onset
                "washout_rate": 0.02,    # very slow washout (irreversible-ish)
            },
        }

        # Optical pipeline (optional)
        self.pipeline = None
        self.live_pipeline = False

    def _compute_width_profile(self) -> np.ndarray:
        """Worm width along body: tapered at head and tail."""
        t = np.linspace(0, 1, self.n_segments)
        # Smooth taper: wide in middle, narrow at ends
        # Head taper (first 15%): smooth rise
        # Tail taper (last 20%): smooth decline
        profile = np.ones(self.n_segments)
        head_len = int(self.n_segments * 0.15)
        tail_len = int(self.n_segments * 0.20)
        # Head: smooth rise using sine
        for i in range(head_len):
            profile[i] = 0.3 + 0.7 * np.sin(np.pi / 2 * i / head_len)
        # Tail: smooth decline
        for i in range(tail_len):
            idx = self.n_segments - tail_len + i
            profile[idx] = 0.3 + 0.7 * np.cos(np.pi / 2 * i / tail_len)
        return profile * self.worm_width

    def _compute_body(self) -> np.ndarray:
        """Compute worm body centerline as array of (x, y) points.

        The worm is defined by its head position and heading.
        A sinusoidal wave propagates from head to tail.
        """
        points = np.zeros((self.n_segments, 2))
        seg_length = self.worm_length / self.n_segments

        # Head is at index 0
        points[0] = self._head_pos

        # Build body backwards from head
        for i in range(1, self.n_segments):
            # Distance along body from head
            s = i * seg_length
            # Sinusoidal lateral displacement
            wave = self.wave_amplitude * np.sin(
                2 * np.pi * s / self.wavelength - self._phase
            )
            # Local heading adjusted by wave
            local_heading = self._heading + np.pi  # body extends behind head
            # Perpendicular direction for wave displacement
            perp = local_heading + np.pi / 2

            points[i, 0] = (points[0, 0]
                            + s * np.cos(local_heading)
                            + wave * np.cos(perp))
            points[i, 1] = (points[0, 1]
                            + s * np.sin(local_heading)
                            + wave * np.sin(perp))

        return points

    def _temp_speed_factor(self) -> float:
        """Temperature-dependent locomotion speed factor.

        C. elegans is ectothermic, optimal at ~20°C.
        Q10 ~ 1.5 for locomotion below optimum.
        Cold arrest below 8°C, heat stress above 25°C, near-lethal above 37°C.
        """
        temp = self._get_temperature()
        if temp < 8:
            return 0.05  # cold arrest
        if temp <= 20:
            return 1.5 ** ((temp - 20) / 10.0)  # Q10 = 1.5
        if temp <= 25:
            # Slight decline above optimum
            return 1.0 - 0.03 * (temp - 20)  # 25°C → 0.85
        # Progressive heat stress
        # 30°C → 0.45, 37°C → 0.12, 42°C → 0.05
        return max(0.05, 0.85 * (0.5 ** ((temp - 25) / 5.0)))

    def step(self, dt: float = 1.0):
        """Advance the worm by one time step."""
        if self.fixed_dt > 0:
            dt = self.fixed_dt

        # Temperature scaling for locomotion
        temp_factor = self._temp_speed_factor()

        # Z-drift accumulation
        self._accumulate_z_drift(dt)

        self._time += dt

        # Update turn dynamics
        self._turn_timer += dt
        if self._turn_timer >= self._next_turn_time:
            # Start a new turn
            self._turn_rate = self.rng.uniform(-0.8, 0.8)
            self._next_turn_time = self.rng.uniform(3.0, 8.0)
            self._turn_timer = 0.0

        # Smooth turn decay
        self._heading += self._turn_rate * dt
        self._turn_rate *= 0.98  # gentle decay

        # Advance phase (wave propagation) — temperature scales wave speed
        self._phase += 2 * np.pi * self.wave_frequency * dt * temp_factor

        # Move head forward — temperature scales crawling speed
        effective_speed = self.speed * temp_factor
        self._head_pos[0] += effective_speed * np.cos(self._heading) * dt
        self._head_pos[1] += effective_speed * np.sin(self._heading) * dt

        # Boundary reflection (soft — reverse heading near edges)
        margin = 200
        if self._head_pos[0] < margin:
            self._heading = np.pi - self._heading + self.rng.uniform(-0.3, 0.3)
            self._head_pos[0] = margin
        elif self._head_pos[0] > self.width - margin:
            self._heading = np.pi - self._heading + self.rng.uniform(-0.3, 0.3)
            self._head_pos[0] = self.width - margin
        if self._head_pos[1] < margin:
            self._heading = -self._heading + self.rng.uniform(-0.3, 0.3)
            self._head_pos[1] = margin
        elif self._head_pos[1] > self.height - margin:
            self._heading = -self._heading + self.rng.uniform(-0.3, 0.3)
            self._head_pos[1] = self.height - margin

        # Update drug effects on locomotion
        self._update_drug_effect()

        # Recompute body
        self._body_points = self._compute_body()

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — same as step (no SLM effects)."""
        self.step(dt)

    def apply_drug(self, name):
        """Apply a drug that modulates locomotion.

        Available drugs:
          - 'levamisole': cholinergic agonist → hypercontraction then paralysis (fast)
          - 'aldicarb': AChE inhibitor → progressive paralysis (slow)
          - 'ivermectin': glutamate-gated Cl channel → flaccid paralysis (irreversible)
        """
        name = name.lower()
        if name not in self._drug_profiles:
            raise ValueError(f"Unknown drug: {name}. Available: {list(self._drug_profiles)}")
        self._drug_active = True
        self._drug_name = name
        self._drug_effect = 0.0
        self._drug_washing_out = False

    def remove_drug(self):
        """Remove drug — begins washout phase."""
        if self._drug_active:
            self._drug_active = False
            self._drug_washing_out = True

    def _update_drug_effect(self):
        """Update drug effect level and apply to locomotion parameters."""
        if not self._drug_active and not self._drug_washing_out:
            return

        profile = self._drug_profiles.get(self._drug_name, {})

        # Update effect level
        if self._drug_washing_out:
            rate = profile.get("washout_rate", 0.05)
            self._drug_effect = max(0.0, self._drug_effect - rate)
            if self._drug_effect <= 0.01:
                self._drug_effect = 0.0
                self._drug_washing_out = False
                self._drug_name = None
        else:
            rate = profile.get("onset_rate", 0.10)
            self._drug_effect = min(1.0, self._drug_effect + rate)

        # Apply effect to locomotion parameters
        eff = self._drug_effect
        speed_mult = 1.0 + (profile.get("speed_mult", 1.0) - 1.0) * eff
        amp_mult = 1.0 + (profile.get("amplitude_mult", 1.0) - 1.0) * eff
        freq_mult = 1.0 + (profile.get("frequency_mult", 1.0) - 1.0) * eff

        self.speed = self._base_speed * max(0.0, speed_mult)
        self.wave_amplitude = self._base_wave_amplitude * max(0.0, amp_mult)
        self.wave_frequency = self._base_wave_frequency * max(0.0, freq_mult)

    def _generate_background(self) -> np.ndarray:
        """Generate agar-like background texture for brightfield."""
        bg = np.full((self.height, self.width), 200, dtype=np.uint8)
        # Add subtle texture
        noise = self._noise_rng.normal(0, 3, (self.height, self.width))
        bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Slight radial vignetting
        cy, cx = self.height / 2, self.width / 2
        Y, X = np.ogrid[:self.height, :self.width]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - 0.08 * (r / r_max) ** 2
        bg = (bg.astype(np.float32) * vignette).clip(0, 255).astype(np.uint8)
        return bg

    # ── Rendering ──

    def _render_bf_full(self) -> np.ndarray:
        """Render brightfield with phase-contrast-like optics.

        Real C. elegans under phase contrast:
        - Body is semi-transparent with visible optical path difference
        - Prominent bright halo around the entire body (phase ring artifact)
        - Dark body outline inside the halo
        - Intestine is the most visible internal feature (dark, granular)
        - Pharynx terminal bulb is very prominent (muscular, opaque)
        - Cuticle creates thin dark edge
        """
        img = cv2.cvtColor(self._bg_texture.copy(), cv2.COLOR_GRAY2BGR)

        body = self._body_points
        n = len(body)
        if n < 2:
            return img

        # Compute normals and outline at each segment
        left_pts = []
        right_pts = []
        normals = []
        for i in range(n):
            if i == 0:
                tangent = body[1] - body[0]
            elif i == n - 1:
                tangent = body[-1] - body[-2]
            else:
                tangent = body[i + 1] - body[i - 1]
            length = np.linalg.norm(tangent)
            if length < 1e-6:
                tangent = np.array([1.0, 0.0])
            else:
                tangent = tangent / length
            normal = np.array([-tangent[1], tangent[0]])
            normals.append(normal)

            w = self._width_profile[i] / 2.0
            left_pts.append(body[i] + normal * w)
            right_pts.append(body[i] - normal * w)

        outline = np.array(left_pts + right_pts[::-1], dtype=np.int32)
        left_arr = np.array(left_pts, dtype=np.int32)
        right_arr = np.array(right_pts, dtype=np.int32)

        # --- Phase contrast bright halo (drawn BEFORE body) ---
        # Outer bright halo: 3px wide around the body outline
        halo_left = []
        halo_right = []
        for i in range(n):
            halo_w = 4.0  # halo width in px
            halo_left.append(body[i] + normals[i] * (self._width_profile[i] / 2.0 + halo_w))
            halo_right.append(body[i] - normals[i] * (self._width_profile[i] / 2.0 + halo_w))
        halo_outline = np.array(halo_left + halo_right[::-1], dtype=np.int32)

        # Draw bright halo band between halo_outline and body outline
        halo_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(halo_mask, [halo_outline], 255)
        cv2.fillPoly(halo_mask, [outline], 0)  # subtract body interior
        halo_region = halo_mask > 0
        img[halo_region] = np.clip(
            img[halo_region].astype(np.int16) + 45, 0, 255
        ).astype(np.uint8)

        # --- Semi-transparent body fill ---
        # Worm body: slightly darker than background (phase object)
        overlay = img.copy()
        cv2.fillPoly(overlay, [outline], (150, 145, 140))
        alpha = 0.30  # increased from 0.15 for better visibility
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Internal structures (pharynx, gut, gonad)
        self._draw_internal_structures_bf(img, body)

        # --- Dark cuticle edge ---
        # The cuticle creates a thin dark boundary around the body
        cv2.polylines(img, [outline], isClosed=True, color=(120, 115, 105),
                       thickness=2, lineType=cv2.LINE_AA)

        return img

    def _draw_internal_structures_bf(self, img: np.ndarray,
                                      body: np.ndarray):
        """Draw pharynx, gut, gonad, and other internal structures in BF.

        These are the main visible features in a transparent worm:
        - Pharynx (head, ~15%): cylindrical with terminal bulb (grinder)
        - Intestine (20-75%): dark granular, birefringent gut granules
        - Gonad (35-65%): faint, extends along one side
        """
        n = len(body)
        rng = self._noise_rng

        # ── Pharynx with terminal bulb ──
        pharynx_end = max(2, int(n * 0.15))
        # Pharynx cylinder: muscular, more opaque than body
        for i in range(1, pharynx_end):
            pt = body[i]
            x, y = int(pt[0]), int(pt[1])
            r = int(self._width_profile[i] * 0.35)
            cv2.circle(img, (x, y), r, (125, 120, 110), -1, cv2.LINE_AA)

        # Terminal bulb (grinder) — the most prominent head structure
        bulb_idx = max(1, pharynx_end - 2)
        bpt = body[bulb_idx]
        bx, by = int(bpt[0]), int(bpt[1])
        bulb_r = int(self.worm_width * 0.45)
        # Bulb body (dark muscular ring)
        cv2.circle(img, (bx, by), bulb_r, (100, 95, 85), -1, cv2.LINE_AA)
        # Grinder plate (very dark crescent inside bulb)
        cv2.circle(img, (bx, by), int(bulb_r * 0.45), (70, 65, 55), -1,
                   cv2.LINE_AA)
        # Bright lumen through center
        cv2.circle(img, (bx, by), int(bulb_r * 0.15), (160, 155, 145), -1)

        # ── Intestine: dark granular (most visible internal feature) ──
        gut_start = int(n * 0.2)
        gut_end = int(n * 0.75)
        # Gut lumen: continuous dark band down the center
        gut_pts = body[gut_start:gut_end:1].astype(np.int32)
        gut_w = int(self.worm_width * 0.35)
        if len(gut_pts) > 1:
            cv2.polylines(img, [gut_pts], isClosed=False,
                          color=(110, 105, 95), thickness=gut_w,
                          lineType=cv2.LINE_AA)
        # Gut granules: birefringent dark spots (prominent under phase contrast)
        for i in range(gut_start, gut_end):
            pt = body[i]
            x, y = int(pt[0]), int(pt[1])
            n_granules = rng.integers(2, 5)
            for _ in range(n_granules):
                gx = x + rng.integers(-int(self._width_profile[i] * 0.25),
                                       int(self._width_profile[i] * 0.25) + 1)
                gy = y + rng.integers(-3, 4)
                gr = rng.integers(1, 3)
                shade = int(80 + rng.integers(-15, 10))
                cv2.circle(img, (gx, gy), gr, (shade, shade - 5, shade - 15),
                           -1, cv2.LINE_AA)

        # ── Gonad: faint curved structure along one side ──
        gonad_start = int(n * 0.35)
        gonad_end = int(n * 0.60)
        gonad_pts = []
        for i in range(gonad_start, gonad_end, 2):
            pt = body[i]
            w = self._width_profile[i] * 0.3
            if i < n - 1:
                tang = body[min(i + 1, n - 1)] - body[max(i - 1, 0)]
                tang_len = np.linalg.norm(tang)
                if tang_len > 1e-6:
                    norm = np.array([-tang[1], tang[0]]) / tang_len
                    gpt = pt + norm * w
                    gonad_pts.append(gpt.astype(np.int32))
        if len(gonad_pts) > 2:
            gp_arr = np.array(gonad_pts, dtype=np.int32)
            cv2.polylines(img, [gp_arr], isClosed=False,
                          color=(155, 150, 140), thickness=3,
                          lineType=cv2.LINE_AA)

    def _render_nuc_full(self) -> np.ndarray:
        """Render GFP-pharynx (nucleus channel): bright pharynx region."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        body = self._body_points
        n = len(body)
        if n < 2:
            return img

        # Pharynx: bright GFP signal in head region
        pharynx_end = max(2, int(n * 0.15))
        for i in range(1, pharynx_end):
            pt = body[i]
            x, y = int(pt[0]), int(pt[1])
            # Intensity decreases from head
            val = int(255 * (1.0 - 0.4 * i / pharynx_end))
            r = int(self._width_profile[i] * 0.4)
            cv2.circle(img, (x, y), r, (val, val, val), -1, cv2.LINE_AA)

        # Gonad: dimmer signal in body center
        gonad_start = int(n * 0.35)
        gonad_end = int(n * 0.55)
        for i in range(gonad_start, gonad_end, 2):
            pt = body[i]
            x, y = int(pt[0]), int(pt[1])
            r = int(self._width_profile[i] * 0.2)
            cv2.circle(img, (x, y), r, (100, 100, 100), -1, cv2.LINE_AA)

        # Apply Gaussian blur for fluorescence spread
        img = cv2.GaussianBlur(img, (5, 5), 1.5)

        return img

    def _render_mem_full(self) -> np.ndarray:
        """Render mCherry-body wall (membrane channel): outline of worm."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        body = self._body_points
        n = len(body)
        if n < 2:
            return img

        # Compute body outline
        left_pts = []
        right_pts = []
        for i in range(n):
            if i == 0:
                tangent = body[1] - body[0]
            elif i == n - 1:
                tangent = body[-1] - body[-2]
            else:
                tangent = body[i + 1] - body[i - 1]
            length = np.linalg.norm(tangent)
            if length < 1e-6:
                tangent = np.array([1.0, 0.0])
            else:
                tangent = tangent / length
            normal = np.array([-tangent[1], tangent[0]])
            w = self._width_profile[i] / 2.0
            left_pts.append(body[i] + normal * w)
            right_pts.append(body[i] - normal * w)

        outline = np.array(left_pts + right_pts[::-1], dtype=np.int32)

        # Draw body wall fluorescence (mCherry on body wall muscles)
        # Bright outline, dim interior
        cv2.polylines(img, [outline], isClosed=True, color=(200, 200, 200),
                       thickness=3, lineType=cv2.LINE_AA)

        # Dim fill for body wall muscle fluorescence
        overlay = img.copy()
        cv2.fillPoly(overlay, [outline], (60, 60, 60))
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # Seam cells: brighter spots along body midline
        for i in range(5, n - 3, 6):
            pt = body[i]
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(img, (x, y), 3, (180, 180, 180), -1, cv2.LINE_AA)

        # Apply blur for fluorescence PSF
        img = cv2.GaussianBlur(img, (3, 3), 1.0)

        return img

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply Poisson + Gaussian noise to fluorescence image."""
        f = img.astype(np.float32)
        # Poisson shot noise
        if f.max() > 0:
            photon_scale = 50.0
            photons = f / 255.0 * photon_scale
            photons = np.clip(photons, 0, None)
            noisy = self._noise_rng.poisson(photons).astype(np.float32)
            f = noisy / photon_scale * 255.0
        # Read noise
        f += self._noise_rng.normal(0, 3, f.shape).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)

    # ── Template-method hooks ──

    def _render_for_mode(self, mode):
        if mode == 0:
            return self._render_bf_full()
        elif mode == 1:
            return self._render_nuc_full()
        elif mode == 2:
            return self._render_mem_full()
        elif mode in self._extra_channels:
            return self._extra_channels[mode]["image"]
        return self._render_bf_full()

    def _get_pad_bg(self) -> int:
        """Background value for out-of-bounds padding.

        Brightfield uses 200 (agar background), fluorescence uses 0.
        """
        return 200 if self.mode == 0 else 0

    def _apply_defocus(self, img: np.ndarray) -> np.ndarray:
        """Apply defocus blur based on focal plane distance."""
        dz = abs(self.focal_plane - self.tissue_z)
        half_dof = self._dof / 2.0
        if dz <= half_dof:
            return img
        defocus_um = dz - half_dof
        blur_scale = self._blur_scale_table.get(self.current_objectiv, 0.5)
        sigma = defocus_um * blur_scale
        if sigma < 0.3:
            return img
        sigma = min(sigma, 30.0)
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        opacity = max(0.2, 1.0 / (1.0 + 0.3 * (defocus_um / max(0.5, self._dof))))
        if opacity < 0.99:
            bg_val = 200 if self.mode == 0 else 0
            bg = np.full_like(blurred, bg_val)
            blurred = cv2.addWeighted(blurred, opacity, bg, 1.0 - opacity, 0)
        return blurred

    def reset(self, seed: int = None):
        """Reset worm to initial state. Optionally with a new seed."""
        super().reset(seed)
        self._head_pos[:] = [self.width / 2.0, self.height / 2.0]
        self._heading = self.rng.uniform(0, 2 * np.pi)
        self._phase = 0.0
        self._time = 0.0
        self._turn_rate = 0.0
        self._turn_timer = 0.0
        self._next_turn_time = self.rng.uniform(3.0, 8.0)
        self._snap_count = 0
        self._body_points = self._compute_body()

    # ── Ground truth for grading ──

    def get_head_position(self) -> tuple:
        """Return worm head position in world coordinates."""
        return (float(self._head_pos[0]), float(self._head_pos[1]))

    def get_body_center(self) -> tuple:
        """Return center of mass of worm body."""
        cx = float(np.mean(self._body_points[:, 0]))
        cy = float(np.mean(self._body_points[:, 1]))
        return (cx, cy)

    def get_heading(self) -> float:
        """Return current heading in degrees."""
        return float(np.degrees(self._heading) % 360)

    def get_body_length(self) -> float:
        """Return actual body length (arc length along centerline)."""
        diffs = np.diff(self._body_points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def get_ground_truth(self) -> dict:
        """Return full ground truth for grading."""
        hx, hy = self.get_head_position()
        cx, cy = self.get_body_center()
        gt = {
            "head_position": [hx, hy],
            "body_center": [cx, cy],
            "heading_deg": self.get_heading(),
            "body_length": self.get_body_length(),
            "time": self._time,
            "n_segments": self.n_segments,
            "speed": round(self.speed, 2),
            "wave_amplitude": round(self.wave_amplitude, 2),
            "wave_frequency": round(self.wave_frequency, 3),
        }

        if self._drug_active or self._drug_washing_out:
            gt["drug"] = {
                "name": self._drug_name,
                "effect": round(self._drug_effect, 3),
                "active": self._drug_active,
                "washing_out": self._drug_washing_out,
            }

        return gt
