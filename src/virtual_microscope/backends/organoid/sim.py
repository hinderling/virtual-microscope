"""
OrganoidSim — 3D epithelial organoid simulation backend.

Simulates a hollow epithelial organoid (intestinal/cystic) embedded in Matrigel:
  - Irregular shell of polarized columnar epithelial cells (not a perfect sphere)
  - Hollow lumen with shed debris (dark center at equatorial Z-planes)
  - Optional crypt-like buds protruding outward
  - Depth-dependent signal attenuation and out-of-focus haze

Channels:
  - mode 0 (brightfield): Phase contrast — dark ring of cells, lighter lumen
  - mode 1 (nucleus-channel): DAPI — elliptical nuclei with nucleolar voids
  - mode 2 (membrane-channel): E-cadherin — apical bright ring + cell boundaries

Z-stack capability:
  - Equatorial planes: annular ring of cells with dark empty lumen
  - Cap planes (near top/bottom): solid disk of cells (no visible lumen)
  - Transition from disk → ring → disk as Z sweeps through organoid

Usage:
    sim = OrganoidSim(outer_radius=120, wall_thickness=20, n_cells=400, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class OrganoidSim(SimBase):
    """3D hollow organoid simulation with Z-stack support."""

    continuous = True

    def __init__(self, outer_radius=120, wall_thickness=20, n_cells=400,
                 n_buds=0, world_size=512, seed=42,
                 viewport_width=512, viewport_height=512,
                 internal_scale: int = 1,
                 lumen_opacity: float = 0.6):
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=1.0,
            auto_step=False, snaps_per_step=1,
            mode_map={
                ("Electra1(402/454)", "CYAN"): 0,      # DIC brightfield
                ("SCFP2(434/474)", "UV"): 1,            # DAPI nuclei
                ("TagGFP2(483/506)", "GREEN"): 2,       # E-cadherin junctions
            },
        )

        self.outer_radius = outer_radius
        self.inner_radius = outer_radius - wall_thickness
        self.wall_thickness = wall_thickness
        self.n_cells = n_cells
        self.n_buds = n_buds
        self.morphology = "budded" if n_buds > 0 else "cystic"
        # Lumen opacity: 0.0 = clear (young cystic), 1.0 = fully opaque
        # (dead cell accumulation). Default 0.6 = typical mature organoid
        # with visible debris. Real organoid lumens are often dark/opaque.
        self.lumen_opacity = max(0.0, min(1.0, lumen_opacity))
        self._seed = seed

        # Override DOF table entry for 20x
        self._dof_table[20] = 3.0

        # Lumen swelling dynamics (forskolin/CFTR assay)
        self._swell_enabled = False
        self._swell_rate = 0.0       # lumen expansion rate (px/step) when drug on
        self._swell_max = 0.0        # max lumen radius expansion (px)
        self._shrink_rate = 0.0      # recovery rate when drug off
        self._lumen_delta = 0.0      # current expansion from baseline
        self._drug_active = False
        self._initial_inner_radius = self.inner_radius
        self._initial_wall_thickness = wall_thickness

        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.4, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 600),
            1: OpticalPipeline(
                psf_sigma=0.7, noise={"photon_scale": 4.0, "read_std": 2.5},
                vignette=0.12, rng_seed=seed + 601),
            2: OpticalPipeline(
                psf_sigma=0.7, noise={"photon_scale": 4.0, "read_std": 2.5},
                vignette=0.12, rng_seed=seed + 602),
        }

        self._rng = np.random.default_rng(seed)

        self._cx = world_size / 2.0
        self._cy = world_size / 2.0
        self._cz = 0.0

        self._generate_shape_perturbation()
        self._generate_cells()
        self._generate_luminal_debris()
        self._generate_matrigel_texture()

    # ── Shape perturbation (irregular boundary) ──

    def _generate_shape_perturbation(self):
        """Generate angular perturbation for non-spherical organoid shape.

        Uses low-order spherical harmonics (Fourier modes on the sphere)
        to create organic-looking bulges and dents. The perturbation is
        stored as coefficients and evaluated per-angle during rendering.
        """
        rng = self._rng
        # 8 angular modes: low-frequency undulations
        n_modes = 8
        self._shape_amp = rng.uniform(-0.06, 0.06, n_modes)
        self._shape_phase_theta = rng.uniform(0, 2 * np.pi, n_modes)
        self._shape_phase_phi = rng.uniform(0, 2 * np.pi, n_modes)
        self._shape_freq = np.arange(2, 2 + n_modes)

    def _perturbed_radius(self, theta, phi, base_radius):
        """Get perturbed radius at given angular coordinates.

        Returns radius with low-frequency bumps applied.
        theta: azimuthal angle, phi: polar angle from equator
        """
        perturbation = np.zeros_like(theta, dtype=np.float64)
        for i, freq in enumerate(self._shape_freq):
            perturbation += self._shape_amp[i] * (
                np.cos(freq * theta + self._shape_phase_theta[i]) *
                np.cos(freq * phi + self._shape_phase_phi[i])
            )
        return base_radius * (1.0 + perturbation)

    def _perturbed_radii_2d(self, dist_2d, cx, cy, base_outer, base_inner):
        """Compute per-pixel perturbed outer/inner radius for a 2D cross-section.

        Given a 2D distance map and center, compute the angular direction of each
        pixel and return the perturbed radii at that angle.
        """
        ih, iw = dist_2d.shape
        yy, xx = np.ogrid[:ih, :iw]
        # Angle in the XY plane (azimuthal)
        theta_2d = np.arctan2(yy - cy, xx - cx).astype(np.float64)
        # For cross-section, phi depends on current Z
        z = self.focal_plane - self.tissue_z
        R = self.outer_radius
        phi_approx = np.arcsin(np.clip(z / max(R, 1e-6), -1, 1))

        r_out = self._perturbed_radius(theta_2d, phi_approx, base_outer)
        r_in = self._perturbed_radius(theta_2d, phi_approx, base_inner)
        return r_out, r_in

    # ── Cell generation ──

    def _generate_cells(self):
        """Place cells on the shell with basal-biased positioning."""
        rng = self._rng
        R = self.outer_radius
        r = self.inner_radius
        n_main = self.n_cells

        # Position cells with basal bias (nuclei sit toward outer edge)
        cells = []
        while len(cells) < n_main:
            batch = rng.uniform(-R, R, size=(n_main * 4, 3))
            dists = np.linalg.norm(batch, axis=1)
            valid = batch[(dists >= r) & (dists <= R)]
            cells.extend(valid[:n_main - len(cells)])

        cells = np.array(cells[:n_main])

        # Bias toward basal (outer) side: push cells outward
        cell_dirs = cells / np.linalg.norm(cells, axis=1, keepdims=True)
        basal_bias = rng.uniform(0.3, 0.8, n_main)[:, None]
        cells = cell_dirs * (r + (R - r) * basal_bias)

        # Perturb cell positions along the shell surface
        for i in range(n_main):
            theta = np.arctan2(cells[i, 1], cells[i, 0])
            phi = np.arcsin(np.clip(cells[i, 2] / max(np.linalg.norm(cells[i]), 1e-6), -1, 1))
            pr = self._perturbed_radius(np.array([theta]), np.array([phi]),
                                        np.linalg.norm(cells[i]))
            scale = pr[0] / max(np.linalg.norm(cells[i]), 1e-6)
            cells[i] *= scale

        self._cell_x = cells[:, 0] + self._cx
        self._cell_y = cells[:, 1] + self._cy
        self._cell_z = cells[:, 2]
        self._cell_dist = np.linalg.norm(cells, axis=1)

        mid_r = (R + r) / 2.0
        self._is_apical = self._cell_dist < mid_r

        # Cell radii (columnar epithelial)
        self._cell_radius = rng.uniform(5, 9, n_main).astype(np.float32)

        # Nuclear aspect ratio (elongated radially, 1.1-1.6)
        self._nuc_aspect = rng.uniform(1.1, 1.6, n_main).astype(np.float32)

        # Nuclear orientation: angle toward organoid center (radial)
        dx = self._cell_x - self._cx
        dy = self._cell_y - self._cy
        self._nuc_angle = np.arctan2(dy, dx).astype(np.float32)

        # DAPI intensity: 2x range for cell cycle (G1=1x DNA, G2=2x DNA)
        # ~60% G1 (dimmer), ~20% S (mid), ~15% G2 (brighter), ~5% M (very bright)
        phase = rng.random(n_main)
        cycle_mult = np.where(phase < 0.60, rng.uniform(0.55, 0.75, n_main),
                     np.where(phase < 0.80, rng.uniform(0.75, 0.90, n_main),
                     np.where(phase < 0.95, rng.uniform(0.90, 1.10, n_main),
                              rng.uniform(1.10, 1.30, n_main))))
        self._dapi = (155.0 * cycle_mult + rng.normal(0, 8, n_main)).clip(
            70, 235).astype(np.float32)

        # Nucleoli: 1-3 per nucleus (dark voids)
        self._n_nucleoli = rng.choice([1, 1, 2, 2, 2, 3], n_main)
        self._nucleoli_offsets = []
        for i in range(n_main):
            offsets = []
            nr = self._cell_radius[i] * 0.7  # nucleus radius
            for _ in range(self._n_nucleoli[i]):
                ox = rng.uniform(-nr * 0.4, nr * 0.4)
                oy = rng.uniform(-nr * 0.4, nr * 0.4)
                size = rng.uniform(0.15, 0.30) * nr
                offsets.append((ox, oy, size))
            self._nucleoli_offsets.append(offsets)

        # E-cadherin intensity: brighter on basal (outer) surface
        self._ecad = np.full(n_main, 100.0, dtype=np.float32)
        self._ecad[~self._is_apical] += rng.uniform(40, 80, (~self._is_apical).sum())
        self._ecad[self._is_apical] += rng.uniform(15, 40, self._is_apical.sum())
        self._ecad = self._ecad.clip(60, 230).astype(np.float32)

        # Bud cells
        self._bud_info = []
        if self.n_buds > 0:
            self._generate_buds()

    def _generate_buds(self):
        """Generate crypt-like bud protrusions."""
        rng = self._rng
        R = self.outer_radius
        bud_radius = R * 0.3

        for b in range(self.n_buds):
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(-0.4, 0.4)
            bud_dir = np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])
            bud_center = bud_dir * (R + bud_radius * 0.5)

            n_bud_cells = max(15, self.n_cells // 10)
            bud_cells = []
            while len(bud_cells) < n_bud_cells:
                batch = rng.normal(0, bud_radius * 0.5, (n_bud_cells * 4, 3))
                dists = np.linalg.norm(batch, axis=1)
                inner_bud = bud_radius * 0.5
                valid = batch[(dists >= inner_bud) & (dists <= bud_radius)]
                dots = np.sum(valid * bud_dir, axis=1)
                outward = valid[dots > 0]
                bud_cells.extend(outward[:n_bud_cells - len(bud_cells)])

            if len(bud_cells) == 0:
                continue

            bud_cells = np.array(bud_cells[:n_bud_cells])
            bud_x = bud_cells[:, 0] + bud_center[0] + self._cx
            bud_y = bud_cells[:, 1] + bud_center[1] + self._cy
            bud_z = bud_cells[:, 2] + bud_center[2]
            n_added = len(bud_cells)

            self._cell_x = np.append(self._cell_x, bud_x)
            self._cell_y = np.append(self._cell_y, bud_y)
            self._cell_z = np.append(self._cell_z, bud_z)
            self._cell_dist = np.append(
                self._cell_dist,
                np.sqrt((bud_x - self._cx)**2 + (bud_y - self._cy)**2 + bud_z**2)
            )
            self._is_apical = np.append(self._is_apical,
                                        np.zeros(n_added, dtype=bool))
            self._cell_radius = np.append(
                self._cell_radius, rng.uniform(4, 7, n_added).astype(np.float32))
            self._nuc_aspect = np.append(
                self._nuc_aspect, rng.uniform(1.1, 1.5, n_added).astype(np.float32))
            dx = bud_x - self._cx
            dy = bud_y - self._cy
            self._nuc_angle = np.append(
                self._nuc_angle, np.arctan2(dy, dx).astype(np.float32))
            self._dapi = np.append(
                self._dapi, rng.uniform(100, 200, n_added).astype(np.float32))
            self._n_nucleoli = np.append(
                self._n_nucleoli, rng.choice([1, 2, 2], n_added))
            for _ in range(n_added):
                nr = rng.uniform(4, 7) * 0.7
                offs = [(rng.uniform(-nr * 0.4, nr * 0.4),
                         rng.uniform(-nr * 0.4, nr * 0.4),
                         rng.uniform(0.15, 0.30) * nr)
                        for __ in range(rng.choice([1, 2, 2]))]
                self._nucleoli_offsets.append(offs)
            self._ecad = np.append(
                self._ecad, rng.uniform(120, 190, n_added).astype(np.float32))

            self._bud_info.append({
                "center": bud_center.tolist(),
                "radius": bud_radius,
                "n_cells": n_added,
            })
            self.n_cells += n_added

    def _generate_luminal_debris(self):
        """Generate apoptotic debris particles inside the lumen."""
        rng = self._rng
        r = self.inner_radius
        n_debris = max(5, self.n_cells // 15)
        # Place inside inner sphere
        debris = []
        while len(debris) < n_debris:
            batch = rng.uniform(-r * 0.85, r * 0.85, (n_debris * 4, 3))
            dists = np.linalg.norm(batch, axis=1)
            valid = batch[dists < r * 0.85]
            debris.extend(valid[:n_debris - len(debris)])

        debris = np.array(debris[:n_debris])
        self._debris_x = debris[:, 0] + self._cx
        self._debris_y = debris[:, 1] + self._cy
        self._debris_z = debris[:, 2]
        self._debris_size = rng.uniform(1.0, 3.0, n_debris).astype(np.float32)
        # Apoptotic debris: condensed chromatin = BRIGHT small puncta in DAPI
        self._debris_dapi = rng.uniform(180, 240, n_debris).astype(np.float32)
        # In BF: dark granular clumps
        self._debris_bf_intensity = rng.uniform(25, 70, n_debris).astype(np.float32)
        self._n_debris = n_debris

    def _generate_matrigel_texture(self):
        """Pre-generate granular Matrigel background.

        Matrigel is a reconstituted basement membrane extract with pore
        sizes ~200 nm — well below the optical resolution limit.  Under
        standard brightfield it appears as a fine granular/mottled
        texture, NOT as visible fibers (unlike collagen I gels).

        The dome shape creates low-frequency intensity gradients from
        varying optical path length.
        """
        rng = np.random.default_rng(self._seed + 8888)

        tex = np.zeros((self.height, self.width), dtype=np.float32)

        # Low-frequency dome gradient (smooth intensity variation)
        y, x = np.mgrid[:self.height, :self.width]
        cx, cy = self.width / 2, self.height / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        dome = 3.0 * np.exp(-r**2 / (2 * (self.width * 0.6)**2))
        # Add slight asymmetry (dome not perfectly centered on organoid)
        off_x, off_y = rng.uniform(-30, 30, 2)
        r_off = np.sqrt((x - cx - off_x)**2 + (y - cy - off_y)**2)
        dome += 1.5 * np.exp(-r_off**2 / (2 * (self.width * 0.4)**2))
        tex += dome.astype(np.float32)

        # Multi-scale granular (isotropic) texture — no directional fibers
        for sigma, amp in [(8.0, 2.0), (3.0, 2.5), (1.2, 1.5)]:
            layer = rng.normal(0, 1, (self.height, self.width)).astype(np.float32)
            layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma)
            tex += layer * amp

        tex = tex.clip(-10, 10)

        # Scatter a few gel particles / specks
        n_specks = rng.integers(8, 25)
        for _ in range(n_specks):
            sx = rng.integers(0, self.width)
            sy = rng.integers(0, self.height)
            sr = rng.integers(1, 3)
            val = rng.uniform(-6, 6)
            cv2.circle(tex, (int(sx), int(sy)), int(sr), float(val), -1)

        self._matrigel_texture = tex

    # ── Rendering ──

    def _render_for_mode(self, mode):
        """Return full-resolution BGR image for the active channel."""
        if mode == 0:
            gray = self._render_brightfield()
        elif mode == 1:
            gray = self._render_dapi()
        elif mode == 2:
            gray = self._render_ecadherin()
        elif mode in self._extra_channels:
            gray = self._extra_channels[mode].get(
                "image", self._render_brightfield())
        else:
            gray = self._render_brightfield()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── Dynamics ──

    # ── Temperature response ──

    def _temp_factor(self) -> float:
        """Temperature-dependent rate factor for organoid biology.

        Mammalian Q10 ~ 2.0, reference 37°C.
        - 37°C: factor = 1.0 (normal)
        - 4°C: factor ≈ 0.0 (metabolic arrest, Matrigel depolymerizes)
        - 20°C: factor ≈ 0.3 (slow)
        - 42°C+: heat stress reduces activity
        """
        temp = self._get_temperature()
        if temp < 10:
            return 0.05  # near-total arrest
        factor = 2.0 ** ((temp - 37) / 10.0)
        if temp > 42:
            # Heat stress: rapid decline above 42°C
            factor *= max(0.05, 1.0 - (temp - 42) * 0.3)
        return factor

    def _matrigel_integrity(self) -> float:
        """Matrigel gel integrity as a function of temperature.

        Matrigel is liquid below ~10°C and gels at 22-37°C.
        At 4°C it depolymerizes — organoid loses structural support.
        Returns 0.0 (liquid) to 1.0 (fully gelled).
        """
        temp = self._get_temperature()
        if temp <= 4:
            return 0.0
        elif temp < 15:
            return (temp - 4) / 11.0  # linear ramp 4→15°C
        else:
            return 1.0  # fully gelled above 15°C

    def enable_swelling(self, swell_rate=2.0, max_expansion=40.0,
                        shrink_rate=0.5):
        """Enable forskolin/CFTR-driven lumen swelling assay.

        When the Perfusion device is set to "Drug", the lumen radius
        increases by *swell_rate* px/step (up to *max_expansion*).
        When drug is removed ("Off"), the lumen slowly shrinks back
        at *shrink_rate* px/step.

        The outer radius grows slightly too (cells push outward),
        but the wall gets thinner since cells flatten.

        Parameters
        ----------
        swell_rate : float
            Lumen expansion rate in px per step when drug is active.
        max_expansion : float
            Maximum lumen radius increase (px).
        shrink_rate : float
            Recovery rate (px/step) when drug is removed.
        """
        self._swell_enabled = True
        self._swell_rate = swell_rate
        self._swell_max = max_expansion
        self._shrink_rate = shrink_rate

    def step(self, dt=1.0):
        """Advance organoid dynamics by *dt* time units."""
        self._time += dt

        # Z-drift (mechanical — not temperature-dependent)
        self._accumulate_z_drift(dt)

        # Temperature scaling for metabolic processes
        tfactor = self._temp_factor()
        effective_dt = dt * tfactor

        # Lumen swelling (CFTR-driven, temperature-dependent)
        if self._swell_enabled:
            perf = self.state_devices.get("Perfusion", {})
            label = perf.get("label", "Off") if isinstance(perf, dict) else str(perf)
            self._drug_active = label == "Drug"

            if self._drug_active:
                # Lumen expands toward max (rate scales with temperature)
                self._lumen_delta = min(
                    self._lumen_delta + self._swell_rate * effective_dt,
                    self._swell_max)
            else:
                # Lumen shrinks back (also temperature-dependent)
                self._lumen_delta = max(
                    0.0, self._lumen_delta - self._shrink_rate * effective_dt)

            # Update radii using area conservation (cells flatten but
            # total cross-sectional area stays constant):
            #   R'^2 - (r+d)^2 = R0^2 - r0^2  =>  R' = sqrt(R0^2 + 2*r0*d + d^2)
            r0 = self._initial_inner_radius
            R0 = r0 + self._initial_wall_thickness
            d = self._lumen_delta
            self.inner_radius = r0 + d
            self.outer_radius = np.sqrt(R0**2 + 2 * r0 * d + d**2)
            self.wall_thickness = self.outer_radius - self.inner_radius

            # Invalidate cached rendering
            self._dirty = True

    def step_autonomous(self, dt=1.0):
        """Step without SLM effects (for RealtimeEngine)."""
        self.step(dt)

    def get_swelling_state(self):
        """Get current swelling state for ground truth."""
        return {
            "enabled": self._swell_enabled,
            "drug_active": self._drug_active,
            "lumen_delta": round(float(self._lumen_delta), 1),
            "current_inner_radius": round(float(self.inner_radius), 1),
            "current_outer_radius": round(float(self.outer_radius), 1),
            "current_wall_thickness": round(float(self.wall_thickness), 1),
            "expansion_fraction": round(
                float(self._lumen_delta / self._swell_max)
                if self._swell_max > 0 else 0.0, 3),
        }

    def _visible_cells(self):
        """Get per-cell opacity and blur based on DOF distance."""
        dz = np.abs(self._cell_z - (self.focal_plane - self.tissue_z))
        half_dof = self._dof / 2.0

        in_focus = dz <= half_dof
        near_focus = (dz > half_dof) & (dz < half_dof * 4)

        opacity = np.zeros(self.n_cells, dtype=np.float32)
        blur = np.zeros(self.n_cells, dtype=np.float32)

        opacity[in_focus] = 1.0

        if near_focus.any():
            defocus = dz[near_focus] - half_dof
            opacity[near_focus] = np.clip(
                1.0 - defocus / (half_dof * 3), 0.05, 0.8)
            blur[near_focus] = np.clip(defocus * 0.5, 0, 8)

        return opacity, blur

    def _slice_radii(self):
        """Compute base (unperturbed) outer and inner slice radii."""
        z = self.focal_plane - self.tissue_z
        R = self.outer_radius
        r = self.inner_radius

        r_out_sq = R * R - z * z
        r_out = np.sqrt(max(0.0, r_out_sq))

        r_in_sq = r * r - z * z
        r_in = np.sqrt(max(0.0, r_in_sq)) if r_in_sq > 0 else 0.0

        return r_out, r_in

    def _oof_haze(self, channel="dapi"):
        """Compute out-of-focus haze from the entire spherical shell.

        The 3D shell contributes diffuse fluorescence to every Z-plane.
        This creates the soft glow that makes thick specimens look realistic.
        """
        ih, iw = self._ih, self._iw
        haze = np.zeros((ih, iw), dtype=np.float32)

        z = self.focal_plane - self.tissue_z
        R = self.outer_radius
        r = self.inner_radius

        cx_int = self._sf(self._cx)
        cy_int = self._sf(self._cy)
        yy, xx = np.ogrid[:ih, :iw]
        dist = np.sqrt((xx - cx_int)**2 + (yy - cy_int)**2)

        # Integrated fluorescence from shell above and below focal plane
        # Model: each Z-slice of the shell contributes blur proportional
        # to distance from focal plane
        n_oof_slices = 8
        for dz_frac in np.linspace(-1.0, 1.0, n_oof_slices):
            z_oof = z + dz_frac * R * 0.8
            if abs(dz_frac) < 0.15:
                continue  # skip near-focus (rendered sharply already)

            r_out_sq = R * R - z_oof * z_oof
            if r_out_sq <= 0:
                continue
            r_out_oof = np.sqrt(r_out_sq)
            r_in_sq = r * r - z_oof * z_oof
            r_in_oof = np.sqrt(max(0.0, r_in_sq)) if r_in_sq > 0 else 0.0

            r_out_int = self._sf(r_out_oof)
            r_in_int = self._sf(r_in_oof)

            in_wall = (dist <= r_out_int) & (dist >= r_in_int)
            defocus_distance = abs(dz_frac) * R * 0.8
            attenuation = 1.0 / (1.0 + (defocus_distance / 8.0)**2)

            base_val = 6.0 if channel == "dapi" else 4.0
            haze += np.where(in_wall, base_val * attenuation, 0.0)

        # Heavy blur to make it diffuse
        blur_sigma = max(3.0, self._sf(6.0))
        haze = cv2.GaussianBlur(haze, (0, 0), sigmaX=blur_sigma)

        return haze

    def _depth_attenuation(self):
        """Compute per-cell depth attenuation factor.

        Cells further from the coverslip (assumed at z = -R) are dimmer
        due to light scattering through tissue and Matrigel.
        Confocal penetration ~100µm: far side significantly dimmer.
        """
        z = self._cell_z
        R = self.outer_radius
        # Coverslip at bottom: cells at z=-R are brightest
        # Normalize: z=-R → 1.0, z=+R → ~0.3
        depth_from_coverslip = (z + R) / (2 * R)  # 0 at bottom, 1 at top
        attenuation = 1.0 - 0.6 * depth_from_coverslip
        return attenuation.clip(0.2, 1.0).astype(np.float32)

    def _ring_depth_attenuation_2d(self, ih, iw):
        """Compute 2D depth attenuation for the analytical ring.

        The ring is a cross-section through the sphere. At the equatorial
        plane, the top of the ring corresponds to deeper tissue (farther
        from coverslip). Returns a 2D map where top=dim, bottom=bright.
        """
        cy_int = self._sf(self._cy)
        R_int = self._sf(self.outer_radius)
        yy = np.arange(ih, dtype=np.float32)
        # Fraction from center: -1 at bottom, +1 at top
        frac = (cy_int - yy) / max(R_int, 1)
        # Bottom (near coverslip) = bright, top (far) = dim
        # Using exponential falloff for more realistic scattering
        atten_1d = np.clip(1.0 - 0.55 * (frac + 1) / 2, 0.25, 1.0)
        return atten_1d[:, np.newaxis]

    def _bf_oof_shadow(self):
        """Diffuse BF shadow from the 3D shell above/below the focal plane.

        In real brightfield, out-of-focus parts of the organoid contribute
        a faint blurred darkening. This is strongest when the focal plane
        is near the equator (most shell material above and below).
        The shadow is a heavily blurred filled disk at the outer radius.
        """
        ih, iw = self._ih, self._iw
        z_from_center = self.focal_plane - self.tissue_z
        R = self.outer_radius

        # Amount of shell volume above+below this plane
        # Maximum at equator (z=0), zero at caps (|z|=R)
        frac_shell = max(0.0, 1.0 - abs(z_from_center) / max(R, 1))

        shadow = np.zeros((ih, iw), dtype=np.float32)
        if frac_shell < 0.05:
            return shadow

        cx_int = int(self._sf(self._cx))
        cy_int = int(self._sf(self._cy))
        r_int = int(self._sf(R))

        # Draw a filled circle for the projected organoid footprint
        cv2.circle(shadow, (cx_int, cy_int), r_int, 1.0, -1)

        # Moderate blur — tight enough that shadow stays within ~10px of edge
        blur_sigma = max(6.0, self._sf(10.0))
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=blur_sigma)

        # Scale: max ~5 intensity units of darkening at equator
        shadow *= 5.0 * frac_shell
        return shadow

    def _render_brightfield(self):
        """Render brightfield with irregular boundary, wall texture, debris."""
        s = self.internal_scale
        ih, iw = self._ih, self._iw
        img = np.full((ih, iw), 178.0, dtype=np.float32)

        # Matrigel background texture (fades when gel depolymerizes at 4°C)
        gel = self._matrigel_integrity()
        if s > 1:
            tex = cv2.resize(self._matrigel_texture, (iw, ih),
                             interpolation=cv2.INTER_LINEAR)
        else:
            tex = self._matrigel_texture
        img += tex * gel

        # Out-of-focus shadow from 3D shell above/below focal plane
        img -= self._bf_oof_shadow()

        r_out_base, r_in_base = self._slice_radii()

        if r_out_base < 1.0:
            return np.clip(img, 0, 255).astype(np.uint8)

        cx_int = self._sf(self._cx)
        cy_int = self._sf(self._cy)
        yy, xx = np.ogrid[:ih, :iw]
        dist = np.sqrt((xx - cx_int)**2 + (yy - cy_int)**2)

        # Perturbed radii (irregular boundary)
        r_out_map, r_in_map = self._perturbed_radii_2d(
            dist, cx_int, cy_int, r_out_base, r_in_base)
        r_out_int = self._sf(r_out_map)
        r_in_int = self._sf(r_in_map)

        in_wall = (dist <= r_out_int) & (dist >= r_in_int)
        in_lumen = (dist < r_in_int) & (r_in_int > 0)

        if r_in_base > 0:
            # Annular wall: optical density gradient (darker at mid-wall)
            wall_mid = (r_out_int + r_in_int) / 2.0
            wall_half_w = np.maximum((r_out_int - r_in_int) / 2.0, 1.0)
            wall_frac = np.where(in_wall,
                                 1.0 - 0.25 * np.abs(dist - wall_mid) / wall_half_w,
                                 0.0)
            img = np.where(in_wall, 178 - wall_frac * 50, img)

            # Wall granular texture (cell-scale)
            rng_wall = np.random.default_rng(self._seed + 3333)
            wall_noise = rng_wall.normal(0, 1, (self.height, self.width)).astype(np.float32)
            wall_noise = cv2.GaussianBlur(wall_noise, (0, 0), sigmaX=1.5)
            if s > 1:
                wall_noise = cv2.resize(wall_noise, (iw, ih),
                                        interpolation=cv2.INTER_LINEAR)
            img = np.where(in_wall, img + wall_noise * 5.0, img)

            # Shade-off artifact: interior of large phase objects trends
            # toward background intensity. The lumen appears brighter than
            # expected because diffracted light fills in.
            shade_off = np.where(in_lumen,
                                 1.0 - 0.3 * (1.0 - dist / np.maximum(r_in_int, 1)),
                                 1.0)
            shade_off = np.clip(shade_off, 0.7, 1.0)

            # Lumen rendering — opacity controls appearance:
            # opacity=0: clear lumen (young cystic) — slightly brighter than wall
            # opacity=1: fully opaque (mature) — dark from dead cell debris
            rng_lumen = np.random.default_rng(self._seed + 4444)
            lumen_noise = rng_lumen.normal(0, 1, (self.height, self.width)).astype(np.float32)
            lumen_noise = cv2.GaussianBlur(lumen_noise, (0, 0), sigmaX=2.0)
            if s > 1:
                lumen_noise = cv2.resize(lumen_noise, (iw, ih),
                                         interpolation=cv2.INTER_LINEAR)
            # Clear lumen base: brighter center (shade-off fills in)
            clear_base = 185 + (192 - 185) * shade_off
            # Opaque lumen: dark from accumulated dead cells (granular texture)
            opaque_base = 105 + 15 * shade_off  # dark center, slightly lighter edges
            lumen_base = clear_base * (1 - self.lumen_opacity) + opaque_base * self.lumen_opacity
            noise_amp = 3.0 + 5.0 * self.lumen_opacity  # more texture when opaque
            img = np.where(in_lumen, lumen_base + lumen_noise * noise_amp, img)
        else:
            # Solid disk at cap: shade-off makes center lighter
            inside = dist <= r_out_int
            norm_d = dist / np.maximum(r_out_int, 1)
            # Optical path thickness through sphere
            thickness = np.where(inside, np.sqrt(np.clip(1 - norm_d**2, 0, 1)), 0)
            # Shade-off: thick center trends toward background
            shade_off_cap = 1.0 - 0.25 * thickness
            effective_dark = thickness * 42 * shade_off_cap
            img = np.where(inside, 178 - effective_dark, img)

        # Apical brush border (microvilli haze) — thin dark+textured band
        # along the inner wall edge (lumen-facing). Intestinal epithelium
        # has dense microvilli creating a fuzzy absorptive appearance.
        if r_in_base > 2:
            brush_w = max(1.5, self._sf(2.5))  # ~2.5 world px band
            apical_zone = ((dist >= r_in_int) &
                           (dist < r_in_int + brush_w) & in_wall)
            # High-frequency noise for fuzzy texture
            rng_brush = np.random.default_rng(self._seed + 8888)
            brush_noise = rng_brush.normal(0, 1, (ih, iw)).astype(np.float32)
            brush_noise = cv2.GaussianBlur(brush_noise, (0, 0), sigmaX=0.8)
            # Slightly darker baseline + texture
            img = np.where(apical_zone, img - 8 + brush_noise * 3.0, img)

        # Phase contrast halos (outer boundary)
        halo_w = max(1.5, 3.0 * s)
        ring_bright = ((dist >= r_out_int - s * 0.5) &
                       (dist < r_out_int + halo_w))
        img = np.where(ring_bright, 218, img)
        dark_w = max(1.5, 3.0 * s)
        ring_dark = ((dist >= r_out_int - dark_w - s) &
                     (dist < r_out_int - s * 0.5))
        img = np.where(ring_dark, img - 14, img)

        # Inner edge halo (lumen boundary)
        if r_in_base > 2:
            inner_bright = ((dist >= r_in_int - s * 0.5) &
                            (dist < r_in_int + halo_w * 0.5))
            img = np.where(inner_bright & in_wall, 202, img)
            inner_dark = ((dist >= r_in_int - halo_w * 0.5 - s) &
                          (dist < r_in_int - s * 0.5))
            img = np.where(inner_dark & in_lumen, img - 8, img)

        # Bud cross-sections
        z = self.focal_plane - self.tissue_z
        for bud in self._bud_info:
            bc = bud["center"]
            br = bud["radius"]
            bud_slice_sq = br**2 - (z - bc[2])**2
            if bud_slice_sq > 0:
                bud_slice_r = np.sqrt(bud_slice_sq)
                bcx = self._sf(bc[0] + self._cx)
                bcy = self._sf(bc[1] + self._cy)
                bud_dist = np.sqrt((xx - bcx)**2 + (yy - bcy)**2)
                bud_outer = bud_dist < self._sf(bud_slice_r)
                bud_inner = bud_dist < self._sf(bud_slice_r * 0.5)
                bud_wall = bud_outer & ~bud_inner
                img = np.where(bud_wall, 138, img)
                img = np.where(bud_inner, 188, img)
                # Bud halo
                bud_halo = ((bud_dist >= self._sf(bud_slice_r) - s) &
                            (bud_dist < self._sf(bud_slice_r) + halo_w * 0.7))
                img = np.where(bud_halo, 210, img)

        # Visible cell features (subtle, in-focus cells in wall)
        opacity, blur = self._visible_cells()
        for i in range(self.n_cells):
            if opacity[i] < 0.15:
                continue
            cx_i = self._s(self._cell_x[i])
            cy_i = self._s(self._cell_y[i])
            r = max(1, self._s(self._cell_radius[i]))
            alpha = opacity[i] * 0.25

            y0 = max(0, cy_i - r)
            y1 = min(ih, cy_i + r + 1)
            x0 = max(0, cx_i - r)
            x1 = min(iw, cx_i + r + 1)
            if y1 > y0 and x1 > x0:
                region = img[y0:y1, x0:x1]
                cmask = np.zeros_like(region)
                cv2.circle(cmask, (cx_i - x0, cy_i - y0), r, 1.0, -1)
                # Cell body slightly darker with bright outline
                inner_val = 148.0 if not self._is_apical[i] else 155.0
                img[y0:y1, x0:x1] = (
                    region * (1 - cmask * alpha) + inner_val * cmask * alpha)

        # Luminal debris in BF (dark clumps)
        if r_in_base > 2:
            for j in range(self._n_debris):
                dz_d = abs(self._debris_z[j] - z)
                if dz_d > self._dof * 3:
                    continue
                dcx = self._s(self._debris_x[j])
                dcy = self._s(self._debris_y[j])
                dr = max(1, self._s(self._debris_size[j]))
                defocus_factor = 1.0 / (1.0 + (dz_d / self._dof)**2)
                blur_r = max(dr, int(dr + dz_d * 0.3))
                cv2.circle(img, (dcx, dcy), blur_r,
                           float(160 - self._debris_bf_intensity[j] * defocus_factor),
                           -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_dapi(self):
        """Render DAPI: analytical nuclear ring + OOF haze + debris.

        Real organoid sections show a dense "necklace" of nuclei at the
        equatorial plane because columnar epithelial cells span the full
        wall thickness. We render this analytically: place nuclei at
        regular intervals along the ring circumference at the basal side.

        Depth-asymmetric: near side (coverslip, bottom) bright, far side dim.
        Apoptotic debris in lumen: bright condensed puncta.
        """
        ih, iw = self._ih, self._iw
        img = np.zeros((ih, iw), dtype=np.float32)

        # Out-of-focus haze from entire shell, depth-attenuated
        haze = self._oof_haze("dapi")
        depth_2d = self._ring_depth_attenuation_2d(ih, iw)
        img += haze * depth_2d

        r_out_base, r_in_base = self._slice_radii()

        if r_out_base > 1.0:
            self._render_dapi_ring(img, r_out_base, r_in_base, depth_2d)

        # Luminal debris: BRIGHT condensed apoptotic nuclei (small + intense)
        z = self.focal_plane - self.tissue_z
        for j in range(self._n_debris):
            dz_d = abs(self._debris_z[j] - z)
            if dz_d > self._dof * 3:
                continue
            dcx = self._s(self._debris_x[j])
            dcy = self._s(self._debris_y[j])
            dr = max(1, self._s(self._debris_size[j] * 0.5))
            defocus = 1.0 / (1.0 + (dz_d / self._dof)**2)
            val_d = self._debris_dapi[j] * defocus
            blur_r = max(dr, int(dr + dz_d * 0.3))
            cv2.circle(img, (dcx, dcy), blur_r, float(min(val_d, 250)), -1)

        img += 2.0  # autofluorescence

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_dapi_ring(self, img, r_out_base, r_in_base, depth_2d):
        """Draw dense analytical nuclear ring at current Z-section.

        In real organoids, columnar epithelial cells span the full wall
        thickness (~20-30µm). At any Z-plane cutting through the ring,
        ALL cells around the circumference contribute their nucleus.
        This gives the characteristic dense "necklace of bright dots"
        pattern seen in confocal sections.

        Nuclear features:
        - Basal positioning (~75% from lumen toward outer edge)
        - Elliptical shape oriented radially
        - Cell-cycle-dependent intensity (G1 dim, G2 bright)
        - Heterochromatin foci and nucleolar voids
        - Depth attenuation (far side dimmer)
        """
        ih, iw = self._ih, self._iw
        cx_int = self._sf(self._cx)
        cy_int = self._sf(self._cy)

        # Nuclear position: 75% from inner to outer edge (basal bias)
        if r_in_base > 2:
            nuc_ring_r = r_in_base + (r_out_base - r_in_base) * 0.75
        else:
            # Cap section: nuclei spread across the solid disk
            nuc_ring_r = r_out_base * 0.65

        # Number of nuclei = circumference / cell width (~7µm)
        circumference = 2 * np.pi * nuc_ring_r
        cell_spacing = 7.0
        n_nuclei = max(8, int(circumference / cell_spacing))

        # Seeded RNG for reproducible nuclei per Z-plane
        z_hash = int(abs(self.focal_plane - self.tissue_z) * 100) % 10000
        rng = np.random.default_rng(self._seed + 7777 + z_hash)

        # Regular angular spacing with jitter
        angles = np.linspace(0, 2 * np.pi, n_nuclei, endpoint=False)
        jitter = rng.uniform(-0.3, 0.3, n_nuclei) * (2 * np.pi / n_nuclei)
        angles += jitter

        # Radial scatter (nuclei don't sit on a perfect ring)
        wall_w = r_out_base - r_in_base if r_in_base > 2 else r_out_base * 0.4
        radial_scatter = rng.normal(0, wall_w * 0.08, n_nuclei)

        # Cell cycle: 60% G1 (dim), 20% S (mid), 15% G2 (bright), 5% M (very bright)
        phase = rng.random(n_nuclei)
        intensity = np.where(
            phase < 0.60, rng.uniform(100, 140, n_nuclei),
            np.where(phase < 0.80, rng.uniform(140, 170, n_nuclei),
                     np.where(phase < 0.95, rng.uniform(170, 200, n_nuclei),
                              rng.uniform(200, 235, n_nuclei))))

        # Perturbed radii for irregular boundary
        r_out_map, r_in_map = self._perturbed_radii_2d(
            np.zeros((1, 1)), cx_int, cy_int, r_out_base, r_in_base)

        for k in range(n_nuclei):
            # Check perturbed radius at this angle
            theta = angles[k]
            r_local = nuc_ring_r + radial_scatter[k]
            r_out_local = self._perturbed_radius(
                np.array([theta]),
                np.arcsin(np.clip(
                    (self.focal_plane - self.tissue_z) / max(self.outer_radius, 1e-6),
                    -1, 1)),
                r_out_base)[0]
            r_in_local = self._perturbed_radius(
                np.array([theta]),
                np.arcsin(np.clip(
                    (self.focal_plane - self.tissue_z) / max(self.outer_radius, 1e-6),
                    -1, 1)),
                r_in_base)[0] if r_in_base > 2 else 0.0

            # Skip if nucleus would be outside the wall at this angle
            if r_in_base > 2 and (r_local < r_in_local or r_local > r_out_local):
                continue
            elif r_in_base <= 2 and r_local > r_out_local:
                continue

            x = self._cx + r_local * np.cos(theta)
            y = self._cy + r_local * np.sin(theta)
            px = int(round(self._sf(x)))
            py = int(round(self._sf(y)))

            if px < 0 or px >= iw or py < 0 or py >= ih:
                continue

            # Nucleus size: ~4µm diameter, elongated radially (aspect 1.2-1.6)
            nuc_size = rng.uniform(3.0, 5.0)
            aspect = rng.uniform(1.2, 1.6)
            ax_major = max(2, self._s(nuc_size))
            ax_minor = max(1, int(ax_major / aspect))
            angle_deg = float(np.degrees(theta))

            # Depth attenuation: top of image (far from coverslip) dimmer
            y_frac = (py / ih)  # 0=top, 1=bottom
            depth_factor = max(0.3, 0.45 + 0.55 * y_frac)

            val = float(intensity[k] * depth_factor)

            # Draw filled ellipse (nucleus body)
            cv2.ellipse(img, (px, py), (ax_major, ax_minor),
                        angle_deg, 0, 360, val, -1)

            # Heterochromatin foci (bright speckles)
            n_speckles = rng.integers(2, 6)
            for _ in range(n_speckles):
                sx = px + int(rng.uniform(-ax_major * 0.4, ax_major * 0.4))
                sy = py + int(rng.uniform(-ax_minor * 0.4, ax_minor * 0.4))
                sr = max(1, self._s(rng.uniform(0.4, 1.2)))
                cv2.circle(img, (sx, sy), sr,
                           float(min(val * rng.uniform(1.05, 1.25), 250)), -1)

            # Nucleolar void (1-2 dark spots)
            n_nuc = rng.choice([1, 1, 2])
            cos_a, sin_a = np.cos(theta), np.sin(theta)
            for _ in range(n_nuc):
                nox = rng.uniform(-nuc_size * 0.3, nuc_size * 0.3)
                noy = rng.uniform(-nuc_size * 0.3, nuc_size * 0.3)
                nx = px + self._s(nox * cos_a - noy * sin_a)
                ny = py + self._s(nox * sin_a + noy * cos_a)
                nr = max(1, self._s(rng.uniform(0.6, 1.2)))
                cv2.circle(img, (nx, ny), nr, float(val * 0.25), -1)

    def _render_ecadherin(self):
        """Render E-cadherin: apical bright line, lateral junctions, cyto fill.

        Real E-cadherin: bright lateral (cell-cell) membranes, apico-lateral
        enrichment (adherens junction belt), absent at free apical/basal surfaces.
        Depth-asymmetric: near side (coverslip) bright, far side dim.
        """
        ih, iw = self._ih, self._iw
        img = np.zeros((ih, iw), dtype=np.float32)

        # OOF haze
        img += self._oof_haze("ecad")

        r_out_base, r_in_base = self._slice_radii()
        depth_2d = self._ring_depth_attenuation_2d(ih, iw)

        # Apical bright line (inner surface of the wall)
        if r_in_base > 2:
            cx_int = self._sf(self._cx)
            cy_int = self._sf(self._cy)
            yy, xx = np.ogrid[:ih, :iw]
            dist = np.sqrt((xx - cx_int)**2 + (yy - cy_int)**2)

            r_out_map, r_in_map = self._perturbed_radii_2d(
                dist, cx_int, cy_int, r_out_base, r_in_base)
            r_out_int = self._sf(r_out_map)
            r_in_int = self._sf(r_in_map)

            in_wall = (dist <= r_out_int) & (dist >= r_in_int)

            # Cytoplasmic E-cadherin fill (faint, depth-attenuated)
            img = np.where(in_wall, np.maximum(img, 12.0 * depth_2d), img)

            # Apico-lateral enrichment: bright band at inner edge of wall
            # (adherens junction belt — brightest E-cadherin signal)
            apical_w = max(1.5, self._sf(2.0))
            apical_ring = (dist >= r_in_int) & (dist < r_in_int + apical_w) & in_wall
            img = np.where(apical_ring, np.maximum(img, 100.0 * depth_2d), img)

            # Lateral membrane zone: radial lines between cells (see cell loop)

            # Basal surface: E-cadherin absent (integrin domain, not cadherin)
            # Only very faint basal outline
            basal_w = max(1.0, self._sf(1.0))
            basal_ring = (dist >= r_out_int - basal_w) & (dist <= r_out_int)
            img = np.where(basal_ring, np.maximum(img, 20.0 * depth_2d), img)

        # Analytical radial dashes (lateral membranes in cross-section)
        # In real organoid equatorial sections, E-cadherin shows "ladder rungs"
        # — short radial line segments crossing the wall at each cell boundary.
        if r_out_base > 2:
            self._render_ecad_radial_dashes(
                img, r_out_base, r_in_base, depth_2d)

        img += 1.5  # autofluorescence

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_ecad_radial_dashes(self, img, r_out_base, r_in_base, depth_2d):
        """Draw radial dash lines at each cell boundary in the wall cross-section.

        Real E-cadherin in equatorial organoid sections shows "ladder rungs":
        short bright radial lines spanning from near the apical surface to the
        basal surface at each cell-cell boundary. Spacing matches the nuclear
        ring (~7µm intervals). Each dash has intensity variation and slight
        angular jitter for a natural look.
        """
        ih, iw = self._ih, self._iw
        cx_int = self._sf(self._cx)
        cy_int = self._sf(self._cy)

        # Same spacing as nuclei
        cell_spacing = 7.0
        if r_in_base > 2:
            ring_r = (r_out_base + r_in_base) / 2.0
        else:
            ring_r = r_out_base * 0.7
        circumference = 2 * np.pi * ring_r
        n_dashes = max(8, int(circumference / cell_spacing))

        z_hash = int(abs(self.focal_plane - self.tissue_z) * 100) % 10000
        rng = np.random.default_rng(self._seed + 9999 + z_hash)

        angles = np.linspace(0, 2 * np.pi, n_dashes, endpoint=False)
        jitter = rng.uniform(-0.2, 0.2, n_dashes) * (2 * np.pi / n_dashes)
        angles += jitter

        # Radial dash extends from inner to outer wall edge
        inner_r = r_in_base if r_in_base > 2 else r_out_base * 0.3
        outer_r = r_out_base

        for k in range(n_dashes):
            theta = angles[k]
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Inner point (near apical surface)
            ix = int(round(cx_int + self._sf(inner_r + 1.0) * cos_t))
            iy = int(round(cy_int + self._sf(inner_r + 1.0) * sin_t))
            # Outer point (near basal surface)
            ox = int(round(cx_int + self._sf(outer_r - 1.0) * cos_t))
            oy = int(round(cy_int + self._sf(outer_r - 1.0) * sin_t))

            if (0 <= ix < iw and 0 <= iy < ih and
                    0 <= ox < iw and 0 <= oy < ih):
                # Depth attenuation based on Y position
                mid_y = (iy + oy) / 2
                y_frac = mid_y / ih
                depth_factor = max(0.3, 0.45 + 0.55 * y_frac)

                val = float(rng.uniform(50, 90) * depth_factor)
                thickness = max(1, self._s(0.8))
                cv2.line(img, (ix, iy), (ox, oy), val, thickness,
                         cv2.LINE_AA)

    # ── Extra channels ──

    def add_channel(self, mode_id, name, led, filt, render_fn):
        self._extra_channels[mode_id] = {
            "name": name, "led": led, "filter": filt,
            "render_fn": render_fn, "image": None,
        }

    # ── Ground truth ──

    def _analytical_nuclei_count(self, r_out, r_in):
        """Count nuclei matching the analytical ring rendering.

        Must stay in sync with ``_render_dapi_ring`` which places nuclei
        at ~7 µm intervals along the circumference of the nuclear ring.
        """
        if r_out < 1.0:
            return 0
        if r_in > 2:
            nuc_ring_r = r_in + (r_out - r_in) * 0.75
        else:
            nuc_ring_r = r_out * 0.65
        circumference = 2 * np.pi * nuc_ring_r
        return max(8, int(circumference / 7.0)) if r_out > 1.0 else 0

    def get_ground_truth(self):
        r_out, r_in = self._slice_radii()
        z_from_center = self.focal_plane - self.tissue_z
        n_visible = self._analytical_nuclei_count(r_out, r_in)

        return {
            "outer_radius": round(float(self.outer_radius), 1),
            "inner_radius": round(float(self.inner_radius), 1),
            "wall_thickness": round(float(self.wall_thickness), 1),
            "n_cells": self.n_cells,
            "morphology": self.morphology,
            "n_buds": self.n_buds,
            "lumen_diameter": round(float(self.inner_radius * 2), 1),
            "current_z": round(float(z_from_center), 1),
            "slice_outer_radius": round(float(r_out), 1),
            "slice_inner_radius": round(float(r_in), 1),
            "lumen_visible": bool(r_in > 0),
            "n_visible_cells": n_visible,
            "temperature_C": round(self._get_temperature(), 1),
            "matrigel_integrity": round(self._matrigel_integrity(), 2),
        }

    def get_z_profile(self, n_slices=20):
        z_values = np.linspace(-self.outer_radius, self.outer_radius, n_slices)
        profile = []
        R = self.outer_radius
        r = self.inner_radius
        for z in z_values:
            r_out_sq = R * R - z * z
            r_out = np.sqrt(max(0.0, r_out_sq))
            r_in_sq = r * r - z * z
            r_in = np.sqrt(max(0.0, r_in_sq)) if r_in_sq > 0 else 0.0
            n_vis = self._analytical_nuclei_count(r_out, r_in)

            profile.append({
                "z": round(float(z), 1),
                "outer_radius": round(float(r_out), 1),
                "inner_radius": round(float(r_in), 1),
                "lumen_visible": bool(r_in > 0),
                "n_visible": n_vis,
            })
        return profile

    def get_morphology(self):
        return self.morphology
