"""
HistologySim — H&E stained tissue section simulation.

Simulates hematoxylin & eosin stained tissue sections with:
  - Nuclei (hematoxylin = dark blue-purple -> dark gray in BF)
  - Cytoplasm/stroma (eosin = pink -> medium-light gray in BF)
  - Tissue architectures: epithelium, glandular, connective, adipose

This is a FIXED, STAINED preparation — no dynamics, no fluorescence.
Brightfield-only. The challenge is tissue architecture recognition
and nuclear counting/measurement.

High-resolution rendering: all channels rendered at internal_scale x world_size
(default 4x). Higher objectives crop from the high-res buffer:
  - 10x: 4:1 bin (overview)
  - 20x: 2:1 bin (cell-level)
  - 40x: native internal resolution (nuclear detail, chromatin texture)

Grayscale mapping (brightfield transmitted light):
  - Background/lumen: ~230-245 (bright, minimal stain)
  - Cytoplasm/ECM: ~180-200 (eosin, medium-light)
  - Nuclei: ~40-80 (hematoxylin, dark)
  - Blood/RBCs in vessels: ~120-140 (medium)
  - Adipocyte interior: ~220-240 (clear, bright)

Usage via SimulationBridge:
    sim = HistologySim(tissue_type="glandular", seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


# -- H&E Color Palette (RGB, float32) --
# Hematoxylin: stains nucleic acids blue-purple
HE_NUCLEUS_DARK = np.array([70, 40, 120], dtype=np.float32)
HE_NUCLEUS_LIGHT = np.array([105, 70, 150], dtype=np.float32)
HE_NUCLEOLUS = np.array([45, 25, 90], dtype=np.float32)
HE_MITOTIC = np.array([50, 28, 85], dtype=np.float32)

# Eosin: stains proteins pink/salmon
HE_STROMA = np.array([225, 170, 180], dtype=np.float32)
HE_CYTOPLASM = np.array([232, 185, 192], dtype=np.float32)
HE_COLLAGEN = np.array([210, 142, 162], dtype=np.float32)
HE_BASEMENT_MEMBRANE = np.array([195, 140, 158], dtype=np.float32)

# Structures
HE_GLASS = np.array([245, 240, 240], dtype=np.float32)
HE_LUMEN = np.array([250, 247, 247], dtype=np.float32)
HE_RBC = np.array([215, 85, 85], dtype=np.float32)  # vivid orange-red
HE_VESSEL_WALL = np.array([188, 140, 158], dtype=np.float32)
HE_ADIPOCYTE = np.array([248, 245, 245], dtype=np.float32)
HE_ADIPOCYTE_WALL = np.array([212, 162, 175], dtype=np.float32)

# Necrosis: coagulative necrosis is *intensely* eosinophilic (bright pink-red)
HE_NECROSIS = np.array([230, 140, 140], dtype=np.float32)  # bright pink-red
HE_KARYORRHEXIS = np.array([90, 55, 105], dtype=np.float32)  # dark nuclear debris


# -- Tissue type configurations --
TISSUE_TYPES = {
    "epithelium": {
        "description": "Simple columnar epithelium on basement membrane",
        "stroma_intensity": 195,
        "stroma_noise": 8,
    },
    "glandular": {
        "description": "Glandular tissue with circular lumens lined by epithelial cells",
        "stroma_intensity": 190,
        "stroma_noise": 10,
    },
    "connective": {
        "description": "Loose connective tissue with scattered fibroblast nuclei",
        "stroma_intensity": 200,
        "stroma_noise": 12,
    },
    "adipose": {
        "description": "Adipose tissue — large clear cells with peripheral nuclei",
        "stroma_intensity": 205,
        "stroma_noise": 6,
    },
}


class HistologySim(SimBase):
    """H&E stained tissue section simulation with high-resolution rendering.

    Channels:
      - mode 0: Brightfield (H&E stain) — the primary channel
      - mode 1: "nucleus channel" — hematoxylin only (nuclei isolated)
      - mode 2: "membrane channel" — eosin only (stroma/cytoplasm)

    Parameters
    ----------
    tissue_type : str
        One of "epithelium", "glandular", "connective", "adipose", or "mixed".
    world_size : int
        Size of the tissue section in world units.
    n_nuclei : int
        Approximate number of nuclei.
    internal_scale : int
        Rendering resolution multiplier (default 4). Internal images are
        world_size * internal_scale pixels. At 40x, output is native resolution.
    seed : int
        Random seed for reproducibility.
    """

    # Grade-dependent nuclear properties
    # (r_range, intensity_range, mitotic_frac, elongation_range, n_nucleoli_pct)
    GRADE_PROFILES = {
        0: {  # Normal
            "r_range": (2.5, 5.5),
            "intensity_range": (40, 85),
            "mitotic_frac": (0.02, 0.05),
            "elong_range": (1.0, 2.5),
            "nucleolus_pct": 0.35,
            "gland_noise_amp": 0.12,
            "gland_spacing": 15,
            "necrosis_frac": 0.0,
            "lymphocyte_frac": 0.0,
        },
        1: {  # Low-grade dysplasia
            "r_range": (2.0, 7.0),
            "intensity_range": (30, 80),
            "mitotic_frac": (0.05, 0.10),
            "elong_range": (1.0, 3.0),
            "nucleolus_pct": 0.50,
            "gland_noise_amp": 0.20,
            "gland_spacing": 10,
            "necrosis_frac": 0.0,
            "lymphocyte_frac": 0.10,
        },
        2: {  # High-grade
            "r_range": (2.0, 9.0),
            "intensity_range": (22, 70),
            "mitotic_frac": (0.10, 0.18),
            "elong_range": (1.0, 4.0),
            "nucleolus_pct": 0.70,
            "gland_noise_amp": 0.35,
            "gland_spacing": 5,
            "necrosis_frac": 0.08,
            "lymphocyte_frac": 0.30,
        },
        3: {  # Undifferentiated
            "r_range": (2.5, 12.0),
            "intensity_range": (18, 55),
            "mitotic_frac": (0.15, 0.25),
            "elong_range": (1.0, 5.0),
            "nucleolus_pct": 0.85,
            "gland_noise_amp": 0.50,
            "gland_spacing": 0,
            "necrosis_frac": 0.15,
            "lymphocyte_frac": 0.50,
        },
    }

    def __init__(
        self,
        tissue_type: str = "glandular",
        world_size: int = 512,
        n_nuclei: int = 200,
        viewport_width: int = 512,
        viewport_height: int = 512,
        seed: int = 42,
        internal_scale: int = 4,
        grade: int = 0,
    ):
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale,
            mode_map={
                ("Electra1(402/454)", "CYAN"): 0,      # H&E composite (brightfield)
                ("SCFP2(434/474)", "UV"): 1,           # hematoxylin
                ("obeYFP(514/528)", "GREEN"): 2,       # eosin
            },
        )

        self.tissue_type = tissue_type
        self.n_nuclei = n_nuclei
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.grade = max(0, min(3, grade))
        self._grade_profile = self.GRADE_PROFILES[self.grade]

        # RGB camera mode — snap_frame returns (H, W, 3) uint8
        self.rgb_mode = True

        # Optical pipeline — shared across all modes (stained slide)
        p = OpticalPipeline()
        self._pipeline = {0: p, 1: p, 2: p}

        # Storage for nuclei positions and properties (world coordinates)
        self._nuclei_x = np.array([], dtype=np.float32)
        self._nuclei_y = np.array([], dtype=np.float32)
        self._nuclei_r = np.array([], dtype=np.float32)
        self._nuclei_intensity = np.array([], dtype=np.float32)
        self._nuclei_elongation = np.array([], dtype=np.float32)
        self._nuclei_angle = np.array([], dtype=np.float32)
        self._nuclei_mitotic = np.array([], dtype=bool)

        # Tissue-specific structures (world coordinates)
        self._glands = []
        self._vessels = []
        self._adipocytes = []
        self._epithelia = []
        self._regions = []

        # Tissue boundary (only for world > 512)
        self._tissue_boundary = None  # (N, 2) int32 polygon
        self._tissue_mask = None      # (H, W) uint8

        # Auto-reduce internal_scale for large worlds to stay under memory limits
        max_dim = world_size * self.internal_scale
        if max_dim > 5000:
            self.internal_scale = max(1, 4096 // world_size)
            self._iw = world_size * self.internal_scale
            self._ih = world_size * self.internal_scale

        # Pre-rendered images at internal resolution
        self._bf_full = None
        self._nuc_full = None
        self._eos_full = None

        # Necrosis patches (world coordinates)
        self._necrosis_patches = []

        # Generate tissue boundary for larger worlds
        if world_size > 512:
            self._generate_tissue_boundary()

        # Generate and render
        self._generate_tissue()
        self._generate_necrosis()
        self._render_full()

    # -- Irregular contour generation --

    def _radial_noise(self, n_points, n_harmonics=5, seed=0):
        """Smooth radial noise for contour deformation. Returns array in [-1, 1]."""
        rng = np.random.default_rng(self._seed + seed + 7777)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        noise = np.zeros(n_points, dtype=np.float32)
        for k in range(2, n_harmonics + 2):
            amp = rng.normal(0, 1.0 / k)
            phase = rng.uniform(0, 2 * np.pi)
            noise += float(amp) * np.sin(k * theta + float(phase))
        mx = max(abs(noise.max()), abs(noise.min()), 1e-6)
        return noise / mx

    def _circle_contour(self, cx, cy, radius, noise_amp=0.15,
                        n_points=48, seed=0):
        """Irregular circle contour at internal resolution. Returns (N, 2) int32."""
        s = self.internal_scale
        noise = self._radial_noise(n_points, seed=seed)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r = radius * s * (1 + noise_amp * noise)
        xs = cx * s + r * np.cos(theta)
        ys = cy * s + r * np.sin(theta)
        return np.column_stack([xs, ys]).astype(np.int32)

    def _ellipse_contour(self, cx, cy, r_major, r_minor, angle,
                         noise_amp=0.12, n_points=32, seed=0):
        """Irregular ellipse contour at internal resolution. Returns (N, 2) int32."""
        s = self.internal_scale
        noise = self._radial_noise(n_points, seed=seed)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        lx = r_major * s * np.cos(theta) * (1 + noise_amp * noise)
        ly = r_minor * s * np.sin(theta) * (1 + noise_amp * noise)
        ca, sa = np.cos(angle), np.sin(angle)
        xs = cx * s + lx * ca - ly * sa
        ys = cy * s + lx * sa + ly * ca
        return np.column_stack([xs, ys]).astype(np.int32)

    def _generate_fiber_field(self, rng):
        """Generate smooth fiber orientation field at world resolution.

        Returns (height, width) float32 angle map in radians.
        Near glands: tangential at grade 0-1, radial at grade 2-3 (TACS).
        """
        w, h = self.width, self.height

        # Base: layered smooth noise for random orientations
        angle_map = np.zeros((h, w), dtype=np.float32)
        for octave in range(3):
            freq = 2 ** octave
            gs = max(3, w // (16 * freq))
            noise = rng.normal(0, 1, (gs + 2, gs + 2)).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (0, 0), 1.5)
            noise_up = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
            angle_map += noise_up * (0.6 ** octave)
        angle_map *= np.pi

        # Gland-proximal orientation (TACS framework)
        if self._glands and self.grade >= 1:
            yy, xx = np.mgrid[0:h, 0:w]
            for cx, cy, outer_r, _, _ in self._glands:
                dx = (xx - cx).astype(np.float32)
                dy = (yy - cy).astype(np.float32)
                dist = np.sqrt(dx ** 2 + dy ** 2)
                zone = (dist > outer_r) & (dist < outer_r + 40)
                if not zone.any():
                    continue
                radial = np.arctan2(dy, dx)
                if self.grade <= 1:
                    target = radial + np.pi / 2  # tangential (TACS-1/2)
                else:
                    target = radial  # radial / perpendicular (TACS-3)
                strength = np.clip(1.0 - (dist - outer_r) / 40, 0, 1)
                blend = strength * 0.6
                angle_map[zone] = (
                    angle_map[zone] * (1 - blend[zone])
                    + target[zone] * blend[zone]
                )

        return angle_map

    def _draw_collagen_fibers(self, img, rgb=True):
        """Draw grade-dependent collagen fiber bundles across the stroma.

        Uses a flow field for natural alignment.  Grade-dependent:
          Grade 0: sparse, wavy, thin (loose connective tissue)
          Grade 1: moderate density, semi-wavy, tangential near glands
          Grade 2: dense, straighter, thick keloid-like bundles
          Grade 3: very dense, straight invasion corridors
        """
        s = self.internal_scale
        rng = np.random.default_rng(self._seed + 5555)
        iw, ih = self._iw, self._ih
        w, h = self.width, self.height

        # Grade-dependent fibre parameters
        _gp = {
            0: {"density": 1.0, "waviness": 0.14, "lt_range": (1, 1),
                "bundle": (2, 5), "keloid_prob": 0.0, "length": (25, 70)},
            1: {"density": 1.5, "waviness": 0.10, "lt_range": (1, 2),
                "bundle": (3, 6), "keloid_prob": 0.0, "length": (30, 80)},
            2: {"density": 2.5, "waviness": 0.05, "lt_range": (1, 3),
                "bundle": (3, 8), "keloid_prob": 0.15, "length": (35, 90)},
            3: {"density": 3.0, "waviness": 0.03, "lt_range": (2, 4),
                "bundle": (4, 10), "keloid_prob": 0.25, "length": (40, 100)},
        }[self.grade]

        # Flow field
        angle_field = self._generate_fiber_field(rng)

        # Structure exclusion mask (world coords)
        exclusion = np.zeros((h, w), dtype=bool)
        yy, xx = np.mgrid[0:h, 0:w]
        for cx, cy, outer_r, _, _ in self._glands:
            exclusion |= (np.hypot(xx - cx, yy - cy) < outer_r + 3)
        for vx, vy, vr in self._vessels:
            exclusion |= (np.hypot(xx - vx, yy - vy) < vr + 3)
        for ax, ay, ar in self._adipocytes:
            exclusion |= (np.hypot(xx - ax, yy - ay) < ar + 2)

        n_bundles = max(5, int(w * h / 5000 * _gp["density"]))
        lt_lo = max(1, _gp["lt_range"][0] * s // 2)
        lt_hi = max(lt_lo, _gp["lt_range"][1] * s // 2)

        for _ in range(n_bundles):
            # Find valid start position (world coords)
            bx_w, by_w = -1.0, -1.0
            for _att in range(10):
                bx_w = rng.uniform(5, w - 5)
                by_w = rng.uniform(5, h - 5)
                ix, iy = int(bx_w) % w, int(by_w) % h
                if not exclusion[iy, ix]:
                    break
            else:
                continue

            base_angle = angle_field[int(by_w) % h, int(bx_w) % w]
            n_fibers = rng.integers(_gp["bundle"][0], _gp["bundle"][1] + 1)
            lt = rng.integers(lt_lo, lt_hi + 1)

            # Occasional keloid-like thick bundles at high grades
            if rng.random() < _gp["keloid_prob"]:
                lt = max(lt, int(3 * s // 2))

            for _fi in range(n_fibers):
                offset = rng.normal(0, 2.5 * s)
                fx = bx_w * s + offset * np.cos(base_angle + np.pi / 2)
                fy = by_w * s + offset * np.sin(base_angle + np.pi / 2)

                length = rng.uniform(_gp["length"][0] * s, _gp["length"][1] * s)
                n_seg = 15
                step = length / n_seg
                pts = []
                a = base_angle

                for _si in range(n_seg + 1):
                    pts.append([int(fx), int(fy)])
                    # Follow flow field
                    wx, wy = fx / s, fy / s
                    if 0 <= wx < w and 0 <= wy < h:
                        fa = angle_field[int(wy) % h, int(wx) % w]
                        diff = (fa - a + np.pi) % (2 * np.pi) - np.pi
                        a += 0.3 * diff  # blend toward field direction
                    a += float(rng.normal(0, _gp["waviness"]))
                    fx += step * np.cos(a)
                    fy += step * np.sin(a)

                if len(pts) >= 2:
                    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    if rgb:
                        fc = np.clip(
                            HE_COLLAGEN + rng.uniform(-6, 6, 3), 0, 255
                        )
                        cv2.polylines(img, [arr], False, fc.tolist(),
                                      lt, cv2.LINE_AA)
                    else:
                        cv2.polylines(img, [arr], False,
                                      float(180 + rng.uniform(-8, 8)),
                                      lt, cv2.LINE_AA)

    def _generate_tissue_boundary(self):
        """Generate irregular tissue section boundary for large worlds.

        Creates a smooth blob covering ~80% of the world area.
        Structures will only be placed inside this boundary.
        """
        w, h = self.width, self.height
        cx, cy = w / 2, h / 2
        base_r = min(w, h) * 0.40

        n_pts = 128
        noise = self._radial_noise(n_pts, n_harmonics=8, seed=99999)
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        radii = base_r * (1 + 0.18 * noise)
        xs = cx + radii * np.cos(theta)
        ys = cy + radii * np.sin(theta)
        self._tissue_boundary = np.column_stack([xs, ys]).astype(np.int32)

        # World-resolution binary mask
        self._tissue_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._tissue_mask, [self._tissue_boundary], 255)

    def _inside_tissue(self, x, y):
        """Check if (x, y) world coordinate is inside the tissue section."""
        if self._tissue_mask is None:
            return True
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self._tissue_mask[iy, ix] > 0

    # -- Tissue generation (all in world coordinates) --

    def _generate_tissue(self):
        """Generate tissue architecture and place nuclei."""
        if self.tissue_type == "mixed":
            self._generate_mixed()
        elif self.tissue_type == "glandular":
            self._generate_glandular()
        elif self.tissue_type == "epithelium":
            self._generate_epithelium()
        elif self.tissue_type == "connective":
            self._generate_connective()
        elif self.tissue_type == "adipose":
            self._generate_adipose()
        else:
            raise ValueError(f"Unknown tissue type: {self.tissue_type}")

    def _generate_glandular(self):
        """Generate glandular tissue with circular lumens.

        At higher grades, glands are more irregular, crowded (back-to-back),
        and have smaller lumens. Grade 3 replaces some glands with solid nests.
        """
        rng = self._rng
        w, h = self.width, self.height
        gp = self._grade_profile

        # Higher grades: more glands, closer together
        base_max = max(1, min(8, int(min(w, h) / 80)))
        grade_extra = [0, 2, 4, 6][self.grade]
        max_glands = base_max + grade_extra
        n_glands = rng.integers(max(1, max_glands - 2), max_glands + 1)
        gland_positions = []

        # Minimum spacing decreases with grade (back-to-back at high grade)
        min_spacing = gp["gland_spacing"]

        margin = min(60, min(w, h) * 0.12)
        for _ in range(n_glands * 30):
            if len(gland_positions) >= n_glands:
                break
            cx = rng.uniform(margin, w - margin)
            cy = rng.uniform(margin, h - margin)
            if not self._inside_tissue(cx, cy):
                continue
            outer_r = rng.uniform(25, 55) if self.grade < 3 else rng.uniform(15, 40)
            ok = True
            for gx, gy, gr, _, _ in gland_positions:
                dist = np.hypot(cx - gx, cy - gy)
                if dist < gr + outer_r + min_spacing:
                    ok = False
                    break
            if ok:
                # Lumen shrinks with grade (solid nests at grade 3)
                if self.grade >= 3 and rng.random() < 0.4:
                    inner_r = outer_r * rng.uniform(0.05, 0.15)  # near-solid
                elif self.grade >= 2:
                    inner_r = outer_r * rng.uniform(0.15, 0.45)
                else:
                    inner_r = outer_r * rng.uniform(0.4, 0.65)
                n_cells = int(2 * np.pi * (outer_r + inner_r) / 2 / 10)
                gland_positions.append((cx, cy, outer_r, inner_r, n_cells))

        self._glands = gland_positions

        nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle = [], [], [], [], [], []

        for cx, cy, outer_r, inner_r, n_cells in gland_positions:
            # Basal polarity: nuclei sit toward the outer (basal) side
            # In real columnar epithelium, nuclei are basally located
            basal_r = inner_r + (outer_r - inner_r) * 0.65
            for i in range(n_cells):
                angle = 2 * np.pi * i / n_cells + rng.normal(0, 0.1)
                r_offset = rng.normal(0, 1.5)
                nx = cx + (basal_r + r_offset) * np.cos(angle)
                ny = cy + (basal_r + r_offset) * np.sin(angle)
                nuc_x.append(nx)
                nuc_y.append(ny)
                nuc_r.append(rng.uniform(3.0, 5.0))
                nuc_int.append(rng.uniform(45, 75))
                nuc_elong.append(rng.uniform(1.0, 1.4))
                nuc_angle.append(angle + np.pi / 2)

        n_stromal = max(0, self.n_nuclei - len(nuc_x))
        for _ in range(n_stromal * 3):
            if len(nuc_x) >= self.n_nuclei:
                break
            nx = rng.uniform(10, w - 10)
            ny = rng.uniform(10, h - 10)
            if not self._inside_tissue(nx, ny):
                continue
            inside_gland = False
            for cx, cy, outer_r, _, _ in gland_positions:
                if np.hypot(nx - cx, ny - cy) < outer_r + 5:
                    inside_gland = True
                    break
            if not inside_gland:
                nuc_x.append(nx)
                nuc_y.append(ny)
                nuc_r.append(rng.uniform(2.5, 4.0))
                nuc_int.append(rng.uniform(55, 85))
                nuc_elong.append(rng.uniform(1.3, 2.5))
                nuc_angle.append(rng.uniform(0, np.pi))

        n_vessels = rng.integers(1, 4)
        for _ in range(n_vessels * 5):
            if len(self._vessels) >= n_vessels:
                break
            vx = rng.uniform(30, w - 30)
            vy = rng.uniform(30, h - 30)
            vr = rng.uniform(8, 18)
            ok = True
            for cx, cy, outer_r, _, _ in gland_positions:
                if np.hypot(vx - cx, vy - cy) < outer_r + vr + 5:
                    ok = False
                    break
            if ok:
                self._vessels.append((vx, vy, vr))

        self._finalize_nuclei(nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle)

    def _generate_epithelium(self):
        """Generate epithelial layers."""
        rng = self._rng
        w, h = self.width, self.height

        nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle = [], [], [], [], [], []

        margin = min(40, h * 0.15)
        max_layers = max(1, min(4, int(h / 60)))
        n_layers = rng.integers(1, max_layers + 1)
        layer_ys = sorted(rng.uniform(margin, h - margin, size=n_layers))

        for y_base in layer_ys:
            thickness = rng.uniform(20, 40)
            self._epithelia.append((y_base, thickness, 0, w))

            cell_spacing = rng.uniform(12, 16)
            n_cells = int(w / cell_spacing)
            for i in range(n_cells):
                nx = i * cell_spacing + rng.normal(0, 1.5)
                ny = y_base + rng.normal(0, thickness * 0.15)
                nuc_x.append(nx)
                nuc_y.append(ny)
                nuc_r.append(rng.uniform(3.5, 5.5))
                nuc_int.append(rng.uniform(40, 70))
                nuc_elong.append(rng.uniform(1.0, 1.8))
                nuc_angle.append(np.pi / 2 + rng.normal(0, 0.15))

        n_stromal = max(0, self.n_nuclei - len(nuc_x))
        for _ in range(n_stromal):
            nx = rng.uniform(10, w - 10)
            ny = rng.uniform(10, h - 10)
            near_layer = False
            for yb, th, _, _ in self._epithelia:
                if abs(ny - yb) < th * 0.6:
                    near_layer = True
                    break
            if not near_layer:
                nuc_x.append(nx)
                nuc_y.append(ny)
                nuc_r.append(rng.uniform(2.5, 4.0))
                nuc_int.append(rng.uniform(60, 90))
                nuc_elong.append(rng.uniform(1.5, 3.0))
                nuc_angle.append(rng.uniform(0, np.pi))

        self._finalize_nuclei(nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle)

    def _generate_connective(self):
        """Generate loose connective tissue with scattered nuclei."""
        rng = self._rng
        w, h = self.width, self.height

        nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle = [], [], [], [], [], []

        for _ in range(self.n_nuclei * 3):
            if len(nuc_x) >= self.n_nuclei:
                break
            nx = rng.uniform(10, w - 10)
            ny = rng.uniform(10, h - 10)
            if not self._inside_tissue(nx, ny):
                continue
            nuc_x.append(nx)
            nuc_y.append(ny)
            nuc_r.append(rng.uniform(2.5, 4.5))
            nuc_int.append(rng.uniform(50, 85))
            nuc_elong.append(rng.uniform(1.5, 3.5))
            nuc_angle.append(rng.uniform(0, np.pi))

        n_bundles = rng.integers(2, 4)
        for _ in range(n_bundles):
            bx = rng.uniform(50, w - 50)
            by = rng.uniform(50, h - 50)
            b_angle = rng.uniform(0, np.pi)
            b_radius = rng.uniform(40, 80)
            for i in range(len(nuc_x)):
                dist = np.hypot(nuc_x[i] - bx, nuc_y[i] - by)
                if dist < b_radius:
                    nuc_angle[i] = b_angle + rng.normal(0, 0.2)

        n_vessels = rng.integers(2, 5)
        for _ in range(n_vessels * 10):
            if len(self._vessels) >= n_vessels:
                break
            vx = rng.uniform(30, w - 30)
            vy = rng.uniform(30, h - 30)
            if not self._inside_tissue(vx, vy):
                continue
            vr = rng.uniform(10, 25)
            ok = all(
                np.hypot(vx - ex, vy - ey) > er + vr + 10
                for ex, ey, er in self._vessels
            )
            if ok:
                self._vessels.append((vx, vy, vr))
                n_endo = int(2 * np.pi * vr / 12)
                for j in range(n_endo):
                    a = 2 * np.pi * j / n_endo + rng.normal(0, 0.1)
                    nuc_x.append(vx + vr * np.cos(a))
                    nuc_y.append(vy + vr * np.sin(a))
                    nuc_r.append(rng.uniform(2.0, 3.0))
                    nuc_int.append(rng.uniform(45, 65))
                    nuc_elong.append(rng.uniform(1.8, 3.0))
                    nuc_angle.append(a + np.pi / 2)

        self._finalize_nuclei(nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle)

    def _generate_adipose(self):
        """Generate adipose tissue — large clear cells with peripheral nuclei."""
        rng = self._rng
        w, h = self.width, self.height

        adipocytes = []
        max_adipo = max(5, int(w * h / (50 * 50)))
        margin_a = min(30, min(w, h) * 0.1)
        for _ in range(max_adipo * 12):
            if len(adipocytes) >= max_adipo:
                break
            cx = rng.uniform(margin_a, w - margin_a)
            cy = rng.uniform(margin_a, h - margin_a)
            if not self._inside_tissue(cx, cy):
                continue
            r = rng.uniform(18, 35)
            ok = True
            for ax, ay, ar in adipocytes:
                if np.hypot(cx - ax, cy - ay) < r + ar + 3:
                    ok = False
                    break
            if ok:
                adipocytes.append((cx, cy, r))

        self._adipocytes = adipocytes

        nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle = [], [], [], [], [], []

        for cx, cy, r in adipocytes:
            angle = rng.uniform(0, 2 * np.pi)
            nx = cx + (r - 3) * np.cos(angle)
            ny = cy + (r - 3) * np.sin(angle)
            nuc_x.append(nx)
            nuc_y.append(ny)
            nuc_r.append(rng.uniform(2.5, 4.0))
            nuc_int.append(rng.uniform(45, 70))
            nuc_elong.append(rng.uniform(1.5, 2.5))
            nuc_angle.append(angle + np.pi / 2)

        n_extra = max(0, self.n_nuclei - len(nuc_x))
        for _ in range(n_extra * 3):
            if len(nuc_x) >= self.n_nuclei:
                break
            nx = rng.uniform(10, w - 10)
            ny = rng.uniform(10, h - 10)
            inside = False
            for cx, cy, r in adipocytes:
                if np.hypot(nx - cx, ny - cy) < r - 2:
                    inside = True
                    break
            if not inside:
                nuc_x.append(nx)
                nuc_y.append(ny)
                nuc_r.append(rng.uniform(2.0, 3.5))
                nuc_int.append(rng.uniform(55, 80))
                nuc_elong.append(rng.uniform(1.5, 3.0))
                nuc_angle.append(rng.uniform(0, np.pi))

        self._finalize_nuclei(nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle)

    def _generate_mixed(self):
        """Generate a mixed tissue section with 2-3 regions."""
        rng = self._rng
        w, h = self.width, self.height

        types = rng.choice(
            ["epithelium", "glandular", "connective", "adipose"],
            size=3,
            replace=False,
        )
        min_h = max(120, h // 4)
        b1 = rng.uniform(min_h, h - 2 * min_h)
        b2 = rng.uniform(b1 + min_h, h - min_h)
        boundaries = [0, b1, b2, h]

        all_nuc = {"x": [], "y": [], "r": [], "int": [], "elong": [], "angle": []}

        for i, ttype in enumerate(types):
            y_start = boundaries[i]
            y_end = boundaries[i + 1]
            region_h = y_end - y_start

            self._regions.append((ttype, (0, y_start, w, y_end)))

            sub = HistologySim.__new__(HistologySim)
            sub._rng = np.random.default_rng(self._seed + i + 100)
            sub.width = w
            sub.height = int(region_h)
            sub.n_nuclei = int(self.n_nuclei * region_h / h)
            sub._glands = []
            sub._vessels = []
            sub._adipocytes = []
            sub._epithelia = []
            sub._tissue_boundary = None
            sub._tissue_mask = None
            sub._grade_profile = self._grade_profile
            sub.grade = self.grade
            sub.internal_scale = self.internal_scale
            sub._seed = self._seed + i + 100
            sub._ih = int(region_h * self.internal_scale)
            sub._iw = int(w * self.internal_scale)

            if ttype == "glandular":
                sub._generate_glandular()
            elif ttype == "epithelium":
                sub._generate_epithelium()
            elif ttype == "connective":
                sub._generate_connective()
            elif ttype == "adipose":
                sub._generate_adipose()

            for j in range(len(sub._nuclei_x)):
                all_nuc["x"].append(float(sub._nuclei_x[j]))
                all_nuc["y"].append(float(sub._nuclei_y[j]) + y_start)
                all_nuc["r"].append(float(sub._nuclei_r[j]))
                all_nuc["int"].append(float(sub._nuclei_intensity[j]))
                all_nuc["elong"].append(float(sub._nuclei_elongation[j]))
                all_nuc["angle"].append(float(sub._nuclei_angle[j]))

            for g in sub._glands:
                self._glands.append((g[0], g[1] + y_start, g[2], g[3], g[4]))
            for v in sub._vessels:
                self._vessels.append((v[0], v[1] + y_start, v[2]))
            for a in sub._adipocytes:
                self._adipocytes.append((a[0], a[1] + y_start, a[2]))
            for e in sub._epithelia:
                self._epithelia.append((e[0] + y_start, e[1], e[2], e[3]))

        self._finalize_nuclei(
            all_nuc["x"],
            all_nuc["y"],
            all_nuc["r"],
            all_nuc["int"],
            all_nuc["elong"],
            all_nuc["angle"],
        )

    def _finalize_nuclei(self, nuc_x, nuc_y, nuc_r, nuc_int, nuc_elong, nuc_angle):
        """Convert nuclei lists to arrays, apply grade-dependent features."""
        self._nuclei_x = np.array(nuc_x, dtype=np.float32)
        self._nuclei_y = np.array(nuc_y, dtype=np.float32)
        self._nuclei_r = np.array(nuc_r, dtype=np.float32)
        self._nuclei_intensity = np.array(nuc_int, dtype=np.float32)
        self._nuclei_elongation = np.array(nuc_elong, dtype=np.float32)
        self._nuclei_angle = np.array(nuc_angle, dtype=np.float32)

        n = len(self._nuclei_x)
        gp = self._grade_profile

        # --- Grade-dependent pleomorphism ---
        if self.grade > 0 and n > 0:
            r_lo, r_hi = gp["r_range"]
            int_lo, int_hi = gp["intensity_range"]
            el_lo, el_hi = gp["elong_range"]
            # Re-draw nuclear sizes with wider variance
            self._nuclei_r = self._rng.uniform(r_lo, r_hi, n).astype(np.float32)
            self._nuclei_intensity = self._rng.uniform(int_lo, int_hi, n).astype(
                np.float32
            )
            self._nuclei_elongation = self._rng.uniform(el_lo, el_hi, n).astype(
                np.float32
            )

        # --- Mitotic figures (grade-dependent frequency) ---
        mf_lo, mf_hi = gp["mitotic_frac"]
        n_mitotic = max(0, int(n * self._rng.uniform(mf_lo, mf_hi)))
        if n_mitotic > 0:
            mitotic_idx = self._rng.choice(n, size=n_mitotic, replace=False)
        else:
            mitotic_idx = np.array([], dtype=int)
        self._nuclei_mitotic = np.zeros(n, dtype=bool)
        if len(mitotic_idx) > 0:
            self._nuclei_mitotic[mitotic_idx] = True
            self._nuclei_intensity[mitotic_idx] = self._rng.uniform(
                25, 45, len(mitotic_idx)
            )
            self._nuclei_r[mitotic_idx] *= 1.2

        # --- Lymphocyte infiltrate (small, round, dark nuclei) ---
        lymph_frac = gp["lymphocyte_frac"]
        if lymph_frac > 0 and n > 0:
            n_lymph = max(0, int(n * lymph_frac))
            lx, ly, lr, li, le, la = [], [], [], [], [], []
            for _ in range(n_lymph * 3):
                if len(lx) >= n_lymph:
                    break
                x = self._rng.uniform(10, self.width - 10)
                y = self._rng.uniform(10, self.height - 10)
                if not self._inside_tissue(x, y):
                    continue
                lx.append(x)
                ly.append(y)
                lr.append(self._rng.uniform(1.5, 2.5))
                li.append(self._rng.uniform(25, 45))
                le.append(self._rng.uniform(1.0, 1.3))
                la.append(self._rng.uniform(0, np.pi))
            if lx:
                self._nuclei_x = np.concatenate(
                    [self._nuclei_x, np.array(lx, dtype=np.float32)]
                )
                self._nuclei_y = np.concatenate(
                    [self._nuclei_y, np.array(ly, dtype=np.float32)]
                )
                self._nuclei_r = np.concatenate(
                    [self._nuclei_r, np.array(lr, dtype=np.float32)]
                )
                self._nuclei_intensity = np.concatenate(
                    [self._nuclei_intensity, np.array(li, dtype=np.float32)]
                )
                self._nuclei_elongation = np.concatenate(
                    [self._nuclei_elongation, np.array(le, dtype=np.float32)]
                )
                self._nuclei_angle = np.concatenate(
                    [self._nuclei_angle, np.array(la, dtype=np.float32)]
                )
                self._nuclei_mitotic = np.concatenate(
                    [self._nuclei_mitotic, np.zeros(len(lx), dtype=bool)]
                )
                self._n_lymphocytes = len(lx)
            else:
                self._n_lymphocytes = 0
        else:
            self._n_lymphocytes = 0

    def _generate_necrosis(self):
        """Generate necrosis patches for high-grade tumors.

        Necrosis appears as amorphous eosinophilic (pink) areas with
        ghost nuclei (faint outlines of dead cells). Placed in stroma,
        avoiding gland centers.
        """
        nf = self._grade_profile["necrosis_frac"]
        if nf <= 0:
            return

        rng = self._rng
        w, h = self.width, self.height
        target_area = w * h * nf
        placed_area = 0

        for _ in range(50):
            if placed_area >= target_area:
                break
            cx = rng.uniform(40, w - 40)
            cy = rng.uniform(40, h - 40)
            if not self._inside_tissue(cx, cy):
                continue
            # Avoid gland centers
            near_gland = False
            for gx, gy, gr, _, _ in self._glands:
                if np.hypot(cx - gx, cy - gy) < gr + 10:
                    near_gland = True
                    break
            if near_gland:
                continue
            rx = rng.uniform(15, 40)
            ry = rng.uniform(15, 40)
            angle = rng.uniform(0, np.pi)
            self._necrosis_patches.append((cx, cy, rx, ry, angle))
            placed_area += np.pi * rx * ry

    # -- Rendering (all at internal resolution) --

    def _render_full(self):
        """Pre-render all channels at internal resolution."""
        self._bf_full = self._render_he()
        self._nuc_full = self._render_nuclei_only()
        self._eos_full = self._render_eosin_only()

    def _render_he(self):
        """Render H&E stained brightfield as RGB at internal resolution.

        Returns (ih, iw, 3) uint8 array in RGB color order.
        """
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        rng = self._rng

        base = TISSUE_TYPES.get(
            self.tissue_type if self.tissue_type != "mixed" else "connective", {}
        )
        noise_std = base.get("stroma_noise", 10)

        # --- Background: eosin-stained stroma (pink) ---
        img = np.empty((ih, iw, 3), dtype=np.float32)
        img[:, :] = HE_STROMA

        # Stromal texture — generate at world res, upscale
        noise = rng.normal(0, noise_std, (self.height, self.width)).astype(
            np.float32
        )
        noise = cv2.GaussianBlur(noise, (15, 15), 4.0)
        if s > 1:
            noise = cv2.resize(noise, (iw, ih), interpolation=cv2.INTER_LINEAR)
        # Correlated staining variation: R varies most, B least
        ch_scale = np.array([1.0, 0.8, 0.7], dtype=np.float32)
        img += noise[:, :, np.newaxis] * ch_scale

        # Collagen fiber texture — anisotropic noise for fibrillar stroma
        # Always present for connective/mixed; also at grade >= 2 (desmoplasia)
        if self.tissue_type in ("connective", "mixed") or self.grade >= 2:
            amp = 4 if self.tissue_type in ("connective", "mixed") else 3
            if self.grade >= 2:
                amp += self.grade  # denser stroma at higher grade
            fiber = rng.normal(0, amp, (self.height, self.width)).astype(
                np.float32
            )
            fiber = cv2.GaussianBlur(fiber, (21, 3), 6.0)
            if s > 1:
                fiber = cv2.resize(fiber, (iw, ih), interpolation=cv2.INTER_LINEAR)
            img += fiber[:, :, np.newaxis] * ch_scale

        # Mixed regions — vary background per region
        if self._regions:
            for ttype, (x0, y0, x1, y1) in self._regions:
                y0i = self._s(y0)
                y1i = self._s(y1)
                cfg = TISSUE_TYPES.get(ttype, {})
                rh = max(1, int(y1 - y0))
                rnoise = rng.normal(
                    0, cfg.get("stroma_noise", 10), (rh, self.width)
                ).astype(np.float32)
                rnoise = cv2.GaussianBlur(rnoise, (15, 15), 4.0)
                if s > 1:
                    rnoise = cv2.resize(
                        rnoise, (iw, y1i - y0i), interpolation=cv2.INTER_LINEAR
                    )
                img[y0i:y1i, :] = HE_STROMA
                img[y0i:y1i] += rnoise[:, :, np.newaxis] * ch_scale

        # --- Collagen fiber bundles in stroma ---
        self._draw_collagen_fibers(img, rgb=True)

        lt = max(1, s // 2)  # scaled line thickness

        # --- Epithelial bands ---
        for y_base, thickness, x_start, x_end in self._epithelia:
            y0 = max(0, self._s(y_base - thickness / 2))
            y1 = min(ih, self._s(y_base + thickness / 2))
            x0 = max(0, self._s(x_start))
            x1 = min(iw, self._s(x_end))
            bh, bw = y1 - y0, x1 - x0
            if bh <= 0 or bw <= 0:
                continue
            band = np.empty((bh, bw, 3), dtype=np.float32)
            band[:, :] = HE_CYTOPLASM
            bnoise = rng.normal(0, 5, (max(1, bh // s), max(1, bw // s))).astype(
                np.float32
            )
            if s > 1:
                bnoise = cv2.resize(bnoise, (bw, bh), interpolation=cv2.INTER_LINEAR)
            band += bnoise[:, :, np.newaxis] * ch_scale
            img[y0:y1, x0:x1] = band
            # Basement membrane
            bm_y = min(ih - 1, self._s(y_base + thickness / 2 + 1))
            if bm_y < ih and bm_y - lt >= 0:
                img[max(0, bm_y - lt) : bm_y, x0:x1] = HE_BASEMENT_MEMBRANE

        # --- Adipocytes (mostly round, noise_amp=0.08) ---
        for ai, (cx, cy, r) in enumerate(self._adipocytes):
            inner = self._circle_contour(cx, cy, r - 1, noise_amp=0.08,
                                         n_points=48, seed=30000 + ai)
            outer = self._circle_contour(cx, cy, r, noise_amp=0.08,
                                         n_points=48, seed=30000 + ai)
            jit = rng.uniform(-3, 3, 3)
            color = np.clip(HE_ADIPOCYTE + jit, 0, 255)
            cv2.fillPoly(img, [inner], color.tolist())
            wall = np.clip(HE_ADIPOCYTE_WALL + rng.uniform(-5, 5, 3), 0, 255)
            cv2.polylines(img, [outer], True, wall.tolist(), lt, cv2.LINE_AA)

        # --- Gland lumens (scalloped lumen, smoother outer) ---
        gland_noise = self._grade_profile["gland_noise_amp"]
        for gi, (cx, cy, outer_r, inner_r, n_cells) in enumerate(self._glands):
            outer = self._circle_contour(cx, cy, outer_r, noise_amp=gland_noise,
                                         n_points=64, seed=10000 + gi)
            inner = self._circle_contour(cx, cy, inner_r,
                                         noise_amp=min(0.50, gland_noise * 2),
                                         n_points=64, seed=10500 + gi)

            # Retraction artifact: thin white gap around gland (fixation shrinkage)
            retract = self._circle_contour(cx, cy, outer_r + 1.5,
                                           noise_amp=gland_noise * 0.5,
                                           n_points=64, seed=10200 + gi)
            retract_c = np.clip(HE_GLASS * 0.98 + rng.uniform(-2, 2, 3), 0, 255)
            cv2.polylines(img, [retract], True, retract_c.tolist(),
                          max(1, s), cv2.LINE_AA)

            # Epithelial ring (pale eosin cytoplasm)
            cyto = np.clip(HE_CYTOPLASM + rng.uniform(-8, 8, 3), 0, 255)
            cv2.fillPoly(img, [outer], cyto.tolist())

            # Lumen with mucin (pale blue-pink secretion product)
            grng = np.random.default_rng(self._seed + 15000 + gi)
            has_mucin = grng.random() < 0.6
            if has_mucin and inner_r > 3:
                mucin_c = np.array([230, 218, 235], dtype=np.float32)
                mucin_c = np.clip(mucin_c + grng.uniform(-5, 5, 3), 0, 255)
                cv2.fillPoly(img, [inner], mucin_c.tolist())
                # Mucin has slight texture variation
                icx_m, icy_m = self._sf(cx), self._sf(cy)
                ir_m = int(inner_r * s)
                for _ in range(max(2, int(inner_r))):
                    ma = grng.uniform(0, 2 * np.pi)
                    mr = grng.uniform(0, inner_r * 0.7) * s
                    mx = int(icx_m + mr * np.cos(ma))
                    my = int(icy_m + mr * np.sin(ma))
                    mrc = max(1, int(grng.uniform(1, 3) * s))
                    mc_j = np.clip(mucin_c + grng.uniform(-8, 8, 3), 0, 255)
                    cv2.circle(img, (mx, my), mrc, mc_j.tolist(), -1, cv2.LINE_AA)
            else:
                lumen = np.clip(HE_LUMEN + rng.uniform(-3, 3, 3), 0, 255)
                cv2.fillPoly(img, [inner], lumen.tolist())

            # Outer basement membrane
            bm = np.clip(HE_BASEMENT_MEMBRANE + rng.uniform(-5, 5, 3), 0, 255)
            cv2.polylines(img, [outer], True, bm.tolist(), lt, cv2.LINE_AA)
            # Epithelial cell boundaries — thin radial lines
            icx, icy = self._sf(cx), self._sf(cy)
            cell_line = np.clip(HE_BASEMENT_MEMBRANE * 0.92, 0, 255)
            for ci in range(n_cells):
                a = 2 * np.pi * ci / n_cells + rng.normal(0, 0.08)
                x1 = int(icx + inner_r * s * np.cos(a))
                y1 = int(icy + inner_r * s * np.sin(a))
                x2 = int(icx + outer_r * s * np.cos(a))
                y2 = int(icy + outer_r * s * np.sin(a))
                cv2.line(img, (x1, y1), (x2, y2),
                         cell_line.tolist(), max(1, s // 3), cv2.LINE_AA)

        # --- Blood vessels with individual RBCs ---
        for vi, (vx, vy, vr) in enumerate(self._vessels):
            self._draw_vessel_rgb(img, vi, vx, vy, vr, lt)

        # --- Necrosis patches (amorphous pink with ghost nuclei) ---
        for ni, (ncx, ncy, nrx, nry, nangle) in enumerate(self._necrosis_patches):
            self._draw_necrosis_rgb(img, ni, ncx, ncy, nrx, nry, nangle)

        # --- Cytoplasm around each nucleus (pale pink halo) ---
        for i in range(len(self._nuclei_x)):
            self._draw_cytoplasm_rgb(img, i)

        # --- Nuclei (hematoxylin blue-purple) ---
        for i in range(len(self._nuclei_x)):
            self._draw_nucleus_rgb(img, i)

        # Staining gradient — real H&E always has subtle unevenness across
        # the section (uneven thickness, fixation, staining bath flow)
        grad_rng = np.random.default_rng(self._seed + 99000)
        grad_angle = grad_rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:ih, 0:iw]
        grad = (np.cos(grad_angle) * (xx - iw / 2) +
                np.sin(grad_angle) * (yy - ih / 2))
        grad = grad / max(1, grad.max() - grad.min())  # normalize to [-0.5, 0.5]
        # Subtle staining intensity modulation (±3% brightness + ±2 color shift)
        stain_mod = 1.0 + grad * 0.06
        img *= stain_mod[:, :, np.newaxis]
        # Slight hue shift along gradient (warmer on thick side, cooler on thin)
        img[:, :, 0] += grad * 3   # R: warmer where thicker
        img[:, :, 2] -= grad * 2   # B: cooler where thinner

        # Subtle overall texture
        texture = rng.normal(0, 2, (self.height, self.width)).astype(np.float32)
        if s > 1:
            texture = cv2.resize(texture, (iw, ih), interpolation=cv2.INTER_LINEAR)
        img += texture[:, :, np.newaxis]

        # --- Tissue boundary: glass slide outside, edge darkening ---
        if self._tissue_mask is not None:
            mask_int = cv2.resize(
                self._tissue_mask, (iw, ih), interpolation=cv2.INTER_LINEAR
            )
            # Glass slide outside tissue
            glass = mask_int < 128
            glass_color = HE_GLASS + rng.normal(0, 1.5, (ih, iw, 3)).astype(
                np.float32
            )
            img[glass] = glass_color[glass]

            # Edge darkening: darker staining at tissue periphery
            dist = cv2.distanceTransform(
                (mask_int >= 128).astype(np.uint8), cv2.DIST_L2, 5
            )
            edge_zone = (dist > 0) & (dist < 15 * s)
            if edge_zone.any():
                darken = np.clip(dist / (15 * s), 0, 1)
                for c in range(3):
                    ch = img[:, :, c]
                    ch[edge_zone] *= (0.88 + 0.12 * darken[edge_zone])

        return np.clip(img, 0, 255).astype(np.uint8)

    def _draw_nucleus(self, img, idx):
        """Draw a single nucleus (dark irregular shape) at internal resolution."""
        x = self._sf(self._nuclei_x[idx])
        y = self._sf(self._nuclei_y[idx])
        r = self._sf(self._nuclei_r[idx])
        intensity = self._nuclei_intensity[idx]
        elongation = self._nuclei_elongation[idx]
        angle = self._nuclei_angle[idx]
        is_mitotic = (
            self._nuclei_mitotic[idx]
            if idx < len(self._nuclei_mitotic)
            else False
        )

        # World-coord params
        wx = self._nuclei_x[idx]
        wy = self._nuclei_y[idx]
        wr = self._nuclei_r[idx]

        contour = self._ellipse_contour(
            wx, wy, wr * elongation, wr, angle,
            noise_amp=0.15 if not is_mitotic else 0.20,
            n_points=32, seed=40000 + idx,
        )

        if is_mitotic:
            cv2.fillPoly(img, [contour], float(intensity))
            rng = self._rng
            for _ in range(3):
                spike_angle = rng.uniform(0, 2 * np.pi)
                spike_r = r * rng.uniform(0.3, 0.6)
                sx = x + spike_r * np.cos(spike_angle)
                sy = y + spike_r * np.sin(spike_angle)
                cv2.circle(
                    img, (int(sx), int(sy)),
                    max(1, int(r * 0.35)),
                    float(max(20, intensity - 10)), -1, cv2.LINE_AA,
                )
        else:
            cv2.fillPoly(img, [contour], float(intensity))
            # Chromatin clumps (darker patches)
            crng = np.random.default_rng(self._seed + 50000 + idx)
            n_clumps = crng.integers(2, 5)
            for _ in range(n_clumps):
                cr = r * crng.uniform(0.1, 0.6)
                ca = crng.uniform(0, 2 * np.pi)
                ccx = int(x + cr * np.cos(ca))
                ccy = int(y + cr * np.sin(ca))
                clump_r = max(1, int(r * crng.uniform(0.12, 0.30)))
                cv2.circle(img, (ccx, ccy), clump_r,
                           float(max(20, intensity * 0.78)), -1, cv2.LINE_AA)
            # Nucleolus
            if crng.random() < self._grade_profile["nucleolus_pct"] and r > 3 * self.internal_scale:
                nx_off = crng.normal(0, r * 0.2)
                ny_off = crng.normal(0, r * 0.2)
                cv2.circle(
                    img, (int(x + nx_off), int(y + ny_off)),
                    max(1, int(r * 0.25)),
                    float(max(20, intensity - 15)), -1, cv2.LINE_AA,
                )

    def _draw_vessel_rgb(self, img, vi, vx, vy, vr, lt):
        """Draw a blood vessel with wall, individual RBCs, and endothelial nuclei."""
        s = self.internal_scale
        vrng = np.random.default_rng(self._seed + 70000 + vi)

        # Retraction artifact around vessel (fixation shrinkage)
        retract_c = self._circle_contour(vx, vy, vr + 3.5, noise_amp=0.12,
                                         n_points=48, seed=20200 + vi)
        retract_col = np.clip(HE_GLASS * 0.97 + vrng.uniform(-2, 2, 3), 0, 255)
        cv2.polylines(img, [retract_c], True, retract_col.tolist(),
                      max(1, s), cv2.LINE_AA)

        # Outer wall and lumen contours
        wall_c = self._circle_contour(vx, vy, vr + 2, noise_amp=0.15,
                                      n_points=48, seed=20000 + vi)
        lumen_c = self._circle_contour(vx, vy, vr, noise_amp=0.15,
                                       n_points=48, seed=20000 + vi)

        # Wall (pink-purple endothelium)
        wall_color = np.clip(HE_VESSEL_WALL + vrng.uniform(-5, 5, 3), 0, 255)
        cv2.fillPoly(img, [wall_c], wall_color.tolist())

        # Lumen background (pale plasma, near-white with slight yellow tint)
        plasma = np.array([248, 240, 235], dtype=np.float32)
        plasma_jit = np.clip(plasma + vrng.uniform(-3, 3, 3), 0, 255)
        cv2.fillPoly(img, [lumen_c], plasma_jit.tolist())

        # Individual RBCs as small eosinophilic discs with pale centers
        cx_i, cy_i = self._sf(vx), self._sf(vy)
        r_i = vr * s  # lumen radius in internal pixels
        rbc_r = max(2, int(2.5 * s))  # RBC radius ~2.5 world units

        # Pack RBCs using Poisson disc sampling (approximate)
        n_rbcs = max(3, int(np.pi * (vr - 1) ** 2 / (3.5 ** 2)))
        placed = []
        for _ in range(n_rbcs * 5):
            if len(placed) >= n_rbcs:
                break
            a = vrng.uniform(0, 2 * np.pi)
            dr = vrng.uniform(0, (vr - 3)) * s
            rx = cx_i + dr * np.cos(a)
            ry = cy_i + dr * np.sin(a)
            # Check distance from center (stay inside lumen)
            if np.hypot(rx - cx_i, ry - cy_i) > r_i - rbc_r:
                continue
            # Minimum spacing between RBCs
            too_close = False
            for px, py in placed:
                if np.hypot(rx - px, ry - py) < rbc_r * 1.6:
                    too_close = True
                    break
            if too_close:
                continue
            placed.append((rx, ry))

            # Draw RBC: dark red disc with pale center
            rbc_color = np.clip(
                HE_RBC + vrng.uniform(-12, 12, 3), 0, 255
            )
            cv2.circle(img, (int(rx), int(ry)), rbc_r,
                       rbc_color.tolist(), -1, cv2.LINE_AA)
            # Pale center (biconcave disc artifact)
            if rbc_r >= 3:
                center_c = np.clip(rbc_color * 1.25, 0, 255)
                cv2.circle(img, (int(rx), int(ry)), max(1, rbc_r // 3),
                           center_c.tolist(), -1, cv2.LINE_AA)

        # Endothelial nuclei lining the wall (thin, elongated, dark)
        n_endo = max(3, int(2 * np.pi * vr / 8))
        for ei in range(n_endo):
            a = 2 * np.pi * ei / n_endo + vrng.normal(0, 0.15)
            ex = cx_i + (vr + 0.5) * s * np.cos(a)
            ey = cy_i + (vr + 0.5) * s * np.sin(a)
            # Thin elongated endothelial nucleus (tangent to wall)
            endo_r = max(1, int(1.5 * s))
            elong = 2.5
            perp_a = a + np.pi / 2
            axes_a = (int(endo_r * elong), int(endo_r))
            angle_deg = int(np.degrees(perp_a))
            endo_c = np.clip(HE_NUCLEUS_DARK * 0.95 + vrng.uniform(-5, 5, 3),
                             0, 255)
            cv2.ellipse(img, (int(ex), int(ey)), axes_a, angle_deg,
                        0, 360, endo_c.tolist(), -1, cv2.LINE_AA)

    def _draw_necrosis_rgb(self, img, ni, cx, cy, rx, ry, angle):
        """Draw a necrosis patch: intensely eosinophilic with nuclear debris.

        Coagulative necrosis in H&E:
          - Bright pink-red (stronger eosin uptake than surrounding stroma)
          - Homogeneous amorphous texture (loss of cellular architecture)
          - Karyorrhexis: dark small fragments of nuclear debris
          - Ghost nuclei: faint pale outlines where nuclei were
          - Inflammatory rim: lymphocytes cluster at necrosis border
        """
        s = self.internal_scale
        nrng = np.random.default_rng(self._seed + 80000 + ni)

        # Irregular elliptical boundary
        contour = self._ellipse_contour(
            cx, cy, rx, ry, angle,
            noise_amp=0.25, n_points=48, seed=80000 + ni,
        )

        # Base necrotic color: intensely eosinophilic (bright pink-red)
        nec_base = HE_NECROSIS.copy()
        nec_color = np.clip(nec_base + nrng.uniform(-8, 8, 3), 0, 255)
        cv2.fillPoly(img, [contour], nec_color.tolist())

        # Build mask for the necrosis region
        mask = np.zeros((self._ih, self._iw), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        y_idx, x_idx = np.where(mask > 0)
        if len(x_idx) == 0:
            return

        # Amorphous granular texture (subtle eosin variation)
        n_blobs = max(8, len(x_idx) // 30)
        for _ in range(n_blobs):
            dx = int(nrng.choice(x_idx))
            dy = int(nrng.choice(y_idx))
            dot_r = max(1, int(nrng.uniform(1, 4) * s))
            # Slight color variation within necrosis (eosinophilic)
            dot_c = np.clip(nec_base + nrng.uniform(-12, 12, 3), 0, 255)
            cv2.circle(img, (dx, dy), dot_r,
                       dot_c.tolist(), -1, cv2.LINE_AA)

        # Karyorrhexis: small dark nuclear debris fragments
        n_fragments = max(5, int(np.pi * rx * ry / 50))
        for _ in range(n_fragments):
            fa = nrng.uniform(0, 2 * np.pi)
            fr = nrng.uniform(0, 0.8)
            fx = cx + fr * rx * np.cos(fa) * np.cos(angle) - \
                 fr * ry * np.sin(fa) * np.sin(angle)
            fy = cy + fr * rx * np.cos(fa) * np.sin(angle) + \
                 fr * ry * np.sin(fa) * np.cos(angle)
            frag_r = max(1, int(nrng.uniform(0.5, 2.0) * s))
            # Dark hematoxylin debris
            frag_c = np.clip(
                HE_KARYORRHEXIS + nrng.uniform(-12, 12, 3), 0, 255
            )
            cv2.circle(img, (self._s(fx), self._s(fy)), frag_r,
                       frag_c.tolist(), -1, cv2.LINE_AA)

        # Ghost nuclei: pale outlines of dead cells (karyolysis)
        n_ghosts = max(3, int(np.pi * rx * ry / 100))
        for gi in range(n_ghosts):
            ga = nrng.uniform(0, 2 * np.pi)
            gr = nrng.uniform(0, 0.7)
            gx = cx + gr * rx * np.cos(ga) * np.cos(angle) - \
                 gr * ry * np.sin(ga) * np.sin(angle)
            gy = cy + gr * rx * np.cos(ga) * np.sin(angle) + \
                 gr * ry * np.sin(ga) * np.cos(angle)
            ghost_r = max(1, int(nrng.uniform(2, 5) * s))
            # Faint hematoxylin outline (karyolysis — pale, washed-out)
            ghost_c = np.clip(
                np.array([170, 130, 155], dtype=np.float32)
                + nrng.uniform(-8, 8, 3), 0, 255
            )
            cv2.circle(img, (self._s(gx), self._s(gy)), ghost_r,
                       ghost_c.tolist(), max(1, s // 2), cv2.LINE_AA)

        # Inflammatory rim: scatter lymphocytes near necrosis border
        n_rim_lymph = max(3, int(np.sqrt(rx * ry) * 0.8))
        for _ in range(n_rim_lymph):
            la = nrng.uniform(0, 2 * np.pi)
            lr = nrng.uniform(0.7, 1.15)  # just inside/outside boundary
            lx = cx + lr * rx * np.cos(la) * np.cos(angle) - \
                 lr * ry * np.sin(la) * np.sin(angle)
            ly = cy + lr * rx * np.cos(la) * np.sin(angle) + \
                 lr * ry * np.sin(la) * np.cos(angle)
            lr_px = max(1, int(nrng.uniform(1.5, 2.5) * s))
            lymph_c = np.clip(
                HE_NUCLEUS_DARK + nrng.uniform(-8, 8, 3), 0, 255
            )
            cv2.circle(img, (self._s(lx), self._s(ly)), lr_px,
                       lymph_c.tolist(), -1, cv2.LINE_AA)

    def _draw_cytoplasm_rgb(self, img, idx):
        """Draw cytoplasm halo around nucleus idx on RGB canvas.

        Real H&E shows visible pink cytoplasm around every nucleus.
        Elongated nuclei (fibroblasts) get spindle-shaped cytoplasm;
        round nuclei get rounder cytoplasm.
        """
        wx = self._nuclei_x[idx]
        wy = self._nuclei_y[idx]
        wr = self._nuclei_r[idx]
        elongation = self._nuclei_elongation[idx]
        angle = self._nuclei_angle[idx]

        # Cytoplasm is 1.8-2.5x nuclear radius, with per-cell jitter
        crng = np.random.default_rng(self._seed + 60000 + idx)
        cyto_scale = crng.uniform(1.8, 2.5)
        r_major = wr * elongation * cyto_scale
        r_minor = wr * cyto_scale

        contour = self._ellipse_contour(
            wx, wy, r_major, r_minor, angle,
            noise_amp=0.10, n_points=24, seed=60000 + idx,
        )

        jitter = crng.uniform(-6, 6, 3).astype(np.float32)
        color = np.clip(HE_CYTOPLASM + jitter, 0, 255)
        cv2.fillPoly(img, [contour], color.tolist())

    def _draw_nucleus_rgb(self, img, idx):
        """Draw a hematoxylin-stained nucleus with irregular boundary on RGB canvas."""
        x = self._sf(self._nuclei_x[idx])
        y = self._sf(self._nuclei_y[idx])
        r = self._sf(self._nuclei_r[idx])
        intensity = self._nuclei_intensity[idx]
        elongation = self._nuclei_elongation[idx]
        angle = self._nuclei_angle[idx]
        is_mitotic = (
            self._nuclei_mitotic[idx]
            if idx < len(self._nuclei_mitotic)
            else False
        )

        # Map grayscale intensity (30-90) → hematoxylin color blend
        t = np.clip((intensity - 30) / 60, 0, 1)
        base_color = HE_NUCLEUS_DARK * (1 - t) + HE_NUCLEUS_LIGHT * t
        jitter = self._rng.uniform(-8, 8, 3).astype(np.float32)

        # World-coord nucleus params for contour generation
        wx = self._nuclei_x[idx]
        wy = self._nuclei_y[idx]
        wr = self._nuclei_r[idx]
        r_major = wr * elongation
        r_minor = wr

        contour = self._ellipse_contour(
            wx, wy, r_major, r_minor, angle,
            noise_amp=0.15 if not is_mitotic else 0.20,
            n_points=32, seed=40000 + idx,
        )

        if is_mitotic:
            mc = np.clip(HE_MITOTIC + jitter, 0, 255)
            cv2.fillPoly(img, [contour], mc.tolist())
            # Condensed chromosome blobs
            for _ in range(3):
                sa = self._rng.uniform(0, 2 * np.pi)
                sr = r * self._rng.uniform(0.3, 0.6)
                sx, sy = x + sr * np.cos(sa), y + sr * np.sin(sa)
                bc = np.clip(HE_MITOTIC * 0.75 + jitter, 0, 255)
                cv2.circle(img, (int(sx), int(sy)), max(1, int(r * 0.35)),
                           bc.tolist(), -1, cv2.LINE_AA)
        else:
            nc = np.clip(base_color + jitter, 0, 255)
            cv2.fillPoly(img, [contour], nc.tolist())
            crng = np.random.default_rng(self._seed + 50000 + idx)

            # Peripheral chromatin condensation (dark rim along nuclear envelope)
            # — the most recognizable feature of real euchromatic nuclei
            if r > 2 * self.internal_scale:
                rim_c = np.clip(nc * 0.72, 0, 255)
                rim_t = max(1, int(r * 0.15))
                cv2.ellipse(img, (int(x), int(y)),
                            (max(1, int(r * elongation - rim_t * 0.5)),
                             max(1, int(r - rim_t * 0.5))),
                            int(np.degrees(angle)), 0, 360,
                            rim_c.tolist(), rim_t, cv2.LINE_AA)

            # Chromatin clumps: 2-4 darker patches within nucleus
            n_clumps = crng.integers(2, 5)
            for _ in range(n_clumps):
                cr = r * crng.uniform(0.1, 0.6)
                ca = crng.uniform(0, 2 * np.pi)
                ccx = int(x + cr * np.cos(ca))
                ccy = int(y + cr * np.sin(ca))
                clump_r = max(1, int(r * crng.uniform(0.12, 0.30)))
                clump_c = np.clip(nc * 0.78 + crng.uniform(-3, 3, 3), 0, 255)
                cv2.circle(img, (ccx, ccy), clump_r,
                           clump_c.tolist(), -1, cv2.LINE_AA)
            # Nucleolus (very dark, prominent)
            if crng.random() < self._grade_profile["nucleolus_pct"] and r > 3 * self.internal_scale:
                nx_off = crng.normal(0, r * 0.2)
                ny_off = crng.normal(0, r * 0.2)
                nlc = np.clip(HE_NUCLEOLUS + jitter, 0, 255)
                cv2.circle(img, (int(x + nx_off), int(y + ny_off)),
                           max(1, int(r * 0.25)), nlc.tolist(), -1, cv2.LINE_AA)

    def _render_nuclei_only(self):
        """Render hematoxylin channel (nuclei only) at internal resolution."""
        iw, ih = self._iw, self._ih
        img = np.full((ih, iw), 240, dtype=np.float32)

        for i in range(len(self._nuclei_x)):
            self._draw_nucleus(img, i)

        img = np.clip(img, 0, 255)
        img = 255 - img

        # Glass slide outside tissue boundary (black = no nuclei)
        if self._tissue_mask is not None:
            mask_int = cv2.resize(
                self._tissue_mask, (iw, ih), interpolation=cv2.INTER_LINEAR
            )
            img[mask_int < 128] = 0

        return img.astype(np.uint8)

    def _render_eosin_only(self):
        """Render eosin channel (stroma/cytoplasm only) at internal resolution."""
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        lt = max(1, s // 2)
        rng = np.random.default_rng(self._seed + 999)

        base = TISSUE_TYPES.get(
            self.tissue_type if self.tissue_type != "mixed" else "connective", {}
        )
        bg_val = base.get("stroma_intensity", 195)

        img = np.full((ih, iw), bg_val, dtype=np.float32)
        noise = rng.normal(0, 8, (self.height, self.width)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (11, 11), 3.0)
        if s > 1:
            noise = cv2.resize(noise, (iw, ih), interpolation=cv2.INTER_LINEAR)
        img += noise

        # Collagen fibers
        self._draw_collagen_fibers(img, rgb=False)

        # Epithelial bands
        for y_base, thickness, x_start, x_end in self._epithelia:
            y0 = max(0, self._s(y_base - thickness / 2))
            y1 = min(ih, self._s(y_base + thickness / 2))
            img[y0:y1, self._s(x_start) : self._s(x_end)] = bg_val - 15

        # Glands (same contours as RGB via deterministic seed)
        for gi, (cx, cy, outer_r, inner_r, _) in enumerate(self._glands):
            outer = self._circle_contour(cx, cy, outer_r, noise_amp=0.12,
                                         n_points=64, seed=10000 + gi)
            inner = self._circle_contour(cx, cy, inner_r, noise_amp=0.25,
                                         n_points=64, seed=10500 + gi)
            cv2.fillPoly(img, [outer], float(rng.uniform(170, 185)))
            cv2.fillPoly(img, [inner], float(rng.uniform(230, 245)))

        # Adipocytes
        for ai, (cx, cy, r) in enumerate(self._adipocytes):
            inner_c = self._circle_contour(cx, cy, r - 1, noise_amp=0.08,
                                           n_points=48, seed=30000 + ai)
            outer_c = self._circle_contour(cx, cy, r, noise_amp=0.08,
                                           n_points=48, seed=30000 + ai)
            cv2.fillPoly(img, [inner_c], float(rng.uniform(225, 240)))
            cv2.polylines(img, [outer_c], True,
                          float(rng.uniform(140, 160)), lt)

        # Vessels
        for vi, (vx, vy, vr) in enumerate(self._vessels):
            lumen_c = self._circle_contour(vx, vy, vr, noise_amp=0.15,
                                           n_points=48, seed=20000 + vi)
            cv2.fillPoly(img, [lumen_c], float(rng.uniform(120, 145)))

        # --- Tissue boundary: glass slide outside ---
        if self._tissue_mask is not None:
            mask_int = cv2.resize(
                self._tissue_mask, (iw, ih), interpolation=cv2.INTER_LINEAR
            )
            glass = mask_int < 128
            img[glass] = 245  # glass slide (near-white)

        return np.clip(img, 0, 255).astype(np.uint8)

    # -- SimulationBridge interface --

    def _render_for_mode(self, mode):
        if mode == 1:
            return self._nuc_full
        elif mode == 2:
            return self._eos_full
        return self._bf_full

    def _apply_exposure(self, viewport, exposure, intensity):
        """Histology is a stained slide — no exposure scaling."""
        return viewport

    def _finalize_output(self, viewport):
        """Return RGB for H&E stain."""
        if viewport.ndim == 2:
            return np.stack([viewport, viewport, viewport], axis=-1)
        return viewport

    # -- Ground truth --

    def _visible_region_world(self):
        """Get visible region in world coordinates for current objective."""
        ox = int(self.camera_offset[0])
        oy = int(self.camera_offset[1])
        obj = self.current_objectiv

        # Stage center in world coords
        cx = ox + self.viewport_width // 2
        cy = oy + self.viewport_height // 2

        # FOV in world units
        fov = self._FOV_MAP.get(obj, 512)

        half = fov // 2
        vx0 = max(0, cx - half)
        vy0 = max(0, cy - half)
        return vx0, vy0, fov, fov

    def get_ground_truth(self):
        """Return ground truth for grading."""
        n_total = len(self._nuclei_x)

        vx0, vy0, vw, vh = self._visible_region_world()

        visible = (
            (self._nuclei_x >= vx0)
            & (self._nuclei_x < vx0 + vw)
            & (self._nuclei_y >= vy0)
            & (self._nuclei_y < vy0 + vh)
        )
        n_visible = int(visible.sum())

        areas = np.pi * self._nuclei_r**2 * self._nuclei_elongation
        mean_area = float(areas.mean()) if n_total > 0 else 0.0

        n_mitotic = (
            int(self._nuclei_mitotic.sum())
            if len(self._nuclei_mitotic) > 0
            else 0
        )

        # Pleomorphism score: coefficient of variation of nuclear area
        if n_total > 1:
            area_cv = float(areas.std() / max(areas.mean(), 1e-6))
        else:
            area_cv = 0.0

        # Mitotic index: mitoses per total nuclei
        mitotic_index = n_mitotic / max(1, n_total)

        result = {
            "tissue_type": self.tissue_type,
            "grade": self.grade,
            "n_nuclei": n_total,
            "n_visible": n_visible,
            "n_mitotic": n_mitotic,
            "mitotic_index": round(mitotic_index, 4),
            "n_glands": len(self._glands),
            "n_vessels": len(self._vessels),
            "n_adipocytes": len(self._adipocytes),
            "n_epithelia": len(self._epithelia),
            "mean_nuclear_area_px": round(mean_area, 1),
            "nuclear_area_cv": round(area_cv, 3),
            "n_necrosis_patches": len(self._necrosis_patches),
            "n_lymphocytes": getattr(self, "_n_lymphocytes", 0),
        }

        if self._regions:
            result["regions"] = [
                (t, (int(b[0]), int(b[1]), int(b[2]), int(b[3])))
                for t, b in self._regions
            ]

        return result

    def get_nuclear_morphometry(self):
        """Return detailed nuclear morphometry for the current viewport."""
        vx0, vy0, vw, vh = self._visible_region_world()
        obj = self.current_objectiv

        zoom = {10: 1, 20: 2, 40: 4}.get(obj, 1)

        visible = (
            (self._nuclei_x >= vx0)
            & (self._nuclei_x < vx0 + vw)
            & (self._nuclei_y >= vy0)
            & (self._nuclei_y < vy0 + vh)
        )
        vis_idx = np.where(visible)[0]

        # Classify: epithelial = near glands or epithelial bands
        is_epithelial = np.zeros(len(self._nuclei_x), dtype=bool)

        for cx, cy, outer_r, inner_r, n_cells in self._glands:
            dist = np.hypot(self._nuclei_x - cx, self._nuclei_y - cy)
            ring = (dist >= inner_r - 2) & (dist <= outer_r + 5)
            is_epithelial |= ring

        for y_base, thickness, x_start, x_end in self._epithelia:
            in_band = (
                (np.abs(self._nuclei_y - y_base) < thickness * 0.7)
                & (self._nuclei_x >= x_start)
                & (self._nuclei_x <= x_end)
            )
            is_epithelial |= in_band

        world_areas = np.pi * self._nuclei_r**2 * self._nuclei_elongation
        rendered_areas = world_areas * (zoom**2)

        vis_areas = rendered_areas[vis_idx]
        vis_elong = self._nuclei_elongation[vis_idx]
        vis_epi = is_epithelial[vis_idx]
        vis_mitotic = self._nuclei_mitotic[vis_idx]

        n_vis = len(vis_idx)
        epi_mask = vis_epi
        stro_mask = ~vis_epi

        result = {
            "n_visible": n_vis,
            "objective": obj,
            "zoom": zoom,
            "mean_area_px": float(vis_areas.mean()) if n_vis > 0 else 0.0,
            "std_area_px": float(vis_areas.std()) if n_vis > 0 else 0.0,
            "mean_elongation": float(vis_elong.mean()) if n_vis > 0 else 0.0,
            "fraction_elongated": (
                float((vis_elong > 1.5).sum() / n_vis) if n_vis > 0 else 0.0
            ),
            "n_mitotic_visible": int(vis_mitotic.sum()),
            "n_epithelial": int(epi_mask.sum()),
            "n_stromal": int(stro_mask.sum()),
        }

        if epi_mask.sum() > 0:
            result["mean_area_epithelial"] = float(vis_areas[epi_mask].mean())
            result["mean_elong_epithelial"] = float(vis_elong[epi_mask].mean())
        if stro_mask.sum() > 0:
            result["mean_area_stromal"] = float(vis_areas[stro_mask].mean())
            result["mean_elong_stromal"] = float(vis_elong[stro_mask].mean())

        return result
