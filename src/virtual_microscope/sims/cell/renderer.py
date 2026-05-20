"""Optimize rendering using OpenCV"""
import numpy as np
import cv2
from typing import List, Tuple, Sequence
from .cell import CellBase
from .cycle import CellCycleNormal
from .chromatin import (
    draw_smooth_chromatin, draw_condensed_chromatin,
    draw_condensed_chromatin_equator, draw_condensed_chromatin_polar,
)
from .apoptosis import draw_apoptosis_phase


class CellCycleRenderer:
    """Fast renderer using OpenCV."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width # image dimension - width
        self.height = height # image dimension - height
        self.contrast = 0.55
        self.brightness = 0.78
        self.margins = 100

        self.dof_map = {10: 6.0, 20: 3.0, 40: 1.5} # fine-tune if necessary
        self.objective = 10 # objective value
        self.dof = self.dof_map[self.objective] # fine-tune if necessary

        # Visual Limits
        self.max_blur_radius = 25 # max Gaussian kernel radius, test if visually looks fine
        self.min_opacity = 0.2 # min visibility when far out focus test if visually looks fine

        # render image according to the objective used
        self.crop_dim = 512 # by default 10x crop
        # create master shape for the chromosome
        self.master_shape = np.array([[-1.25,-5], [0,-1.25], [1.25,-5], [1.25,5], [0,1.25], [-1.25,5]], dtype=np.float32)

        # Pre-allocated per-cell scratch buffer (avoids N allocations per frame)
        self._cell_buf = np.zeros((height, width, 3), dtype=np.uint8)

        # Cached substrate background (recomputed only when camera moves)
        self._substrate_cache_key = None  # (ox, oy) of last computation
        self._substrate_cache = None


    def render_cells(self, cells: List[CellBase], mode: int = 0,
                     camera_offset: Tuple[float, float] = (0,0),
                     focal_plane: float = 0.0) -> np.ndarray:
        """Render cells to image array using OpenCV."""
        # Create base image with substrate texture
        img = self._make_substrate_background(camera_offset)

        # Get visibile cells
        visible_cells = self._get_visible_cells(cells, camera_offset)

        for cell in visible_cells:
            self._draw_cell(img, cell, mode, camera_offset, focal_plane)

        # Apply brightfield contrast/brightness adjustments
        if mode == 0:
            img = self._apply_brightfield_look(img)

        # crop and rescale base on the objective used
        img = self._crop_and_rescale(img)

        return img

    def render_cell_cycle(self, cells: Sequence[CellCycleNormal], mode: int = 0,
                          camera_offset: Tuple[float, float] = (0,0),
                          focal_plane: float = 0.0):
        """Render cell cycle to image array using OpenCV"""
        # Create base image with substrate texture
        img = self._make_substrate_background(camera_offset)

        # Get visibile cells
        visible_cells = self._get_visible_cells(cells, camera_offset)

        for cell in visible_cells:
            self._draw_cell_cycle(img, cell, camera_offset, focal_plane, mode)  # type: ignore

        # Apply brightfield contrast/brightness adjustments
        if mode == 0:
            img = self._apply_brightfield_look(img)

        # crop and rescale based on the objective used
        img = self._crop_and_rescale(img)

        return img

    def _get_visible_cells(self, cells: Sequence[CellBase | CellCycleNormal], camera_offset: Tuple[float, float]) -> Sequence[CellBase | CellCycleNormal]:
        """Filter cells visible in current viewport"""

        margin = self.margins
        vx, vy = camera_offset

        viewport_width = self.width # always leave viewport dimension to 512
        viewport_height = self.height # always leave viewport dimension to 512

        visible = []
        for cell in cells:
            # only append those cell in or near viewport
            if (vx - margin <= cell.center[0] <= vx + viewport_width + margin and
                vy - margin <= cell.center[1] <= vy + viewport_height + margin):
                visible.append(cell)

        return visible

    def _draw_smooth_cell(self, img: np.ndarray, center: np.ndarray,
                         vertices: np.ndarray,color: tuple,
                         thickness: int = -1) -> None:
        """Draw a smooth cell shape using cubic spline interpolation for rounder borders."""
        n_interp = 200  # Increased for even smoother appearance
        angles_interp = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)

        # Get original angles and radii
        angles_orig = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
        radii_orig = np.linalg.norm(vertices - center, axis=1)

        # Duplicate first point at the end to handle periodic boundary for spline
        angles_extended = np.append(angles_orig, angles_orig[0] + 2 * np.pi)
        radii_extended = np.append(radii_orig, radii_orig[0])

        # Use cubic spline interpolation for much rounder, smoother curves
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(angles_extended, radii_extended, bc_type='periodic')
        radii_interp = cs(angles_interp)

        # Ensure radii stay positive
        radii_interp = np.maximum(radii_interp, 0.01)

        # Generate smooth vertices
        smooth_verts = np.zeros((n_interp, 2))
        smooth_verts[:, 0] = center[0] + radii_interp * np.cos(angles_interp)
        smooth_verts[:, 1] = center[1] + radii_interp * np.sin(angles_interp)
        smooth_pts = smooth_verts.astype(np.int32)

        # Draw the smooth polygon
        if thickness == -1: # filled
            cv2.fillPoly(img, [smooth_pts], color, lineType=cv2.LINE_AA)  # LINE_AA for anti-aliasing
        else:
            cv2.polylines(img, [smooth_pts], True, color, thickness, lineType=cv2.LINE_AA)


    def _draw_cell_cycle(self, img: np.ndarray, cell: CellCycleNormal,
                         camera_offset: Tuple[float, float], focal_plane: float,
                         mode: int = 0) -> None:
        """Draw a single cell in cell cycle using OpenCV with proper focal plane effect."""
        # compute blur and opacity depth of field and z-distance
        kernel_size, opacity = self._compute_blur_and_opacity(cell.z_position, focal_plane)

        # Get vertex position adjusted for camera (convert to screen space)
        vertices = (cell.vertices_positions - camera_offset)
        center_screen = (cell.center - camera_offset)

        # Convert chromatin points to screen space (consistent with vertices)
        chromatin_pts_screen = [(pt[0] - camera_offset[0], pt[1] - camera_offset[1]) for pt in cell.chromatin_pts]

        # Adjust cell radius for zoom
        cell_radius = cell.base_r

        # Skip if center is outside viewport
        if (center_screen[0] < -self.margins or center_screen[0] > self.width + self.margins or
            center_screen[1] < -self.margins or center_screen[1] > self.height + self.margins):
            return

        # Handle apoptosis rendering
        if cell.is_dying:
            draw_apoptosis_phase(
                img, cell, center_screen, vertices, chromatin_pts_screen,
                camera_offset, opacity, kernel_size,
                self.height, self.width, self._draw_smooth_cell,
                self.master_shape)
            return

        # Dispatch to mode-specific renderer
        if mode == 1:
            self._draw_cell_cycle_nucleus(img, cell, center_screen, vertices, chromatin_pts_screen, cell_radius, opacity, kernel_size, camera_offset)
        elif mode == 2:
            self._draw_cell_cycle_membrane(img, cell, center_screen, vertices, cell_radius, opacity, kernel_size)
        else:
            self._draw_cell_cycle_brightfield(img, cell, center_screen, vertices, chromatin_pts_screen, cell_radius, opacity, kernel_size, camera_offset)

    def _draw_cell_cycle_brightfield(self, img, cell, center_screen, vertices,
                                      chromatin_pts_screen, cell_radius, opacity,
                                      kernel_size, camera_offset):
        """Brightfield: gradient body fill + dark nucleus + chromatin."""
        cell_img = self._cell_buf
        cell_img[:] = 0
        layers = 10

        for i in range(layers, 0, -1):
            s = i / layers
            shade = 80 + int(100 * s)
            color = (shade, shade, 255)
            scaled_verts = center_screen + (vertices - center_screen) * s
            self._draw_smooth_cell(cell_img, center_screen, scaled_verts, color, thickness=-1)

        self._draw_smooth_cell(cell_img, center_screen, vertices, (0, 0, 0), thickness=2)

        nucleus_pos = tuple(center_screen.astype(int))
        nucleus_radius = int(0.4 * cell_radius)

        center_for_chromatin = np.array(cell.center) - np.array(camera_offset)

        if cell.cell_mitosis_state == 'Interphase' and cell.cell_cycle_state == 'G1':
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (240, 240, 240), -1, lineType=cv2.LINE_AA)
        elif cell.cell_mitosis_state == 'Interphase' and cell.cell_cycle_state == 'S':
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (240, 240, 240), -1, lineType=cv2.LINE_AA)
        elif cell.cell_mitosis_state == 'Interphase' and cell.cell_cycle_state == 'G2':
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (240, 240, 240), -1, lineType=cv2.LINE_AA)
        elif cell.cell_mitosis_state == 'Prophase':
            draw_condensed_chromatin(cell_img, center_for_chromatin, cell.base_r,
                                    (0, 0), self.master_shape, num_chromosomes=10)
        elif cell.cell_mitosis_state == 'Metaphase':
            draw_condensed_chromatin_equator(cell_img, center_for_chromatin, cell.base_r,
                                            self.master_shape, num_chromosomes=10)
        elif cell.cell_mitosis_state == 'Anaphase':
            draw_condensed_chromatin_polar(cell_img, center_for_chromatin, cell.base_r,
                                          self.master_shape, num_chromosomes=10)
        elif cell.cell_mitosis_state == 'Telophase':
            draw_condensed_chromatin_polar(cell_img, center_for_chromatin, cell.base_r,
                                          self.master_shape, num_chromosomes=10)
            pole_offset = int(cell_radius * 0.5)
            cv2.circle(cell_img, (nucleus_pos[0], nucleus_pos[1] - pole_offset),
                       int(nucleus_radius * 0.7), (240, 240, 240), -1, lineType=cv2.LINE_AA)
            cv2.circle(cell_img, (nucleus_pos[0], nucleus_pos[1] + pole_offset),
                       int(nucleus_radius * 0.7), (240, 240, 240), -1, lineType=cv2.LINE_AA)
        elif cell.cell_mitosis_state == 'Cytokinesis':
            # Nuclear envelope has reformed — show two decondensed nuclei, not condensed chromosomes
            pole_offset = int(cell_radius * 0.5)
            nuc_r = int(0.4 * cell_radius * 0.7)
            cv2.circle(cell_img, (nucleus_pos[0], nucleus_pos[1] - pole_offset),
                       nuc_r, (240, 240, 240), -1, lineType=cv2.LINE_AA)
            cv2.circle(cell_img, (nucleus_pos[0], nucleus_pos[1] + pole_offset),
                       nuc_r, (240, 240, 240), -1, lineType=cv2.LINE_AA)

        if kernel_size > 0:
            cell_img = cv2.GaussianBlur(cell_img, (kernel_size, kernel_size), kernel_size / 3.0)
        img[:] = cv2.addWeighted(img, 1.0, cell_img, opacity, 0)

    def _draw_cell_cycle_nucleus(self, img, cell, center_screen, vertices,
                                  chromatin_pts_screen, cell_radius, opacity,
                                  kernel_size, camera_offset):
        """Nucleus fluorescence: bright chromatin/nucleus on black background.

        Mimics DAPI/Hoechst staining — compact bright spots where DNA is.
        Interphase: diffuse nucleus glow. Mitotic: bright condensed chromosomes.
        """
        cell_img = self._cell_buf
        cell_img[:] = 0
        nucleus_pos = tuple(center_screen.astype(int))
        nucleus_radius = int(0.4 * cell_radius)
        # DAPI-like blue-white color
        nuc_color = (255, 200, 120)  # BGR: bright blue with some green

        center_for_chromatin = np.array(cell.center) - np.array(camera_offset)

        if cell.cell_mitosis_state == 'Interphase':
            # Diffuse nucleus glow — interphase chromatin is decondensed
            intensity = 160 if cell.cell_cycle_state == 'G1' else 200  # S/G2 brighter (more DNA)
            nuc_col = (intensity, int(intensity * 0.75), int(intensity * 0.45))
            cv2.circle(cell_img, nucleus_pos, nucleus_radius, nuc_col, -1, lineType=cv2.LINE_AA)
            # Subtle chromatin texture inside
            for pt in chromatin_pts_screen[:20]:
                pt_int = (int(pt[0]), int(pt[1]))
                cv2.circle(cell_img, pt_int, 1, nuc_color, -1, lineType=cv2.LINE_AA)
        elif cell.cell_mitosis_state == 'Prophase':
            # Condensing chromosomes — brighter, speckled
            draw_condensed_chromatin(cell_img, center_for_chromatin, cell.base_r,
                                    (0, 0), self.master_shape, num_chromosomes=10)
            # Brighten the chromatin spots
            mask = cell_img[:, :, 0] > 0
            cell_img[mask] = [255, 200, 120]
        elif cell.cell_mitosis_state == 'Metaphase':
            # Bright metaphase plate — aligned chromosomes
            draw_condensed_chromatin_equator(cell_img, center_for_chromatin, cell.base_r,
                                            self.master_shape, num_chromosomes=10)
            mask = cell_img[:, :, 0] > 0
            cell_img[mask] = [255, 220, 140]
        elif cell.cell_mitosis_state == 'Anaphase':
            # Separated chromosome masses moving to poles
            draw_condensed_chromatin_polar(cell_img, center_for_chromatin, cell.base_r,
                                          self.master_shape, num_chromosomes=10)
            mask = cell_img[:, :, 0] > 0
            cell_img[mask] = [255, 200, 120]
        elif cell.cell_mitosis_state in ('Telophase', 'Cytokinesis'):
            # Two reforming nuclei with decondensed chromatin (nuclear envelope has reformed)
            pole_offset = int(cell_radius * 0.5)
            nuc_r = int(nucleus_radius * 0.7)
            cv2.circle(cell_img, (nucleus_pos[0], nucleus_pos[1] - pole_offset),
                       nuc_r, nuc_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(cell_img, (nucleus_pos[0], nucleus_pos[1] + pole_offset),
                       nuc_r, nuc_color, -1, lineType=cv2.LINE_AA)

        if kernel_size > 0:
            cell_img = cv2.GaussianBlur(cell_img, (kernel_size, kernel_size), kernel_size / 3.0)
        img[:] = cv2.addWeighted(img, 1.0, cell_img, opacity, 0)

    def _draw_cell_cycle_membrane(self, img, cell, center_screen, vertices,
                                   cell_radius, opacity, kernel_size):
        """Membrane fluorescence: thin bright outline on black background.

        Mimics CellMask/DiI staining — bright plasma membrane boundary only.
        """
        cell_img = self._cell_buf
        cell_img[:] = 0
        # Membrane marker color (red-ish, like miRFP670)
        mem_color = (80, 80, 255)  # BGR: red

        # Draw membrane outline only (thin line, not filled)
        self._draw_smooth_cell(cell_img, center_screen, vertices, mem_color, thickness=2)

        if kernel_size > 0:
            cell_img = cv2.GaussianBlur(cell_img, (kernel_size, kernel_size), kernel_size / 3.0)
        img[:] = cv2.addWeighted(img, 1.0, cell_img, opacity, 0)


    def _draw_cell(self, img: np.ndarray, cell: CellBase, mode: int,
                   camera_offset: Tuple[float, float], focal_plane: float) -> None:
        """Draw a single cell using OpenCV with proper focal plane effect."""
        # compute blur and opacity using depth of field and z-distance
        kernel_size, opacity = self._compute_blur_and_opacity(cell.z_position, focal_plane)

        # Get vertex postion adjusted for camera
        vertices = (cell.vertices_positions - camera_offset)
        center_screen = (cell.center - camera_offset)

        # Adjust cell radius for zoom
        cell_radius = cell.base_r

        # Skip if center is outside viewport
        if (center_screen[0] < -self.margins or center_screen[0] > self.width + self.margins or
            center_screen[1] < -self.margins or center_screen[1] > self.height + self.margins):
            return

        # fluorescence mode
        if mode == 0: # brightfield
            # reuse pre-allocated scratch buffer
            cell_img = self._cell_buf
            cell_img[:] = 0
            layers = 10 #6 # numbers of layers

            for i in range(layers, 0, -1):
                s = i / layers
                shade = 80 + int(100 * s)
                color = (shade, shade, 255)

                scaled_verts = center_screen + (vertices - center_screen) * s

                self._draw_smooth_cell(cell_img, center_screen, scaled_verts,
                                       color, thickness=-1)

            self._draw_smooth_cell(cell_img, center_screen, vertices,
                                (0, 0, 0), thickness=2)

            # Draw nucleus — bright color inverts to dark in _apply_brightfield_look
            nucleus_pos = tuple(center_screen.astype(int))
            nucleus_radius = int(0.4 * cell_radius)

            cv2.circle(cell_img, nucleus_pos, nucleus_radius, (240, 240, 240), -1, lineType=cv2.LINE_AA)

            # Apply blur based on focal plane
            if kernel_size > 0:
                cell_img = cv2.GaussianBlur(cell_img,
                                            (kernel_size, kernel_size),
                                            kernel_size / 3.0)

            cell_opacity = opacity * 1.0
            img[:] = cv2.addWeighted(img, 1.0, cell_img, cell_opacity, 0)

        elif mode == 1: # nucleus fluorescence
            if cell.nucleus_fluorescence > 0:
                # reuse pre-allocated scratch buffer
                fluor_img = self._cell_buf
                fluor_img[:] = 0

                nucleus_pos = tuple(center_screen.astype(int))
                nucleus_radius = int(0.55 * cell_radius)

                # Drawing glowing nucleus with proper color
                fluorescence_intensity = cell.nucleus_fluorescence * opacity

                # Multiple layers for glow effect
                for i in range(5, 0, -1):
                    radius = int(nucleus_radius * (1.0 + 0.2 * (5 - i)))
                    intensity = int(255 * fluorescence_intensity * (i / 5.0))
                    # red fluorescence mScarlet - BGR format
                    color = (8, 8, intensity)
                    cv2.circle(fluor_img, nucleus_pos, radius, color,
                                -1, lineType=cv2.LINE_AA)

                # Add brighter center spot
                cv2.circle(fluor_img, nucleus_pos, int(nucleus_radius * 0.6),
                           (8, 8, 255), -1, lineType=cv2.LINE_AA)

                base_blur = 21
                if kernel_size > 0:
                    # Add extra blur for out of ocus
                    total_blur = base_blur + (kernel_size * 2)
                    total_blur = total_blur if total_blur % 2 == 1 else total_blur + 1
                    fluor_img = cv2. GaussianBlur(fluor_img, (total_blur, total_blur),
                                                  kernel_size * 0.8)
                else:
                    # Apply blur for glow
                    fluor_img = cv2.GaussianBlur(fluor_img, (base_blur,base_blur), 7)

                # Blend with main image
                img[:] = cv2.add(img, fluor_img)

        elif mode == 2: # membrane fluorescence
            if np.any(cell.membrane_fluorescence > 0):

                fluor_img = self._cell_buf
                fluor_img[:] = 0
                avg_fluorescence = np.mean(cell.membrane_fluorescence) * opacity

                membrane_color = (int(8 * avg_fluorescence),
                                  int(8 * avg_fluorescence),
                                  int(255 * avg_fluorescence)) # old 136 8 8

                self._draw_smooth_cell(fluor_img, center_screen, vertices, membrane_color, thickness=-1)

                base_blur = 7
                if kernel_size > 0:
                    # Add extra blur for out-of-focus
                    total_blur = base_blur + (base_blur * 2)
                    total_blur = total_blur if total_blur % 2 == 1 else total_blur + 1
                    fluor_img = cv2.GaussianBlur(fluor_img, (total_blur, total_blur),
                                                 kernel_size * 0.6)
                else:
                    # Apply blur effect for glow
                    fluor_img = cv2.GaussianBlur(fluor_img, (7, 7), 2)

                # Blend with main image
                img[:] = cv2.add(img, fluor_img)

    def _apply_brightfield_look(self, img: np.ndarray) -> np.ndarray:
        """Apply brightfield contrast and brightness adjustments.

        This is a rendering concern (visual tuning for BF appearance),
        not an optical effect — those are handled by OpticalPipeline.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        img = (img - 128.0) * self.contrast + 128.0
        img = np.clip(img, 0, 255)
        img = img * self.brightness
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = 255 - img  # invert: bright background (transmitted light), dark cells
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _make_substrate_background(self, camera_offset: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """Generate a subtle substrate texture for the culture surface.

        Uses position-seeded noise so the texture is spatially consistent
        (same world region always produces the same pattern).
        Result is cached — only recomputed when camera_offset changes.
        """
        ox, oy = int(camera_offset[0]), int(camera_offset[1])
        key = (ox, oy)
        if self._substrate_cache_key == key and self._substrate_cache is not None:
            return self._substrate_cache.copy()

        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.uint8)

        rng = np.random.RandomState(seed=(abs(ox * 7919 + oy * 104729)) % (2**31))

        # Low-frequency substrate pattern (large-scale texture)
        coarse = rng.randn(h // 16 + 2, w // 16 + 2).astype(np.float32)
        coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        coarse = cv2.GaussianBlur(coarse, (31, 31), 0)

        # Mid-frequency detail (scratches/dust on culture surface)
        mid = rng.randn(h // 4 + 2, w // 4 + 2).astype(np.float32)
        mid = cv2.resize(mid, (w, h), interpolation=cv2.INTER_CUBIC)
        mid = cv2.GaussianBlur(mid, (7, 7), 0)

        # Combine: subtle texture, amplitude ~3-5 gray levels
        texture = coarse * 2.0 + mid * 1.5
        texture = np.clip(texture, -6, 6).astype(np.int8)

        # Apply to all channels
        for ch in range(3):
            img[:, :, ch] = np.clip(texture.astype(np.int16), 0, 255).astype(np.uint8)

        self._substrate_cache_key = key
        self._substrate_cache = img
        return img.copy()

    def set_objective(self, mag: int, dof: float) -> None:
        """Set the current objective and its depth of field"""
        self.objective = mag # set magnitude objective
        self.dof = dof # set depth of field


    def _compute_blur_and_opacity(self, cell_z: float, focal_plane: float) -> Tuple[int, float]:
        """
        Convert a cell z-position and focal_plane (both in microns) to
        a blur kernel radius (odd integer kernel = 2*r+1) and opacity scalar.

        Returns (kernel_size, opacity) where kernel_size is 0 for no blur.

        """
        # Calculate focus effects based on z-distance from focal plane
        # z positions from focal plane
        z_dist_um = abs(cell_z - focal_plane)

        # Inside DOF -> sharp
        if z_dist_um <= self.dof:
            return 0, 1.0

        # Out of focus amount
        out_um = z_dist_um - self.dof

        # blur amoun increases with distance from focal plane
        # Map out_um to blur radius (pixel radius)
        # Tunable mapping: gentle growth near DOF, faster for larger distances
        # normalized by DOF to be objective-aware
        norm = out_um / max(1.0, self.dof * 4.0)
        radius = int(min(self.max_blur_radius, (norm ** 0.75) * self.max_blur_radius))

        # Ensure odd kernel = 2*radius+1 later; return radius as kernel radius
        if radius > 0:
            kernel = radius if radius % 2 == 1 else radius + 1
        else:
            kernel = 0

        # opacity decreases with distance
        opacity = max(self.min_opacity, 1.0 / (1.0 + 0.12 * (out_um / max(1.0, self.dof))))

        return kernel, float(opacity)

    def _crop_and_rescale(self, img: np.ndarray) -> np.ndarray:
        """
        Render the image according to the current objective selected.

        Higher magnification objectives crop a smaller region from the center
        of the base render and rescale to the full viewport size.

        The stage position represents the center of the FOV (parcentric):
            world = stage + (pixel - 256) * pixel_size_um
        where pixel_size_um = 10 / objective_mag.
        """
        # Compute crop size: higher mag → smaller crop
        if self.objective == 10:
            self.crop_dim = 512
            return img

        self.crop_dim = int(512 * (10 / self.objective))

        # Crop from center so switching objectives keeps the same FOV center
        mid = 256
        half = self.crop_dim // 2
        crop_img = img[mid - half:mid + half, mid - half:mid + half]

        # Rescale back to full viewport dimensions
        rescaled_img = cv2.resize(crop_img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return rescaled_img


    def _draw_cytokenesis_furrow(self, img: np.ndarray, cell: CellCycleNormal,
                                 center: np.ndarray, vertices: np.ndarray,
                                 furrow_depth: float = 0.3) -> None:
        """Draw the cleavage furrow during telophase/cytokenesis"""
        # Calculate perpendicular direction to division axis
        # For simplicity, assuming vertical division

        # Draw inward pinching at equator
        equator_y = center[1]
        left_bound = int(center[0] - np.max(np.abs(vertices[:, 0] - center[0])))
        right_bound = int(center[0] + np.max(np.abs(vertices[:, 0] - center[0])))

        # Deepening furrow as cytokenesis progress
        furrow_depth_px = int(cell.base_r * furrow_depth)

        # draw dark line to show cleavage
        cv2.line(img, (left_bound, int(equator_y)), (right_bound, int(equator_y)),
                 (50, 50, 50), 2, lineType=cv2.LINE_AA)

        # Draw subtle shading above and below furrow
        pts_top = np.array([
            [left_bound, int(equator_y - furrow_depth_px)],
            [right_bound, int(equator_y - furrow_depth_px)],
            [right_bound, int(equator_y)],
            [left_bound, int(equator_y)]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts_top], (40, 40, 80), lineType=cv2.LINE_AA)


    def _draw_separating_cells(self, img: np.ndarray, cell: CellCycleNormal,
                               center: np.ndarray, vertices: np.ndarray,
                               separation_progress: float = 0.5, camera_offset: Tuple[float, float] = (0, 0)) -> None:
        """Draw two separating cells during cytokenesis with connection bridge."""
        # separation_progress: 0.0 = fully connected, 1.0 = fully separated

        # Use original radius (unconstricted) to calculate proper daughter cell size
        # This avoids issues from cytokinesis constriction deforming the cells
        cell_radius = cell.original_base_r if hasattr(cell, 'original_base_r') else cell.base_r
        separation_distance = cell_radius * separation_progress

        # Top daughter cell center
        center_top = center + np.array([0, -separation_distance])

        # Bottom daughter cell center
        center_bottom = center + np.array([0, separation_distance])

        # Reconstruct vertices from original radius (avoid using constricted vertices)
        # This ensures daughter cells have proper circular shape, not pinched/deformed
        scale_factor = np.sqrt(0.5)  # Each daughter cell has ~half the area
        if hasattr(cell, 'original_r'):
            # Use original radii to get proper unconstricted shape
            angles = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
            reconstructed_radii = cell.original_r * scale_factor
            vertices_scaled = np.zeros_like(vertices)
            for i, angle in enumerate(angles):
                vertices_scaled[i] = np.array([
                    reconstructed_radii[i] * np.cos(angle),
                    reconstructed_radii[i] * np.sin(angle)
                ])
        else:
            # Fallback: use current vertices but scaled
            vertices_scaled = (vertices - center) * scale_factor

        # Draw top cell with blue gradient layers
        # Reduce brightness for daughter cells (two cells rendered = reduce to ~half brightness each)
        vertices_top = vertices_scaled + center_top
        layers = 10
        brightness_reduction = 0.6  # Reduce to 60% brightness to compensate for two cells
        for i in range(layers, 0, -1):
            s = i / layers
            shade = int((80 + int(100 * s)) * brightness_reduction)
            color = (shade, shade, int(255 * brightness_reduction))
            scaled_verts_top = (vertices_scaled - center_top) * (i / layers) + center_top
            self._draw_smooth_cell(img, center_top, scaled_verts_top, color, thickness=-1)
        self._draw_smooth_cell(img, center_top, vertices_top, (0, 0, 0), thickness=2)

        # draw nuclei with uncondensed chromatin
        nucleus_radius = int(0.4 * cell_radius * scale_factor)

        # Scale chromatin offsets (not absolute pts) for daughter cells
        chromatin_offset_scaled = [(offset[0] * scale_factor, offset[1] * scale_factor) for offset in cell.chromatin_offset]

        # Top nucleus (always drawn - it's in viewport)
        cv2.circle(img, tuple(center_top.astype(int)), nucleus_radius,
                   (150, 60, 60), -1, lineType=cv2.LINE_AA)
        # center_top is already in screen space, so don't subtract camera_offset again
        chromatin_top = [(center_top[0] + offset[0], center_top[1] + offset[1]) for offset in chromatin_offset_scaled]
        draw_smooth_chromatin(img, chromatin_top, num_strands=46)

        # Only draw bottom cell if it's within visible viewport bounds
        # Skip if center_bottom is outside viewport to avoid shadow artifacts
        if not (center_bottom[1] < -self.margins or center_bottom[1] > self.height + self.margins):
            vertices_bottom = vertices_scaled + center_bottom
            for i in range(layers, 0, -1):
                s = i / layers
                shade = int((80 + int(100 * s)) * brightness_reduction)
                color = (shade, shade, int(255 * brightness_reduction))
                scaled_verts_bottom = (vertices_scaled - center_bottom) * (i / layers) + center_bottom
                self._draw_smooth_cell(img, center_bottom, scaled_verts_bottom, color, thickness=-1)
            self._draw_smooth_cell(img, center_bottom, vertices_bottom, (0, 0, 0), thickness=2)

            # Bottom nucleus
            cv2.circle(img, tuple(center_bottom.astype(int)), nucleus_radius,
                       (150, 60, 60), -1, lineType=cv2.LINE_AA)
            # center_bottom is already in screen space, so don't subtract camera_offset again
            chromatin_bottom = [(center_bottom[0] + offset[0], center_bottom[1] + offset[1]) for offset in chromatin_offset_scaled]
            draw_smooth_chromatin(img, chromatin_bottom, num_strands=46)

            # Draw connecting bridge (narrowing as separation progresses)
            if separation_progress < 0.95:
                bridge_width = int(cell_radius * 0.3 * (1.0 - separation_progress))
                bridge_color = (120, 120, 180)

                top_connect = center_top + np.array([0, cell_radius * scale_factor])
                bottom_connect = center_bottom + np.array([0, -cell_radius * scale_factor])

                cv2.line(img, tuple(top_connect.astype(int)),
                        tuple(bottom_connect.astype(int)),
                        bridge_color, max(1, bridge_width), lineType=cv2.LINE_AA)
