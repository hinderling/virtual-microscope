# Optimized microscope simulation based on the nwe opto-loop implmentation
import numpy as np
from typing import Optional, List, Union, Literal, Sequence
from virtual_microscope.base import SimBase
from virtual_microscope.sims.cell.optogenetic import OptogeneticCell
from virtual_microscope.sims.cell.drug import DrugResponseCell
from virtual_microscope.sims.cell.renderer import CellCycleRenderer
from virtual_microscope.sims.cell.spatial_grid import SpatialGrid
from virtual_microscope.sims.cell.cell import CellBase, update_all_cells_parallel, check_collision
from virtual_microscope.sims.cell.normal import NormalCell
from virtual_microscope.sims.cell.cycle import CellCycleNormal
from virtual_microscope.sims.cell.cycle_manager import CellCycleManager
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline
import cv2


class ScatteredCellSim(SimBase):
    """Optmized microscope simulation with modular cell types."""

    continuous = True

    def __init__(self, width: int = 1500, height: int = 1500,
                 n_cells: int = 240, cell_type: str = "optogenetic",
                 viewport_width: int = 512, viewport_height: int = 512,
                 base_radius: float = 20.0, seed: int = 0,
                 cell_mix: Optional[dict] = None, concentration: float = 0.01, drug_type: Literal["growth", "mobility", "apoptosis"] = "growth",
                 brownian_d: float = 15.0):
        super().__init__(
            width=width, height=height,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=1,
            fixed_dt=0.005,
            mode_map={
                ("Electra1(402/454)", "CYAN"): 0,       # brightfield / DIC
                ("SCFP2(434/474)", "UV"): 1,            # DAPI
                ("mScarlet3(569/582)", "ORANGE"): 2,    # membrane
            },
        )

        self.n_cells = n_cells
        self.base_radius = base_radius
        self.cell_type = cell_type
        self.cell_mix = cell_mix
        self.concentration = concentration
        self.drug_type = drug_type
        self.brownian_d = brownian_d
        # Initialize components
        self.renderer = CellCycleRenderer(viewport_width, viewport_height)
        self.spatial_grid = SpatialGrid(width, height, base_radius * 3)

        # Per-channel optical pipelines (noise, PSF, vignette)
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.4, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 300),
            1: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 301),
            2: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 302),
        }

        # Initialize cells based on type
        self._rng = np.random.RandomState(seed)

        # Create cell objects
        self._cells = self._create_cells()
        self._resolve_initial_overlaps()
        self._init_numpy_arrays()
        self._step_count = 0

        # Initialize cell cycle manager if using cell cycle cells
        self.cycle_manager: Optional[CellCycleManager] = None
        if self.cell_type == "cycle":
            self.cycle_manager = CellCycleManager(max_divisions=10, track_stats=True)


    def _create_cells(self) -> List[Union[OptogeneticCell, DrugResponseCell, NormalCell, CellCycleNormal]]:
        """Create cells of specific type."""
        cells = []
        if self.cell_mix is None:

            for i in range(self.n_cells):
                seed = self._rng.randint(0, 10000)

                if self.cell_type == "optogenetic":
                    cell = OptogeneticCell(self.width, self.height, self.base_radius,vertices=24, seed=seed) # to check

                elif self.cell_type == "drug":
                    cell = DrugResponseCell(self.width, self.height, self.base_radius, vertices=24, seed=seed) # to check
                elif self.cell_type == "normal":
                    cell = NormalCell(self.width, self.height, self.base_radius, vertices = 24, seed= seed)
                elif self.cell_type == "mixed":
                    # Default mixed population
                        cell_choice = self._rng.choice(['normal', 'optogenetic'], p=[0.7, 0.3])
                        if cell_choice == 'normal':
                            cell = NormalCell(
                                self.width, self.height, self.base_radius,
                                vertices=24, seed=seed
                            )
                        else:
                            cell = OptogeneticCell(
                                self.width, self.height, self.base_radius,
                                vertices=24, seed=seed
                            )
                elif self.cell_type == "cycle":
                    cell = CellCycleNormal(self.width, self.height, self.base_radius, vertices=24, seed=seed, initial_state='M', initial_mitosis='Metaphase', initial_divisions=10, initial_time=496)
                else:
                    raise ValueError(f"Unknow celly type: {self.cell_type}")
                
                cells.append(cell)
        else:
            # cell_mix = {"normal": 60, "optogenetic": 40, "drug": 20}
            #total = sum(self.cell_mix.values())
            for cell_type, count in self.cell_mix.items():
                for _ in range(count):
                    seed = self._rng.randint(0, 10000)
                    
                    if cell_type == "normal":
                        cell = NormalCell(
                            self.width, self.height, self.base_radius,
                            vertices=24, seed=seed
                        )
                    elif cell_type == "optogenetic":
                        cell = OptogeneticCell(
                            self.width, self.height, self.base_radius,
                            vertices=24, seed=seed
                        )
                    elif cell_type == "drug":
                        cell = DrugResponseCell(
                            self.width, self.height, self.base_radius,
                            vertices=24, seed=seed
                        )
                    else:
                        continue
                    
                    cells.append(cell)
            
            # Shuffle cells for random positioning
            self._rng.shuffle(cells)
            #self.n_cells = len(cells)

        return cells

    def _resolve_initial_overlaps(self, min_gap: float = 2.0,
                                  max_attempts: int = 500) -> None:
        """Reposition cells so none overlap at initialisation.

        For each cell, rejection-sample a new position until it is at
        least ``base_r_i + base_r_j + min_gap`` from every previously
        placed cell (with periodic wrapping).  Falls back to the
        original position if *max_attempts* is exhausted.
        """
        placed: list[tuple[np.ndarray, float]] = []  # (center, base_r)
        w, h = self.width, self.height

        for cell in self._cells:
            for _ in range(max_attempts):
                candidate = np.array([
                    self._rng.uniform(0, w),
                    self._rng.uniform(0, h),
                ], dtype=np.float64)

                ok = True
                for other_center, other_r in placed:
                    dvec = candidate - other_center
                    dvec[0] -= w * round(dvec[0] / w)
                    dvec[1] -= h * round(dvec[1] / h)
                    dist = np.sqrt(dvec[0] ** 2 + dvec[1] ** 2)
                    if dist < cell.base_r + other_r + min_gap:
                        ok = False
                        break

                if ok:
                    cell.center = candidate
                    break

            placed.append((cell.center.copy(), cell.base_r))

    def _init_numpy_arrays(self):
        """Initialize numpy arrays from cell objects for fast physics."""
        self.centers = np.zeros((self.n_cells, 2), dtype=np.float64)
        self.velocities = np.zeros((self.n_cells, 2), dtype=np.float64)
        self.radii = np.zeros((self.n_cells, 24), dtype=np.float64)
        self.base_radii = np.zeros(self.n_cells, dtype=np.float64)
        self.areas = np.zeros(self.n_cells, dtype=np.float64)
        self.angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        
        # Copy data from cell objects to arrays
        for i, cell in enumerate(self._cells):
            self.centers[i] = cell.center
            self.velocities[i] = cell.vel
            self.radii[i] = cell.r
            self.base_radii[i] = cell.base_r
            self.areas[i] = cell.area0

    def _sync_arrays_to_cells(self):
        """Sync numpy arrays back to cell objects."""
        for i, cell in enumerate(self._cells):
            cell.center = self.centers[i].copy()
            cell.vel = self.velocities[i].copy()
            cell.r = self.radii[i].copy()
    
    def _resync_arrays_after_division(self):
        """Rebuild numpy arrays when cell count changes due to divisions/apoptosis."""
        n_cells = len(self._cells)
        
        # Only rebuild if size changed
        if n_cells != len(self.centers):
            # Create new arrays with new size
            self.centers = np.zeros((n_cells, 2), dtype=np.float64)
            self.velocities = np.zeros((n_cells, 2), dtype=np.float64)
            self.radii = np.zeros((n_cells, 24), dtype=np.float64)
            self.base_radii = np.zeros(n_cells, dtype=np.float64)
            self.areas = np.zeros(n_cells, dtype=np.float64)
            
            # Copy data from all cells
            for i, cell in enumerate(self._cells):
                self.centers[i] = cell.center
                self.velocities[i] = cell.vel
                self.radii[i] = cell.r
                self.base_radii[i] = cell.base_r
                self.areas[i] = cell.area0
            
            # Update tracking variable
            self.n_cells = n_cells
    

    def step(self, dt: float = 0.005) -> None:
        """Advance physics by *dt*.  Called by RealtimeEngine or manually."""
        update_all_cells_parallel(
            self.centers, self.velocities, self.radii, self.angles,
            self.base_radii, self.areas, self.width, self.height, dt,
            step_count=self._step_count,
            brownian_d=self.brownian_d,
        )
        self._step_count += 1

        # Sync back to cell objects
        self._sync_arrays_to_cells()

        # Call cell-specific behaviours (safe - no list modification during iteration)
        for cell in self._cells:
            if hasattr(cell, "update_behavior"):
                cell.update_behavior(dt) # type: ignore

        # Handle collisions
        self._handle_collisions_with_spatial_grid()

        # Sync collision/behavior changes back to arrays so next step
        # starts from the corrected positions and velocities
        for i, cell in enumerate(self._cells):
            self.centers[i] = cell.center
            self.velocities[i] = cell.vel

        # Update cell cycle dynamics (divisions, apoptosis)
        # This modifies self._cells (adds/removes cells), so do it last
        if self.cycle_manager is not None:
            self.cycle_manager.update(self._cells, self) # type: ignore
            # Resync numpy arrays after cell list changes
            # This protects against index misalignment by rebuilding arrays completely
            self._resync_arrays_after_division()

    def _handle_collisions_with_spatial_grid(self) -> None:
        """Use SpatialGrid for collision detection."""
        self.spatial_grid.clear()
        
        # Add all cells to spatial grid
        for i, cell in enumerate(self._cells):
            self.spatial_grid.add_cell(cell, i)
        
        # Check collisions only for nearby cells
        checked_pairs = set()
        for i, cell in enumerate(self._cells):
            potential = self.spatial_grid.get_potential_collisions(cell, i)
            
            for j in potential:
                if (i, j) not in checked_pairs and (j, i) not in checked_pairs:
                    if cell.check_collision(self._cells[j]):
                        # Additional collision response for normal cells
                        if isinstance(cell, NormalCell):
                            cell.respond_to_collision(self._cells[j])
                        #if isinstance(self._cells[j], NormalCell):
                        #    self._cells[j].respond_to_collision(cell)
                    
                    checked_pairs.add((i, j))

    def apply_optogenetic_stimulator(self, mask: np.ndarray) -> None:
        """Apply optogenetic stimulation to cells.

        The mask is in viewport/camera space.  Cell vertex positions are
        converted from world to viewport coordinates using camera_offset
        so the mask always corresponds to what the camera sees.
        """
        if self.cell_type != "optogenetic":
            return

        offset = tuple(self.camera_offset)

        for cell in self._cells:
            if isinstance(cell, OptogeneticCell):
                cell.stimulate(mask, camera_offset=offset)

        # Sync changes back to arrays
        for i, cell in enumerate(self._cells):
            self.radii[i] = cell.r
            self.velocities[i] = cell.vel


    def apply_drug(self, concentration: float, drug_type: str = "growth") -> None:
        """Apply drug to all cells"""
        if self.cell_type != "drug":
            return
        
        for cell in self._cells:
            if isinstance(cell, DrugResponseCell):
                cell.apply_drug(concentration, drug_type)
        
        # Sync changes back to arrays
        for i, cell in enumerate(self._cells):
            self.base_radii[i] = cell.base_r
            self.radii[i] = cell.r
            self.areas[i] = cell.area0

    
    def _handle_mask(self, mask: np.ndarray) -> None:
        """Process an SLM mask for optogenetic stimulation or drug response."""
        if self.cell_type == "optogenetic":
            self.apply_optogenetic_stimulator(mask)
        elif self.cell_type == "drug":
            if np.any(mask):
                self.apply_drug(self.concentration, self.drug_type)

    def _render_for_mode(self, mode: int) -> np.ndarray:
        """Render the current cell state for the given channel mode.

        Updates per-cell fluorescence, then delegates to the CellCycleRenderer.
        Returns a BGR uint8 image at viewport resolution (internal_scale=1,
        so _crop_fov is a no-op identity).
        """
        # Update cell fluorescence based on mode
        for cell in self._cells:
            self._update_cell_fluorescence(cell, mode)

        # Render frame based on cell type
        if self.cell_type == "cycle":
            img = self.renderer.render_cell_cycle(
                self._cells, mode,  # type: ignore
                tuple(self.camera_offset),
                self.focal_plane)
        else:
            img = self.renderer.render_cells(
                self._cells, mode,  # type: ignore
                tuple(self.camera_offset),
                self.focal_plane)

        return img

    def _apply_exposure(self, viewport: np.ndarray, exposure: float,
                        intensity: float) -> np.ndarray:
        """Apply uniform exposure scaling for all modes.

        ScatteredCellSim uses a single formula (intensity * 0.01 * exposure)
        for all channels, unlike the base class which uses 2x for BF mode.
        """
        scale = intensity * 0.01 * exposure
        return (viewport.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)


    def _on_objective_changed(self, mag: int, dof: float) -> None:
        """Forward objective changes to the cell renderer."""
        self.renderer.set_objective(mag, dof)
            

    def _update_cell_fluorescence(self, cell: CellBase, mode: int) -> None:
        """Update cell fluorescence based on mode."""
        if isinstance(cell, NormalCell):
            # NormalCell and subclasses (cycle, etc.): respect per-cell markers
            if mode == 0:
                pass  # keep markers, brightfield needs no fluorescence update
            elif mode == 1:
                if not cell.has_nucleus_marker:
                    cell.nucleus_fluorescence = 0.0
            elif mode == 2:
                if not cell.has_membrane_marker:
                    cell.membrane_fluorescence[:] = 0.0
        else:
            # Optogenetic / drug cells: always show fluorescence
            if mode == 0:
                cell.nucleus_fluorescence = 0.0
                cell.membrane_fluorescence[:] = 0.0
            elif mode == 1:
                cell.nucleus_fluorescence = 1.0
            elif mode == 2:
                cell.membrane_fluorescence[:] = 1.0


    def get_visible_cells(self) -> Sequence[CellBase | CellCycleNormal]:
        """Get cells visible in current viewport."""
        return self.renderer._get_visible_cells(self._cells, tuple(self.camera_offset))
    
    def reset(self) -> None:
        """Reset simulation."""
        self._cells = self._create_cells()
        self._resolve_initial_overlaps()
        self._init_numpy_arrays()
        self._step_count = 0