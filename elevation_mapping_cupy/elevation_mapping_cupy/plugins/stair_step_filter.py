#
# Climbable steps are traversable (2026-11-03, after RAEM arXiv 2608.25366: a quadruped's terrain map should read a
# staircase as ground it can walk, not as a 59-degree slope).
#
# At 0.05 m cells the slope filter sees every riser as a cliff and the roughness filter sees the tread/riser sawtooth,
# so a flight the robot climbs without trouble came out lethal and had to be overridden by the registered stairs prior.
# This layer recognises the step regime directly, per cell, from the window the step filter already uses:
#   two levels      the heights in the foot-scale window cluster at two values (riser face cells aside): the variance
#                   of a two-level patch is p(1-p) span^2, so var / (span^2 / 4) is ~1 for an even split and 1/3 for
#                   heights spread evenly (ramp, rubble, vegetation) -> ratio >= bimodal_min
#   climbable       span_min <= span <= step_climb (one riser; walls and ledges are taller, flat noise is lower)
#   not a cliff     the plane fitted over coarse_window (two treads) is at most slope_max_deg
# Output 1.0 on such cells, 0.0 elsewhere, NaN where the window has too little data. drivability_filter lifts its score
# on the 1.0 cells (override_value), so they reach the costmap as traversable instead of lethal. Everything the rule
# does not match keeps the slope / step / roughness verdict. What it cannot do: see the tread below an edge the lidar
# has not observed yet (unknown stays unknown) - that is what the mapping-time registration still covers for descents.
#
import cupy as cp
from cupyx.scipy import ndimage

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class StairStepFilter(PluginBase):
    """1.0 where the local surface is a climbable step between two levels, 0.0 otherwise, NaN unobserved.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        window_size (int): odd box width of the step window (5 at 0.05 m = 0.25 m: at most one riser of a 0.3 m tread).
        median_size (int): 3x3 median before the span, as in step_filter (0 = off).
        coarse_window (int): odd box width of the plane fit that measures the overall slope (13 = 0.65 m, two treads).
        step_climb (float): largest riser the robot steps up [m] (D1 MaxPro climbed 0.174 on 2026-10-29; 0.20 default).
        span_min (float): below this span the window is flat noise, not a step; the normal filters decide [m].
        slope_max_deg (float): coarse slope above which a staircase is a ramp too steep to walk (30 deg flights pass).
        bimodal_min (float): var / (span^2/4) threshold for the two-level test (1 = even split, 1/3 = evenly spread).
        min_valid_count (int): valid cells required in the step window.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        window_size: int = 5,
        median_size: int = 3,
        coarse_window: int = 13,
        step_climb: float = 0.20,
        span_min: float = 0.06,
        slope_max_deg: float = 35.0,
        bimodal_min: float = 0.5,
        min_valid_count: int = 10,
        min_samples: int = 0,
        **kwargs,
    ):
        self.cell_n = int(cell_n)
        self.resolution = float(resolution)
        self.window_size = int(window_size)
        self.median_size = int(median_size)
        self.coarse_window = int(coarse_window)
        self.step_climb = float(step_climb)
        self.span_min = float(span_min)
        self.slope_max = float(cp.tan(cp.radians(float(slope_max_deg))))
        self.bimodal_min = float(bimodal_min)
        self.min_valid_count = int(min_valid_count)
        self.min_samples = int(min_samples)   # 0 = off; cells measured by fewer clouds are unknown (2026-11-04)
        idx = cp.arange(self.cell_n, dtype=cp.float32) * self.resolution
        self._coord_a = cp.broadcast_to(idx[:, None], (self.cell_n, self.cell_n)).copy()
        self._coord_b = cp.broadcast_to(idx[None, :], (self.cell_n, self.cell_n)).copy()

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names,
        plugin_layers: cp.ndarray,
        plugin_layer_names,
        semantic_map: cp.ndarray,
        semantic_layer_names,
        *args,
        **kwargs,
    ) -> cp.ndarray:
        elevation = elevation_map[0]
        valid = elevation_map[2] > 0.5
        if self.min_samples > 0 and "sample_count" in layer_names:
            valid = valid & (elevation_map[layer_names.index("sample_count")] >= self.min_samples)
        m = valid.astype(cp.float32)
        w = self.window_size
        z = elevation
        if self.median_size >= 3:
            e0 = cp.where(valid, elevation, 0.0).astype(cp.float32)
            cnt = ndimage.uniform_filter(m, size=self.median_size, mode="nearest")
            mean = ndimage.uniform_filter(e0, size=self.median_size, mode="nearest") / cp.maximum(cnt, 1e-6)
            z = ndimage.median_filter(cp.where(valid, elevation, mean), size=self.median_size, mode="nearest")
        z0 = cp.where(valid, z, 0.0).astype(cp.float32)

        # step window: span and the two-level ratio
        neg = cp.where(valid, z, -cp.inf).astype(cp.float32)
        pos = cp.where(valid, z, cp.inf).astype(cp.float32)
        span = ndimage.maximum_filter(neg, size=w, mode="nearest") - ndimage.minimum_filter(pos, size=w, mode="nearest")
        n = ndimage.uniform_filter(m, size=w, mode="nearest")
        n_safe = cp.maximum(n, 1e-6)
        ez = ndimage.uniform_filter(z0, size=w, mode="nearest") / n_safe
        ezz = ndimage.uniform_filter(z0 * z0, size=w, mode="nearest") / n_safe
        var = cp.maximum(ezz - ez * ez, 0.0)
        ratio = var / cp.maximum(span * span / 4.0, 1e-6)
        count = n * (w * w)

        # coarse plane over two treads: slope magnitude
        wc = self.coarse_window
        a, b = self._coord_a, self._coord_b

        def cmean(x):
            return ndimage.uniform_filter(x * m, size=wc, mode="nearest")
        nc = cp.maximum(ndimage.uniform_filter(m, size=wc, mode="nearest"), 1e-6)
        Ea, Eb, Ez = cmean(a) / nc, cmean(b) / nc, cmean(z0) / nc
        Saa = cmean(a * a) / nc - Ea * Ea
        Sbb = cmean(b * b) / nc - Eb * Eb
        Sab = cmean(a * b) / nc - Ea * Eb
        Saz = cmean(a * z0) / nc - Ea * Ez
        Sbz = cmean(b * z0) / nc - Eb * Ez
        det = Saa * Sbb - Sab * Sab
        ok = det > 1e-12
        alpha = cp.where(ok, (Sbz * -Sab + Saz * Sbb) / cp.maximum(det, 1e-12), 0.0)
        beta = cp.where(ok, (Sbz * Saa - Saz * Sab) / cp.maximum(det, 1e-12), 0.0)
        coarse_slope = cp.sqrt(alpha * alpha + beta * beta)

        enough = valid & (count >= self.min_valid_count) & cp.isfinite(span)
        step = (span >= self.span_min) & (span <= self.step_climb) & (ratio >= self.bimodal_min) & ok & (coarse_slope <= self.slope_max)
        out = cp.where(step, 1.0, 0.0).astype(cp.float32)
        return cp.where(enough, out, cp.nan).astype(cp.float32)
