#
# Detrended surface roughness: RMS residual from a local plane fit, meters.
#
# Plain local standard deviation counts slope as roughness -- a smooth 12
# degree ramp would read rough while being perfectly walkable. Fitting a plane
# per window and measuring what is left removes the trend, so the layer only
# responds to texture: gravel, rubble, grass, sensor noise.
#
import cupy as cp
from cupyx.scipy import ndimage

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class RoughnessFilter(PluginBase):
    """RMS of elevation residuals about the best-fit local plane.

    The fit is closed-form from windowed moments, so it is a handful of box
    filters and elementwise math -- no per-cell solver. With E[.] the masked
    window mean and (a, b) the cell coordinates in meters:

        cov [[Saa, Sab], [Sab, Sbb]] . (alpha, beta) = (Saz, Sbz)
        residual variance = Szz - alpha * Saz - beta * Sbz

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        window_size (int): odd box width of the fit neighborhood.
        min_valid_count (int): valid cells needed for a meaningful fit; a
            plane through fewer points has no residual worth reporting.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        window_size: int = 5,
        min_valid_count: int = 6,
        **kwargs,
    ):
        self.cell_n = int(cell_n)
        self.resolution = float(resolution)
        self.window_size = int(window_size)
        self.min_valid_count = int(min_valid_count)
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
        w = self.window_size

        m = valid.astype(cp.float32)
        z = cp.where(valid, elevation, 0.0).astype(cp.float32)
        a, b = self._coord_a, self._coord_b

        def wmean(x):
            return ndimage.uniform_filter(x * m, size=w, mode="nearest")

        n = ndimage.uniform_filter(m, size=w, mode="nearest")
        n_safe = cp.maximum(n, 1e-6)
        Ea, Eb, Ez = wmean(a) / n_safe, wmean(b) / n_safe, wmean(z) / n_safe
        Saa = wmean(a * a) / n_safe - Ea * Ea
        Sbb = wmean(b * b) / n_safe - Eb * Eb
        Sab = wmean(a * b) / n_safe - Ea * Eb
        Saz = wmean(a * z) / n_safe - Ea * Ez
        Sbz = wmean(b * z) / n_safe - Eb * Ez
        Szz = wmean(z * z) / n_safe - Ez * Ez

        det = Saa * Sbb - Sab * Sab
        # A degenerate window (single row/column of valid cells) has no plane;
        # fall back to the raw variance there rather than dividing by ~zero.
        ok = det > 1e-12
        alpha = cp.where(ok, (Sbz * -Sab + Saz * Sbb) / cp.maximum(det, 1e-12), 0.0)
        beta = cp.where(ok, (Sbz * Saa - Saz * Sab) / cp.maximum(det, 1e-12), 0.0)
        resid_var = Szz - alpha * Saz - beta * Sbz
        roughness = cp.sqrt(cp.maximum(resid_var, 0.0))

        count = n * (w * w)
        enough = count >= self.min_valid_count
        return cp.where(valid & enough, roughness, cp.nan).astype(cp.float32)
