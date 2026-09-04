#
# Slope angle of the terrain surface, in degrees.
#
# The CNN traversability layer scores geometry it was trained on; for
# navigation we want terms we can verify against ground truth and threshold per
# robot. This is the first of them: the local surface inclination.
#
import cupy as cp
from cupyx.scipy import ndimage

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SlopeFilter(PluginBase):
    """Surface slope in degrees from central differences of the elevation.

    The gradient axis order does not matter for the magnitude, so this stays
    out of the row/col-vs-x/y question entirely.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        window_size (int): odd box width for pre-smoothing and validity. At a
            height discontinuity the finite difference is the step divided by
            the cell size, not a surface inclination -- that regime belongs to
            the step layer, and smoothing keeps it from bleeding far sideways.
        min_valid_ratio (float): fraction of the window that must hold valid
            cells for the output to count. Below it the cell reads NaN.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        window_size: int = 3,
        min_valid_ratio: float = 0.5,
        **kwargs,
    ):
        self.resolution = float(resolution)
        self.window_size = int(window_size)
        self.min_valid_ratio = float(min_valid_ratio)

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

        # Normalized masked mean: invalid cells contribute nothing, and the
        # result divides by how much of the window was actually observed.
        mask = valid.astype(cp.float32)
        z = cp.where(valid, elevation, 0.0).astype(cp.float32)
        w = self.window_size
        sum_z = ndimage.uniform_filter(z, size=w, mode="nearest")
        sum_m = ndimage.uniform_filter(mask, size=w, mode="nearest")
        smooth = cp.where(sum_m > 0, sum_z / cp.maximum(sum_m, 1e-6), 0.0)

        # Central differences on the smoothed surface. Where a neighbor was
        # entirely unobserved the smoothed value is a hole; the validity ratio
        # below removes those cells rather than letting the edge read as slope.
        dzd0 = (cp.roll(smooth, -1, axis=0) - cp.roll(smooth, 1, axis=0)) / (2.0 * self.resolution)
        dzd1 = (cp.roll(smooth, -1, axis=1) - cp.roll(smooth, 1, axis=1)) / (2.0 * self.resolution)
        slope_deg = cp.degrees(cp.arctan(cp.sqrt(dzd0 ** 2 + dzd1 ** 2)))

        enough = sum_m >= self.min_valid_ratio
        # The roll wraps at the border; the outermost ring is padding anyway.
        enough[0, :] = enough[-1, :] = False
        enough[:, 0] = enough[:, -1] = False
        return cp.where(valid & enough, slope_deg, cp.nan).astype(cp.float32)
