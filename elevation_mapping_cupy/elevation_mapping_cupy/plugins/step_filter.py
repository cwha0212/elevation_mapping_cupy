#
# Local step height: the elevation span inside a foot-scale window.
#
# Slope reads a 0.12 m curb crossed in one cell as ~67 degrees, which is not
# an inclination at all. Discontinuities are their own regime, and this layer
# measures them directly: max minus min elevation within the window, which for
# a clean step equals the riser height.
#
import cupy as cp
from cupyx.scipy import ndimage

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class StepFilter(PluginBase):
    """Height span (max - min) of valid cells inside a square window, meters.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        window_size (int): odd box width. Size it to the robot's foot reach:
            5 cells at 0.05 m spans 0.25 m, so a riser within a step's length
            of the cell shows its full height.
        min_valid_count (int): valid cells required in the window; fewer and
            the output is NaN instead of a span computed from almost nothing.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        window_size: int = 5,
        min_valid_count: int = 4,
        **kwargs,
    ):
        self.window_size = int(window_size)
        self.min_valid_count = int(min_valid_count)

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

        # Invalid cells must not win either extreme, so they enter the max as
        # -inf and the min as +inf.
        neg = cp.where(valid, elevation, -cp.inf).astype(cp.float32)
        pos = cp.where(valid, elevation, cp.inf).astype(cp.float32)
        local_max = ndimage.maximum_filter(neg, size=w, mode="nearest")
        local_min = ndimage.minimum_filter(pos, size=w, mode="nearest")
        span = local_max - local_min

        count = ndimage.uniform_filter(valid.astype(cp.float32), size=w, mode="nearest") * (w * w)
        enough = count >= self.min_valid_count
        return cp.where(valid & enough & cp.isfinite(span), span, cp.nan).astype(cp.float32)
