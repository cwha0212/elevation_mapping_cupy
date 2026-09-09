#
# A face too tall to step and too broken to walk.
#
# Every other verdict deliberately steps around this terrain. drivability
# refuses it but that refusal proved unable to hold the 2D grid on its own
# (the hill's south border came back with two raw cells after a full pass).
# drop exempts everything within reach of a ramp flag -- it has to, or the
# crest between two ramps grows an unerasable stripe -- and a hill's side
# face sits exactly inside that exemption hole. stairs and ramp both exclude
# it by definition, so nothing climbs it, but nothing protects it either.
#
# The definition here is the user's, stated as a measurement: ground that
# rises like a wall and offers no tread. On the fine window one threshold
# does all the sorting, which is why this filter has no exemption logic at
# all -- a riser spans 0.15, a kerb 0.12, a 45 degree bank 0.14 on the
# diagonal, and none of them clear 0.30; a structure's vertical side spans
# its own height and does. No exemption is the point: this layer exists to
# hold marks precisely where drop's flag-radius exemption cannot.
#
import cupy as cp
from cupyx.scipy import ndimage

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class WallFilter(PluginBase):
    """Height span of the fine window where it exceeds a riser, in meters.

    Zero where the ground is anything a gait could handle, the span itself
    where a wall-sized face stands, NaN where unmeasured. Consumers demote
    (traversability cloud) and veto erasure (fuse) on wall_min; the value
    stays in meters rather than a confidence because the number means
    something by itself.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        window (int): odd box width for the span, cells. 3 keeps the test
            local enough that stairs and banks stay under the threshold.
        wall_min (float): spans at or below this are not walls. 0.30 is
            max_riser and the platform's step limit -- above it no gait
            applies, so no new tuning axis is introduced.
        max_range (float): beyond this from the map centre the span is too
            noisy to assert a wall from; 0 disables the gate.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        window: int = 3,
        wall_min: float = 0.30,
        max_range: float = 6.0,
        **kwargs,
    ):
        self.resolution = float(resolution)
        self.window = int(window)
        self.wall_min = float(wall_min)
        self.max_range = float(max_range)

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

        # Invalid cells must not win either extreme.
        big = cp.where(valid, elevation, -cp.inf).astype(cp.float32)
        small = cp.where(valid, elevation, cp.inf).astype(cp.float32)
        span = ndimage.maximum_filter(
            big, size=self.window, mode="nearest"
        ) - ndimage.minimum_filter(small, size=self.window, mode="nearest")

        wall = cp.where(cp.isfinite(span) & (span > self.wall_min), span, 0.0)

        if self.max_range > 0.0:
            h, w = elevation.shape
            rows = cp.arange(h, dtype=cp.float32) - h / 2.0 + 0.5
            cols = cp.arange(w, dtype=cp.float32) - w / 2.0 + 0.5
            dist = cp.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2) * self.resolution
            wall = cp.where(dist <= self.max_range, wall, 0.0)

        return cp.where(valid, wall, cp.nan).astype(cp.float32)
