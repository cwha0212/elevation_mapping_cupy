#
# Stairs: terrain a quadruped can climb, but only after changing gait.
#
# Drivability folds a stair into "0.25, nearly at the limit", which reads as
# merely bad ground; the mode executive needs to know it is looking at a
# STAIR -- a region that becomes traversable if the robot stops, switches
# gait, and takes it head on. That is a discrete kind, not a grade, so it
# gets its own layer for the supervisor and the stairs grid to consume.
#
import cupy as cp
from cupyx.scipy import ndimage

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class StairsFilter(PluginBase):
    """Flag cells that belong to a climbable stair flight.

    Two conditions, each rejecting a different impostor:

      riser band      step (foot-scale max-min) within [min_riser, max_riser].
                      Flat ground and ramps sit far below the band (a 12 degree
                      ramp spans ~5 cm across the foot window); a wall face
                      spans far above it. A curb passes -- which is why the
                      second condition exists.

      sustained climb elevation span across a body-length window of at least
                      min_total_gain. A curb climbs its 0.12 m exactly once
                      and fails; four 0.15 m risers gain ~0.45 m over the same
                      metre and pass.

      no wall nearby  no cell within the window may carry a step above
                      max_riser. This is what keeps object edges out: a foot
                      window clipping a box side at an angle can land inside
                      the riser band by accident, but the cells straight at
                      that edge measure the full face height -- and a real
                      flight contains no unclimbable edge anywhere in it.

    Output: 1.0 on stair cells, 0.0 on other observed cells, NaN unobserved.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        min_riser (float): smallest step that counts as a riser, meters.
        max_riser (float): largest climbable riser, meters. Above it the
            edge is a wall no gait will fix.
        gain_window (int): odd width of the sustained-climb window, cells.
        min_total_gain (float): elevation span that window must contain.
        min_valid_ratio (float): observed fraction of the gain window needed
            before its span is trusted.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        min_riser: float = 0.09,
        max_riser: float = 0.22,
        gain_window: int = 21,
        min_total_gain: float = 0.25,
        min_valid_ratio: float = 0.4,
        **kwargs,
    ):
        self.min_riser = float(min_riser)
        self.max_riser = float(max_riser)
        self.gain_window = int(gain_window)
        self.min_total_gain = float(min_total_gain)
        self.min_valid_ratio = float(min_valid_ratio)
        # The manager computes this lazily before us.
        self.input_layer_names = ["step"]

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

        step = self.get_layer_data(
            elevation_map, layer_names, plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names, "step",
        )
        if step is None:
            raise ValueError("stairs_filter: input layer 'step' not found.")

        riser = cp.isfinite(step) & (step >= self.min_riser) & (step <= self.max_riser)
        wall = cp.isfinite(step) & (step > self.max_riser)

        w = self.gain_window
        neg = cp.where(valid, elevation, -cp.inf).astype(cp.float32)
        pos = cp.where(valid, elevation, cp.inf).astype(cp.float32)
        span = (
            ndimage.maximum_filter(neg, size=w, mode="nearest")
            - ndimage.minimum_filter(pos, size=w, mode="nearest")
        )
        observed = ndimage.uniform_filter(
            valid.astype(cp.float32), size=w, mode="nearest"
        )
        sustained = (
            cp.isfinite(span)
            & (span >= self.min_total_gain)
            & (observed >= self.min_valid_ratio)
        )

        wall_near = ndimage.maximum_filter(
            wall.astype(cp.float32), size=w, mode="nearest"
        ) > 0.5
        stairs = (riser & sustained & ~wall_near).astype(cp.float32)
        return cp.where(valid, stairs, cp.nan).astype(cp.float32)
