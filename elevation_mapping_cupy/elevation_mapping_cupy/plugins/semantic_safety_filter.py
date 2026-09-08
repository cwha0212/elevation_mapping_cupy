#
# Safety = what the geometry allows, minus what the semantics forbid.
#
# Geometry answers "can the robot physically get across this cell". It cannot
# answer "should it". A roadway is as flat as the sidewalk beside it, and a
# quadruped will happily step off the curb onto it; only the label separates
# them. This layer is where that second judgement enters.
#
import cupy as cp

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticSafetyFilter(PluginBase):
    """Combine a geometric base layer with semantic hazard classes.

    A hazard drags the score down on its own, whatever the geometry says, so
    flat ground that is labelled forbidden comes out unsafe. The hazard ramp
    is graded rather than a hard cut: segmentation confidence is a continuous
    thing and a costmap can use the gradient.

    Missing hazard layers are not an error. The same chain runs on setups with
    no camera at all, and there safety is simply the geometry.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        base_layer (str): geometric drivability layer to start from.
        hazard_layers (list): semantic class layers that make a cell unsafe.
        hazard_low (float): probability at which a hazard starts to count.
        hazard_high (float): probability at which it zeroes the score.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        base_layer: str = "drivability",
        hazard_layers: list = [],
        hazard_low: float = 0.25,
        hazard_high: float = 0.60,
        hazard_range: float = 0.0,
        **kwargs,
    ):
        self.base_layer = base_layer
        self.hazard_layers = list(hazard_layers)
        self.hazard_low = float(hazard_low)
        self.hazard_high = float(hazard_high)
        # How far out the camera is allowed to have an opinion, in meters
        # from the map centre. Past a few meters one pixel covers a long
        # smear of ground, and an object's verdict lands on whatever is
        # behind it; the geometry keeps working out there either way.
        # 0 means no limit.
        self.hazard_range = float(hazard_range)
        self._cell_n = cell_n
        self._resolution = resolution
        if self.hazard_high <= self.hazard_low:
            raise ValueError(
                "semantic_safety_filter: hazard_high must exceed hazard_low "
                f"(got {hazard_low} and {hazard_high})."
            )
        # Only the base layer is a plugin layer the manager can compute for us;
        # the hazards are semantic layers, filled by the camera.
        self.input_layer_names = [base_layer]

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
        base = self.get_layer_data(
            elevation_map, layer_names, plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names, self.base_layer,
        )
        if base is None:
            raise ValueError(
                f"semantic_safety_filter: base layer '{self.base_layer}' not found."
            )

        hazard = None
        for name in self.hazard_layers:
            if name not in semantic_layer_names:
                continue
            layer = semantic_map[semantic_layer_names.index(name)]
            layer = cp.where(cp.isfinite(layer), layer, 0.0)
            hazard = layer if hazard is None else cp.maximum(hazard, layer)

        if hazard is None:
            return base.astype(cp.float32)

        if self.hazard_range > 0.0:
            n = self._cell_n
            idx = cp.arange(n, dtype=cp.float32) - n / 2.0 + 0.5
            dist = cp.sqrt(idx[None, :] ** 2 + idx[:, None] ** 2) * self._resolution
            hazard = cp.where(dist <= self.hazard_range, hazard, 0.0)

        span = self.hazard_high - self.hazard_low
        semantic_term = 1.0 - cp.clip((hazard - self.hazard_low) / span, 0.0, 1.0)
        # The camera votes only on ground the lidar has measured. A pixel does
        # not carry a range: its verdict lands on the map at whatever depth the
        # surface underneath it says, so on a cell with no measured surface the
        # depth is a guess and the mark is at a made-up place -- typically
        # smeared out behind the very object being labelled. Letting the
        # verdict stand there was inventing obstacles out of colour alone.
        # Where geometry has measured, the two combine and the camera can only
        # ever make a cell worse, which is the veto it is meant to be.
        combined = cp.minimum(base, semantic_term)
        return cp.where(cp.isfinite(base), combined, cp.nan).astype(cp.float32)
