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
        persist_frames: int = 1,
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
        # A cell's veto only counts once the camera has said so on
        # persist_frames consecutive observations of that cell. One frame's
        # verdict on a cell the camera saw once, at a grazing angle, at the
        # edge of its field, is the false veto that ends up on ground the
        # robot then walks over. Consecutive observations, not consecutive
        # evaluations: the filter runs on every lidar update while the
        # camera is slower, so the count moves only where the hazard value
        # changed. State lives in the semantic map's elements_to_shift so it
        # travels with the map. 1 keeps the old behaviour.
        self.persist_frames = int(persist_frames)
        self._state_key = "semantic_safety_persist"
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
        rotation=None,
        elements_to_shift=None,
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

        if self.persist_frames > 1 and elements_to_shift is not None:
            st = elements_to_shift.get(self._state_key)
            if st is None or st.shape[1:] != hazard.shape:
                # [0] = last hazard seen, [1] = consecutive hits
                st = cp.zeros((2,) + hazard.shape, dtype=cp.float32)
                elements_to_shift[self._state_key] = st
            changed = hazard != st[0]
            hit = hazard > self.hazard_low
            st[1] = cp.where(changed & hit, st[1] + 1.0,
                             cp.where(changed & ~hit, 0.0, st[1]))
            st[0] = hazard
            hazard = cp.where(st[1] >= self.persist_frames, hazard, 0.0)

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
