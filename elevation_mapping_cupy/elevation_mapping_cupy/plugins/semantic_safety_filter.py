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
        unobserved_cap (float): ceiling on safety where no camera evidence has
            ever landed (every hazard layer still at its 0.0 init). 1.0
            disables it. Set between the costmap's obstacle and free
            thresholds and camera-unseen ground stays unknown instead of
            being promoted to free on geometry alone.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        base_layer: str = "drivability",
        hazard_layers: list = [],
        hazard_low: float = 0.25,
        hazard_high: float = 0.60,
        unobserved_cap: float = 1.0,
        veto_logit: float = float("nan"),
        **kwargs,
    ):
        self.base_layer = base_layer
        self.hazard_layers = list(hazard_layers)
        self.hazard_low = float(hazard_low)
        self.hazard_high = float(hazard_high)
        self.unobserved_cap = float(unobserved_cap)
        # Direct verdict mode: when set, the graded ramp is bypassed entirely
        # and the rule is simply "untrav above this logit -> forbidden, below
        # -> the geometry's call". One knob, and it is in the model's own
        # units; 0 is SAM-TP's decision boundary, 0.5 adds a margin against
        # residual silhouette noise.
        self.veto_logit = float(veto_logit)
        # The camera judges the ground it is ABOUT to drive over. At range a
        # grazing pixel covers tens of centimetres and one strong verdict
        # paints a swath, so the veto only applies near the robot (the map is
        # robot-centred); farther cells wait until the approach.
        self.veto_range = float(kwargs.get("veto_range", 3.0))
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
        observed = None
        for name in self.hazard_layers:
            if name not in semantic_layer_names:
                continue
            layer = semantic_map[semantic_layer_names.index(name)]
            layer = cp.where(cp.isfinite(layer), layer, 0.0)
            hazard = layer if hazard is None else cp.maximum(hazard, layer)
            # Exactly 0.0 is the fusion init value; the EMA of real logits
            # never returns there, so it doubles as the never-observed flag.
            seen = layer != 0.0
            observed = seen if observed is None else (observed | seen)

        if hazard is None:
            return base.astype(cp.float32)

        if self.veto_logit == self.veto_logit:  # veto mode (nan-safe check)
            n = self._cell_n
            idx = cp.arange(n, dtype=cp.float32) - n / 2 + 0.5
            dist = cp.sqrt(idx[None, :] ** 2 + idx[:, None] ** 2) * self._resolution
            veto = (hazard > self.veto_logit) & (dist <= self.veto_range)
            combined = cp.where(veto, 0.0,
                                cp.where(cp.isfinite(base), base, cp.nan))
            known = cp.isfinite(base) | veto
            return cp.where(known, combined, cp.nan).astype(cp.float32)

        span = self.hazard_high - self.hazard_low
        semantic_term = 1.0 - cp.clip((hazard - self.hazard_low) / span, 0.0, 1.0)
        # A hazard on a cell the geometry never measured is still a hazard, so
        # the semantic verdict stands where the base layer is NaN.
        combined = cp.where(cp.isfinite(base), cp.minimum(base, semantic_term), semantic_term)
        if self.unobserved_cap < 1.0:
            # Camera-unseen ground: geometry may only promise so much. The
            # cap keeps it out of the costmap's free class without calling
            # it an obstacle -- unseen stays unknown, not forbidden.
            combined = cp.where(observed, combined, cp.minimum(combined, self.unobserved_cap))
        known = cp.isfinite(base) | (hazard > 0.0)
        return cp.where(known, combined, cp.nan).astype(cp.float32)
