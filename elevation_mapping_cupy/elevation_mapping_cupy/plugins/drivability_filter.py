#
# Drivability: slope, step, and roughness folded into one graded score.
#
# 1.0 is flat clean ground, 0.0 is at or past a limit, and the worst term
# wins -- a gentle slope does not excuse an over-limit step. Costmap layers
# downstream read this directly: cost scales with (1 - drivability) and
# NaN stays NaN, which is what unknown must remain for navigation.
#
import cupy as cp

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class DrivabilityFilter(PluginBase):
    """min over terms of (1 - value / limit), clamped to [0, 1].

    The limits are the robot's physical envelope and belong in the per-robot
    plugin config, not here: a quadruped that climbs 0.15 m steps and a wheeled
    base that stalls at 0.03 m share this code with different numbers.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        layers (list): plugin layer names to combine, e.g. [slope, step, roughness].
        limits (list): per-layer limit at which that term alone zeroes the
            score. Same length and order as ``layers``.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        layers: list = ["slope", "step", "roughness"],
        limits: list = [30.0, 0.15, 0.05],
        **kwargs,
    ):
        if len(layers) != len(limits):
            raise ValueError(
                f"drivability_filter: {len(layers)} layers but {len(limits)} limits."
            )
        self.layers = list(layers)
        self.limits = [float(v) for v in limits]
        # The manager reads this to compute the inputs lazily before this runs.
        self.input_layer_names = list(layers)

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
        score = None
        known = None
        for name, limit in zip(self.layers, self.limits):
            layer = self.get_layer_data(
                elevation_map, layer_names, plugin_layers, plugin_layer_names,
                semantic_map, semantic_layer_names, name,
            )
            if layer is None:
                raise ValueError(f"drivability_filter: input layer '{name}' not found.")
            term = cp.clip(1.0 - layer / limit, 0.0, 1.0)
            fin = cp.isfinite(layer)
            # A term with no data neither helps nor hurts; only measured terms
            # take part, and a cell no term measured stays NaN.
            term = cp.where(fin, term, 1.0)
            score = term if score is None else cp.minimum(score, term)
            known = fin if known is None else (known | fin)

        return cp.where(known, score, cp.nan).astype(cp.float32)
