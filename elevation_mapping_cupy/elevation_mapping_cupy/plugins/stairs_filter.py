#
# Stairs = the terrain a quadruped can climb once it changes gait. The step
# filter alone cannot say that: a 0.15 m riser and a 0.15 m ledge score the
# same. What separates a stair FLIGHT is repetition -- riser-sized steps that
# keep gaining height across the window, with nothing wall-sized among them.
#
import cupy as cp

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class StairsFilter(PluginBase):
    """Flag cells that belong to a climbable stair flight.

    A cell is a stair cell when, in its neighbourhood:
      * the local step sits in the riser band (min_riser..max_riser) -- below
        is ordinary rough ground, above is a wall no gait climbs;
      * the elevation keeps climbing: the window's total height gain reaches
        min_total_gain, so a lone curb or ledge does not qualify;
      * nothing in the window steps taller than max_riser -- that vetoes
        object edges and walls that would otherwise mimic risers.

    Output is 1.0 on stair cells, 0.0 elsewhere. Downstream may treat flagged
    cells as conditionally drivable; this layer only answers "is this a stair
    flight".

    Args:
        step_layer (str): step layer to read riser sizes from.
        min_riser / max_riser (float): riser band, meters.
        gain_window (int): window (cells) over which the climb must persist.
        min_total_gain (float): height gain across the window, meters.
        min_valid_ratio (float): required fraction of valid cells in window.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        step_layer: str = "step",
        min_riser: float = 0.09,
        max_riser: float = 0.22,
        gain_window: int = 21,
        min_total_gain: float = 0.25,
        min_valid_ratio: float = 0.4,
        **kwargs,
    ):
        self.step_layer = step_layer
        self.min_riser = float(min_riser)
        self.max_riser = float(max_riser)
        self.gain_window = int(gain_window)
        self.min_total_gain = float(min_total_gain)
        self.min_valid_ratio = float(min_valid_ratio)
        self.input_layer_names = [step_layer]

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names,
        plugin_layers: cp.ndarray,
        plugin_layer_names,
        semantic_map,
        semantic_params,
        *args,
        **kwargs,
    ) -> cp.ndarray:
        import cupyx.scipy.ndimage as ndi

        elevation = elevation_map[0]
        valid = elevation_map[2] > 0.5
        step = self.get_layer_data(
            elevation_map, layer_names, plugin_layers, plugin_layer_names,
            semantic_map, semantic_params, self.step_layer,
        )
        if step is None:
            return cp.zeros_like(elevation)

        w = self.gain_window
        elev = cp.where(valid, elevation, cp.nan)
        # Sustained climb: max minus min of the (valid) elevation across the
        # window. ndimage min/max ignore nothing, so run them on padded copies.
        emax = ndi.maximum_filter(cp.where(valid, elevation, -1e6), size=w)
        emin = ndi.minimum_filter(cp.where(valid, elevation, 1e6), size=w)
        gain = emax - emin
        enough_valid = (
            ndi.uniform_filter(valid.astype(cp.float32), size=w)
            >= self.min_valid_ratio
        )

        riser = cp.isfinite(step) & (step >= self.min_riser) & (step <= self.max_riser)
        # Wall veto: any step above the riser band nearby disqualifies the
        # whole neighbourhood -- flights are made of risers, walls are not.
        tall = cp.isfinite(step) & (step > self.max_riser)
        wall_near = ndi.maximum_filter(tall.astype(cp.float32), size=w) > 0.5

        flight = (
            riser
            & enough_valid
            & (gain >= self.min_total_gain)
            & (gain < 1e5)
            & ~wall_near
        )
        return cp.where(valid, flight.astype(cp.float32), cp.nan)
