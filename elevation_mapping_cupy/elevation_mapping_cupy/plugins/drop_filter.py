#
# How far the ground falls away from here.
#
# The step layer is max minus min inside a box, and a box span cannot tell a
# 0.12 m rise the robot steps up from a 0.12 m drop it falls off. It also has
# to see both sides: at a kerb the lip itself goes unmeasured, so a 0.25 m
# window centred on the last measured pavement cell reaches 0.125 m, never
# crosses the gap, and reports 0.007 for a boundary that is plainly 0.12 m in
# the elevation layer. Measured on the bench, the pavement edge came back with
# a step of 0.049 at its worst -- a third of the limit that would have blocked
# it, so free space walked straight out onto the roadway.
#
# A drop is asymmetric, and stating it that way fixes both problems at once:
# how far below this cell does the ground get within reach. It needs only the
# lower side measured, so the unmeasured lip costs nothing, and beside a wall
# it reads zero, because a wall is higher and not lower. That last property is
# why this and not a wider span: widening the span to 0.65 m caught 48% of the
# kerb but quadrupled what the pavement reported and thickened the skirt along
# the facade, while this caught 91% of the kerb with the pavement flat at
# 0.09% and the facade strip at 0.07%, unchanged across every radius tried.
#
# Output is the drop in meters, zero where there is nothing to fall off, NaN
# where unmeasured. Metres rather than a confidence because the number means
# something on its own and every consumer wants a different cut from it.
#
import cupy as cp
from cupyx.scipy import ndimage

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class DropFilter(PluginBase):
    """Height the ground falls away by within reach of each cell, in meters.

    Flights and ramps are exempt, and so is everything within a radius of one.
    A staircase is a drop by construction and its landing is the edge of one,
    so without this the layer would close the top of every flight the stairs
    work exists to keep open. Exempting the structure alone is not enough --
    the cells that flag are the ones *beside* the drop, which on a flight is
    the landing the robot has to stand on.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        radius (float): how far to look for lower ground, meters. Doubles as
            the standoff the layer asks for: a cell this far back from an edge
            still flags, which is the margin a walking robot wants anyway.
        min_drop (float): drops shallower than this are ground texture.
        stairs_layer (str), ramp_layer (str): exemption sources; missing
            layers are simply not applied.
        exempt_confidence (float): magnitude at which a stairs or ramp verdict
            counts. Both layers are signed, so this reads their absolute value
            and a descending flight exempts itself the same as a climbing one.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        radius: float = 0.50,
        exempt_radius: float = 0.80,
        min_drop: float = 0.08,
        stairs_layer: str = "stairs",
        ramp_layer: str = "ramp",
        exempt_confidence: float = 0.2,
        **kwargs,
    ):
        self.resolution = float(resolution)
        self.radius = float(radius)
        # Wider than the measuring radius on purpose. A crest between two
        # ramps is flat, so it carries no flag of its own, and its centre
        # sits exactly one measuring radius from the flags on either face:
        # at the same radius the exemption dies on the boundary line and the
        # crest grows a lethal stripe nothing can erase. 0.8 clears that
        # while staying well short of the first real cliff past a landing,
        # which sits a full landing-length from the nearest flag.
        self.exempt_radius = float(exempt_radius)
        self.min_drop = float(min_drop)
        self.stairs_layer = stairs_layer
        self.ramp_layer = ramp_layer
        self.exempt_confidence = float(exempt_confidence)
        self.input_layer_names = [stairs_layer, ramp_layer]

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
        k = int(round(2.0 * self.radius / self.resolution)) | 1

        # Invalid cells must not win the minimum, so they enter it as +inf.
        pos = cp.where(valid, elevation, cp.inf).astype(cp.float32)
        lowest = ndimage.minimum_filter(pos, size=k, mode="nearest")
        drop = cp.where(valid & cp.isfinite(lowest), elevation - lowest, cp.nan)
        drop = cp.where(drop >= self.min_drop, drop, 0.0)

        exempt = None
        for name in (self.stairs_layer, self.ramp_layer):
            layer = self.get_layer_data(
                elevation_map, layer_names, plugin_layers, plugin_layer_names,
                semantic_map, semantic_layer_names, name,
            )
            if layer is None:
                continue
            flagged = cp.isfinite(layer) & (cp.abs(layer) >= self.exempt_confidence)
            exempt = flagged if exempt is None else (exempt | flagged)
        if exempt is not None:
            k_ex = int(round(2.0 * self.exempt_radius / self.resolution)) | 1
            reach = ndimage.maximum_filter(exempt, size=k_ex, mode="nearest")
            drop = cp.where(reach, 0.0, drop)

        return cp.where(valid, drop, cp.nan).astype(cp.float32)
