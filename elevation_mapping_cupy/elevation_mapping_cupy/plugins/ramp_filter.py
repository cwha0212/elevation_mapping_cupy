#
# A ramp is the other thing a quadruped changes gait for.
#
# The stairs filter answers "is this a flight" and, having been taught that a
# staircase is riser AND tread alternating, correctly says no to a bank. But
# "not stairs" is not the same as "ordinary ground": a sustained 20 degree
# slope is walkable and needs a different posture than flat pavement, and the
# planner and the supervisor both want to know it is coming.
#
# So this is the complement of the stairs test, stated positively. Where a
# flight is broken -- riser, flat, riser, flat -- a ramp is continuous: its
# surface is tilted everywhere, smooth, and free of riser-sized steps. That
# last clause is what keeps a staircase out of this layer, and the tread
# fraction is what keeps a ramp out of the stairs layer, so the two are
# mutually exclusive by construction rather than by tuning.
#
# Output follows the same convention as stairs: magnitude is confidence, sign
# is whether the slope climbs or drops away from the robot, NaN is unmeasured.
#
import numpy as np

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase
from elevation_mapping_cupy.plugins.stairs_filter import _climb_sign, _gpu_modules, _host

DEFAULTS = dict(
    # Below min_slope it is just ground with a lean on it; above max_slope no
    # gait helps, and drivability has already refused it.
    min_slope=8.0,
    max_slope=45.0,
    max_roughness=0.05,
    # A ramp is a continuous surface, so nothing in it steps like a riser.
    # Measured on the same fine window the stairs filter uses, not on the
    # step layer: that one spans 0.20 m, so a plane's own rise fills it and
    # a 25 degree bank reads 0.093 -- riser-sized, and the ramp would exclude
    # itself. Over 0.10 m the same bank reads 0.047, while a 0.15 m riser
    # still reads 0.15.
    max_riser_step=0.12,
    riser_window=3,
    struct_window=21,
    # The window cannot be half sloped within half a window of a bank's edge,
    # so this is really "how much of the border do we throw away". Measured
    # head-on, dropping it from 0.50 took the 25 deg bank from 73% to 85% of
    # what was observed and cost nothing: flat pavement flagged 0.00% either
    # way. Shrinking the window instead is not an option -- a shallow ramp
    # needs the full 1.05 m to clear min_total_gain at all, and the 11 deg
    # one collapses from 56% to 14% at 0.55 m.
    min_ramp_ratio=0.35,
    min_total_gain=0.20,
    min_valid_ratio=0.5,
    min_valid_ratio_confirmed=0.65,
    max_range=4.5,
    min_component_cells=60,
    # A ramp is planar: its slope does not wander. Rubble and vegetation give
    # plenty of tilted cells but no consistency.
    max_slope_std=8.0,
    conf_candidate=0.4,
    conf_confirmed=0.8,
)


def detect_ramps(elevation, valid, step, slope, roughness, resolution, params=None):
    """Signed ramp confidence per cell, same convention as the stairs layer."""
    import scipy.ndimage as host_ndi

    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULTS})

    xp, ndi, is_gpu = _gpu_modules(elevation)
    step, slope, roughness, valid = (
        xp.asarray(a) for a in (step, slope, roughness, valid)
    )
    h, w = elevation.shape
    f32 = xp.float32

    rows = xp.arange(h, dtype=f32) - h / 2.0 + 0.5
    cols = xp.arange(w, dtype=f32) - w / 2.0 + 0.5
    dist = xp.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2) * resolution

    admissible = valid & (dist <= p["max_range"])
    big = xp.where(valid, elevation, -1e6)
    small = xp.where(valid, elevation, 1e6)
    rw = p["riser_window"]
    fine = ndi.maximum_filter(big, size=rw, mode="nearest") - ndi.minimum_filter(
        small, size=rw, mode="nearest"
    )
    sloped = (
        admissible
        & xp.isfinite(slope)
        & (slope >= p["min_slope"])
        & (slope <= p["max_slope"])
        & xp.isfinite(roughness)
        & (roughness < p["max_roughness"])
        & (fine < p["max_riser_step"])
    )

    sw = p["struct_window"]
    # Fraction of what was observed, not of the window -- same reasoning as
    # the stairs filter: a bank seen from the side hides its own far half,
    # and a window-based denominator counts every hidden cell as "not
    # sloped", failing ground whose observed part is unambiguous. The
    # min_valid_ratio floor below still demands enough observation to judge.
    valid_frac = ndi.uniform_filter(valid.astype(f32), size=sw, mode="nearest")
    ramp_frac = ndi.uniform_filter(
        sloped.astype(f32), size=sw, mode="nearest"
    ) / xp.maximum(valid_frac, 1e-3)
    gain = ndi.maximum_filter(big, size=sw, mode="nearest") - ndi.minimum_filter(
        small, size=sw, mode="nearest"
    )

    candidate = (
        sloped
        & (ramp_frac >= p["min_ramp_ratio"])
        & (gain >= p["min_total_gain"])
        & (valid_frac >= p["min_valid_ratio"])
    )

    cand = _host(candidate, is_gpu)
    adm = _host(admissible, is_gpu)
    smooth = _host(fine < p["max_riser_step"], is_gpu)
    elev_h = _host(elevation, is_gpu)
    valid_h = _host(valid, is_gpu)
    slope_h = _host(slope, is_gpu)
    vfrac_h = _host(valid_frac, is_gpu)
    dist_h = _host(dist, is_gpu)

    ones3 = np.ones((3, 3), bool)
    # Re-intersect with the no-riser-step test after every morphological op,
    # not just with admissible. The candidate stage excludes a slope's side
    # rim exactly -- the fine window sees the 0.6 m lateral drop -- but
    # closing glued those rim cells back on, they inherited the component's
    # confirmed grade, and downstream that grade is a licence to erase
    # occupancy: the marks guarding the side edge were being deleted by
    # flags the detector never actually earned there.
    mask = host_ndi.binary_opening(cand, structure=ones3) & adm & smooth
    mask = host_ndi.binary_closing(mask, structure=ones3) & adm & smooth

    conf = np.zeros_like(elev_h, dtype=np.float32)
    labels, n = host_ndi.label(mask)
    for k in range(1, n + 1):
        comp = labels == k
        if comp.sum() < p["min_component_cells"]:
            continue
        core = host_ndi.binary_erosion(comp, structure=ones3)
        if core.sum() < p["min_component_cells"]:
            continue
        sl = slope_h[comp]
        sl = sl[np.isfinite(sl)]
        if sl.size == 0:
            continue
        confirmed = (
            float(sl.std()) <= p["max_slope_std"]
            and float(vfrac_h[comp].mean()) >= p["min_valid_ratio_confirmed"]
        )
        grade = p["conf_confirmed"] if confirmed else p["conf_candidate"]
        conf[comp] = _climb_sign(comp, elev_h, dist_h) * grade

    out = np.where(valid_h, conf, np.nan).astype(np.float32)
    return xp.asarray(out) if is_gpu else out


class RampFilter(PluginBase):
    """Flag sustained walkable slopes, graded and signed.

    Mutually exclusive with the stairs layer by construction: a cell only
    counts here if it has no riser-sized step, and the stairs layer only
    fires where risers and treads sit together.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        **kwargs: any key in DEFAULTS overrides that threshold.
    """

    def __init__(
        self,
        cell_n: int = 100,
        resolution: float = 0.05,
        step_layer: str = "step",
        slope_layer: str = "slope",
        roughness_layer: str = "roughness",
        **kwargs,
    ):
        self.step_layer = step_layer
        self.slope_layer = slope_layer
        self.roughness_layer = roughness_layer
        self.resolution = float(resolution)
        self.params = {}
        for key, value in kwargs.items():
            if key in DEFAULTS:
                default = DEFAULTS[key]
                self.params[key] = int(value) if isinstance(default, int) else float(value)
        self.input_layer_names = [step_layer, slope_layer, roughness_layer]

    def __call__(
        self,
        elevation_map,
        layer_names,
        plugin_layers,
        plugin_layer_names,
        semantic_map,
        semantic_params,
        *args,
        **kwargs,
    ):
        import cupy as cp

        elevation = elevation_map[0]
        valid = elevation_map[2] > 0.5

        def layer(name):
            return self.get_layer_data(
                elevation_map, layer_names, plugin_layers, plugin_layer_names,
                semantic_map, semantic_params, name,
            )

        step = layer(self.step_layer)
        slope = layer(self.slope_layer)
        roughness = layer(self.roughness_layer)
        if step is None or slope is None or roughness is None:
            return cp.zeros_like(elevation)

        return detect_ramps(
            elevation, valid, step, slope, roughness, self.resolution, self.params
        )
