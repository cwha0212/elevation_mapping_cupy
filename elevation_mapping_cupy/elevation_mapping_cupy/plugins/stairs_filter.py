#
# Stairs = the terrain a quadruped can climb once it changes gait.
#
# The thing that makes a staircase a staircase is that risers and flat treads
# ALTERNATE. Crediting only the risers, as this filter used to, has two
# consequences that turn out to be the same bug seen from two sides:
#
#   * the flag lands on the vertical faces and skips the treads, so a flight
#     comes out as stripes rather than as the solid region it is; and
#   * a plain slope is riser-and-nothing-else, so it passes. With the step
#     layer's 5 cell span, any plane between roughly 18 and 47 degrees sits
#     inside the riser band with enough relief to clear the climb test and
#     nothing tall enough to trip the wall veto. Measured on a 25 degree bank
#     in the demo world: 203 of 998 observed cells flagged; on 35 degrees,
#     243 of 1026. Those cells were then lifted to passable, overriding
#     drivability's correct refusal. A robot would have been walked onto an
#     embankment it cannot stand on.
#
# So the window has to see BOTH: riser-sized steps and genuinely flat, smooth
# tread between them. Requiring both makes the flagged set cover the whole
# footprint by construction, no morphology needed to fake it, and it puts a
# bank's tread fraction at exactly zero.
#
# Everything else here is about not being confidently wrong. A flagged cell
# becomes PASSABLE downstream, and a false region does more than invite the
# robot in: the free-space fan marches over the lifted values, so bearings
# through it read clear and octomap erases whatever was correctly mapped
# behind. Hence the range gate, the component tests, and the rule that
# morphology may only ever reclassify cells that were individually admissible.
#
import math

import numpy as np

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase

DEFAULTS = dict(
    # riser evidence, measured on a fine window of the elevation itself
    min_riser=0.09,
    max_riser=0.30,
    riser_window=3,
    # tread evidence: flat, level, smooth
    max_tread_step=0.04,
    max_tread_slope=15.0,
    max_tread_roughness=0.03,
    # the structural window both fractions are counted over
    struct_window=21,
    # Both fractions oscillate as the window slides over the tread period, so
    # thresholds set near the mean fragment the region into one blob per
    # riser. They can be low without costing anything: the separation does
    # not come from how much tread there is, it comes from a plane having
    # exactly none. A 25 degree bank scores tread_frac 0.000.
    min_riser_ratio=0.10,
    min_tread_ratio=0.08,
    min_total_gain=0.25,
    min_valid_ratio=0.5,
    min_valid_ratio_confirmed=0.65,
    # a cell touching a wall-sized face is never stairs
    wall_window=3,
    interior_erosion=5,
    # trust nothing far away: at grazing incidence one pixel of noise becomes
    # a riser, and morphology would turn that into confident structure
    max_range=4.5,
    max_variance=0.05,
    # shape of the region
    open_size=3,
    close_size=3,
    max_hole_cells=100,
    min_component_cells=60,
    # repetition is what separates a flight from a ledge or a terrace
    min_risers=2,
    min_risers_confirmed=3,
    max_rise_std=0.04,
    min_tread_depth=0.20,
    max_tread_depth=0.60,
    conf_candidate=0.4,
    conf_confirmed=0.8,
)


def _gpu_modules(a):
    """cupy and its ndimage if that is what we are holding, else numpy's."""
    try:
        import cupy

        if isinstance(a, cupy.ndarray):
            import cupyx.scipy.ndimage as ndi

            return cupy, ndi, True
    except ImportError:
        pass
    import scipy.ndimage as ndi

    return np, ndi, False


def _host(a, is_gpu):
    if not is_gpu:
        return np.asarray(a)
    import cupy

    return cupy.asnumpy(a)


def detect_stairs(elevation, valid, step, slope, roughness, resolution, params=None):
    """Signed stair confidence per cell.

    Positive is a flight that climbs away from the robot, negative one that
    drops away, zero is not stairs, NaN is unmeasured. Magnitude is
    ``conf_candidate`` or ``conf_confirmed``.

    Works on cupy or numpy arrays; the windowed arithmetic runs wherever the
    input lives and only the per-component reasoning comes to the host, where
    there are a handful of regions rather than fifty thousand cells.
    """
    import scipy.ndimage as host_ndi

    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULTS})

    xp, ndi, is_gpu = _gpu_modules(elevation)
    # The manager hands layers over from a few different places and they do
    # not all arrive on the same device. Pull them onto whichever one the
    # elevation lives on before anything touches them together.
    step, slope, roughness, valid = (
        xp.asarray(a) for a in (step, slope, roughness, valid)
    )
    h, w = elevation.shape
    f32 = xp.float32

    # ---- what the camera-free geometry is allowed to speak about ---------
    rows = xp.arange(h, dtype=f32) - h / 2.0 + 0.5
    cols = xp.arange(w, dtype=f32) - w / 2.0 + 0.5
    dist = xp.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2) * resolution

    tall = xp.isfinite(step) & (step > p["max_riser"])
    wall_near = (
        ndi.maximum_filter(tall.astype(f32), size=p["wall_window"], mode="nearest") > 0.5
    )
    admissible = valid & (dist <= p["max_range"]) & ~wall_near

    # ---- riser and tread evidence, on a window sized for stair structure --
    # The step layer's window is foot reach, which smears a riser across five
    # cells and eats the tread core on any real stair with a tread under
    # 0.30 m. A three cell window keeps them apart, and it raises the angle at
    # which a plane starts looking like a riser from about 18 to about 32
    # degrees for free.
    big = xp.where(valid, elevation, -1e6)
    small = xp.where(valid, elevation, 1e6)
    rw = p["riser_window"]
    fine = ndi.maximum_filter(big, size=rw, mode="nearest") - ndi.minimum_filter(
        small, size=rw, mode="nearest"
    )
    fine_ok = (
        ndi.uniform_filter(valid.astype(f32), size=rw, mode="nearest") > 0.5
    )

    riser = admissible & fine_ok & (fine >= p["min_riser"]) & (fine <= p["max_riser"])
    tread = (
        admissible
        & fine_ok
        & (fine < p["max_tread_step"])
        & xp.isfinite(slope)
        & (slope < p["max_tread_slope"])
        & xp.isfinite(roughness)
        & (roughness < p["max_tread_roughness"])
    )

    # ---- both must be present in the same neighbourhood ------------------
    sw = p["struct_window"]
    riser_frac = ndi.uniform_filter(riser.astype(f32), size=sw, mode="nearest")
    tread_frac = ndi.uniform_filter(tread.astype(f32), size=sw, mode="nearest")
    gain = ndi.maximum_filter(big, size=sw, mode="nearest") - ndi.minimum_filter(
        small, size=sw, mode="nearest"
    )
    valid_frac = ndi.uniform_filter(valid.astype(f32), size=sw, mode="nearest")

    candidate = (
        admissible
        & (riser_frac >= p["min_riser_ratio"])
        & (tread_frac >= p["min_tread_ratio"])
        & (gain >= p["min_total_gain"])
        & (valid_frac >= p["min_valid_ratio"])
    )

    # ---- the rest is small-region reasoning; do it on the host -----------
    cand = _host(candidate, is_gpu)
    adm = _host(admissible, is_gpu)
    riser_h = _host(riser, is_gpu)
    tall_h = _host(tall, is_gpu)
    elev_h = _host(elevation, is_gpu)
    valid_h = _host(valid, is_gpu)
    fine_h = _host(fine, is_gpu)
    vfrac_h = _host(valid_frac, is_gpu)

    ones3 = np.ones((p["open_size"], p["open_size"]), bool)
    mask = host_ndi.binary_opening(cand, structure=ones3) & adm
    mask = host_ndi.binary_closing(
        mask, structure=np.ones((p["close_size"], p["close_size"]), bool)
    ) & adm
    mask = _fill_safe_holes(mask, adm, tall_h, valid_h, p["max_hole_cells"], host_ndi)

    conf = np.zeros_like(elev_h, dtype=np.float32)
    dist_h = _host(dist, is_gpu)

    labels, n = host_ndi.label(mask)
    for k in range(1, n + 1):
        comp = labels == k
        grade = _grade_component(
            comp, riser_h, tall_h, elev_h, fine_h, vfrac_h, dist_h, resolution, p,
            host_ndi,
        )
        if grade:
            conf[comp] = grade

    out = np.where(valid_h, conf, np.nan).astype(np.float32)
    if is_gpu:
        return xp.asarray(out)
    return out


def _fill_safe_holes(mask, admissible, tall, valid, max_cells, ndi):
    """Fill enclosed gaps, but never fill one that is hiding something.

    A hole in a flight is usually a tread the ratios just missed. It can also
    be a person standing on the stairs, a stairwell void, or a bollard, and
    those arrive enclosed by stairs exactly the same way. So each hole is
    judged: anything containing a wall-sized step or an unmeasured cell, or
    simply too big to be a gap in the evidence, stays a hole.
    """
    holes = ndi.binary_fill_holes(mask) & ~mask
    if not holes.any():
        return mask
    labels, n = ndi.label(holes)
    keep = np.zeros_like(mask)
    for k in range(1, n + 1):
        hole = labels == k
        if hole.sum() > max_cells:
            continue
        if tall[hole].any() or (~valid[hole]).any():
            continue
        keep |= hole
    return (mask | keep) & admissible


def _climb_sign(comp, elevation, dist, min_corr=0.2):
    """Does this flight rise or fall as you walk away from the robot?

    Asked as the correlation between a cell's height and its range, which is
    the question itself rather than a proxy for it. Comparing against the
    ground under the robot sounds simpler but is measured exactly where it is
    least reliable: approaching a descent the robot stands at the lip, half
    its footprint on each level, and the estimate lands on the wrong side.

    Near zero correlation means the flight runs past the robot rather than
    away from it, so it is standing on the flight. That returns positive: a
    robot already on stairs has to be able to keep going.
    """
    e = elevation[comp].astype(np.float64)
    d = dist[comp].astype(np.float64)
    ok = np.isfinite(e) & np.isfinite(d)
    if ok.sum() < 3:
        return 1.0
    e, d = e[ok], d[ok]
    se, sd = e.std(), d.std()
    if se < 1e-6 or sd < 1e-6:
        return 1.0
    corr = float(((e - e.mean()) * (d - d.mean())).mean() / (se * sd))
    return -1.0 if corr < -min_corr else 1.0


def _riser_lines(comp, riser, ndi, min_cells=3):
    """The individual riser faces inside one region, with their centroids."""
    labels, n = ndi.label(riser & comp)
    lines = []
    for k in range(1, n + 1):
        sel = labels == k
        cnt = int(sel.sum())
        if cnt < min_cells:
            continue
        rr, cc = np.nonzero(sel)
        lines.append((cnt, float(rr.mean()), float(cc.mean()), sel))
    return lines


def _tread_depth(lines, resolution):
    """Spacing between riser faces: the tread depth, in meters.

    Measured along whichever axis the risers actually march down, so it does
    not matter which way the flight is turned.
    """
    if len(lines) < 2:
        return None
    rs = np.array([ln[1] for ln in lines])
    cs = np.array([ln[2] for ln in lines])
    spread_r, spread_c = rs.max() - rs.min(), cs.max() - cs.min()
    along = rs if spread_r >= spread_c else cs
    return float(along.max() - along.min()) / (len(lines) - 1) * resolution


def _grade_component(comp, riser, tall, elevation, fine, valid_frac, dist,
                     resolution, p, ndi):
    """Confidence for one candidate region, or 0 if it is not a flight."""
    if comp.sum() < p["min_component_cells"]:
        return 0.0
    # A curb line has plenty of area but no width; erosion is the cheap test
    # that rejects long thin things and speckle in one go.
    core = ndi.binary_erosion(comp, structure=np.ones((3, 3), bool))
    if core.sum() < p["min_component_cells"]:
        return 0.0

    lines = _riser_lines(comp, riser, ndi)
    n_risers = len(lines)
    if n_risers < p["min_risers"]:
        return 0.0

    sign = _climb_sign(comp, elevation, dist)

    confirmed = True
    if n_risers < p["min_risers_confirmed"]:
        confirmed = False
    else:
        rises = np.array([float(np.median(fine[ln[3]])) for ln in lines])
        if float(rises.std()) > p["max_rise_std"]:
            confirmed = False
        depth = _tread_depth(lines, resolution)
        if depth is None or not (
            p["min_tread_depth"] <= depth <= p["max_tread_depth"]
        ):
            confirmed = False
        # a wall along the edge is ordinary -- stairwell sides, the top lip,
        # a handrail foot. A wall through the middle means this is not one
        # flight.
        interior = ndi.binary_erosion(
            comp, structure=np.ones((p["interior_erosion"], p["interior_erosion"]), bool)
        )
        if interior.any() and tall[interior].any():
            confirmed = False
        if float(valid_frac[comp].mean()) < p["min_valid_ratio_confirmed"]:
            confirmed = False

    conf = p["conf_confirmed"] if confirmed else p["conf_candidate"]
    return sign * conf


class StairsFilter(PluginBase):
    """Flag stair flights, graded and signed.

    The layer is not a yes/no mask. Its magnitude says how sure the geometry
    is and its sign says which way the flight runs, because the two consumers
    want opposite errors: the occupancy grid lifts stairs to passable and must
    never be wrong about it, while the keepout channel closes ground off and
    would rather over-mark than let the robot onto stairs in a walking gait.
    A graded layer lets each pick its own cut, the way drivability and safety
    already do.

    Args:
        cell_n (int): map width/height in cells (injected by the manager).
        resolution (float): cell size in meters (injected by the manager).
        step_layer / slope_layer / roughness_layer (str): inputs.
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
        # Coerced against the default's own type, the way every other filter
        # in this chain does it: what the manager hands over for an
        # extra_param is not necessarily a plain number, and a threshold that
        # is not one blows up on first contact with a cupy array.
        self.params = {}
        for key, value in kwargs.items():
            if key not in DEFAULTS:
                continue
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

        return detect_stairs(
            elevation, valid, step, slope, roughness, self.resolution, self.params
        )
