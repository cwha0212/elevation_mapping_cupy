#
# Stairs from the elevation histogram, not from riser lines.
#
# A staircase quantises height. Inside a window that spans a flight, the
# observed elevations pile up on the tread levels, evenly spaced by the
# riser; a bank fills the same span continuously, a wall makes two piles a
# metre apart, a kerb makes two piles and nothing more. So the verdict is
# asked of the histogram itself -- how many well-separated levels, how much
# of the mass sits on them, and whether the level spacing is riser-sized.
#
# The previous detector counted riser LINES in the map plane and demanded
# riser/tread alternation ratios, which made the verdict a function of the
# viewpoint: a flight seen side-on from the driving lane showed two lines
# where the geometry demanded three, and the answer flickered with DDS
# timing because every gate sat on a knife edge (measured: one confirming
# frame in three otherwise-identical replays). The histogram is what the
# flight looks like from anywhere, so the lane view confirms steadily --
# 45-53 cells per frame across the whole north-lane pass on the canonical
# bag, where the line counter never confirmed once.
#
# What it deliberately does not do: confirm a thin side silhouette. From
# the far lane the flight's flank puts too few samples on each level to
# clear peak_frac, and lowering that bar was measured to hand the 25 degree
# bank a confirmed-stairs verdict through lidar ring aliasing (528 cells at
# peak_frac 0.06, 3 at 0.10). Ring piles and tread piles are near-identical
# at range; the sample-share bar is what tells them apart, so it stays.
#
import numpy as np

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase

DEFAULTS = dict(
    # 1.45 m: enough for FOUR levels of a 0.40 m-tread flight to fit with a
    # countable share each -- at 1.05 m the fourth level was geometrically
    # impossible and confirmation could never fire. The longer reach also
    # raises the absolute per-level sample bar (peak_frac x a bigger window),
    # which thins out lidar ring piles before they can impersonate treads.
    struct_window=29,
    # Height bin. Half the minimum riser: coarse enough that a noisy tread
    # stays one pile, fine enough that two adjacent levels stay two.
    bin_w=0.025,
    # Level spacing that counts as a riser. A level only counts at all when
    # another level sits a riser away from it -- the levels must CHAIN. A
    # stairwell wall's top is a perfectly good pile of samples a metre above
    # the flight, and it simply has no riser-spaced neighbour, so it drops
    # out of the chain instead of poisoning a mean-gap or total-gain test.
    min_riser=0.08,
    max_riser=0.33,
    # A level exists when it holds enough of the window's observed samples.
    # peak_frac is the bank/stairs discriminator -- see the header note.
    peak_min_cells=3,
    peak_frac=0.10,
    # A level must stand this many times over its neighbouring valley --
    # see the prominence note at the peak test.
    peak_prominence=2.0,
    # Share of the window's samples sitting on the levels. A staircase is
    # nothing but its levels; anything continuous leaks mass between them.
    cover_cand=0.55,
    cover_conf=0.65,
    # Three levels = two risers reads as a candidate; four = three risers
    # confirms, same repetition bar the line counter used.
    levels_cand=3,
    levels_conf=4,
    # Beyond this the camera-free geometry stops voting. 6.0 rather than the
    # old 4.5: with the 128-channel front lidar and the corrected ride
    # height, the lane-distance evidence is real (validated on the canonical
    # bag: confirmation appears, the banks stay silent).
    max_range=6.0,
    min_component_cells=40,
    max_levels=7,
    # A flight includes its own foot. The level trims hand the verdict to
    # cells ON the treads, which leaves the first riser's line and the half
    # tread of ground before it outside every region the gait eraser may
    # touch -- and the riser's occupancy marks live exactly there. Cells
    # within this reach of a component, at heights within a riser of its
    # lowest tread, are annexed; a stairwell wall (a metre up) and the top
    # platform (a flight up) both fail the height test by construction.
    foot_cells=4,
    conf_candidate=0.4,
    conf_confirmed=0.8,
)


def _gpu_modules(a):
    """scipy or cupyx, matching where the elevation actually lives."""
    try:
        import cupy as cp

        if isinstance(a, cp.ndarray):
            import cupyx.scipy.ndimage as ndi

            return cp, ndi, True
    except ImportError:
        pass
    import scipy.ndimage as ndi

    return np, ndi, False


def _host(a, is_gpu):
    return a.get() if is_gpu else a


def detect_stairs(elevation, valid, step, slope, roughness, resolution, params=None):
    """Stairs layer: +-conf_confirmed / +-conf_candidate / 0, nan unobserved.

    step/slope/roughness are accepted for interface compatibility and
    ignored: the histogram is asked of the elevation alone.
    """
    p = dict(DEFAULTS)
    if params:
        for key, value in params.items():
            if key in DEFAULTS:
                d = DEFAULTS[key]
                p[key] = int(value) if isinstance(d, int) else float(value)

    xp, ndi, is_gpu = _gpu_modules(elevation)
    valid = xp.asarray(valid)
    h, w = elevation.shape
    f32 = xp.float32
    sw = p["struct_window"]

    rows = xp.arange(h, dtype=f32) - h / 2.0 + 0.5
    cols = xp.arange(w, dtype=f32) - w / 2.0 + 0.5
    dist = xp.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2) * resolution

    small = xp.where(valid, elevation, 1e6)
    local_min = ndi.minimum_filter(small, size=sw, mode="nearest")
    nvalid = ndi.uniform_filter(valid.astype(f32), size=sw, mode="nearest") * sw * sw

    # ---- the histogram, one uniform_filter per height bin ----------------
    finite = _host(valid, is_gpu)
    if not finite.any():
        return xp.full((h, w), xp.nan, dtype=f32)
    el_host = _host(elevation, is_gpu)
    lo = float(np.nanmin(np.where(finite, el_host, np.nan)))
    hi = float(np.nanmax(np.where(finite, el_host, np.nan)))
    nb = min(int((hi - lo) / p["bin_w"]) + 2, 160)
    cnt = xp.zeros((nb, h, w), dtype=f32)
    for b in range(nb):
        m = valid & (elevation >= lo + b * p["bin_w"]) \
            & (elevation < lo + (b + 1) * p["bin_w"])
        cnt[b] = ndi.uniform_filter(m.astype(f32), size=sw,
                                    mode="nearest") * sw * sw

    # A level is a local maximum along the height axis, no closer to the
    # next than a riser, holding a real share of the window's samples.
    k = max(3, 2 * int(p["min_riser"] / p["bin_w"] / 2) + 1)
    nms = ndi.maximum_filter(cnt, size=(k, 1, 1), mode="nearest")
    thresh = xp.maximum(p["peak_min_cells"], p["peak_frac"] * nvalid)
    # Prominence: a tread stands over EMPTY neighbouring bins, because a
    # riser face is nearly vertical and leaves almost nothing between the
    # levels. A smooth slope sliced by the bin grid makes stripes of equal
    # mass -- peaks with full valleys -- and dies here, which is what keeps
    # a plain bank from impersonating a flight one bin-width at a time.
    valley = ndi.minimum_filter(cnt, size=(k, 1, 1), mode="nearest")
    peaks = ((cnt >= nms) & (cnt >= thresh)
             & (cnt >= p["peak_prominence"] * valley + p["peak_min_cells"]))

    # ...and it only counts as part of a FLIGHT when another level sits a
    # riser away. A wall top, a lone ledge, the platform seam: piles with no
    # riser-spaced neighbour, gone from the chain without a special case.
    gmin = max(1, int(round(p["min_riser"] / p["bin_w"])))
    gmax = max(gmin, int(round(p["max_riser"] / p["bin_w"])))
    neighbour = xp.zeros_like(peaks)
    for g in range(gmin, gmax + 1):
        neighbour[g:] |= peaks[:-g]
        neighbour[:-g] |= peaks[g:]
    chained = peaks & neighbour
    levels = chained.sum(axis=0)

    # Coverage: of the samples inside the chain's height span, how many sit
    # on the levels. The denominator stops at the span so a wall towering
    # over a stairwell does not dilute a perfectly combed flight below it,
    # while a bank -- which fills its own span continuously -- still fails.
    idx = xp.arange(nb, dtype=f32)[:, None, None]
    bmin = xp.min(xp.where(chained, idx, xp.inf), axis=0)
    bmax = xp.max(xp.where(chained, idx, -xp.inf), axis=0)
    within = (idx >= bmin[None] - 1) & (idx <= bmax[None] + 1)
    near = ndi.maximum_filter(chained.astype(f32), size=(3, 1, 1),
                              mode="nearest")
    mass_chain = (cnt * near).sum(axis=0)
    mass_span = (cnt * within).sum(axis=0)
    cover = mass_chain / xp.maximum(mass_span, 1e-3)

    ok_levels = (levels >= p["levels_cand"]) & (levels <= p["max_levels"])
    cand = (valid & (dist <= p["max_range"]) & ok_levels
            & (cover >= p["cover_cand"]))
    confirmed = cand & (levels >= p["levels_conf"]) & (cover >= p["cover_conf"])

    # The verdict belongs to the flight, not to its audience: a lane cell
    # beside the stairs passes every window test by looking at them. A cell
    # keeps its verdict only if its own height sits on one of the levels and
    # above the window's base, which no spectator on the surrounding ground
    # does and every tread above the first does.
    rel_bin = xp.clip(
        ((xp.nan_to_num(elevation, nan=lo) - lo) / p["bin_w"]).astype(xp.int32),
        0, nb - 1,
    )
    on_level = xp.take_along_axis(
        near, rel_bin[None, :, :].astype(xp.int64), axis=0
    )[0] > 0.5
    raised = elevation > (local_min + p["min_riser"] / 2.0)
    cand = cand & on_level & raised
    confirmed = confirmed & on_level & raised

    # ---- components, sign, grade: small-region work on the host ----------
    import scipy.ndimage as host_ndi

    cand_h = _host(cand, is_gpu)
    conf_h = _host(confirmed, is_gpu)
    elev_h = el_host
    dist_h = _host(dist, is_gpu)

    # Label over a closed copy so observation gaps do not shatter one flight
    # into sub-threshold shards -- the lane view arrives striped -- but the
    # verdict and the size belong to the cells that actually passed. An
    # opening here was measured to erase the entire (real) staircase: 116
    # surviving cells, all of them in stripes thinner than its structure.
    bridged = host_ndi.binary_closing(cand_h, structure=np.ones((3, 3), bool))
    out = np.zeros((h, w), dtype=np.float32)
    labels, n = host_ndi.label(bridged)
    for c in range(1, n + 1):
        comp = (labels == c) & cand_h
        size = int(comp.sum())
        if size < p["min_component_cells"]:
            continue
        sign = _climb_sign(comp, elev_h, dist_h)
        strong = int((conf_h & comp).sum())
        grade = p["conf_confirmed"] if strong >= max(
            p["min_component_cells"] // 2, size // 4
        ) else p["conf_candidate"]
        # ...and the flight's foot with it: the ground ring at the lowest
        # tread's level, first riser line included. Height does the safety
        # work -- see the foot_cells note in DEFAULTS.
        if p["foot_cells"] > 0:
            lowest = float(np.nanmin(np.where(comp, elev_h, np.nan)))
            ring = host_ndi.binary_dilation(
                comp, iterations=int(p["foot_cells"])
            ) & ~comp & finite
            foot = ring & (np.abs(elev_h - lowest) <= p["max_riser"])
            comp = comp | foot
        out[comp] = sign * grade

    out = np.where(finite, out, np.nan).astype(np.float32)
    return xp.asarray(out) if is_gpu else out


def _climb_sign(comp, elevation, dist, min_step=0.05):
    """Does this flight rise or fall as you walk away from the robot?

    Asked as the angle between the region's own uphill direction -- the
    least-squares height gradient over its cells -- and the direction from
    the robot to the region. Range-based readings (correlation with range,
    near half against far half) looked like the same question and were not:
    on a full-width flight most of the range is sideways distance, which
    diluted the trend below any usable threshold (correlation -0.18 on a
    plain descending flight against a -0.2 bar). The gradient is immune to
    the width of the flight.

    A flat gradient, or a region the robot is standing in, means the flight
    runs past rather than away. That returns positive: a robot already on
    stairs has to be able to keep going.
    Comparing against the ground under the robot sounds simpler but is
    measured exactly where it is least reliable: approaching a descent the
    robot stands at the lip, half its footprint on each level, and the
    estimate lands on the wrong side.
    """
    rr, cc = np.nonzero(comp)
    e = elevation[rr, cc].astype(np.float64)
    ok = np.isfinite(e)
    if ok.sum() < 6:
        return 1.0
    rr, cc, e = rr[ok].astype(np.float64), cc[ok].astype(np.float64), e[ok]
    r0, c0 = rr - rr.mean(), cc - cc.mean()
    # least-squares height gradient over the region: which way is uphill
    srr, scc, src = (r0 * r0).sum(), (c0 * c0).sum(), (r0 * c0).sum()
    det = srr * scc - src * src
    if det < 1e-9:
        return 1.0
    ser, sec = (r0 * e).sum(), (c0 * e).sum()
    gr = (scc * ser - src * sec) / det
    gc = (srr * sec - src * ser) / det
    g = float(np.hypot(gr, gc))
    if g < 1e-4:                       # flat to the gradient: runs past us
        return 1.0
    h, w = elevation.shape
    ar, ac = rr.mean() - h / 2.0 + 0.5, cc.mean() - w / 2.0 + 0.5
    away = float(np.hypot(ar, ac))
    if away < 1.0:                     # standing on it: keep going
        return 1.0
    heads_up = (gr * ar + gc * ac) / (g * away)
    return -1.0 if heads_up < -0.2 else 1.0


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
        step_layer / slope_layer / roughness_layer (str): accepted for
            interface compatibility; the histogram reads elevation alone.
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
