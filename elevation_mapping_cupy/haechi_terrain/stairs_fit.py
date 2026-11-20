"""Stair flights from a MAPPING run: the robot's own 3D trajectory + the points it accumulated (library + offline CLI).

Why not per-frame detection: navi_lidar downsamples the merged cloud to 0.15 m voxels, so a single frame cannot separate
0.165 m risers (the elevation stairs_filter confirmed 0 % of 6 flights on the 2026-10-27 bag). During mapping the robot
walks every staircase it maps across, which gives two cues that do not depend on cloud density:

  1. flights from the trajectory   foot height (lidar z - sensor_height) vs arc length: stretches with |dz/ds| > 0.2
                                    over >= 0.6 m are stair traversals; repeated traversals of one flight are merged;
                                    flights chained top-to-bottom (< 0.3 m, < 4 m apart) form a staircase whose levels
                                    (floors, landings) are the robot's own flat-stretch heights
  2. steps from accumulated points  the clouds of many frames (each 0.15 m voxel grid in a different phase) sampled in
                                    5 cm voxels: per flight a ramp gives the axis, the lateral histogram the width, and a
                                    staircase blurred by the 0.15 m voxel (box filter) is fitted to the binned
                                    low-quantile profile: riser count n, rise, tread, first riser s0

Robustness (2026-10-29, Isaac Sim mapping run sim1): a flight cut in two by a pause on a step is re-joined (slope-mask gap
closing, transitive flight merge on height overlap, same-walk traversal merge); the robot's own body/legs are removed by
travelled distance (not time/frames, which fail at 0.3 m/s on stairs) and carved above 0.30 m on sloped stretches; the
ramp is a RANSAC line through the lane's low-quantile profile (slope 0.25-1.2) and the plane axis uses ramp inliers only.

Port of haechi_sim/scripts/terrain_stairs.py steps 2-3 (offline: 6/6 flights, residual MAD 4-6 cm on that bag), made
self-contained (numpy + scipy) so the mapping recorder can run it at save time.

Output (map_stairs.yaml, same flight schema as haechi_sim's stairs.json): per flight p0 / axis / lat (map frame), s0,
tread, n, rise, z_bot, z_top, l_lo, l_hi, width, residual; per staircase the flight indices (1-based) and level heights.

CLI: scripts/stairs_fit.py --record <map_dir>/stairs_record.npz --out <map_dir>     (refit a saved recording offline)
"""
import math
import os
import time

import numpy as np
from scipy import ndimage

from . import config

DEFAULTS = dict(sensor_height=0.74, min_slope=0.2, min_run=0.6, rise_min=0.10, rise_max=0.22, tread_min=0.22, tread_max=0.55,
                blur=0.15, rise_prior=0.165, n_tol=0.05, stair_margin=0.05, clearance=0.30,
                # ramp guard (2026-10-30): a sloped stretch whose low-quantile lane profile a straight line explains to within
                # ramp_mad_min (median |residual|) is a plain ramp, not stairs. Measured: synthetic 12-30 deg ramps with 1 cm
                # noise 0.002-0.005 m; real flights (bag_live 6, sim1 6) 0.014-0.120 m (the per-bin low quantile smooths the
                # riser sawtooth, so the margin is 3x, not rise/4). A rough outdoor ramp (> 1 cm) may still pass as stairs:
                # harmless for the footprint (cost lane), costs the scan filter / controller hold on it. Such a flight is
                # neither point-fitted nor registered from the trajectory.
                # 0.010 -> 0.006 (2026-11-20): with the smoother LIO z of the new IMU covariance (NAVI v2.1.0) real 1F bag
                # flights came down to 0.006-0.009 m and 0-2 of 6 fell out as "plain ramp" per run; 0.006 registers 6/6 on
                # every recorded run. Margin over the synthetic ramps is now 1 mm (0.005), not 2x.
                ramp_mad_min=0.006)
DEFAULTS.update(config.load("stairs_fit"))      # config/setups/haechi/stairs.yaml wins


class Cfg(dict):
    __getattr__ = dict.__getitem__


def cfg(**kw):
    c = Cfg(DEFAULTS); c.update({k: v for k, v in kw.items() if v is not None}); return c


# ------------------------------------------------------------------------------------------------ trajectory
def find_flights(t, foot, min_slope=0.2, min_run=0.6, close_m=0.3):
    """Sloped stretches of the trajectory (foot height vs arc length). One dict per traversal."""
    z = ndimage.median_filter(foot[:, 2], size=11, mode="nearest")
    ds = np.r_[0, np.hypot(np.diff(foot[:, 0]), np.diff(foot[:, 1]))]
    s = np.cumsum(ds)
    lo = np.searchsorted(s, s - 0.4); hi = np.minimum(np.searchsorted(s, s + 0.4), len(s) - 1)
    run = s[hi] - s[lo]
    slope = np.where(run > 0.3, (z[hi] - z[lo]) / np.maximum(run, 1e-6), 0.0)
    lab, n = ndimage.label(np.abs(slope) > min_slope)
    segs = []
    for k in range(1, n + 1):
        idx = np.flatnonzero(lab == k); i0, i1 = idx[0], idx[-1]
        if s[i1] - s[i0] < min_run:
            continue
        up = bool(z[i1] > z[i0])
        # one walk over a flight cut by a momentary dip of the windowed slope (robot hesitating on a step, SLAM z
        # jitter): re-join stretches of the same direction < close_m apart (a landing leaves >= 0.5 m even when the
        # +-0.4 m slope window spills onto it); slope blips shorter than min_run are dropped, never joined
        if segs and up == segs[-1][2] and s[i0] - s[segs[-1][1]] < close_m:
            segs[-1] = (segs[-1][0], i1, up)
        else:
            segs.append((i0, i1, up))
    trav = []
    for i0, i1, up in segs:
        trav.append(dict(i0=int(i0), i1=int(i1), t0=float(t[i0]), t1=float(t[i1]), p0=foot[i0, :2].tolist(), p1=foot[i1, :2].tolist(),
                         z0=float(z[max(i0 - 5, 0)]), z1=float(z[min(i1 + 5, len(z) - 1)]), up=up))
    return trav, z, slope


def _same_flight(a, b, zl, zh, fl):
    """Segment a->b (uphill) with heights zl..zh lies on flight fl: collinear within 25 deg / 1.5 m, heights overlap."""
    fa, fb = np.array(fl["p0"]), np.array(fl["p1"])
    d = fb - fa; L = np.linalg.norm(d); u = d / max(L, 1e-6); lat = np.array([-u[1], u[0]])
    v = b - a; cosang = abs(v @ u) / max(np.linalg.norm(v), 1e-6)
    off = max(abs((a - fa) @ lat), abs((b - fa) @ lat))
    ov = min(zh, fl["z_hi"]) - max(zl, fl["z_lo"])
    return cosang > math.cos(math.radians(25)) and off < 1.5 and ov > 0.3 * min(zh - zl, fl["z_hi"] - fl["z_lo"])


def _absorb(fl, a, b, zl, zh, trav):
    fa, fb = np.array(fl["p0"]), np.array(fl["p1"])
    d = fb - fa; L = np.linalg.norm(d); u = d / max(L, 1e-6)
    ss = [0.0, L, (a - fa) @ u, (b - fa) @ u]
    fl["p0"] = (fa + u * min(ss)).tolist(); fl["p1"] = (fa + u * max(ss)).tolist()
    fl["z_lo"], fl["z_hi"] = min(zl, fl["z_lo"]), max(zh, fl["z_hi"])
    fl["trav"] += trav


def merge_traversals(trav, walk_gap_s=3.0):
    """Same flight walked several times -> one flight (collinear within 1.5 m / 25 deg, overlapping heights)."""
    flights = []
    for tr in trav:
        a, b = np.array(tr["p0"]), np.array(tr["p1"])
        zl, zh = sorted((tr["z0"], tr["z1"]))
        if not tr["up"]:
            a, b = b, a
        for fl in flights:
            if _same_flight(a, b, zl, zh, fl):
                _absorb(fl, a, b, zl, zh, [tr]); break
        else:
            flights.append(dict(p0=a.tolist(), p1=b.tolist(), z_lo=zl, z_hi=zh, trav=[tr]))
    # the greedy pass compares a traversal only with the flights seeded before it: the two halves of a walk cut by a
    # pause (z 0.15-0.80 and 0.46-1.03 on sim1) seed two flights that overlap in height once their partners joined ->
    # merge flights pairwise until stable (adjacent real flights share a level and never overlap)
    changed = True
    while changed:
        changed = False
        for i in range(len(flights)):
            for j in range(i + 1, len(flights)):
                g = flights[j]
                if _same_flight(np.array(g["p0"]), np.array(g["p1"]), g["z_lo"], g["z_hi"], flights[i]):
                    _absorb(flights[i], np.array(g["p0"]), np.array(g["p1"]), g["z_lo"], g["z_hi"], g["trav"])
                    del flights[j]; changed = True; break
            if changed:
                break
    # pieces of one walk over the same flight (same direction, < walk_gap_s apart) are one traversal again, so that the
    # level search around traversal ends does not look at the pause in the middle of the flight
    for fl in flights:
        out = []
        for tr in sorted(fl["trav"], key=lambda x: x["t0"]):
            p = out[-1] if out else None
            if p is not None and tr["up"] == p["up"] and tr["t0"] - p["t1"] < walk_gap_s:
                out[-1] = dict(i0=min(p["i0"], tr["i0"]), i1=max(p["i1"], tr["i1"]), t0=min(p["t0"], tr["t0"]), t1=max(p["t1"], tr["t1"]),
                               p0=p["p0"], p1=tr["p1"], z0=p["z0"], z1=tr["z1"], up=p["up"])
            else:
                out.append(dict(tr))
        fl["trav"] = out
    return flights


def chain_flights(flights):
    """Staircases: flights whose top meets the next one's bottom (< 0.3 m in z, < 4 m in xy), lowest first."""
    order = sorted(range(len(flights)), key=lambda i: flights[i]["z_lo"])
    chains = []
    for i in order:
        fl = flights[i]
        for ch in chains:
            g = flights[ch[-1]]
            if abs(g["z_hi"] - fl["z_lo"]) < 0.3 and np.linalg.norm(np.array(g["p1"]) - np.array(fl["p0"])) < 4.0:
                ch.append(i); break
        else:
            chains.append([i])
    return chains


# ------------------------------------------------------------------------------------------------ surfaces
def column_surfaces(xyz, res, clearance, x0, y0, z0):
    """Top of every occupied stack with >= clearance free above. Returns ix, iy, iz(top voxel), zs(refined)."""
    ijk = np.floor((xyz - [x0, y0, z0]) / res).astype(np.int64)
    key = (ijk[:, 0] << 40) | (ijk[:, 1] << 20) | ijk[:, 2]
    order = np.argsort(key, kind="stable"); ks = key[order]
    uk, start = np.unique(ks, return_index=True)
    col = uk >> 20; iz = uk & 0xFFFFF
    nxt_same = np.r_[col[1:] == col[:-1], False]
    gap = np.where(nxt_same, np.r_[iz[1:] - iz[:-1], 0], 1 << 20)
    surf = gap > int(round(clearance / res))
    zsorted = xyz[order, 2]
    idx = np.flatnonzero(surf)
    zs = np.empty(len(idx))
    bounds = np.r_[start, len(ks)]
    colid = col[idx]
    for n, u in enumerate(idx):
        top = zsorted[bounds[u]:bounds[u + 1]].max()
        lo = u
        while lo > 0 and col[lo - 1] == colid[n] and iz[u] - iz[lo - 1] <= int(round(0.15 / res)) + 1:
            lo -= 1
        zz = zsorted[bounds[lo]:bounds[u + 1]]
        zs[n] = np.median(zz[zz >= top - 0.15])
    ix = (colid >> 20).astype(np.int64); iy = (colid & 0xFFFFF).astype(np.int64)
    return ix, iy, iz[idx].astype(np.int64), zs


def surface_points_5cm(P, clearance=0.30):
    if len(P) == 0:
        return P
    o0 = P.min(0) - 0.05
    ix, iy, iz, zs = column_surfaces(P, 0.05, clearance, *o0)
    return np.c_[o0[0] + (ix + 0.5) * 0.05, o0[1] + (iy + 0.5) * 0.05, zs]


# ------------------------------------------------------------------------------------------------ step fit
def huber(r, d=0.05):
    a = np.abs(r)
    return np.where(a < d, 0.5 * a * a, d * (a - 0.5 * d))


LAST_FAIL = ""
LAST_RAMP = None      # unconstrained line fit of the last lane profile between the levels: dict(mad, slope, bins)


def line_residual(cc, prof, z_lo, z_hi):
    """Median |profile - straight line| over the bins between the two levels (None when < 8 bins)."""
    m = np.isfinite(prof) & (prof > z_lo + 0.08) & (prof < z_hi - 0.08)
    if m.sum() < 8:
        return None
    line = np.polyfit(cc[m], prof[m], 1)
    return dict(mad=float(np.median(np.abs(prof[m] - np.polyval(line, cc[m])))), slope=float(line[0]), bins=int(m.sum()))


def _fail(msg):
    global LAST_FAIL
    LAST_FAIL = msg
    return None


def lane_profile(s, z, m, s_lo, s_hi, step=0.05, q=35, min_pts=3):
    """Low-quantile height per step-wide bin of s over the mask m (robust to the robot's own body above the steps)."""
    e = np.arange(s_lo, s_hi + step, step)
    cc = 0.5 * (e[:-1] + e[1:]); prof = np.full(len(cc), np.nan)
    idx = np.digitize(s[m], e) - 1; zz = z[m]
    for k in range(len(cc)):
        v = zz[idx == k]
        if len(v) >= min_pts:
            prof[k] = np.percentile(v, q)
    return cc, prof


def ransac_ramp(cc, prof, z_lo, z_hi, slope_lo=0.25, slope_hi=1.2, band=0.10, min_bins=5):
    """Line through the profile bins between the levels with the most inliers among lines of stair-like slope
    (0.25-1.2, i.e. 14-50 deg): landings inside the height band (biased trajectory levels) and body ghosts cannot win."""
    ok = np.isfinite(prof) & (prof > z_lo + 0.08) & (prof < z_hi - 0.08)
    x, y = cc[ok], prof[ok]
    if len(x) < min_bins:
        return None
    best = None
    for i in range(len(x)):
        dx = x[i + 1:] - x[i]
        for j in np.flatnonzero(dx > 0.2) + i + 1:
            k = (y[j] - y[i]) / (x[j] - x[i])
            if not slope_lo <= k <= slope_hi:
                continue
            r = np.abs(y - (y[i] + k * (x - x[i]))); inl = r < band
            score = (int(inl.sum()), -float(r[inl].sum()))
            if best is None or score > best[0]:
                best = (score, inl)
    if best is None or best[0][0] < min_bins:
        return None
    inl = best[1]; ramp = np.polyfit(x[inl], y[inl], 1)
    for _ in range(2):
        inl2 = np.abs(y - np.polyval(ramp, x)) < band
        if inl2.sum() < min_bins:
            break
        ramp2 = np.polyfit(x[inl2], y[inl2], 1)
        if not slope_lo <= ramp2[0] <= slope_hi:
            break
        inl, ramp = inl2, ramp2
    return ramp, int(inl.sum())


def flight_ramp(fl, P, c, lane=0.5):
    """Ramp + levels + axis of a flight from points P. Returns dict or None (reason in LAST_FAIL)."""
    p0, p1 = np.array(fl["p0"], float), np.array(fl["p1"], float)
    ax = p1 - p0; L = np.linalg.norm(ax); ax /= max(L, 1e-6); lat = np.array([-ax[1], ax[0]])
    s_all = (P[:, :2] - p0) @ ax; l_all = (P[:, :2] - p0) @ lat
    box = (s_all > -2.0) & (s_all < L + 2.0) & (np.abs(l_all) < 2.5) & (P[:, 2] > fl["z_lo"] - 0.5) & (P[:, 2] < fl["z_hi"] + 0.5)
    SP = surface_points_5cm(P[box], c.clearance)
    if len(SP) < 50:
        return _fail("%d surface points around the flight" % len(SP))
    z = SP[:, 2]; ax_tr = ax.copy()
    lc = fl.get("l_c", 0.0)
    if fl.get("axis_fixed") is not None:
        ax = np.array(fl["axis_fixed"]); lat = np.array([-ax[1], ax[0]])
    s = (SP[:, :2] - p0) @ ax; l = (SP[:, :2] - p0) @ lat
    ramp = None
    for it in range(3):
        core = np.abs(l - lc) < lane
        cc, prof = lane_profile(s, z, core & (z > fl["z_lo"] - 0.15) & (z < fl["z_hi"] + 0.15), -1.0, L + 1.0)
        global LAST_RAMP
        LAST_RAMP = line_residual(cc, prof, fl["z_lo"], fl["z_hi"])
        rr = ransac_ramp(cc, prof, fl["z_lo"], fl["z_hi"])
        if rr is None:
            return _fail("no stair-like ramp in the +-%.1f m lane (%d profile bins between the levels)" % (
                lane, int(np.sum(np.isfinite(prof) & (prof > fl["z_lo"] + 0.08) & (prof < fl["z_hi"] - 0.08)))))
        ramp = rr[0]
        if fl.get("axis_fixed") is not None or it == 2:
            break
        # axis = uphill gradient of a plane through the surface points on the ramp (the walked heading is off by degrees)
        sa_ = (fl["z_lo"] + 0.08 - ramp[1]) / ramp[0]; sb_ = (fl["z_hi"] - 0.08 - ramp[1]) / ramp[0]
        m = (np.abs(l - lc) < 1.5) & (np.abs(z - np.polyval(ramp, s)) < 0.15) & (s > sa_ - 0.1) & (s < sb_ + 0.1)
        if m.sum() < 30:
            break
        A = np.c_[SP[m, 0], SP[m, 1], np.ones(m.sum())]
        gx, gy, _ = np.linalg.lstsq(A, z[m], rcond=None)[0]
        g = np.array([gx, gy]); gn = np.linalg.norm(g)
        if gn < 0.15 or g @ ax_tr < math.cos(math.radians(25)) * gn:
            break
        if g @ ax > math.cos(math.radians(1.0)) * gn:
            break
        ax = g / gn; lat = np.array([-ax[1], ax[0]])
        s = (SP[:, :2] - p0) @ ax; l = (SP[:, :2] - p0) @ lat
    if ramp is None or ramp[0] < 0.15:
        return _fail("ramp slope %.2f" % (0.0 if ramp is None else ramp[0]))
    core = np.abs(l - lc) < lane
    s_a = (fl["z_lo"] - ramp[1]) / ramp[0]; s_b = (fl["z_hi"] - ramp[1]) / ramp[0]
    lo_m = core & (s > s_a - 0.6) & (s < s_a - 0.05) & (np.abs(z - fl["z_lo"]) < 0.25)
    hi_m = core & (s > s_b + 0.05) & (s < s_b + 0.6) & (np.abs(z - fl["z_hi"]) < 0.25)
    z_bot = float(np.median(z[lo_m])) if lo_m.sum() >= 5 else fl["z_lo"]
    z_top = float(np.median(z[hi_m])) if hi_m.sum() >= 5 else fl["z_hi"]
    s_a = (z_bot - ramp[1]) / ramp[0]; s_b = (z_top - ramp[1]) / ramp[0]
    s_tr = sorted(((np.array(fl["p0"]) - p0) @ ax, (np.array(fl["p1"]) - p0) @ ax))
    return dict(p0=p0, ax=ax, lat=lat, L=L, s=s, l=l, z=z, ramp=ramp, z_bot=z_bot, z_top=z_top, s_a=s_a, s_b=s_b, s_tr=s_tr)


def flight_width(fl, P, c):
    """Lateral extent: sloped surface points within 0.12 m of the ramp in the middle of the flight, dense histogram core."""
    r = flight_ramp(fl, P, c)
    if r is None:
        return None
    s, l, z, ramp = r["s"], r["l"], r["z"], r["ramp"]
    mid = (np.abs(z - np.polyval(ramp, s)) < 0.12) & (s > r["s_a"] + 0.25) & (s < r["s_b"] - 0.25) & (np.abs(l) < 2.5)
    hb = np.arange(-2.5, 2.55, 0.05); hcount, _ = np.histogram(l[mid], hb)
    if (hcount > 0).sum() < 4:
        return _fail("width: %d points on the ramp" % int(mid.sum()))
    dense = hcount >= max(2, 0.35 * np.percentile(hcount[hcount > 0], 80))
    labs, _ = ndimage.label(ndimage.binary_closing(dense, iterations=1))
    cb = int(np.argmin(np.abs(hb[:-1] + 0.025)))
    k = labs[cb] if labs[cb] else int(np.argmax(np.bincount(labs[labs > 0])))
    sel = np.flatnonzero(labs == k)
    return float(hb[sel[0]]) + c.stair_margin, float(hb[sel[-1] + 1]) - c.stair_margin, r["ax"].tolist()


def blurred_profile(S, s0, tread, n, z_bot, rise, w):
    """Staircase seen through a w-wide box filter (0.15 m voxel centroids: a riser is a w-wide linear transition)."""
    k = np.arange(n)
    sk = np.asarray(s0)[..., None] + k * np.asarray(tread)[..., None]
    x = np.clip((np.asarray(S)[..., None] - sk) / max(w, 1e-3) + 0.5, 0.0, 1.0)
    return z_bot + rise * x.sum(-1)


def isotonic(y, w=None):
    y = np.asarray(y, float); w = np.ones(len(y)) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            vals[-2:] = [v]; wts[-2:] = [wts[-2] + wts[-1]]; cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(vals, cnt)


def median_profile(S, Z, z_bot, z_top, step=0.05, q=35):
    e = np.arange(S.min(), S.max() + step, step)
    idx = np.digitize(S, e) - 1
    cc = 0.5 * (e[:-1] + e[1:])
    med = np.full(len(cc), np.nan); cnt = np.zeros(len(cc))
    for k in range(len(cc)):
        zz = Z[idx == k]
        if len(zz) >= 4:
            med[k] = np.percentile(zz, q); cnt[k] = len(zz)
    ok = np.isfinite(med)
    cc, med, cnt = cc[ok], med[ok], cnt[ok]
    if not len(cc):
        return cc, med
    return cc, np.clip(isotonic(med, np.sqrt(cnt)), z_bot, z_top)


def crossing(cc, m, level):
    j = np.flatnonzero(m >= level)
    if not len(j):
        return None
    j = j[0]
    if j == 0:
        return float(cc[0])
    return float(cc[j - 1] + (level - m[j - 1]) * (cc[j] - cc[j - 1]) / max(m[j] - m[j - 1], 1e-6))


def fit_steps(r, l_lo, l_hi, c, levels=None):
    """Blurred-staircase fit on the lane [l_lo, l_hi]: riser count n (rise prior when counts tie), tread, first riser s0."""
    s, l, z, z_bot, z_top = r["s"], r["l"], r["z"], r["z_bot"], r["z_top"]
    if levels is not None:
        z_bot, z_top = levels
    H = z_top - z_bot
    if H < 0.2:
        return _fail("levels %.2f m apart" % H)
    lane = (l > l_lo + 0.05) & (l < l_hi - 0.05) & (z > z_bot - 0.15) & (z < z_top + 0.15) & (s > r["s_tr"][0] - 1.0) & (s < r["s_tr"][1] + 1.0)
    S, Z = s[lane], z[lane]
    if len(S) < 30:
        return _fail("%d lane points for the step fit" % len(S))
    if len(S) > 4000:
        pick = np.random.default_rng(0).choice(len(S), 4000, replace=False); S, Z = S[pick], Z[pick]
    cc, med = median_profile(S, Z, z_bot, z_top)
    if len(cc) < 5:
        return _fail("%d profile bins" % len(cc))
    s_in = crossing(cc, med, z_bot + 0.1 * H); s_out = crossing(cc, med, z_top - 0.1 * H)
    if s_in is None or s_out is None or s_out <= s_in:
        return _fail("profile does not climb from z_bot to z_top (%s, %s)" % (s_in, s_out))
    keep = (S > s_in - 0.8) & (S < s_out + 0.8)
    S, Z = S[keep], Z[keep]
    per_n = {}
    n_prior = max(1, int(round(H / c.rise_prior)))
    for n in range(max(2, int(H / c.rise_max)), int(H / c.rise_min) + 1):
        rise = H / n
        if not (c.rise_min <= rise <= c.rise_max) and n != n_prior:
            continue
        # the profile may stop just short of z_top - rise/2 (sparse top step): aim at its own top instead of dropping n
        sa_ = crossing(cc, med, z_bot + 0.5 * rise); sb_ = crossing(cc, med, min(z_top - 0.5 * rise, float(med.max()) - 0.01))
        if sa_ is None or sb_ is None:
            continue
        t0 = (sb_ - sa_) / max(n - 1, 1) if n > 1 else 0.3
        treads = np.clip(np.arange(t0 - 0.06, t0 + 0.0601, 0.005), c.tread_min, c.tread_max)
        s0s = np.arange(sa_ - 0.2, sa_ + 0.2001, 0.01)
        T_, S0_ = np.meshgrid(treads, s0s, indexing="ij")
        prof = blurred_profile(S[None, None, :], S0_[..., None], T_[..., None], n, z_bot, rise, c.blur)
        cost = huber(Z[None, None, :] - prof, 0.03).mean(-1)
        i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
        per_n[n] = dict(cost=float(cost[i, j]), n=n, rise=rise, tread=float(T_[i, j]), s0=float(S0_[i, j]))
    if not per_n:
        return _fail("no riser count fits the profile")
    cmin = min(b["cost"] for b in per_n.values())
    clear = [b for b in per_n.values() if all(b["cost"] * (1.0 + c.n_tol) < o["cost"] for o in per_n.values() if o is not b)]
    # second opinion from the robust binned profile (immune to the body scatter above the steps that flattens the point
    # cost): the same grid search against (cc, med). Where single steps are resolved (5 cm sim voxels) it has a sharp
    # minimum at the true n; on 0.15 m voxel clouds it slides towards more, smaller steps, so it only counts when it is
    # a strict local minimum (>= 1.3x against both neighbours) within one step of the rise prior
    pk = (cc > s_in - 0.4) & (cc < s_out + 0.4)
    prof_n = {}
    for n, b in per_n.items():
        sa_ = crossing(cc, med, z_bot + 0.5 * b["rise"])
        treads = np.clip(np.arange(b["tread"] - 0.06, b["tread"] + 0.0601, 0.005), c.tread_min, c.tread_max)
        s0s = np.arange(sa_ - 0.2, sa_ + 0.2001, 0.01) if sa_ is not None else np.array([b["s0"]])
        T_, S0_ = np.meshgrid(treads, s0s, indexing="ij")
        pm = blurred_profile(cc[pk][None, None, :], S0_[..., None], T_[..., None], n, z_bot, b["rise"], c.blur)
        prof_n[n] = float(huber(med[pk][None, None, :] - pm, 0.03).mean(-1).min()) if pk.sum() >= 5 else float("nan")
    pmin = min(prof_n.values()) if prof_n and np.isfinite(list(prof_n.values())).all() else float("nan")
    n_prof = None
    if np.isfinite(pmin) and pmin > 0:
        for n, v in prof_n.items():
            nb = [prof_n[k] for k in (n - 1, n + 1) if k in prof_n]
            if nb and all(o > 1.3 * v for o in nb) and abs(n - n_prior) <= 1:
                n_prof = n
    if clear and len(per_n) > 1:
        best = dict(clear[0]); best["n_basis"] = "data"
    elif n_prof is not None:
        best = dict(per_n[n_prof]); best["n_basis"] = "profile"
    elif n_prior in per_n:
        best = dict(per_n[n_prior]); best["n_basis"] = "rise prior"
    else:
        best = dict(min(per_n.values(), key=lambda b: abs(b["rise"] - c.rise_prior))); best["n_basis"] = "nearest to rise prior"
    resid = Z - blurred_profile(S, best["s0"], best["tread"], best["n"], z_bot, best["rise"], c.blur)
    best.update(z_bot=float(z_bot), z_top=float(z_top), H=float(H), resid_mad=float(np.median(np.abs(resid))), n_pts=int(len(S)),
                slope_deg=float(np.degrees(np.arctan2(best["rise"], best["tread"]))),
                cost_by_n={str(k): round(v["cost"] / cmin, 4) for k, v in sorted(per_n.items())},
                prof_by_n={str(k): round(v / pmin, 4) for k, v in sorted(prof_n.items())} if np.isfinite(pmin) and pmin > 0 else {})
    best["_section"] = (S, Z, cc, med)
    return best


def fit_flight(fl, P, c, levels=None):
    global LAST_RAMP
    LAST_RAMP = None
    def plain_ramp():   # the lane profile between the levels is a straight line: a ramp, whatever else failed or fitted
        return LAST_RAMP is not None and LAST_RAMP["mad"] < c.ramp_mad_min
    w = flight_width(fl, P, c)
    if w is None:
        if plain_ramp():
            return _fail("plain ramp: a line explains the lane profile (residual %.3f m over %d bins, slope %.2f)" % (
                LAST_RAMP["mad"], LAST_RAMP["bins"], LAST_RAMP["slope"])) or "ramp"
        return None
    l_lo, l_hi, axis = w
    fl = dict(fl, l_c=0.5 * (l_lo + l_hi), axis_fixed=axis)
    r = flight_ramp(fl, P, c)
    f = fit_steps(r, l_lo, l_hi, c, levels) if r is not None else None
    line_mad = LAST_RAMP["mad"] if LAST_RAMP is not None else float("nan")
    if f is None:
        if plain_ramp():
            return _fail("plain ramp: a line explains the lane profile (residual %.3f m over %d bins, slope %.2f)" % (
                LAST_RAMP["mad"], LAST_RAMP["bins"], LAST_RAMP["slope"])) or "ramp"
        return None
    S, Z = f["_section"][0], f["_section"][1]
    f["ramp_mad"] = float(np.median(np.abs(Z - np.polyval(r["ramp"], S))))   # section points vs the RANSAC ramp line
    f["line_mad"] = line_mad                                                 # low-quantile lane profile vs a free line
    if plain_ramp() and f["resid_mad"] > 0.5 * line_mad:
        return _fail("plain ramp: lane profile is a line (residual %.3f m), step model %.3f m (no sawtooth)" % (line_mad, f["resid_mad"])) or "ramp"
    f.update(l_lo=l_lo, l_hi=l_hi, width=l_hi - l_lo, p0=r["p0"].tolist(), axis=r["ax"].tolist(), lat=r["lat"].tolist(), L=float(r["L"]),
             source="points", conf="high" if f["n_basis"] in ("data", "profile") else "medium")
    return f


# ------------------------------------------------------------------------------------------------ cleaning
def clean_points(xyz, frames, t_first, t_last, t, lidar_xyz, c=None, log=print, dyn_span=2.0, dyn_frames=8, carve_r=0.45,
                 near_r=1.5, near_frames=6, self_travel=0.8, slope_h=0.30):
    """Static points only (terrain_stairs.py rules): drop voxels seen briefly (< dyn_span s and < dyn_frames clouds: people
    walking by), voxels 0.08-1.9 m above the foot within near_r of the path that were seen over < self_travel m of robot
    travel (the robot's own body/legs, the operator next to it: time/frame counts fail when it walks stairs at 0.3 m/s)
    and, on flat stretches, that band within carve_r or seen in < near_frames clouds; on sloped stretches the body column
    is carved only above slope_h (the next step is up to 0.25 m above the foot within carve_r)."""
    from scipy.spatial import cKDTree
    c = c or cfg()
    keep = ~((t_last - t_first < dyn_span) & (frames < dyn_frames))
    foot = lidar_xyz.copy(); foot[:, 2] -= c.sensor_height
    _, zf, slope = find_flights(t, foot, c.min_slope, c.min_run)
    flat = np.abs(slope) < 0.1
    dd, fi = cKDTree(foot[:, :2]).query(xyz[:, :2], distance_upper_bound=near_r)
    ok = np.isfinite(dd); fi = np.minimum(fi, len(foot) - 1)
    h = xyz[:, 2] - (zf[fi] + 0.03)
    s_path = np.r_[0, np.cumsum(np.hypot(np.diff(foot[:, 0]), np.diff(foot[:, 1])))]
    travel = np.interp(t_last, t, s_path) - np.interp(t_first, t, s_path)
    band = ok & (h > 0.08) & (h < 1.9)
    self_g = band & (travel < self_travel)
    ghost = band & flat[fi] & ((dd < carve_r) | (frames < near_frames))
    ghost |= ok & ~flat[fi] & (dd < carve_r) & (h > slope_h) & (h < 1.9)
    log("cleaning: %d voxels, transient %d, self/body %d, path ghosts %d" % (
        len(xyz), int((~keep).sum()), int((self_g & keep).sum()), int((ghost & keep & ~self_g).sum())))
    return keep & ~ghost & ~self_g


# ------------------------------------------------------------------------------------------------ whole run
def fit_run(t, lidar_xyz, pts, c=None, log=print):
    """t (s), lidar positions (N,3, map frame), accumulated points (M,3, map frame) -> (flights, staircases, info)."""
    c = c or cfg()
    foot = lidar_xyz.copy(); foot[:, 2] -= c.sensor_height
    trav, zf, slope_t = find_flights(t, foot, c.min_slope, c.min_run)
    flights = merge_traversals(trav)
    log("stair traversals %d -> flights %d" % (len(trav), len(flights)))
    if not flights:
        return [], [], dict(traversals=0)
    # surface-above-foot offset on flat stretches (SLAM foot height vs the surface points under it)
    flat = np.abs(slope_t) < 0.08
    from scipy.spatial import cKDTree
    SP = surface_points_5cm(pts, c.clearance) if len(pts) else np.zeros((0, 3))
    foot_off = 0.0
    if len(SP):
        tr_ = cKDTree(SP[:, :2]); offs = []
        for j in np.flatnonzero(flat)[::5]:
            q = tr_.query_ball_point(foot[j, :2], 0.2)
            zz = [SP[k, 2] - zf[j] for k in q if abs(SP[k, 2] - zf[j]) < 0.15]
            if zz:
                offs.append(np.median(zz))
        foot_off = float(np.median(offs)) if offs else 0.0
    log("surface above the trajectory foot on flat stretches: %+.3f m" % foot_off)

    def level_at(spans):
        zs_ = []
        for ta_, tb_ in spans:
            m = (t >= ta_) & (t <= tb_) & flat
            zs_ += list(zf[m])
        return float(np.median(zs_)) + foot_off if len(zs_) >= 3 else None

    fits, staircases = [], []
    n_ramp = n_skipped = 0
    for ch in chain_flights(flights):
        lv = []
        for k in range(len(ch) + 1):
            spans = []
            if k > 0:
                spans += [(tr["t1"], tr["t1"] + 4.0) if tr["up"] else (tr["t0"] - 4.0, tr["t0"]) for tr in flights[ch[k - 1]]["trav"]]
            if k < len(ch):
                spans += [(tr["t0"] - 4.0, tr["t0"]) if tr["up"] else (tr["t1"], tr["t1"] + 4.0) for tr in flights[ch[k]]["trav"]]
            z_ = level_at(spans)
            if z_ is None:
                z_ = flights[ch[k - 1]]["z_hi"] if k > 0 else flights[ch[k]]["z_lo"]
            lv.append(z_)
        idx = []
        for k, i in enumerate(ch):
            f = fit_flight(flights[i], pts, c, levels=(lv[k], lv[k + 1]))
            if f == "ramp":
                log("flight z %.2f -> %.2f: %s -> not registered" % (lv[k], lv[k + 1], LAST_FAIL)); n_ramp += 1; continue
            if f is None:
                log("flight z %.2f -> %.2f: point fit failed (%s; lane line residual %s)" % (lv[k], lv[k + 1], LAST_FAIL,
                    "%.3f m / %d bins" % (LAST_RAMP["mad"], LAST_RAMP["bins"]) if LAST_RAMP else "n/a"))
                f = trajectory_flight(flights[i], lv[k], lv[k + 1], c, log)
                if f is None:
                    log("flight z %.2f -> %.2f: too few points and no usable traversal, skipped" % (lv[k], lv[k + 1])); n_skipped += 1; continue
            f["traversals"] = [dict(t0=tr["t0"], t1=tr["t1"], up=tr["up"]) for tr in flights[i]["trav"]]
            fits.append(f); idx.append(len(fits) - 1)
            log("flight %d: %d steps rise %.3f tread %.3f (%.1f deg) width %.2f m, z %.2f -> %.2f, residual %.3f m (n by %s), ramp line %.3f / lane line %.3f m" % (
                len(fits), f["n"], f["rise"], f["tread"], f["slope_deg"], f["width"], f["z_bot"], f["z_top"], f["resid_mad"], f["n_basis"], f.get("ramp_mad", float("nan")), f.get("line_mad", float("nan"))))
        if idx:
            staircases.append(dict(name="s%d" % (len(staircases) + 1), flights=[i + 1 for i in idx], levels=[round(v, 3) for v in lv]))
    return fits, staircases, dict(traversals=len(trav), flights_walked=len(flights), foot_off=foot_off, not_registered_ramp=n_ramp, not_registered_no_data=n_skipped)


def trajectory_flight(fl, z_bot, z_top, c, log=print):
    """Fallback when the point fit fails (sparse or noisy clouds): the robot DID walk this flight, so register it from the
    trajectory alone. Axis = walked direction, first riser = where the foot height left the lower level, run = where it
    reached the upper level, n = round(H / rise_prior), lane = +-traj_lane m around the walked line. Marked
    source 'trajectory' and conf 'low': good enough for the navigation footprint, not for riser-accurate geometry."""
    p0, p1 = np.array(fl["p0"], float), np.array(fl["p1"], float)
    L = np.linalg.norm(p1 - p0)
    H = z_top - z_bot
    if L < 0.5 or H < 0.2:
        return None
    ax = (p1 - p0) / L; lat = np.array([-ax[1], ax[0]])
    n = max(2, int(round(H / c.rise_prior))); rise = H / n
    run = L
    tread = float(np.clip(run / max(n - 1, 1), c.tread_min, c.tread_max))
    # the walked segment starts at the lower level and ends on the upper level: first riser ~0.3 m in, last riser ~0.3 m before the end
    s0 = 0.3
    if s0 + (n - 1) * tread > L - 0.3:
        tread = max(c.tread_min, (L - 0.6) / max(n - 1, 1))
    half = float(c.get("traj_lane", 0.8))
    log("flight z %.2f -> %.2f: point fit failed -> trajectory-only flight (%d x %.3f m, tread %.3f, lane +-%.1f m)" % (z_bot, z_top, n, rise, tread, half))
    return dict(n=n, rise=rise, tread=tread, s0=s0, z_bot=float(z_bot), z_top=float(z_top), H=float(H), resid_mad=float("nan"), n_pts=0,
                slope_deg=float(np.degrees(np.arctan2(rise, tread))), n_basis="trajectory", cost_by_n={}, l_lo=-half, l_hi=half, width=2 * half,
                p0=p0.tolist(), axis=ax.tolist(), lat=lat.tolist(), L=float(L), source="trajectory", conf="low")


def write_yaml(path, fits, staircases, meta):
    """map_stairs.yaml (plain YAML, flights in the map frame)."""
    import yaml
    keys = ("p0", "axis", "lat", "s0", "tread", "n", "rise", "z_bot", "z_top", "l_lo", "l_hi", "width", "H", "slope_deg", "resid_mad",
            "n_basis", "n_pts", "traversals", "source", "conf", "ramp_mad", "line_mad")

    def clean(v):
        if isinstance(v, (np.floating, float)):
            return round(float(v), 4)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        return v
    doc = dict(meta, frame="map", flights=[{k: clean(f[k]) for k in keys if k in f} for f in fits], staircases=clean(staircases))
    with open(path, "w") as fh:
        fh.write("# stair flights registered during mapping (stairs_prior_recorder / stairs_fit.py). Review map_stairs.png before use.\n")
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=None)


def figure(path, fits, staircases, t, lidar_xyz, pts, c=None):
    """Review figure: top view (trajectory, flights) + one section per flight (points, fitted steps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = c or cfg()
    nf = max(1, len(fits))
    fig = plt.figure(figsize=(6 + 4 * min(nf, 3), 5 + 3.2 * ((nf + 2) // 3)))
    gs = fig.add_gridspec(1 + (nf + 2) // 3, 3, height_ratios=[5] + [3.2] * ((nf + 2) // 3))
    ax = fig.add_subplot(gs[0, :])
    if len(pts):
        sub = pts[:: max(1, len(pts) // 200000)]
        ax.scatter(sub[:, 0], sub[:, 1], c=sub[:, 2], s=0.2, cmap="Greys", alpha=0.4)
    sc = ax.scatter(lidar_xyz[:, 0], lidar_xyz[:, 1], c=lidar_xyz[:, 2] - c.sensor_height, s=2, cmap="viridis")
    plt.colorbar(sc, ax=ax, fraction=0.03, label="foot z [m]")
    for k, f in enumerate(fits):
        p0, a_, la = np.array(f["p0"]), np.array(f["axis"]), np.array(f["lat"]); s1 = f["s0"] + (f["n"] - 1) * f["tread"] + 0.2
        cr = np.array([p0 + a_ * s + la * l for s, l in ((f["s0"], f["l_lo"]), (s1, f["l_lo"]), (s1, f["l_hi"]), (f["s0"], f["l_hi"]), (f["s0"], f["l_lo"]))])
        ax.plot(cr[:, 0], cr[:, 1], "r-", lw=1.5)
        ax.annotate("", xy=p0 + a_ * s1, xytext=p0 + a_ * f["s0"], arrowprops=dict(arrowstyle="->", color="r"))
        ax.text(*(p0 + a_ * (f["s0"] + s1) / 2 + la * (f["l_hi"] + 0.3)), "%d" % (k + 1), color="r", fontsize=10, weight="bold")
    ax.set_aspect("equal"); ax.set_title("registered stair flights (red, arrow = up) on the mapping trajectory", fontsize=10)
    for k, f in enumerate(fits):
        a2 = fig.add_subplot(gs[1 + k // 3, k % 3])
        if "_section" in f:
            S, Z, cc, med = f["_section"]
            a2.plot(S, Z, ".", ms=1.5, color="0.6"); a2.plot(cc, med, "-", color="b", lw=1, label="binned profile")
            ss = np.linspace(S.min(), S.max(), 400)
            a2.plot(ss, blurred_profile(ss, f["s0"], f["tread"], f["n"], f["z_bot"], f["rise"], 0.0), "r-", lw=1.2, label="fitted steps")
        a2.set_title("flight %d: %d x %.3f m rise, %.3f m tread, width %.2f m, resid %.3f%s" % (
            k + 1, f["n"], f["rise"], f["tread"], f["width"], f["resid_mad"], " (trajectory only)" if f.get("source") == "trajectory" else ""), fontsize=8)
        a2.set_xlabel("s along flight [m]", fontsize=7); a2.set_ylabel("z [m]", fontsize=7); a2.tick_params(labelsize=7)
        a2.legend(fontsize=6); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)
