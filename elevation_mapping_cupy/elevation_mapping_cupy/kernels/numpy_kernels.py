#
# CPU versions of the CUDA kernels in custom_kernels.py / custom_image_kernels.py.
#
# Same call signatures as the cupy ElementwiseKernels so elevation_mapping.py
# does not care which it holds. Same conventions: maps are (layers, H, W),
# row = Y, col = X, flat index = W * iy + ix, the outermost ring is padding.
#
# Two kernels walk a ray per point (visibility cleanup in add_points, the
# occlusion check in the image projection). Those are plain loops under
# numba when it is installed; without numba a vectorised fallback does the
# same work, slower. Everything else is ordinary numpy.
#
import math
import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    import numba
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    numba = None
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # noqa: D401 - decorator shim
        if args and callable(args[0]):
            return args[0]

        def wrap(f):
            return f

        return wrap


# --------------------------------------------------------------------------
# shared geometry helpers
# --------------------------------------------------------------------------
def _cell_index(x, y, cx, cy, resolution, width, height):
    """Clamped flat index per point, plus the 'inside' (not on the padding ring) mask."""
    ix = np.floor((x - cx) / resolution + 0.5 * (width - 1) + 0.5).astype(np.int64)
    iy = np.floor((y - cy) / resolution + 0.5 * (height - 1) + 0.5).astype(np.int64)
    inside = (ix > 0) & (ix < width - 1) & (iy > 0) & (iy < height - 1)
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)
    return width * iy + ix, inside


def _transform(p, R, t):
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t, dtype=np.float32).reshape(3)
    return p[:, :3].astype(np.float32, copy=False) @ R.T + t


def _is_valid(x, y, z, t, min_valid_distance, max_height_range, a, b, c):
    sx, sy, sz = float(t[0]), float(t[1]), float(t[2])
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    dx, dy, dz = x - sx, y - sy, z - sz
    d2 = dx * dx + dy * dy + dz * dz
    dxy = np.maximum(np.sqrt(dx * dx + dy * dy) - b, 0.0)
    ok = finite & (d2 >= min_valid_distance * min_valid_distance)
    ok &= ~(dz > dxy * a + c)
    ok &= ~(dz > max_height_range)
    return ok


# --------------------------------------------------------------------------
# add_points: fuse points into newmap, mark visibility along their rays
# --------------------------------------------------------------------------
@njit(cache=True)
def _visibility_walk(
    xs, ys, zs, tx, ty, tz, cx, cy, resolution, width, height, max_ray_length, cleanup_step,
    cleanup_cos_thresh, wall_num_thresh, map_flat, norm_flat, newmap_flat, vis_flat,
):
    layer = width * height
    n = xs.shape[0]
    for i in range(n):
        vx = xs[i] - tx
        vy = ys[i] - ty
        vz = zs[i] - tz
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if norm > 0.0:
            rx, ry, rz = vx / norm, vy / norm, vz / norm
        else:
            rx = ry = rz = 0.0
        ray_length = min(norm, max_ray_length)
        end_x = tx + rx * ray_length
        end_y = ty + ry * ray_length
        end_z = tz + rz * ray_length
        start_gx = (tx - cx) / resolution + 0.5 * (width - 1) + 0.5
        start_gy = (ty - cy) / resolution + 0.5 * (height - 1) + 0.5
        end_gx = (end_x - cx) / resolution + 0.5 * (width - 1) + 0.5
        end_gy = (end_y - cy) / resolution + 0.5 * (height - 1) + 0.5
        gdx = end_gx - start_gx
        gdy = end_gy - start_gy
        cell_x = int(math.floor(start_gx))
        cell_y = int(math.floor(start_gy))
        end_cell_x = int(math.floor(end_gx))
        end_cell_y = int(math.floor(end_gy))
        step_x = (1 if gdx > 0.0 else 0) - (1 if gdx < 0.0 else 0)
        step_y = (1 if gdy > 0.0 else 0) - (1 if gdy < 0.0 else 0)
        delta_x = 1.0e30 if step_x == 0 else abs(1.0 / gdx)
        delta_y = 1.0e30 if step_y == 0 else abs(1.0 / gdy)
        boundary_x = cell_x + (1.0 if step_x > 0 else 0.0)
        boundary_y = cell_y + (1.0 if step_y > 0 else 0.0)
        next_x = 1.0e30 if step_x == 0 else (boundary_x - start_gx) / gdx
        next_y = 1.0e30 if step_y == 0 else (boundary_y - start_gy) / gdy
        for _ in range(width + height):
            if next_x < next_y:
                entry = next_x
                next_x += delta_x
                cell_x += step_x
            elif next_y < next_x:
                entry = next_y
                next_y += delta_y
                cell_y += step_y
            else:
                entry = next_x
                next_x += delta_x
                next_y += delta_y
                cell_x += step_x
                cell_y += step_y
            if entry >= 1.0 or (cell_x == end_cell_x and cell_y == end_cell_y):
                break
            if cell_x <= 0 or cell_x >= width - 1 or cell_y <= 0 or cell_y >= height - 1:
                continue
            nidx = cell_y * width + cell_x
            exit_ = min(min(next_x, next_y), 1.0)
            sample = 0.5 * (entry + exit_)
            nz = tz + (end_z - tz) * sample
            nmap_h = map_flat[nidx]
            nmap_v = map_flat[layer + nidx]
            nmap_valid = map_flat[2 * layer + nidx]
            non_updated_t = map_flat[4 * layer + nidx]
            if nmap_valid < 0.5:
                if nz < vis_flat[2 * layer + nidx]:
                    vis_flat[2 * layer + nidx] = nz
                continue
            if non_updated_t < 0.5:
                continue
            if nmap_h > nz + 0.01 - min(nmap_v, 1.0) * 0.05:
                product = rx * norm_flat[nidx] + ry * norm_flat[layer + nidx] + rz * norm_flat[2 * layer + nidx]
                if abs(product) < cleanup_cos_thresh:
                    continue
                num_points = newmap_flat[3 * layer + nidx]
                if num_points > wall_num_thresh and non_updated_t < 1.0:
                    continue
                vis_flat[nidx] += cleanup_step / (ray_length / max_ray_length)
                vis_flat[layer + nidx] += 1.0
                if nz < vis_flat[2 * layer + nidx]:
                    vis_flat[2 * layer + nidx] = nz


def _visibility_walk_vectorised(
    xs, ys, zs, t, cx, cy, resolution, width, height, max_ray_length, cleanup_step,
    cleanup_cos_thresh, wall_num_thresh, map_, norm_map, newmap, visibility,
):
    """numpy fallback for the ray walk: all rays advance one cell per iteration."""
    tx, ty, tz = (float(v) for v in t)
    vx, vy, vz = xs - tx, ys - ty, zs - tz
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    nz_ = norm > 0
    rx = np.where(nz_, vx / np.where(nz_, norm, 1), 0.0)
    ry = np.where(nz_, vy / np.where(nz_, norm, 1), 0.0)
    rz = np.where(nz_, vz / np.where(nz_, norm, 1), 0.0)
    ray_length = np.minimum(norm, max_ray_length)
    end_x, end_y, end_z = tx + rx * ray_length, ty + ry * ray_length, tz + rz * ray_length
    sgx = (tx - cx) / resolution + 0.5 * (width - 1) + 0.5
    sgy = (ty - cy) / resolution + 0.5 * (height - 1) + 0.5
    egx = (end_x - cx) / resolution + 0.5 * (width - 1) + 0.5
    egy = (end_y - cy) / resolution + 0.5 * (height - 1) + 0.5
    gdx, gdy = egx - sgx, egy - sgy
    cell_x = np.full(xs.shape, int(math.floor(sgx)), np.int64)
    cell_y = np.full(xs.shape, int(math.floor(sgy)), np.int64)
    end_cx, end_cy = np.floor(egx).astype(np.int64), np.floor(egy).astype(np.int64)
    step_x = np.sign(gdx).astype(np.int64)
    step_y = np.sign(gdy).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_x = np.where(step_x == 0, 1e30, np.abs(1.0 / gdx))
        delta_y = np.where(step_y == 0, 1e30, np.abs(1.0 / gdy))
        bx = cell_x + (step_x > 0)
        by = cell_y + (step_y > 0)
        next_x = np.where(step_x == 0, 1e30, (bx - sgx) / gdx)
        next_y = np.where(step_y == 0, 1e30, (by - sgy) / gdy)
    active = np.ones(xs.shape, bool)
    layer = width * height
    m0, m1, m2, m4 = map_[0].ravel(), map_[1].ravel(), map_[2].ravel(), map_[4].ravel()
    n0, n1, n2 = norm_map[0].ravel(), norm_map[1].ravel(), norm_map[2].ravel()
    np3 = newmap[3].ravel()
    v0, v1, v2 = visibility[0].ravel(), visibility[1].ravel(), visibility[2].ravel()
    weight = cleanup_step / np.maximum(ray_length / max_ray_length, 1e-9)
    for _ in range(width + height):
        if not active.any():
            break
        lt = next_x < next_y
        gt = next_y < next_x
        eq = ~lt & ~gt
        entry = np.where(lt | eq, next_x, next_y)
        next_x = np.where(lt | eq, next_x + delta_x, next_x)
        next_y = np.where(gt | eq, next_y + delta_y, next_y)
        cell_x = np.where(lt | eq, cell_x + step_x, cell_x)
        cell_y = np.where(gt | eq, cell_y + step_y, cell_y)
        done = (entry >= 1.0) | ((cell_x == end_cx) & (cell_y == end_cy))
        active &= ~done
        inside = active & (cell_x > 0) & (cell_x < width - 1) & (cell_y > 0) & (cell_y < height - 1)
        if not inside.any():
            continue
        sel = np.flatnonzero(inside)
        nidx = cell_y[sel] * width + cell_x[sel]
        exit_ = np.minimum(np.minimum(next_x[sel], next_y[sel]), 1.0)
        nz = tz + (end_z[sel] - tz) * 0.5 * (entry[sel] + exit_)
        invalid = m2[nidx] < 0.5
        if invalid.any():
            np.minimum.at(v2, nidx[invalid], nz[invalid])
        cand = ~invalid & (m4[nidx] >= 0.5)
        cand &= m0[nidx] > nz + 0.01 - np.minimum(m1[nidx], 1.0) * 0.05
        if cand.any():
            s2 = sel[cand]
            ni = nidx[cand]
            product = rx[s2] * n0[ni] + ry[s2] * n1[ni] + rz[s2] * n2[ni]
            keep = np.abs(product) >= cleanup_cos_thresh
            keep &= ~((np3[ni] > wall_num_thresh) & (m4[ni] < 1.0))
            if keep.any():
                ni, s2 = ni[keep], s2[keep]
                np.add.at(v0, ni, weight[s2])
                np.add.at(v1, ni, 1.0)
                np.minimum.at(v2, ni, nz[cand][keep])


def add_points_kernel(
    resolution, width, height, sensor_noise_factor, mahalanobis_thresh, outlier_variance,
    wall_num_thresh, max_ray_length, cleanup_step, min_valid_distance, max_height_range,
    cleanup_cos_thresh, ramped_height_range_a, ramped_height_range_b, ramped_height_range_c,
    enable_edge_shaped=True, enable_visibility_cleanup=True,
):
    width, height = int(width), int(height)
    layer = width * height

    def run(center_x, center_y, R, t, map_, norm_map, p, newmap, visibility, size=None):
        n = p.shape[0] if size is None else int(size)
        if n == 0:
            return
        pts = p[:n, :3]
        xyz = _transform(pts, R, t)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        v = sensor_noise_factor * (pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2)
        cx, cy = float(center_x[0]), float(center_y[0])
        idx, inside = _cell_index(x, y, cx, cy, resolution, width, height)
        valid = _is_valid(
            x, y, z, t, min_valid_distance, max_height_range,
            ramped_height_range_a, ramped_height_range_b, ramped_height_range_c,
        )

        sel = valid & inside
        if sel.any():
            si = idx[sel]
            zs, vs = z[sel], v[sel]
            m0, m1 = map_[0].ravel(), map_[1].ravel()
            n4 = newmap[4].ravel()
            map_h, map_v, num_points = m0[si], m1[si], n4[si]
            outlier = np.abs(map_h - zs) > map_v * mahalanobis_thresh
            fuse = ~outlier
            if enable_edge_shaped:
                with np.errstate(divide="ignore", invalid="ignore"):
                    edge = (num_points > wall_num_thresh) & (
                        zs < map_h - map_v * mahalanobis_thresh / np.where(num_points > 0, num_points, 1.0)
                    )
                fuse &= ~edge
            if outlier.any():
                newmap[5].ravel()[:] += np.bincount(si[outlier], minlength=layer).astype(np.float32)
            if fuse.any():
                fi = si[fuse]
                mh, mv, zz, vv = map_h[fuse], map_v[fuse], zs[fuse], vs[fuse]
                new_h = (mh * vv + zz * mv) / (mv + vv)
                new_v = (mv * vv) / (mv + vv)
                newmap[0].ravel()[:] += np.bincount(fi, weights=new_h, minlength=layer).astype(np.float32)
                newmap[1].ravel()[:] += np.bincount(fi, weights=new_v, minlength=layer).astype(np.float32)
                newmap[2].ravel()[:] += np.bincount(fi, minlength=layer).astype(np.float32)

        if enable_visibility_cleanup and valid.any():
            xs, ys, zs = x[valid].astype(np.float64), y[valid].astype(np.float64), z[valid].astype(np.float64)
            if HAS_NUMBA:
                _visibility_walk(
                    xs, ys, zs, float(t[0]), float(t[1]), float(t[2]), cx, cy, float(resolution),
                    width, height, float(max_ray_length), float(cleanup_step), float(cleanup_cos_thresh),
                    float(wall_num_thresh), map_.reshape(-1), norm_map.reshape(-1), newmap.reshape(-1),
                    visibility.reshape(-1),
                )
            else:
                _visibility_walk_vectorised(
                    xs, ys, zs, t, cx, cy, float(resolution), width, height, float(max_ray_length),
                    float(cleanup_step), float(cleanup_cos_thresh), float(wall_num_thresh),
                    map_, norm_map, newmap, visibility,
                )

    return run


# --------------------------------------------------------------------------
# error_counting: drift statistics and per-cell hit counts
# --------------------------------------------------------------------------
def error_counting_kernel(
    resolution, width, height, sensor_noise_factor, mahalanobis_thresh, outlier_variance,
    traversability_inlier, min_valid_distance, max_height_range, ramped_height_range_a,
    ramped_height_range_b, ramped_height_range_c,
):
    width, height = int(width), int(height)
    layer = width * height

    def run(map_, p, center_x, center_y, R, t, newmap, error, error_cnt, size=None):
        n = p.shape[0] if size is None else int(size)
        if n == 0:
            return
        xyz = _transform(p[:n, :3], R, t)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        valid = _is_valid(
            x, y, z, t, min_valid_distance, max_height_range,
            ramped_height_range_a, ramped_height_range_b, ramped_height_range_c,
        )
        idx, inside = _cell_index(x, y, float(center_x[0]), float(center_y[0]), resolution, width, height)
        sel = valid & inside
        if not sel.any():
            return
        si, zs = idx[sel], z[sel]
        m0, m1, m2, m3 = (map_[k].ravel()[si] for k in range(4))
        inlier = (m2 > 0.5) & (np.abs(m0 - zs) < m1 * mahalanobis_thresh)
        inlier &= (m1 < outlier_variance / 2.0) & (m3 > traversability_inlier)
        if inlier.any():
            e = zs[inlier] - m0[inlier]
            error[0] += np.float32(e.sum())
            error_cnt[0] += np.float32(inlier.sum())
            newmap[3].ravel()[:] += np.bincount(si[inlier], minlength=layer).astype(np.float32)
        newmap[4].ravel()[:] += np.bincount(si, minlength=layer).astype(np.float32)

    return run


# --------------------------------------------------------------------------
# finalize: turn the per-frame proposals into the next map
# --------------------------------------------------------------------------
def finalize_map_kernel(width, height, max_variance, initial_variance, outlier_variance):
    def run(previous, newmap, visibility, map_, size=None):
        map_[...] = previous
        new_count = newmap[2]
        outlier_count = newmap[5]
        has = new_count > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            new_height = np.where(has, newmap[0] / np.where(has, new_count, 1.0), 0.0)
            new_variance = np.where(has, newmap[1] / np.where(has, new_count, 1.0), 0.0)
        # cells with new measurements
        map_[0] = np.where(has, new_height, map_[0])
        map_[1] = np.where(has, new_variance, map_[1])
        map_[2] = np.where(has, 1.0, map_[2])
        map_[4] = np.where(has, 0.0, map_[4])
        map_[5] = np.where(has, new_height, map_[5])
        map_[6] = np.where(has, 0.0, map_[6])
        blown = has & (new_variance > max_variance)
        map_[0] = np.where(blown, 0.0, map_[0])
        map_[1] = np.where(blown, initial_variance, map_[1])
        map_[2] = np.where(blown, 0.0, map_[2])
        # cells without: outliers, visibility cleanup, upper bound
        no = ~has
        map_[1] = np.where(no, map_[1] + outlier_count * outlier_variance, map_[1])
        cleanup = visibility[0]
        cl = no & (cleanup > 0.0)
        map_[2] = np.where(cl, map_[2] - cleanup, map_[2])
        map_[1] = np.where(cl, map_[1] + visibility[1] * outlier_variance, map_[1])
        proposed = visibility[2]
        prev_ub, prev_has = previous[5], previous[6]
        take = no & np.isfinite(proposed) & ((prev_has < 0.5) | (proposed < prev_ub))
        map_[5] = np.where(take, proposed, map_[5])
        map_[6] = np.where(take, 1.0, map_[6])
        # invalidated cells reset
        inv = map_[2] < 0.5
        map_[0] = np.where(inv, 0.0, map_[0])
        map_[1] = np.where(inv, initial_variance, map_[1])
        map_[2] = np.where(inv, 0.0, map_[2])

    return run


# --------------------------------------------------------------------------
# dilation: fill invalid cells from the nearest valid one within a radius
# --------------------------------------------------------------------------
def dilation_filter_kernel(width, height, dilation_size):
    if dilation_size < 0:
        raise ValueError("dilation_size must be non-negative")

    def run(map_in, mask_in, map_out, mask_out, size=None):
        valid = mask_in > 0.5
        if dilation_size == 0:
            map_out[...] = map_in
            mask_out[...] = valid
            return
        seeds = valid.copy()
        seeds[[0, -1], :] = False
        seeds[:, [0, -1]] = False
        if not seeds.any():
            map_out[...] = map_in
            mask_out[...] = valid
            return
        distances, indices = distance_transform_edt(~seeds, return_indices=True)
        rows = np.clip(indices[0], 0, height - 1)
        cols = np.clip(indices[1], 0, width - 1)
        nearest = map_in[rows, cols]
        replace = ~valid & (distances <= dilation_size)
        map_out[...] = np.where(replace, nearest, map_in)
        mask_out[...] = (valid | replace).astype(map_out.dtype)

    return run


# --------------------------------------------------------------------------
# normals from forward differences, interior cells only
# --------------------------------------------------------------------------
def normal_filter_kernel(width, height, resolution):
    def run(map_, mask, newmap, size=None):
        h = map_
        valid = mask > 0.5
        dzdx = np.zeros_like(h)
        dzdy = np.zeros_like(h)
        dzdx[:, :-1] = h[:, 1:] - h[:, :-1]
        dzdy[:-1, :] = h[1:, :] - h[:-1, :]
        nx = -dzdx / resolution
        ny = -dzdy / resolution
        norm = np.sqrt(nx * nx + ny * ny + 1.0)
        ok = valid.copy()
        ok[0, :] = ok[-1, :] = False
        ok[:, 0] = ok[:, -1] = False
        # the CUDA kernel requires the +x and +y neighbours to be interior too
        ok[-2, :] = False
        ok[:, -2] = False
        newmap[0] = np.where(ok, nx / norm, newmap[0])
        newmap[1] = np.where(ok, ny / norm, newmap[1])
        newmap[2] = np.where(ok, 1.0 / norm, newmap[2])

    return run


# --------------------------------------------------------------------------
# image -> map correspondence with an occlusion walk toward the camera cell
# --------------------------------------------------------------------------
@njit(cache=True)
def _occlusion_walk(rows, cols, z0s, x1, y1, z1, map0, map2, width, height, tol, out):
    # Bresenham from each cell (x0=row, y0=col) toward the camera cell (x1, y1),
    # exactly as the CUDA kernel does it.
    for k in range(rows.shape[0]):
        x0 = rows[k]
        y0 = cols[k]
        x0c, y0c = x0, y0
        z0 = z0s[k]
        total = math.sqrt(float((x0c - x1) ** 2 + (y0c - y1) ** 2))
        delta_z = z1 - z0
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        ok = True
        while True:
            if x0 == x1 and y0 == y1:
                break
            if 0 <= x0 < height and 0 <= y0 < width:
                if map2[x0, y0] != 0.0:
                    dis = math.sqrt(float((x0c - x0) ** 2 + (y0c - y0) ** 2))
                    rayheight = z0 + (dis / total * delta_z) if total > 0 else z0
                    if map0[x0, y0] - tol > rayheight:
                        ok = False
                        break
            e2 = 2 * err
            if e2 >= dy:
                if x0 == x1:
                    break
                err += dy
                x0 += sx
            if e2 <= dx:
                if y0 == y1:
                    break
                err += dx
                y0 += sy
        out[k] = ok


def _occlusion_viewshed(rows, cols, z0s, x1, y1, z1, map0, map2, tol):
    """numpy fallback: angular-bin viewshed from the camera cell.

    A cell is visible when no valid cell closer to the camera, in the same
    angular bin, rises above the sight line by more than ``tol``. Slightly
    coarser than the per-cell Bresenham walk, but vectorised.
    """
    vr, vc = np.nonzero(map2 > 0.5)
    dr, dc = vr.astype(np.float64) - x1, vc.astype(np.float64) - y1
    r = np.sqrt(dr * dr + dc * dc)
    nbins = 1440
    b = np.floor((np.arctan2(dc, dr) + np.pi) / (2 * np.pi) * nbins).astype(np.int64) % nbins
    with np.errstate(divide="ignore", invalid="ignore"):
        occ = np.where(r > 0, (map0[vr, vc] - tol - z1) / np.where(r > 0, r, 1), -np.inf)
    order = np.lexsort((r, b))
    b_s, s_s = b[order], occ[order]
    # horizon seen from the camera before reaching each cell: running max of
    # the occluder slopes of the closer cells in the same bin
    shifted = np.r_[-np.inf, s_s[:-1]]
    run = np.empty_like(shifted)
    starts = np.r_[0, np.flatnonzero(np.diff(b_s)) + 1]
    ends = np.r_[starts[1:], len(b_s)]
    for a, e in zip(starts, ends):
        seg = shifted[a:e].copy()
        seg[0] = -np.inf
        run[a:e] = np.maximum.accumulate(seg)
    horizon = np.full(map2.shape, -np.inf)
    horizon[vr[order], vc[order]] = run
    dq = np.sqrt((rows - x1) ** 2.0 + (cols - y1) ** 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sight = np.where(dq > 0, (z0s - z1) / np.where(dq > 0, dq, 1), np.inf)
    return sight >= horizon[rows, cols]


def image_to_map_correspondence_kernel(resolution, width, height, tolerance_z_collision):
    width, height = int(width), int(height)

    def run(map_, x1, y1, z1, P, K, D, image_height, image_width, center, uv_correspondence, valid_correspondence, size=None):
        P = np.asarray(P, dtype=np.float64).reshape(-1)
        K = np.asarray(K, dtype=np.float64).reshape(-1)
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        img_h, img_w = float(image_height), float(image_width)
        x1i, y1i, z1f = int(x1), int(y1), float(z1)
        c = np.asarray(center, dtype=np.float64).reshape(-1)
        valid_cells = map_[2] == 1
        rows, cols = np.nonzero(valid_cells)
        if rows.size == 0:
            return
        # row = Y (x0 in the CUDA code), col = X (y0)
        p1 = (cols - width / 2) * resolution + c[0]
        p2 = (rows - height / 2) * resolution + c[1]
        p3 = map_[0][rows, cols].astype(np.float64) + c[2]
        u = p1 * P[0] + p2 * P[1] + p3 * P[2] + P[3]
        v = p1 * P[4] + p2 * P[5] + p3 * P[6] + P[7]
        d = p1 * P[8] + p2 * P[9] + p3 * P[10] + P[11]
        front = d > 0
        if not front.any():
            return
        rows, cols, u, v, d = rows[front], cols[front], u[front], v[front], d[front]
        u, v = u / d, v / d
        if np.any(D[:5] != 0):
            k1, k2, pp1, pp2, k3 = D[:5]
            fx, fy, cx, cy = K[0], K[4], K[2], K[5]
            xn, yn = (u - cx) / fx, (v - cy) / fy
            r2 = xn * xn + yn * yn
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            uc = xn * radial + 2 * pp1 * xn * yn + pp2 * (r2 + 2 * xn * xn)
            vc = yn * radial + 2 * pp2 * xn * yn + pp1 * (r2 + 2 * yn * yn)
            u, v = fx * uc + cx, fy * vc + cy
        inimg = (u >= 0) & (v >= 0) & (u < img_w) & (v < img_h)
        if not inimg.any():
            return
        rows, cols, u, v = rows[inimg], cols[inimg], u[inimg], v[inimg]
        z0 = map_[0][rows, cols].astype(np.float64)
        ok = np.empty(rows.shape, dtype=np.bool_)
        if HAS_NUMBA:
            _occlusion_walk(
                rows.astype(np.int64), cols.astype(np.int64), z0, x1i, y1i, z1f,
                map_[0].astype(np.float64), map_[2].astype(np.float64), width, height,
                float(tolerance_z_collision), ok,
            )
        else:
            ok = _occlusion_viewshed(rows, cols, z0, x1i, y1i, z1f, map_[0].astype(np.float64), map_[2], float(tolerance_z_collision))
        uv_correspondence[0][rows, cols] = u.astype(np.float32)
        uv_correspondence[1][rows, cols] = v.astype(np.float32)
        valid_correspondence[rows, cols] = ok

    return run


# --------------------------------------------------------------------------
# image fusion kernels
# --------------------------------------------------------------------------
def exponential_correspondences_to_map_kernel(resolution, width, height, alpha):
    def run(sem_map, map_idx, image_mono, uv_correspondence, valid_correspondence, image_height, image_width, new_sem_map, size=None):
        k = int(map_idx)
        w = int(image_width)
        flat = np.asarray(image_mono).reshape(-1)
        valid = valid_correspondence
        new_sem_map[k] = sem_map[k]
        if not valid.any():
            return
        uu = uv_correspondence[0][valid].astype(np.int64)
        vv = uv_correspondence[1][valid].astype(np.int64)
        idx = np.clip(uu + vv * w, 0, flat.size - 1)
        new_sem_map[k][valid] = sem_map[k][valid] * (1 - alpha) + alpha * flat[idx]

    return run


def color_correspondences_to_map_kernel(resolution, width, height):
    def run(sem_map, map_idx, image_rgb, uv_correspondence, valid_correspondence, image_height, image_width, new_sem_map, size=None):
        k = int(map_idx)
        w, h = int(image_width), int(image_height)
        flat = np.asarray(image_rgb).reshape(-1)
        valid = valid_correspondence
        new_sem_map[k] = sem_map[k]
        if not valid.any():
            return
        uu = uv_correspondence[0][valid].astype(np.int64)
        vv = uv_correspondence[1][valid].astype(np.int64)
        idx = np.clip(uu + vv * w, 0, w * h - 1)
        r = flat[idx].astype(np.uint32)
        g = flat[w * h + idx].astype(np.uint32)
        b = flat[2 * w * h + idx].astype(np.uint32)
        rgb = (r << 16) + (g << 8) + b
        new_sem_map[k][valid] = rgb.view(np.float32)

    return run


if not HAS_NUMBA:
    warnings.warn(
        "elevation_mapping_cupy numpy backend: numba not found; the ray walks run "
        "vectorised and slower. pip install numba to speed them up.",
        RuntimeWarning,
    )
