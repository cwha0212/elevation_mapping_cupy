import cupy as cp

from elevation_mapping_cupy.kernels.custom_kernels import map_utils


def _probe_get_idx_kernel(axis: str, width: int, height: int, resolution: float):
    if axis not in ("x", "y"):
        raise ValueError(f"Unsupported axis '{axis}'. Expected 'x' or 'y'.")
    op = "out[i] = get_x_idx(coords[i], center[0]);" if axis == "x" else "out[i] = get_y_idx(coords[i], center[0]);"
    return cp.ElementwiseKernel(
        in_params="raw U coords, raw U center",
        out_params="raw int32 out",
        preamble=map_utils(
            resolution=resolution,
            width=width,
            height=height,
            sensor_noise_factor=0.0,
            min_valid_distance=0.0,
            max_height_range=1000.0,
            ramped_height_range_a=0.0,
            ramped_height_range_b=0.0,
            ramped_height_range_c=1000.0,
        ),
        operation=op,
        name=f"probe_get_{axis}_idx_kernel",
    )


def test_get_x_idx_rounds_left_of_boundary_before_clamp():
    kernel = _probe_get_idx_kernel(axis="x", width=200, height=200, resolution=1.0)
    coords = cp.asarray([-100.2], dtype=cp.float32)
    center = cp.asarray([0.0], dtype=cp.float32)
    out = cp.zeros((1,), dtype=cp.int32)
    kernel(coords, center, out, size=1)
    assert int(out[0].item()) == -1


def test_get_y_idx_at_center_matches_middle_cell():
    kernel = _probe_get_idx_kernel(axis="y", width=200, height=200, resolution=1.0)
    coords = cp.asarray([0.0], dtype=cp.float32)
    center = cp.asarray([0.0], dtype=cp.float32)
    out = cp.zeros((1,), dtype=cp.int32)
    kernel(coords, center, out, size=1)
    assert int(out[0].item()) == 100
