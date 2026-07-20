#!/usr/bin/env python3
"""Compare the same deterministic map-finalization algorithm in CuPy and Warp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cupy as cp
import numpy as np
import warp as wp


@wp.kernel
def finalize_warp(
    previous: wp.array(dtype=wp.float32),
    proposals: wp.array(dtype=wp.float32),
    visibility: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
    cells: int,
    max_variance: float,
    initial_variance: float,
    outlier_variance: float,
):
    i = wp.tid()
    for layer in range(7):
        output[layer * cells + i] = previous[layer * cells + i]

    new_count = proposals[2 * cells + i]
    outlier_count = proposals[5 * cells + i]
    if new_count > 0.0:
        new_height = proposals[i] / new_count
        new_variance = proposals[cells + i] / new_count
        output[i] = new_height
        output[cells + i] = new_variance
        output[2 * cells + i] = 1.0
        output[4 * cells + i] = 0.0
        output[5 * cells + i] = new_height
        output[6 * cells + i] = 0.0
        if new_variance > max_variance:
            output[i] = 0.0
            output[cells + i] = initial_variance
            output[2 * cells + i] = 0.0
    else:
        output[cells + i] = output[cells + i] + outlier_count * outlier_variance
        cleanup = visibility[i]
        if cleanup > 0.0:
            output[2 * cells + i] = output[2 * cells + i] - cleanup
            output[cells + i] = output[cells + i] + visibility[cells + i] * outlier_variance

        proposed_upper_bound = visibility[2 * cells + i]
        previous_upper_bound = previous[5 * cells + i]
        previous_has_upper_bound = previous[6 * cells + i]
        if wp.isfinite(proposed_upper_bound) and (
            previous_has_upper_bound < 0.5 or proposed_upper_bound < previous_upper_bound
        ):
            output[5 * cells + i] = proposed_upper_bound
            output[6 * cells + i] = 1.0

    if output[2 * cells + i] < 0.5:
        output[i] = 0.0
        output[cells + i] = initial_variance
        output[2 * cells + i] = 0.0


def summarize(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }


def time_cuda(operation, warmups: int, repetitions: int) -> dict[str, float]:
    for _ in range(warmups):
        operation()
    cp.cuda.get_current_stream().synchronize()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    samples = []
    for _ in range(repetitions):
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, stop)))
    return summarize(samples)


def make_inputs(size: int):
    cells = size * size
    rng = cp.random.RandomState(17 + size)
    previous = cp.zeros((7, cells), dtype=cp.float32)
    previous[0] = rng.normal(0.0, 0.5, cells, dtype=cp.float32)
    previous[1].fill(0.1)
    previous[2] = (rng.random_sample(cells, dtype=cp.float32) > 0.2).astype(cp.float32)
    previous[4].fill(1.0)
    previous[5] = previous[0]

    proposals = cp.zeros_like(previous)
    update = rng.random_sample(cells, dtype=cp.float32) < 0.08
    count = rng.randint(1, 20, cells, dtype=cp.int32).astype(cp.float32) * update
    height = rng.normal(0.0, 0.5, cells, dtype=cp.float32)
    variance = rng.uniform(0.01, 0.5, cells, dtype=cp.float32)
    proposals[0] = height * count
    proposals[1] = variance * count
    proposals[2] = count
    proposals[5] = rng.randint(0, 4, cells, dtype=cp.int32).astype(cp.float32)

    visibility = cp.zeros((3, cells), dtype=cp.float32)
    visibility[0] = rng.uniform(0.0, 0.2, cells, dtype=cp.float32) * (
        rng.random_sample(cells, dtype=cp.float32) < 0.05
    )
    visibility[1] = rng.randint(0, 5, cells, dtype=cp.int32).astype(cp.float32)
    visibility[2].fill(cp.inf)
    upper_update = rng.random_sample(cells, dtype=cp.float32) < 0.05
    visibility[2] = cp.where(upper_update, height - 0.1, visibility[2])
    return previous.reshape(-1), proposals.reshape(-1), visibility.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[202, 602])
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root / "elevation_mapping_cupy"))
    from elevation_mapping_cupy.kernels.custom_kernels import finalize_map_kernel

    wp.init()
    device = "cuda:0"
    stream = wp.Stream(device, cuda_stream=cp.cuda.get_current_stream().ptr)
    report = {"cupy": cp.__version__, "warp": wp.__version__, "workloads": {}}

    for size in args.sizes:
        cells = size * size
        previous, proposals, visibility = make_inputs(size)
        cupy_output = cp.empty(7 * cells, dtype=cp.float32)
        warp_output = cp.empty_like(cupy_output)
        cupy_finalize = finalize_map_kernel(size, size, 1.0, 10.0, 0.01)

        def run_cupy():
            cupy_finalize(previous, proposals, visibility, cupy_output, size=cells)

        warp_arguments = [
            wp.from_dlpack(previous),
            wp.from_dlpack(proposals),
            wp.from_dlpack(visibility),
            wp.from_dlpack(warp_output),
            cells,
            1.0,
            10.0,
            0.01,
        ]

        run_cupy()
        wp.launch(finalize_warp, dim=cells, inputs=warp_arguments, device=device, stream=stream)
        cp.cuda.get_current_stream().synchronize()
        cp.testing.assert_allclose(warp_output, cupy_output, rtol=1e-6, atol=1e-6)

        result = {
            "cupy": time_cuda(run_cupy, args.warmups, args.repetitions),
            "warp": {},
        }
        for block_size in (64, 128, 256):
            def run_warp(block_size=block_size):
                wp.launch(
                    finalize_warp,
                    dim=cells,
                    inputs=warp_arguments,
                    device=device,
                    stream=stream,
                    block_dim=block_size,
                )

            result["warp"][str(block_size)] = time_cuda(
                run_warp,
                args.warmups,
                args.repetitions,
            )
        report["workloads"][f"{size}x{size}"] = result

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
