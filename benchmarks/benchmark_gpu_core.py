#!/usr/bin/env python3
"""Reproducible CUDA-event benchmark for elevation_mapping_cupy's core path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np


def percentile(samples: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(samples), value))


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "max_ms": max(samples),
    }


def time_cuda(operation, warmups: int, repetitions: int) -> dict[str, float]:
    for _ in range(warmups):
        operation()
    cp.cuda.Device().synchronize()

    samples = []
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    for _ in range(repetitions):
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, stop)))
    return summarize(samples)


def make_cloud(point_count: int) -> cp.ndarray:
    rng = np.random.default_rng(7 + point_count)
    x = rng.uniform(0.5, 7.0, point_count)
    y = rng.uniform(-6.0, 6.0, point_count)
    z = -1.5 + 0.03 * x + rng.normal(0.0, 0.02, point_count)
    return cp.asarray(np.column_stack((x, y, z)), dtype=cp.float32)


def benchmark_callbacks(ElevationMap, Parameter, config_root: Path, args) -> dict[str, object]:
    results = {}
    for point_count in args.point_counts:
        param = Parameter(
            use_chainer=False,
            weight_file=str(config_root / "weights.dat"),
            plugin_config_file=str(config_root / "plugin_config.yaml"),
        )
        param.resolution = 0.1
        param.map_length = 20.0
        param.dilation_size = 3
        param.max_ray_length = 7.0
        param.enable_drift_compensation = False
        param.update()
        elevation_map = ElevationMap(param)
        cloud = make_cloud(point_count)
        rotation = cp.eye(3, dtype=cp.float32)
        translation = cp.asarray([0.0, 0.0, 2.0], dtype=cp.float32)

        def update():
            elevation_map.input_pointcloud(
                cloud.copy(),
                ["x", "y", "z"],
                rotation,
                translation,
                0.0,
                0.0,
            )

        for _ in range(args.warmups):
            update()
        cp.cuda.Device().synchronize()

        cuda_samples = []
        wall_samples = []
        start_event = cp.cuda.Event()
        stop_event = cp.cuda.Event()
        for _ in range(args.repetitions):
            start_event.record()
            wall_start = time.perf_counter()
            update()
            stop_event.record()
            stop_event.synchronize()
            wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
            cuda_samples.append(float(cp.cuda.get_elapsed_time(start_event, stop_event)))

        results[str(point_count)] = {
            "cuda": summarize(cuda_samples),
            "wall": summarize(wall_samples),
        }
    return results


def benchmark_dilation(dilation_filter_kernel, args) -> dict[str, object]:
    results = {}
    for cells, radius in ((202, 3), (602, 70)):
        values = cp.arange(cells * cells, dtype=cp.float32).reshape(cells, cells)
        mask = cp.zeros_like(values)
        step = max(cells // 20, 1)
        mask[step:-step:step, step:-step:step] = 1.0
        output = cp.empty_like(values)
        output_mask = cp.empty_like(mask)
        operation = dilation_filter_kernel(cells, cells, radius)

        def update():
            operation(values, mask, output, output_mask, size=cells * cells)

        results[f"{cells}x{cells}_radius_{radius}"] = time_cuda(
            update,
            args.warmups,
            args.repetitions,
        )
    return results


def git_head(source_root: Path) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        text=True,
    ).strip()
    return f"{head}-dirty" if dirty else head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository checkout to benchmark.",
    )
    parser.add_argument("--point-counts", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "elevation_mapping_cupy"))
    from elevation_mapping_cupy import ElevationMap, Parameter
    from elevation_mapping_cupy.kernels.custom_kernels import dilation_filter_kernel

    device = cp.cuda.runtime.getDeviceProperties(0)
    report = {
        "source_root": str(source_root),
        "git_head": git_head(source_root),
        "gpu": device["name"].decode(),
        "cupy": cp.__version__,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "callbacks": benchmark_callbacks(
            ElevationMap,
            Parameter,
            source_root / "elevation_mapping_cupy" / "config" / "core",
            args,
        ),
        "dilation": benchmark_dilation(dilation_filter_kernel, args),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
