#!/usr/bin/env python3
"""Benchmark the ROS ingress callback and in-process publication preparation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np


def summarize(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }


def git_head(source_root: Path) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip()
    return f"{head}-dirty" if dirty else head


def make_cloud_message(PointCloud2, PointField, point_count: int):
    rng = np.random.default_rng(7 + point_count)
    x = rng.uniform(0.5, 7.0, point_count)
    y = rng.uniform(-6.0, 6.0, point_count)
    z = -1.5 + 0.03 * x + rng.normal(0.0, 0.02, point_count)
    xyz = np.ascontiguousarray(np.column_stack((x, y, z)), dtype=np.float32)

    message = PointCloud2()
    message.header.frame_id = "odom"
    message.height = 1
    message.width = point_count
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = point_count * message.point_step
    message.data = xyz.tobytes()
    message.is_dense = True
    return message


class SinkPublisher:
    """Publisher-shaped sink that keeps DDS transport outside the benchmark."""

    def __init__(self) -> None:
        self.message_count = 0

    def get_subscription_count(self) -> int:
        return 1

    def publish(self, _message) -> None:
        self.message_count += 1


def time_operation(operation, warmups: int, repetitions: int) -> dict[str, float]:
    for _ in range(warmups):
        operation()
    cp.cuda.Device().synchronize()

    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        operation()
        cp.cuda.Device().synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return summarize(samples)


def benchmark_node(node, messages, args) -> dict[str, object]:
    subscriber_key = next(iter(node.my_subscribers))
    publisher_key = args.publisher_key
    if publisher_key not in node.my_publishers:
        raise KeyError(
            f"Publisher '{publisher_key}' is not configured; available: {sorted(node.my_publishers)}"
        )

    node._publishers_dict[publisher_key] = SinkPublisher()
    node._map_q = object()
    results = {}
    for point_count, message in messages.items():
        results[str(point_count)] = {
            "pointcloud_callback": time_operation(
                lambda: node.pointcloud_callback(message, subscriber_key),
                args.warmups,
                args.repetitions,
            ),
            "callback_and_gridmap_preparation": time_operation(
                lambda: (
                    node.pointcloud_callback(message, subscriber_key),
                    node.publish_map(publisher_key),
                ),
                args.warmups,
                args.repetitions,
            ),
        }
    return results


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
    parser.add_argument("--publisher-key", default="elevation_map_filter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "elevation_mapping_cupy"))

    import rclpy
    from elevation_mapping_cupy.elevation_mapping_node import ElevationMappingNode
    from sensor_msgs.msg import PointCloud2, PointField

    config_root = source_root / "elevation_mapping_cupy" / "config"
    rclpy.init(
        args=[
            "--ros-args",
            "--params-file",
            str(config_root / "core" / "core_param.yaml"),
            "--params-file",
            str(config_root / "setups" / "menzi" / "base.yaml"),
        ]
    )
    node = ElevationMappingNode()
    try:
        messages = {
            count: make_cloud_message(PointCloud2, PointField, count)
            for count in args.point_counts
        }
        device = cp.cuda.runtime.getDeviceProperties(0)
        report = {
            "source_root": str(source_root),
            "git_head": git_head(source_root),
            "gpu": device["name"].decode(),
            "cupy": cp.__version__,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "publisher_key": args.publisher_key,
            "scope": (
                "In-process ROS callback through GridMap message construction; "
                "executor scheduling, serialization, and DDS transport excluded."
            ),
            "results": benchmark_node(node, messages, args),
        }
    finally:
        node.destroy_node()
        rclpy.shutdown()

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
