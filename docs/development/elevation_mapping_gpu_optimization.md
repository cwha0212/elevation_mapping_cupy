# elevation_mapping_cupy GPU Optimization Worklog

## Outcome

The correctness-first CuPy redesign is complete on branch
`elevation-gpu-throughput`. It keeps the existing public ROS interfaces and
does not add NVIDIA Warp as a runtime dependency.

On an RTX 4090, the median of three independent runs reduced the deterministic
core callback p95 by 55.3--64.2% for 10k--100k point clouds. An in-process ROS
cycle containing `PointCloud2` parsing, mapping, filtered plugin export, and
GridMap message construction improved p95 by 26.3--65.8%.

The baseline is `f6ed873a7ed0825028595d9ded6e456faf7be1f9`. Measurements were
run from its code-equivalent parent `deb1c86`; `f6ed873` only adds
`elevation_mapping_gpu_handover.md`.

## Changes

### Correctness

- Restored float32 geometry and ray intermediates and reject NaN/Inf points
  before index or range calculations.
- Replaced the incorrect signed-offset dilation with exact Euclidean nearest
  valid-cell selection. Small radii use a direct kernel and large radii use
  `cupyx.scipy.ndimage.distance_transform_edt`.
- Removed point-order races by reading an immutable map snapshot, accumulating
  endpoint and visibility proposals, and applying them once per cell in a
  deterministic finalization kernel. Endpoint fusion wins over same-frame
  visibility cleanup.
- Replaced fixed-step visibility sampling with exact 2D grid DDA traversal.
- Fixed the PointCloud2 fast parser for organized clouds with row padding and
  added fail-loudly validation for truncated buffers.
- Fixed the min-filter plugin's same-buffer launch race and row/column bounds.

### Throughput

- Reuse point-fusion proposal buffers and per-callback error counters.
- Compute point validity once, return early for invalid points, and remove
  unused point reductions and map loads.
- Make XYZ contiguous once before the raw point kernel.
- Cache plugin results by a CPU-side map generation counter, removing full-map
  NaN scans, forced host synchronizations, and redundant plugin evaluation.
- Use ping-pong buffers for the min filter.
- Encode ROS `Float32MultiArray` payloads directly from contiguous bytes rather
  than creating Python float lists.
- Skip GridMap construction when a publisher has no subscribers.
- Use CuPy's default device memory pool. Periodic pool trimming is now opt-in
  (`cupy_memory_pool_trim_interval_s: 0.0` by default).

## Measurement protocol

- GPU: NVIDIA GeForce RTX 4090, 24 GiB
- Driver: 580.159.03
- CuPy: 13.6.0
- Warmups: 30 per workload
- Measurements: 200 per workload
- Repeated runs: 3 fresh processes; tables report the median p50 and p95
- Clouds: deterministic synthetic LiDAR-like XYZ distributions with fixed
  seeds at 10k, 50k, and 100k points
- Maintained harnesses:
  - `benchmarks/benchmark_gpu_core.py`
  - `benchmarks/benchmark_ros_pipeline.py`
  - `benchmarks/benchmark_finalize_cupy_warp.py`
- Machine-readable summary:
  `benchmarks/results/rtx4090_20260720_summary.json`

No recorded raw PointCloud2 bag was present in the available workspace. The
reproducible workload satisfies the development gate, but a representative
Ouster bag replay remains the deployment gate for the target robot.

## Results

### Core mapping callback wall time

| Points | Baseline p50 | Optimized p50 | Baseline p95 | Optimized p95 | p95 reduction |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 1.005 ms | 0.581 ms | 1.987 ms | 0.764 ms | 61.6% |
| 50,000 | 1.266 ms | 0.586 ms | 1.744 ms | 0.779 ms | 55.3% |
| 100,000 | 1.787 ms | 0.597 ms | 2.309 ms | 0.827 ms | 64.2% |

### In-process ROS pipeline

`pointcloud_callback` includes PointCloud2 parsing, CPU-to-GPU transfer, and
mapping. The full cycle additionally prepares the configured six-layer
`elevation_map_filter` GridMap. Executor scheduling, ROS serialization, and DDS
transport are deliberately excluded.

| Points | Scope | Baseline p95 | Optimized p95 | p95 reduction |
|---:|---|---:|---:|---:|
| 10,000 | callback | 1.839 ms | 1.777 ms | 3.4% |
| 10,000 | callback + filtered GridMap | 54.544 ms | 40.194 ms | 26.3% |
| 50,000 | callback | 2.306 ms | 1.734 ms | 24.8% |
| 50,000 | callback + filtered GridMap | 32.036 ms | 15.933 ms | 50.3% |
| 100,000 | callback | 3.428 ms | 2.641 ms | 23.0% |
| 100,000 | callback + filtered GridMap | 32.609 ms | 11.146 ms | 65.8% |

The 10k callback's p50 still improves from 1.443 ms to 1.076 ms; its p95 is
dominated by host-side tail latency. Publication preparation remains the
largest end-to-end cost after the kernel fixes.

### Dilation and message encoding

| Workload | Baseline p95 | Optimized p95 | p95 reduction |
|---|---:|---:|---:|
| 202 x 202, radius 3 | 0.0092 ms | 0.0082 ms | 11.1% |
| 602 x 602, radius 70 | 8.7038 ms | 0.5897 ms | 93.2% |

For a 600 x 600 single-layer GridMap payload, 200 samples reduced message
construction from 5.82 ms p50 / 10.80 ms p95 to 0.86 ms p50 / 0.90 ms p95.

## CuPy versus Warp

The deterministic finalizer was implemented with the same algorithm and
validated output in both frameworks. Across three 30-warmup/200-sample runs:

| Grid | CuPy p95 | Best Warp p95 | Result |
|---|---:|---:|---|
| 202 x 202 | 0.01322 ms | 0.01331 ms | no Warp gain |
| 602 x 602 | 0.02150 ms | 0.02662 ms | CuPy faster |

The differences are tiny relative to callback latency and Warp does not
provide a repeatable p95 benefit. Keep the redesigned CuPy RawKernel and do
not add Warp.

## Verification

- Direct GPU unit suite: 126 passed; 24 existing warnings from erosion's
  constant-array normalization.
- Isolated ROS overlay build: `elevation_map_msgs`, `semantic_sensor`, and
  `elevation_mapping_cupy` passed.
- Isolated `colcon test`: 77 tests, 0 errors, 0 failures, 0 skipped. This
  includes 17 elevation-mapping CTest targets, TF/GridMap integration,
  save/load services, and the synthetic demo launch.
- Added regression coverage for dilation geometry, float32 behavior at large
  map offsets, NaN/Inf rejection, dense-cell input-order invariance,
  deterministic finalization, plugin generation caching, min-filter aliasing,
  and padded/truncated PointCloud2 parsing.

The optional semantic runtime smoke script cannot launch the semantic image
node in this host environment because `torchvision` is absent. The semantic
package's three configuration/launch tests pass, and the missing optional host
dependency does not exercise the elevation mapping path.

## Reproduction

Run from a ROS environment containing this checkout's packages:

```bash
python3 benchmarks/benchmark_gpu_core.py --output /tmp/gpu-core.json
python3 benchmarks/benchmark_ros_pipeline.py --output /tmp/ros-pipeline.json
python3 benchmarks/benchmark_finalize_cupy_warp.py --output /tmp/cupy-warp.json
```

For deployment acceptance, record or select a representative raw Ouster
PointCloud2 bag, run the same baseline/optimized A/B with fixed clocks and
configuration, inspect map equivalence, and measure actual DDS publication
latency and missed deadlines.
