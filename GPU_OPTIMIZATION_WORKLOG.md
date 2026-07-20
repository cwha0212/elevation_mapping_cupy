# elevation_mapping_cupy GPU Optimization Worklog

## Goal

Improve correctness and end-to-end p95 throughput relative to `f6ed873`, using
the correctness gates and measurement protocol in
`ELEVATION_MAPPING_GPU_HANDOVER.md`. Warp is not a default dependency; it must
show a repeatable benefit for an equivalent redesigned kernel.

## Baseline environment

- GPU: NVIDIA GeForce RTX 4090, 24 GiB
- Driver: 580.159.03
- CuPy: 13.6.0
- Baseline commit: `f6ed873a7ed0825028595d9ded6e456faf7be1f9`
- Baseline GPU unit suite: 113 passed, 24 pre-existing erosion warnings

Initial dilation probe before changes:

| Workload | Current p50 | CuPy EDT p50 |
|---|---:|---:|
| 202 x 202, radius 3 | 0.010 ms | 0.214 ms |
| 202 x 202, radius 10 | 0.039 ms | 0.209 ms |
| 602 x 602, radius 70 | 7.930 ms | 0.255 ms |

This supports a direct nearest-neighbor kernel for small radii and EDT for the
large-radius regime. The maintained benchmark is
`benchmarks/benchmark_gpu_core.py`; committed result files will be added after
the correctness changes stabilize.

## Current implementation

- Restored float32 geometry and ray intermediates.
- Added finite-point rejection and one validity calculation per point.
- Made XYZ contiguous before the raw point kernel.
- Reused callback error counters and removed unused point reductions.
- Replaced signed-distance/row-wrapping dilation with an alias-safe hybrid:
  exact direct search for small radii and CuPy EDT for large radii.

## Verification

- Focused geometry, dilation, and kernel smoke tests: 13 passed.
- Full GPU unit suite after the first change: 119 passed, 24 pre-existing
  erosion warnings.
- Full GPU unit suite after publication/plugin changes: 120 passed, 24
  pre-existing erosion warnings.
- 600 x 600 single-layer message construction, 200 samples:
  - Baseline: 5.82 ms p50, 10.80 ms p95.
  - Optimized: 0.86 ms p50, 0.90 ms p95.
- Core callback A/B, 30 warmups and 200 measured callbacks:

| Points | Baseline wall p95 | Optimized wall p95 | Improvement |
|---:|---:|---:|---:|
| 10,000 | 1.987 ms | 0.800 ms | 59.7% |
| 50,000 | 1.520 ms | 0.993 ms | 34.7% |
| 100,000 | 2.309 ms | 1.485 ms | 35.7% |

- Large dilation, 30 warmups and 200 samples:
  - Baseline 602 x 602/radius 70: 8.601 ms p95.
  - Optimized: 0.637 ms p95 (92.6% reduction).
