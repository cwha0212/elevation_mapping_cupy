# Elevation Mapping GPU Performance Handover

## Executive recommendation

Do not start with a one-for-one NVIDIA Warp port. The current implementation
already executes its main point update as compiled CUDA through CuPy. Most of
the likely runtime is instead caused by serial ray marching, divergent control
flow, contended/global map updates, host synchronization, repeated temporary
allocations, CPU-only filtering, and ROS message construction.

The recommended order is:

1. Fix the correctness and determinism problems described below.
2. Apply the low-risk CuPy and publication improvements.
3. Replace the dilation and visibility algorithms.
4. Benchmark a fused CuPy `RawKernel` against a Warp implementation of the
   same redesigned algorithm.
5. Keep Warp only where it demonstrates a clear end-to-end benefit.

Warp is still useful as a targeted experiment. CuPy arrays implement the CUDA
Array Interface, so selected kernels can be prototyped in Warp without copying
the map or point cloud.

## Review scope and limitations

This review covered the core mapping kernels, plugin pipeline, ROS point-cloud
ingestion and GridMap publication. The active Moleworks configurations observed
during the review were:

- Local map: 20 m length, 0.1 m resolution, dilation radius 3, ray length 7 m.
- Survey/global map: 60 m length, 0.1 m resolution, dilation radius 70, ray
  length 7 m.
- Map publication: two three-layer maps plus one four-layer filtered map at
  5 Hz, for ten published 2-D float layers per cycle.

The review host did not have a compatible CUDA driver. Kernel timings must
therefore be collected on the GPU machine. A CPU-only GridMap construction
microbenchmark was run; its result is included below. No performance estimate
in this document should replace measurement on representative recorded clouds.

## Correctness blockers

These should be resolved before using a new implementation as a performance
reference.

### 1. Point fusion and visibility cleanup have data races

The point kernel concurrently reads and modifies map variance, validity,
timestamps, and upper bounds. In particular, the upper-bound minimum is a
non-atomic read/modify/write, and fusion decisions can observe variance written
by another point in the same launch.

Relevant code:

- [`custom_kernels.py`, point update](elevation_mapping_cupy/elevation_mapping_cupy/kernels/custom_kernels.py#L181)
- [`custom_kernels.py`, visibility cleanup](elevation_mapping_cupy/elevation_mapping_cupy/kernels/custom_kernels.py#L224)

Recommended design:

1. Read the previous map from an immutable snapshot.
2. Bin or sort observations by target cell.
3. Reduce point observations and ray-cleanup proposals into per-cell values.
4. Finalize each output cell once in a separate kernel.

This removes scheduling-dependent classification and should reduce atomic
contention for dense clouds.

### 2. Geometry is explicitly reduced to binary16

Coordinate transforms, map indexing, validity checks, and ray traversal use
CuPy's `float16` type even though the input arrays are float32.

Relevant code:

- [`custom_kernels.py`, geometry helpers](elevation_mapping_cupy/elevation_mapping_cupy/kernels/custom_kernels.py#L22)
- [`custom_kernels.py`, ray intermediates](elevation_mapping_cupy/elevation_mapping_cupy/kernels/custom_kernels.py#L209)

At a coordinate of 200 m, binary16 spacing is approximately 0.125 m, which is
larger than the configured 0.1 m cell size. Convert these helpers and all
geometry intermediates to CUDA `float`. Add tests around cell boundaries and at
large world-coordinate offsets.

### 3. Dilation does not select the nearest valid cell

The current comparison uses signed `dx + dy` rather than a distance magnitude.
For a fully valid neighborhood with radius 3, it can select `(-3, -3)` instead
of an adjacent cell. Flattened neighbor addressing can also wrap between map
rows, and the initializer aliases the dilation input and output.

Relevant code:

- [`custom_kernels.py`, dilation](elevation_mapping_cupy/elevation_mapping_cupy/kernels/custom_kernels.py#L416)
- [`elevation_mapping.py`, initialization call](elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping.py#L983)

The survey setting is especially problematic: a 602 x 602 map with radius 70
has a 141 x 141 search window, or about 7.2 billion candidate checks in the
worst case.

Preferred replacement: use `cupyx.scipy.ndimage.distance_transform_edt` with
nearest indices and validate its mask semantics. If its runtime or memory is
unsuitable, benchmark a jump-flood nearest-seed implementation.

### 4. The Mahalanobis gate needs mathematical validation

The current gate compares absolute height error with `map_variance * threshold`
and ignores the point measurement variance.

Relevant code:

- [`custom_kernels.py`, fusion gate](elevation_mapping_cupy/elevation_mapping_cupy/kernels/custom_kernels.py#L184)
- [`custom_kernels.py`, error counting](elevation_mapping_cupy/elevation_mapping_cupy/kernels/custom_kernels.py#L327)

If the configured threshold is intended to be a standard-deviation multiplier,
the innovation test should normally have the form:

```text
error^2 > threshold^2 * (map_variance + measurement_variance)
```

Confirm the desired sensor model before changing this because it changes map
behavior, not only performance. Dense-cell fusion should also be reviewed: it
averages independent point posteriors and may intentionally avoid decreasing
variance with point count, but that should be documented and tested.

### 5. Non-default input paths are currently unsafe

- The image input method references correspondence kernels and semantic-map
  fields which are never initialized.
- For point clouds with extra channels, `points_all[:, :3]` is a strided CuPy
  view while the raw point kernel assumes tightly packed XYZ values.
- The ROS `PointCloud2` parser does not honor organized-cloud `row_step`
  padding.

Relevant code:

- [`elevation_mapping.py`, point slicing](elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping.py#L350)
- [`elevation_mapping.py`, image input](elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping.py#L544)
- [`elevation_mapping_node.py`, PointCloud2 parsing](elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping_node.py#L46)

Either repair these interfaces with regression tests or fail loudly when they
are selected. The default XYZ Moleworks path avoids the strided-view problem.

## Performance work, in recommended order

### Phase 1: low-risk reductions

1. Delete the unused point-cloud `min`, `max`, and `mean` reductions in
   [`elevation_mapping.py`](elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping.py#L479).
2. Compute point validity once, early-exit invalid points, and remove the unused
   traversability load from the ray loop.
3. Apply the configured voxel/downsampling policy before the update kernel.
4. Reuse error counters, temporary maps, CUDA streams, and small device scalar
   buffers rather than allocating them per callback.
5. Replace plugin cache-validity scans such as `cp.isnan(...).all()` with a CPU
   generation counter or dirty flag.
6. Avoid map transforms and full-map `cp.roll` operations when the snapped map
   center and height have not changed.

### Phase 2: visibility traversal

At a 7 m ray limit and 0.1 m resolution, fixed stepping performs about 99
serial iterations per valid point. A 100k-point cloud can therefore execute
approximately 9.9 million divergent loop iterations before considering
contention or duplicate rays.

Implement exact 2-D grid DDA traversal and benchmark:

- One ray per original point.
- One ray per endpoint grid cell.
- One ray per angular/range bin.

Endpoint fusion and free-space cleanup should be separate passes. This permits
independent compaction and avoids races between endpoint writes and ray-clearing
writes.

### Phase 3: plugin fusion

The highest-value plugin candidates are:

- [`min_filter.py`](elevation_mapping_cupy/elevation_mapping_cupy/plugins/min_filter.py#L59):
  currently reads and writes neighboring output cells in one launch, without a
  grid-wide barrier. Use ping-pong buffers or a different propagation
  algorithm. Its per-iteration `.all()` also causes a host synchronization and
  should be removed or checked infrequently on the interior only.
- [`positive_spike_filter.py`](elevation_mapping_cupy/elevation_mapping_cupy/plugins/positive_spike_filter.py#L104):
  currently creates 24 shifted full-map views and a median stack. Replace this
  with one fused neighborhood kernel. This is a strong CuPy RawKernel versus
  Warp trial.
- [`inpainting.py`](elevation_mapping_cupy/elevation_mapping_cupy/plugins/inpainting.py#L109):
  currently performs full GPU-to-CPU-to-GPU transfers and CPU OpenCV work.
  Preserve float32 and either process only small hole bounding boxes or use a
  GPU fill algorithm.

Do not replace optimized PyTorch/cuDNN convolutions with handwritten Warp
kernels unless a representative benchmark shows a benefit. Framework uniformity
alone is not a performance result.

### Phase 4: publication and allocation

GridMap conversion currently uses `.tolist()`, creating a Python object for
every cell before converting those objects back to float32 storage.

Relevant code:

- [`gridmap_utils.py`](elevation_mapping_cupy/elevation_mapping_cupy/gridmap_utils.py#L18)
- [`elevation_mapping_node.py`, publication](elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping_node.py#L451)

On a synthetic 600 x 600 NumPy layer, the reviewed path took approximately
10.45 ms per layer. Copying contiguous transposed bytes directly into an
`array('f')` took approximately 0.37 ms per layer, about 28 times faster for
this CPU construction step. This is not an end-to-end ROS benchmark.

Recommended publication design:

1. Materialize one consistent map snapshot per publication generation.
2. Batch requested layers into one device-to-pinned-host transfer.
3. Reuse a persistent pinned host buffer and CUDA stream.
4. Reuse that snapshot for publishers requesting identical layers.
5. Skip all preparation when a publisher has no subscribers.
6. Consider latest-only QoS for large maps if downstream semantics permit it.

The package also globally selects CUDA managed memory and, by default, releases
CuPy, pinned-memory, and Torch allocator caches every five seconds:

- [`elevation_mapping.py`, allocator](elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping.py#L44)
- [`elevation_mapping.py`, cache trimming](elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping.py#L711)

Benchmark the default CuPy device pool against managed allocation with all
other variables fixed. Disable periodic trimming by default, or trigger it only
when idle or under measured memory pressure. Frequent cache destruction is a
memory-footprint/latency tradeoff and is expected to create recurring allocation
and page-migration jitter.

## CuPy versus Warp experiment

Compare these variants rather than comparing frameworks with different
algorithms:

1. Current CuPy implementation.
2. Exact Warp translation of one representative kernel.
3. Redesigned/fused algorithm implemented as a CuPy `RawKernel`.
4. The same redesigned/fused algorithm implemented in Warp.

Suggested first candidates:

- Exact point-update translation, to establish framework overhead/parity.
- Fused positive-spike filter, to measure a realistic kernel-fusion win.
- Redesigned deterministic per-cell reduction after correctness is established.

A full Warp port is justified only if it wins end-to-end callback latency,
maintains output equivalence, and the benefit is large enough to cover the new
dependency and stream/allocator integration. A suggested decision threshold is
at least a repeatable 15% p95 end-to-end improvement, unless Warp provides a
separate maintainability advantage.

Useful references:

- [CuPy performance and CUDA-event timing](https://docs.cupy.dev/en/stable/user_guide/performance.html)
- [CuPy memory pools](https://docs.cupy.dev/en/stable/user_guide/memory.html)
- [Warp kernels](https://nvidia.github.io/warp/stable/user_guide/basics.html)
- [Warp and CuPy interoperability](https://nvidia.github.io/warp/stable/user_guide/interoperability.html)

## GPU benchmark protocol

### Workloads

Use saved production clouds representing p10, p50, and p90 point counts. Test at
least:

- 202 x 202 map, dilation radius 3, ray length 7 m.
- 602 x 602 map, dilation radius 70, ray length 7 m.
- The 602 x 602 map again after replacing dilation, so the rest of the pipeline
  can be measured without the pathological neighborhood scan.

Restore the input point buffer between repetitions because the current update
kernel modifies point values in place.

### Measurement

1. Record GPU model, driver, CUDA, CuPy, PyTorch, and Warp versions.
2. Hold allocator, stream, precision, block size, input, and initial map fixed
   between variants.
3. Measure cold compilation/startup separately.
4. Warm each variant for at least 30 callbacks.
5. Record at least 200 measured callbacks.
6. Use CUDA events for individual GPU stages.
7. Record end-to-end ROS callback and publication p50, p95, and maximum latency.
8. Record peak allocated bytes and memory-pool bytes.
9. Use Nsight Systems to identify launches, synchronization, and transfers.
10. Use Nsight Compute on the dominant kernels for divergence, atomic
    contention, occupancy, and register pressure.

Sweep block sizes 64, 128, and 256 where applicable. Add NVTX ranges around
point preparation, endpoint fusion, visibility cleanup, dilation, each plugin,
device-to-host transfer, and message construction.

### Correctness gates

Before accepting a performance result:

- Verify exact validity masks and cell-selection indices.
- Compare elevation and variance using documented absolute/relative tolerances.
- Replay multiple consecutive frames, not only an empty-map update.
- Repeat runs to detect scheduling-dependent outputs.
- Test large coordinate offsets and cell-boundary coordinates.
- Include duplicate points, dense single-cell points, NaN/Inf points, padded
  organized clouds, and points near map borders.

## Proposed implementation sequence

- [ ] Add focused tests for dilation, float32 coordinate mapping, fusion
      determinism, and the statistical gate.
- [ ] Fix binary16 geometry, dilation correctness, and non-contiguous XYZ input.
- [ ] Split point fusion/visibility cleanup and remove fusion races.
- [ ] Remove unused reductions and repeated validity work.
- [ ] Add effective point/ray deduplication and DDA traversal.
- [ ] Replace plugin synchronization and large temporary stacks.
- [ ] Batch/pin GridMap publication and eliminate `.tolist()`.
- [ ] Benchmark device-pool versus managed-memory allocation and disable periodic
      trimming for the latency baseline.
- [ ] Run the controlled CuPy RawKernel versus Warp comparison.
- [ ] Keep a small replay/performance harness as a regression test.
