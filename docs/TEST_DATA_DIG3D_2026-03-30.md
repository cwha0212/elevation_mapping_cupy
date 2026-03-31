# DIG3D Test Data For `elevation_mapping_cupy`

This note points to the current March 26 DIG3D real bags that are useful when
changing `elevation_mapping_cupy`.

## Current local digging chain

When replaying the split DIG bags with the current local digging profile, the
effective height-processing chain is:

- `elevation -> near_base_filtered -> despiked -> inpaint -> excavation_mapping`

Important detail:

- excavation mapping currently patches from `inpaint`, not directly from
  `despiked`
- so a despike change should usually be checked in both
  `/mole/elevation_map_filter` and `/mole/excavation_mapping/grid_map`

## Existing dataset docs

- Main bag manifest:
  - `/home/lorenzo/mcap/dig3d_2026-03-26/bag_manifest_initial_2026-03-28.md`
- Single-scoop playback priority:
  - `/home/lorenzo/mcap/dig3d_2026-03-26/split_single_scoops_obs_2026-03-28/PLAYBACK_PRIORITY_2026-03-28.md`
- Single-scoop manifest:
  - `/home/lorenzo/mcap/dig3d_2026-03-26/split_single_scoops_obs_2026-03-28/single_scoop_manifest_2026-03-28.csv`
- Current despike validation summary:
  - `/home/lorenzo/mcap/dig3d_2026-03-26/analysis/elevation_despike_2026-03-29/README.md`

## Bag layouts

There are two layouts in this dataset.

### 1. Split full-run layout

Use this when you want to replay raw lidar and run the perception stack live.

Example:

- `/home/lorenzo/mcap/dig3d_2026-03-26/dig3d_real_run_2026-03-26_21-09-18/`

Expected structure:

- `sensors/`
- `state/`
- `commands/`
- `lidar/`
- `elevation_map/`
- optional `camera/`

This is the cleanest layout for:

- `robot_self_filter`
- live `elevation_mapping_cupy`
- live excavation mapping on top of the `inpaint` layer built from the despiked
  elevation-map chain

### 2. Monolithic bag layout

Use this when you want a short repro bag or a single whole-run bag in one
folder.

Examples:

- `/home/lorenzo/mcap/dig3d_2026-03-26/trenching_single_2026-03-26_21-23-04/`
- `/home/lorenzo/mcap/dig3d_2026-03-26/analysis/single_scoop_splits_2026-03-28/trenching_single_2026-03-26_21-39-24/trenching_single_2026-03-26_21-39-24__scoop_03/`

Expected structure:

- `<name>.mcap`
- `metadata.yaml`

This is the easiest layout for:

- short Foxglove review
- targeted repro of one failure case
- offline analysis scripts

## Recommended test data

### Clean baseline full replay

Use:

- `/home/lorenzo/mcap/dig3d_2026-03-26/dig3d_real_run_2026-03-26_21-09-18/`

Why:

- split layout is already clean
- short controlled run
- good reference when checking that a filter does not damage reasonable terrain

### Spike-heavy full-run repro

Use:

- `/home/lorenzo/mcap/dig3d_2026-03-26/trenching_single_2026-03-26_21-23-04/`

Why:

- strongest near-machine positive spike behavior
- primary bag for validating spike rejection

Related metrics:

- `/home/lorenzo/mcap/dig3d_2026-03-26/analysis/elevation_despike_2026-03-29/despike_summary_212304.json`

### Short focused spike repro

Use:

- `/home/lorenzo/mcap/dig3d_2026-03-26/analysis/single_scoop_splits_2026-03-28/trenching_single_2026-03-26_21-23-04/trenching_single_2026-03-26_21-23-04__scoop_05/`

Why:

- short repro bag
- good when iterating quickly in Foxglove

### Short mixed-quality comparison case

Use:

- `/home/lorenzo/mcap/dig3d_2026-03-26/analysis/single_scoop_splits_2026-03-28/trenching_single_2026-03-26_21-39-24/trenching_single_2026-03-26_21-39-24__scoop_03/`

Why:

- useful comparison case after the spike-heavy repro
- also contains the pitch-jerk behavior investigated elsewhere

## Fast guidance for future agents

If the task is "change `elevation_mapping_cupy` and validate on real DIG data",
point the agent to:

- clean reference:
  - `/home/lorenzo/mcap/dig3d_2026-03-26/dig3d_real_run_2026-03-26_21-09-18/`
- spike repro:
  - `/home/lorenzo/mcap/dig3d_2026-03-26/trenching_single_2026-03-26_21-23-04/`
- short repro:
  - `/home/lorenzo/mcap/dig3d_2026-03-26/analysis/single_scoop_splits_2026-03-28/trenching_single_2026-03-26_21-23-04/trenching_single_2026-03-26_21-23-04__scoop_05/`

The full-run split layout is the best target when the agent needs live replay
with:

- bag TF
- raw lidar replay
- `robot_self_filter`
- local `elevation_mapping_cupy`

The monolithic bags are better when the agent only needs one concise repro case.
