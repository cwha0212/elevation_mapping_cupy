# elevation_mapping_cupy (haechi fork)

GPU elevation mapping and terrain judgement for the haechi quadruped. The map
feeds Nav2's local costmap as an occupancy grid; the global map stays NAVI's.

Packages

- `elevation_mapping_cupy` — the mapper (`elevation_mapping_node.py`), the
  SAM-TP camera node (`samtp_node.py`), the terrain plugin chain and
  `terrain_grid_node` (C++), which cuts the `safety` layer into
  `/terrain/local_grid` (free / camera-cost / lethal / unknown).
- `elevation_map_msgs` — `ChannelInfo`, the channel list a semantic image carries.
- `samtp/` — SAM-TP model assets: fetch script, checksums and engine build (the files themselves are a GitHub Release).

Model assets (once per board)

    bash samtp/fetch_assets.sh

Dependencies on the robot

    rosdep install --from-paths src/elevation_mapping_cupy --ignore-src -r -y

Build on the robot

    colcon build --packages-select elevation_map_msgs elevation_mapping_cupy \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

Run on the robot

NAVI starts and stops this package itself. The haechi robot profile
(`navi_lidar/config/profile/haechi.yaml`, `elevation:` section) names
`haechi_nav.launch.py` and its per-mode arguments; NAVI's mode orchestrator
launches it with every localization/mapping mode and shuts it down with SIGINT
(so the prior/stairs recorders save into the map folder), and `navi_lidar
nav2.launch.py` picks `nav2_params_elevation.yaml` from the same profile.
If the grid stops, the orchestrator reports it and cancels Nav2 goals.
By hand (debugging, NAVI lidar + localization already up):

    ros2 launch elevation_mapping_cupy haechi_nav.launch.py

Bag replay and modes

    ros2 launch elevation_mapping_cupy haechi.launch.py            # navigation chain
    ros2 launch elevation_mapping_cupy haechi.launch.py audit:=true # extra layers for driven_audit

Board install and run procedure: `HAECHI_BOARD.md`. Changes: `CHANGELOG.md`.

Branches follow the NAVI repos: work on `chang_feature`, integrate into `dev`
(`Dev vX.Y.Z — … 반영` merge commits), release from `main`. The board runs `chang_feature`.

Configuration lives in `elevation_mapping_cupy/config/setups/haechi/`:
`haechi.yaml` (frames, topics, publishers), `plugin_config.yaml` (slope, step,
roughness, drivability, semantic safety) and `stairs.yaml` (stairs record/fit/raster).
Shared mapper defaults: `config/core/core_param.yaml`.
