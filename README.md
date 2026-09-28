# elevation_mapping_cupy (haechi fork)

GPU elevation mapping and terrain judgement for the haechi quadruped. The map
feeds Nav2's local costmap as an occupancy grid; the global map stays NAVI's.

Packages

- `elevation_mapping_cupy` — the mapper (`elevation_mapping_node.py`), the
  SAM-TP camera node (`samtp_node.py`), the terrain plugin chain and
  `terrain_grid_node` (C++), which cuts the `safety` layer into
  `/terrain/local_grid` (free / camera-cost / lethal / unknown).
- `elevation_map_msgs` — `ChannelInfo`, the channel list a semantic image carries.
- `gz_demo` (`elevation_mapping_gz_demo`) — the Gazebo Fortress bench: worlds,
  sim configs, the stairs/gait channel nodes and the Nav2 sim parameters.
  Not needed on the robot.

Build on the robot

    colcon build --packages-select elevation_map_msgs elevation_mapping_cupy \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

Run on the robot (NAVI lidar + localization already up)

    ros2 launch elevation_mapping_cupy haechi_nav.launch.py
    ros2 launch navi_lidar nav2.launch.py robot:=haechi map:=<map.yaml> \
      params_file:=$(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/params/nav2_params_elevation.yaml

Bag replay and modes

    ros2 launch elevation_mapping_cupy haechi.launch.py            # navigation chain
    ros2 launch elevation_mapping_cupy haechi.launch.py gait:=true # + stairs/ramp/drop + octomap
    ros2 launch elevation_mapping_cupy haechi.launch.py audit:=true # extra layers for driven_audit

Board install and run procedure: `HAECHI_BOARD.md`.

Configuration lives in `elevation_mapping_cupy/config/setups/haechi/`:
`haechi.yaml` (frames, topics, publishers), `plugin_config.yaml` (slope, step,
roughness, drivability, semantic safety) and `plugin_config_gait.yaml`.
