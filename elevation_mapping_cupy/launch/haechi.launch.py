"""Elevation mapping for haechi: 3 merged LiDARs plus a front RGB camera.

    ros2 launch elevation_mapping_cupy haechi.launch.py

Geometry comes from /points/merged_deskewed (frame `lidar_frame`), semantics
from the front camera. Bring the geometry up on its own first --
`use_semantics:=false` -- because the camera labels are projected onto the
elevation surface, so a wrong surface moves the labels with it.

Camera extrinsic below is copied from ~/haechi_data/calib/haechi_calibration.yaml
(2026-09-03), section `camera.extrinsic`, which is T_lidarframe_camera. Its
rotation maps camera x/y/z onto lidar_frame -Y / -Z / +X, i.e. right/down/front:
the optical convention, so it publishes directly against the camera's own
`camera_color_optical_frame`. Intrinsics from the same file live in
semantic_sensor/config/haechi.yaml; re-running the calibration means updating
both.
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# T_lidarframe_camera. Degrees here, radians at the call site -- the same
# convention navi_lidar's robot profiles use.
CAMERA_TRANSLATION = (0.683797, -0.063316, -0.023919)
CAMERA_RPY_DEG = (-86.6693, -3.3177, -87.4774)
CAMERA_PARENT_FRAME = "lidar_frame"
CAMERA_CHILD_FRAME = "camera_color_optical_frame"

# haechi_data/calib/haechi_calibration.yaml, `camera:` -- the robot publishes
# no CameraInfo, so these travel with the launch instead. Row-major K, the
# resolution it was calibrated at, and t_reference = t_camera + offset.
CAMERA_K = [632.05027422, 0.0, 626.09047259,
            0.0, 633.89951429, 343.05708742,
            0.0, 0.0, 1.0]
CAMERA_SIZE = [1280, 720]
CAMERA_TIME_OFFSET_S = 0.019554


def _find_key(d, key):
    """First value of `key` anywhere in a nested dict (the profile keeps sensor_height under its lidar section)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def generate_launch_description():
    share_dir = get_package_share_directory("elevation_mapping_cupy")
    core_param_path = os.path.join(share_dir, "config", "core", "core_param.yaml")
    robot_param_path = os.path.join(share_dir, "config", "setups", "haechi", "haechi.yaml")
    # Navigation terrain chain (slope/step/roughness/drivability/safety).
    # gait:=true merges the stairs/ramp/drop chain on top and brings the
    # octomap path up with it; audit:=true adds the layers the board's
    # driven-ground audit reads. Whatever the mode, there is exactly one
    # terrain publisher at terrain_fps: the safety filter's persistence
    # counter advances once per chain evaluation, so a second publisher
    # naming a plugin layer would change the map, not just the traffic.
    plugin_config_path = os.path.join(share_dir, "config", "setups", "haechi", "plugin_config.yaml")
    gait_config_path = os.path.join(share_dir, "config", "setups", "haechi", "plugin_config_gait.yaml")
    for path in (core_param_path, robot_param_path, plugin_config_path, gait_config_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} does not exist")

    use_semantics = LaunchConfiguration("use_semantics")
    use_sim_time = LaunchConfiguration("use_sim_time")
    gait = LaunchConfiguration("gait")
    audit = LaunchConfiguration("audit")
    gait_on = PythonExpression(["'", gait, "'.lower() in ('true', '1')"])
    audit_on = PythonExpression(["'", audit, "'.lower() in ('true', '1')"])
    plugin_files = PythonExpression([
        "['", plugin_config_path, "', '", gait_config_path, "'] if ", gait_on,
        " else ['", plugin_config_path, "']"])
    terrain_layers = PythonExpression([
        "['elevation', 'variance', 'slope', 'step', 'roughness', 'drivability', 'safety']"
        " + (['stairs', 'ramp', 'drop'] if ", gait_on, " else []) if ", audit_on,
        " else (['slope', 'drivability', 'safety', 'stairs', 'drop'] if ", gait_on,
        " else ['drivability', 'safety'])"])
    terrain_basic = PythonExpression(["['elevation'] if ", audit_on, " else ['safety']"])

    x, y, z = CAMERA_TRANSLATION
    roll, pitch, yaw = (math.radians(v) for v in CAMERA_RPY_DEG)

    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_frame_to_camera",
        output="screen",
        condition=IfCondition(use_semantics),
        arguments=[
            "--x", str(x),
            "--y", str(y),
            "--z", str(z),
            "--roll", str(roll),
            "--pitch", str(pitch),
            "--yaw", str(yaw),
            "--frame-id", CAMERA_PARENT_FRAME,
            "--child-frame-id", CAMERA_CHILD_FRAME,
        ],
    )

    # SAM-TP, the same model the simulated path runs. This branch used to be
    # semantic_sensor's Cityscapes classifier and was left behind when SAM-TP
    # replaced it: the real robot has been running with no camera verdict at
    # all, which is why `safety` here has been nothing but the geometry.
    #
    # haechi publishes no CameraInfo, so the calibration is injected -- K and
    # size straight from haechi_calibration.yaml, and the camera-to-reference
    # time offset with them.
    semantic_node = Node(
        package="elevation_mapping_cupy",
        executable="samtp_node.py",
        namespace="front_cam",
        name="samtp_node",
        output="screen",
        condition=IfCondition(use_semantics),
        parameters=[
            {
                "engine_path": LaunchConfiguration("samtp_engine"),
                # Straight off the camera's own topic. SAM-TP decodes JPEG
                # itself, so the decode hop that used to sit here is gone --
                # and with it a QoS mismatch that silently starved the whole
                # branch: image_transport's republish subscribes reliably,
                # the camera publishes best effort, and nothing crossed.
                "image_topic": "/camera/image_raw/compressed",
                "camera_k": CAMERA_K,
                "camera_size": CAMERA_SIZE,
                "time_offset_s": CAMERA_TIME_OFFSET_S,
                "max_rate": LaunchConfiguration("samtp_max_rate"),
                "use_sim_time": use_sim_time,
            }
        ],
    )

    # Self-body cut, the same polygon navi_lidar v0.6.4 applies at merge time
    # (footprint, horizontal, z-independent), for bags recorded before that
    # version existed; on the robot it is redundant and off. The box is
    # navi's haechi footprint in lidar_frame: nav2.footprint (base_link)
    # shifted by odom_2d_to_base_link (+0.23, -0.105), i.e. x -0.53..0.77,
    # y -0.38..0.18 -- 1.30 x 0.56 m, measured 2026-09-16. It runs inside
    # the mapper's cloud callback; nothing else is filtered.
    # navi_lidar's own self-body polygon: profile nav2.footprint (base_link) + odom_2d_to_base_link -> lidar_frame,
    # the same list lidar.launch.py injects as footprint_polygon. One source of truth for the driver cut, the
    # mapper's body cut / leg ring / standing prior and the terrain grid's under-body clearing.
    body_polygon, sensor_height = [], 0.74
    try:
        import yaml
        robot = os.environ.get("ROBOT_TYPE", "").strip() or "haechi"
        prof_path = os.path.join(get_package_share_directory("navi_lidar"), "config", "profile", robot + ".yaml")
        with open(prof_path) as f:
            prof = yaml.safe_load(f) or {}
        nv_fp = (prof.get("nav2") or {}).get("footprint")
        if nv_fp:
            poly = yaml.safe_load(nv_fp) if isinstance(nv_fp, str) else nv_fp
            bl = (prof.get("base_link") or {}).get("odom_2d_to_base_link") or {}
            ox, oy = float(bl.get("x", 0.0)), float(bl.get("y", 0.0))
            for vx, vy in poly:
                body_polygon += [float(vx) + ox, float(vy) + oy]
        sh = _find_key(prof, "sensor_height")
        sensor_height = float(sh) if sh is not None else 0.74
    except Exception as e:  # noqa: BLE001 - profile missing: fall back to the hard-coded box below
        print("[haechi.launch] navi_lidar profile not used (%s): body box fallback" % e)
    body_filter = LaunchConfiguration("body_filter")
    body_margin = LaunchConfiguration("body_margin")
    # The box is always handed over: the leg ring is defined around it even
    # when the merge-time cut upstream has already emptied it.
    body_min = PythonExpression(["[-0.53 - ", body_margin, ", -0.38 - ", body_margin, ", -10.0]"])
    body_max = PythonExpression(["[0.77 + ", body_margin, ", 0.18 + ", body_margin, ", 10.0]"])

    elevation_mapping_node = Node(
        package="elevation_mapping_cupy",
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        parameters=[
            core_param_path,
            robot_param_path,
            {
                "use_sim_time": use_sim_time,
                "plugin_config_file": plugin_files,
                "map_length": LaunchConfiguration("map_length"),
                "publishers.elevation_map_terrain.layers": terrain_layers,
                "publishers.elevation_map_terrain.basic_layers": terrain_basic,
                "publishers.elevation_map_terrain.fps": LaunchConfiguration("terrain_fps"),
                "body_filter": body_filter,
                "body_filter_min": body_min,
                "body_filter_max": body_max,
                "leg_ring": LaunchConfiguration("leg_ring"),
                "leg_rise": LaunchConfiguration("leg_rise"),
                "dedup_voxel": LaunchConfiguration("dedup_voxel"),
                "body_polygon": body_polygon,
                "sensor_height": sensor_height,
                "body_prior": LaunchConfiguration("body_prior"),
            },
        ],
    )

    # Local costmap feed: the safety layer as an OccupancyGrid, cut at the
    # same threshold the bearing fan and the octomap use, so Nav2's local
    # window gets free / lethal / unknown straight from the judged map.
    terrain_grid = Node(
        package="elevation_mapping_cupy",
        executable="terrain_grid_node",
        name="terrain_grid_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "layer": "safety",
                "base_layer": "drivability",
                "threshold": LaunchConfiguration("grid_threshold"),
                "veto_cost": LaunchConfiguration("veto_cost"),
                "graded_free": LaunchConfiguration("graded_free"),
                "graded_lethal": LaunchConfiguration("graded_lethal"),
                "body_polygon": body_polygon,
                "body_frame": "lidar_frame",
                "body_margin": LaunchConfiguration("body_margin_grid"),
                "body_tail": 0.40,
            }
        ],
    )

    # Global memory of the terrain grid on the Nav2 map geometry (merged with /map) for the GLOBAL costmap: the
    # elevation map is a 10 m window, the planner needs what the robot has already judged (2026-10-20).
    terrain_global = Node(
        package="elevation_mapping_cupy",
        executable="terrain_global_node.py",
        name="terrain_global_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("global_grid")),
        parameters=[{"use_sim_time": use_sim_time, "input_topic": "/terrain/local_grid", "map_topic": "/map",
                     "output_topic": "/terrain/global_grid", "mode": "last", "rate_hz": 1.0,
                     "merge": LaunchConfiguration("global_merge"), "prior_file": LaunchConfiguration("terrain_prior")}],
    )

    # LOCALIZATION with a terrain prior: replace Nav2's 2D map with map_terrain.yaml from the prior's folder (SLAM walls +
    # terrain cost, terrain_map_merge.py; built on the fly from map.yaml + the prior if missing) through
    # /map_server/load_map, and keep it there if the orchestrator reloads map.yaml. Nothing runs without terrain_prior.
    terrain_prior = LaunchConfiguration("terrain_prior")
    map_loader = Node(
        package="elevation_mapping_cupy",
        executable="terrain_map_loader.py",
        name="terrain_map_loader",
        output="screen",
        condition=IfCondition(PythonExpression(["'", terrain_prior, "'.strip() != '' and '", LaunchConfiguration("load_nav2_map"),
                                                "'.lower() in ('true', '1')"])),
        parameters=[{"use_sim_time": use_sim_time, "terrain_prior": terrain_prior, "map2d_yaml": LaunchConfiguration("nav2_map2d_yaml"),
                     "load_map_service": LaunchConfiguration("load_map_service")}],
    )

    # MAPPING: record the terrain analysis and save it as the terrain prior (map_terrain_prior.pgm/yaml + npz) that the
    # navigation stack loads through terrain_prior / terrain_global_node (2026-10-21).
    prior_recorder = Node(
        package="elevation_mapping_cupy",
        executable="terrain_prior_recorder.py",
        name="terrain_prior_recorder",
        output="screen",
        condition=IfCondition(LaunchConfiguration("prior_record")),
        parameters=[{"use_sim_time": use_sim_time, "out_dir": LaunchConfiguration("prior_out_dir"),
                     "res": LaunchConfiguration("prior_res")}],
    )

    # The geometry-only grid beside it, for the board's diff and reach probes
    # (what the camera changed is the difference between the two).
    terrain_grid_geom = Node(
        package="elevation_mapping_cupy",
        executable="terrain_grid_node",
        name="terrain_grid_geom",
        output="screen",
        condition=IfCondition(LaunchConfiguration("geom_grid")),
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "layer": "drivability",
                "base_layer": "drivability",
                "output_topic": "/terrain/local_grid_geom",
                "threshold": LaunchConfiguration("grid_threshold"),
                "veto_cost": LaunchConfiguration("veto_cost"),
            }
        ],
    )

    # The bearing fan and octomap: the gait channel's global memory. Only
    # with gait:=true; the navigation split leaves the global map to NAVI.
    octomap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share_dir, "launch", "haechi_octomap.launch.py")),
        condition=IfCondition(gait),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_semantics",
                default_value="true",
                description="Run the front camera semantic branch. Set false to bring "
                "up LiDAR-only geometry first.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "body_filter",
                default_value="true",
                description="Drop returns inside the robot's own footprint before "
                "mapping. Needed for bags recorded before navi_lidar v0.6.4; "
                "harmless after, since the polygon is the same.",
            ),
            DeclareLaunchArgument(
                "body_margin",
                default_value="0.0",
                description="Metres added around the footprint box on every side in x and y "
                "(cuts ground too; superseded by leg_ring, kept as an experiment knob).",
            ),
            DeclareLaunchArgument(
                "leg_ring",
                default_value="[0.50, 0.15, 0.0]",
                description="[front, side, back] metres outside the box where a return "
                "standing leg_rise above the cell height the map already holds is dropped "
                "as a leg. Ground, slopes and stair risers measured on approach pass. "
                "Measured 2026-09-28: leg bursts reach 0.25 m ahead, half that sideways, none behind.",
            ),
            DeclareLaunchArgument("global_grid", default_value="true",
                                  description="terrain_global_node: accumulate the terrain grid on the /map geometry -> /terrain/global_grid "
                                              "for Nav2's global costmap (nav2_params_elevation.yaml static_layer.map_topic)."),
            DeclareLaunchArgument("prior_record", default_value="false",
                                  description="MAPPING: run terrain_prior_recorder (accumulates the terrain grid with per-cell vote "
                                              "counters, saves map_terrain_prior.pgm/yaml + terrain_prior.npz to prior_out_dir on "
                                              "shutdown or ~/save)."),
            DeclareLaunchArgument("prior_out_dir", default_value=os.path.expanduser("~/terrain_prior")),
            DeclareLaunchArgument("prior_res", default_value="0.1"),
            DeclareLaunchArgument("terrain_prior", default_value="",
                                  description="terrain_global_node: map_terrain_prior.yaml (+pgm, mode scale) baked at mapping time "
                                              "(terrain_map_bake.py); the 2D grid the elevation stack plans on: live terrain > prior > /map. "
                                              "Prior cost is capped at 89 (never a wall). Empty = no prior."),
            DeclareLaunchArgument("load_nav2_map", default_value="true",
                                  description="with terrain_prior: load <prior dir>/map_terrain.yaml into Nav2's map_server "
                                              "(/map_server/load_map) and keep it loaded. false = Nav2 keeps map.yaml."),
            DeclareLaunchArgument("nav2_map2d_yaml", default_value="",
                                  description="the 2D map file to load instead of <prior dir>/map_terrain.yaml."),
            DeclareLaunchArgument("load_map_service", default_value="/map_server/load_map"),
            DeclareLaunchArgument("global_merge", default_value="override",
                                  description="terrain_global_node: 'override' = a terrain-judged cell replaces the prior /map value "
                                              "(dirty SLAM pgm is cleaned as the robot re-observes it); 'max' = /map lethal is permanent."),
            DeclareLaunchArgument("body_prior", default_value="true",
                                  description="write the standing ground (sensor z - sensor_height) under the body polygon on every "
                                              "pose update: the lidars never re-measure those cells (2026-10-20)."),
            DeclareLaunchArgument("graded_free", default_value="0.6",
                                  description="terrain grid: safety >= this -> cost 0; <= graded_lethal -> 100; linear between "
                                              "(graded_lethal < 0 = old binary cut at grid_threshold)."),
            DeclareLaunchArgument("graded_lethal", default_value="0.2"),
            DeclareLaunchArgument("body_margin_grid", default_value="0.10",
                                  description="terrain grid: cells under the body polygon + this margin (+0.4 m tail) forced free."),
            DeclareLaunchArgument("leg_rise", default_value="0.08",
                                  description="Height above the known cell that marks a ring point as a leg."),
            DeclareLaunchArgument(
                "dedup_voxel",
                default_value="0.02",
                description="One point per voxel of this size after the body cut (0 = off). "
                "Below the map cell it only merges coincident returns of the three lidars; "
                "navi_lidar already voxelises at 0.15 m, so the robot preset turns it off.",
            ),
            DeclareLaunchArgument(
                "samtp_engine",
                default_value=os.path.expanduser("~/samtp/samtp_512_fp16.engine"),
                description="TensorRT engine for SAM-TP. Machine specific, so "
                "it lives outside the repo and is rebuilt per device with "
                "trtexec --onnx=... --fp16.",
            ),
            DeclareLaunchArgument(
                "grid_threshold",
                default_value="0.4",
                description="safety below this is lethal in /terrain/local_grid; "
                "matches the octomap's threshold so local and global agree.",
            ),
            DeclareLaunchArgument(
                "veto_cost",
                default_value="70",
                description="Grid value for a cell the geometry passes and only the "
                "camera fails: a cost the planner may pay, not a wall. -1 = lethal.",
            ),
            DeclareLaunchArgument("gait", default_value="false",
                                  description="Add the stairs/ramp/drop chain and the octomap path."),
            DeclareLaunchArgument("audit", default_value="false",
                                  description="Publish the extra terrain layers the driven-ground audit reads."),
            DeclareLaunchArgument("geom_grid", default_value="true",
                                  description="Also publish the geometry-only grid on /terrain/local_grid_geom."),
            DeclareLaunchArgument("map_length", default_value="10.0", description="Map side in metres."),
            DeclareLaunchArgument("terrain_fps", default_value="10.0", description="Terrain publisher rate."),
            DeclareLaunchArgument("samtp_max_rate", default_value="4.0", description="SAM-TP inference rate cap."),
            camera_tf,
            semantic_node,
            elevation_mapping_node,
            terrain_grid,
            terrain_global,
            map_loader,
            prior_recorder,
            terrain_grid_geom,
            octomap,
        ]
    )
