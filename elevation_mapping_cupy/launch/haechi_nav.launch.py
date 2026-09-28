"""haechi on the robot: elevation + SAM-TP + terrain grids for Nav2's local costmap.

    ros2 launch elevation_mapping_cupy haechi_nav.launch.py
    ros2 launch elevation_mapping_cupy haechi_nav.launch.py lite:=true

No octomap: under the navigation split the global map is NAVI's map_server.
navi_lidar >= v0.6.4 cuts the exact footprint polygon out of the merged cloud
at merge time, so the box cut here is off. Legs in mid-stride reach past the
polygon (measured: the cell under the robot went lethal in 2.2% of frames,
almost all while walking); the mapper's leg ring handles that by dropping
ring points that stand above the cell height it already holds, which keeps
ground, slopes and stair risers. The dedup is off because navi_lidar
already voxelises at 0.15 m.

The measured configuration (10 m map at 0.05, terrain at 3 Hz, SAM-TP at
4 Hz) is the default. lite:=true is the fallback for a loaded board: an 8 m
map and SAM-TP at 2 Hz. Nav2 only ever reads the 5 x 5 m window either way.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    share_dir = get_package_share_directory("elevation_mapping_cupy")
    lite = LaunchConfiguration("lite")
    lite_on = PythonExpression(["'", lite, "'.lower() in ('true', '1')"])
    return LaunchDescription([
        DeclareLaunchArgument("lite", default_value="false",
                              description="8 m map and SAM-TP at 2 Hz for a loaded board."),
        DeclareLaunchArgument("use_semantics", default_value="true"),
        DeclareLaunchArgument("body_filter", default_value="false"),
        DeclareLaunchArgument("leg_ring", default_value="[0.25, 0.10, 0.0]"),
        DeclareLaunchArgument("samtp_engine",
                              default_value=os.path.expanduser("~/samtp/samtp_512_fp16.engine")),
        DeclareLaunchArgument("grid_threshold", default_value="0.4"),
        DeclareLaunchArgument("veto_cost", default_value="70"),
        DeclareLaunchArgument("geom_grid", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share_dir, "launch", "haechi.launch.py")),
            launch_arguments={
                "use_sim_time": "false",
                "use_semantics": LaunchConfiguration("use_semantics"),
                "body_filter": LaunchConfiguration("body_filter"),
                "leg_ring": LaunchConfiguration("leg_ring"),
                "dedup_voxel": "0.0",
                "samtp_engine": LaunchConfiguration("samtp_engine"),
                "grid_threshold": LaunchConfiguration("grid_threshold"),
                "veto_cost": LaunchConfiguration("veto_cost"),
                "geom_grid": LaunchConfiguration("geom_grid"),
                "gait": "false",
                "audit": "false",
                "map_length": PythonExpression(["'8.0' if ", lite_on, " else '10.0'"]),
                "terrain_fps": "3.0",
                "samtp_max_rate": PythonExpression(["'2.0' if ", lite_on, " else '4.0'"]),
            }.items(),
        ),
    ])
