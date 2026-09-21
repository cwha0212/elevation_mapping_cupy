"""The terrain judgement inside a Nav2 costmap, as a grid.

    ros2 launch elevation_mapping_cupy haechi_nav2_costmap.launch.py

Runs on top of haechi.launch.py + haechi_octomap.launch.py, which produce the
terrain grid, and next to navi_indoor, which produces /scan and the pose. The
costmap here is a standalone nav2_costmap_2d -- the same class navigation
would run -- so what comes out is what a planner would see.

The relay in front of it exists because Nav2's static layer speaks
transient-local and a live grid publisher does not; see grid_latch_node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = get_package_share_directory("elevation_mapping_cupy")
    params = os.path.join(
        share_dir, "config", "setups", "haechi", "terrain_costmap.yaml"
    )
    if not os.path.exists(params):
        raise FileNotFoundError(f"Config file {params} does not exist")

    use_sim_time = LaunchConfiguration("use_sim_time")

    relay = Node(
        package="elevation_mapping_cupy",
        executable="grid_latch_node.py",
        name="terrain_grid_latch",
        output="screen",
        parameters=[{
            "input_topic": LaunchConfiguration("grid_topic"),
            "output_topic": "/terrain/map",
            "unknown": "keep",
            "use_sim_time": use_sim_time,
        }],
    )

    costmap = Node(
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        name="costmap",
        output="screen",
        parameters=[params, {"use_sim_time": use_sim_time}],
    )

    # A costmap is a lifecycle node and stays inert until something walks it
    # through configure and activate.
    lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_terrain_costmap",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": ["costmap/costmap"],
            "bond_timeout": 0.0,
            # The static layer blocks on its map, and a transition that
            # blocks long enough simply times out -- leaving the costmap
            # inactive, publishing nothing, with one warning to show for
            # it. Start this after the grid exists, and check the state
            # afterwards rather than assuming autostart took.
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "grid_topic",
            default_value="/terrain/projected_map",
            description="The live terrain grid to hand the static layer.",
        ),
        relay,
        costmap,
        lifecycle,
    ])
