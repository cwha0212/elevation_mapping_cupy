"""Traversability-driven octomap 2D grid for haechi.

    ros2 launch elevation_mapping_cupy haechi_octomap.launch.py

Runs next to haechi.launch.py. The drivability layer becomes the obstacle
cloud, so /projected_map carries what the robot cannot drive on rather than
what is tall -- which is the difference between a curb and a ramp, and the
reason a quadruped's map should not be built from a height band.

octomap_server2 comes from ~/dependencies/octomap_ws; source its setup.bash
first. The elevation map is a rolling 10 m window, so the octree is also what
gives the 2D grid any memory beyond it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    threshold = LaunchConfiguration("threshold")

    trav_cloud = Node(
        package="elevation_mapping_cupy",
        executable="traversability_cloud_node.py",
        name="traversability_cloud",
        output="screen",
        parameters=[{
            "input_topic": "/elevation_mapping_node/elevation_map_terrain",
            "layer": "drivability",
            "threshold": threshold,
            # haechi maps in odom, per config/setups/haechi/haechi.yaml.
            "map_frame": "odom",
        }],
    )

    octomap = Node(
        package="octomap_server2",
        executable="octomap_server",
        name="octomap_server",
        output="screen",
        parameters=[{
            "frame_id": "odom",
            "base_frame_id": "base_link",
            # Matches the elevation map, so a grid cell is a map cell.
            "resolution": 0.05,
            # The elevation map is 10 m square, so nothing useful arrives from
            # further than its half-width.
            "sensor_model/max_range": 5.0,
            "filter_ground": False,
            # The cloud is flat by construction; the band only covers z=0.
            "occupancy_min_z": -0.10,
            "occupancy_max_z": 0.10,
        }],
        remappings=[("cloud_in", "/traversability/obstacles")],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "threshold",
            default_value="0.4",
            description="drivability below this becomes an obstacle. Against the "
            "0.20 m step limit, 0.15 m risers score 0.25: 0.4 calls stairs an "
            "obstacle, 0.2 leaves them climbable.",
        ),
        trav_cloud,
        octomap,
    ])
