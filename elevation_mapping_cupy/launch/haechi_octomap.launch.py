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
            "layer": "safety",
            "threshold": threshold,
            # haechi maps in odom, per config/setups/haechi/haechi.yaml.
            "map_frame": "odom",
            "scan_topic": LaunchConfiguration("scan_topic"),
            # Outdoors the safety score is dominated by vegetation, and a
            # quadruped walks through what it can step over. Only something
            # standing this far above the ground under the robot may block a
            # bearing -- the sim keeps the old behaviour, where a 0.12 m kerb
            # is exactly what must stop it.
            "min_obstacle_rise": LaunchConfiguration("min_obstacle_rise"),
            # navi's own 2D scan blanks the rear; the fan does the same,
            # for the cloud octomap reads as well as the scan.
            "rear_blank_deg": LaunchConfiguration("rear_blank_deg"),
        }],
    )

    octomap = Node(
        package="octomap_server2",
        executable="octomap_server",
        name="octomap_server",
        namespace="terrain",
        output="screen",
        parameters=[{
            # `map`, not `odom`. This stack's odom rides ON the robot -- a
            # static identity to lidar_frame -- so an octree anchored there
            # is anchored to the robot: measured over a 56 m route, the grid
            # came out 9x10 m, the size of the elevation window, while
            # navi's own map of the same run covered 65x96 m. The fixed
            # world frame here is `map`, the one LIO corrects.
            "frame_id": "map",
            "base_frame_id": "lidar_frame",
            # Matches the elevation map, so a grid cell is a map cell.
            "resolution": 0.05,
            # The elevation map is 10 m square, so nothing useful arrives from
            # further than its half-width.
            "sensor_model/max_range": 5.0,
            "filter_ground": False,
            # No height band. The cloud is a flat sheet by construction and
            # the verdict on it is already altitude-free, so there is nothing
            # for a band to filter -- while the sheet itself rides at the
            # robot's own height, which walks up and down with the ground.
            # Measured: with a +-0.10 m band about world z=0 the projection
            # held only the first stretch of a 56 m route, because after that
            # the robot was no longer standing where it started. The octree
            # kept the cells; the 2D projection dropped them.
            "occupancy_min_z": -50.0,
            "occupancy_max_z": 50.0,
        }],
        remappings=[("cloud_in", "/traversability/obstacles")],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "threshold",
            default_value="0.4",
            description="safety below this becomes an obstacle. Against the "
            "0.20 m step limit, 0.15 m risers score 0.25: 0.4 calls stairs an "
            "obstacle, 0.2 leaves them climbable.",
        ),
        DeclareLaunchArgument(
            "min_obstacle_rise",
            default_value="0.30",
            description="Height above the ground under the robot before a "
            "cell may block a bearing. 0 disables the test, which is what "
            "the simulated setups want.",
        ),
        DeclareLaunchArgument(
            "rear_blank_deg",
            default_value="40.0",
            description="Half-angle of the rear arc the fan stays silent "
            "about, in the cloud and the scan alike, matching navi's own 2D "
            "scan. 0 keeps the full circle.",
        ),
        DeclareLaunchArgument(
            "scan_topic",
            default_value="",
            description="Publish the fan as a LaserScan here too (e.g. "
            "/terrain_scan) for a 2D navigation stack. Empty = off.",
        ),
        trav_cloud,
        octomap,
    ])
