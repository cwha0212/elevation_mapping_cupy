"""Octomap-based 2D occupancy grid, run alongside gz_fortress_demo.launch.py.

    ros2 launch elevation_mapping_cupy gz_octomap.launch.py

This is the flat-world baseline the navi stack builds its 2D grid with: feed
the lidar into octomap, project occupied voxels inside a fixed height band to
an OccupancyGrid (/projected_map). On one floor it works; the point of running
it next to the elevation map is to watch what happens when the floor itself
moves -- drive up the ramp and the surface you stand on enters the obstacle
band, because "obstacle" was defined as a z-interval of the map frame, not as
something the robot cannot traverse.

The elevation pipeline never uses that assumption: slope/step/roughness are
relative measures, so a 12 degree ramp stays drivable at any altitude.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    octomap = Node(
        package="octomap_server",
        executable="octomap_server_node",
        name="octomap_server",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "frame_id": "odom",
            "base_frame_id": "base_link",
            "resolution": 0.05,
            # Raw 360 lidar in; octomap raycasts free space itself.
            "sensor_model.max_range": 12.0,
            "filter_ground": False,
            # Occupied voxels inside this odom-frame band become obstacles in
            # /projected_map. 5 cm floor clearance, robot height on top: the
            # standard single-floor band, kept deliberately so the climbing
            # experiment shows its failure mode instead of hiding it.
            "occupancy_min_z": 0.05,
            "occupancy_max_z": 1.2,
        }],
        remappings=[("cloud_in", "/lidar/points")],
    )
    return LaunchDescription([octomap])
