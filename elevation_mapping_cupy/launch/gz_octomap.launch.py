"""Octomap 2D occupancy grid, run alongside gz_fortress_demo.launch.py.

    ros2 launch elevation_mapping_cupy gz_octomap.launch.py
    ros2 launch elevation_mapping_cupy gz_octomap.launch.py source:=lidar

Two ways to feed the same in-house octomap_server2 (source
~/dependencies/octomap_ws/install/setup.bash before launching):

  source:=traversability  (default)
      The drivability layer becomes the obstacle cloud. "Obstacle" means a
      cell the robot cannot drive on, at any altitude, so a ramp stays free
      while a curb does not. octomap accumulates those cells with its log-odds
      sensor model, making /projected_map a probabilistic costmap over
      traversability -- and a global one, outliving the elevation map's
      rolling window.

  source:=lidar
      The raw cloud with a fixed odom-frame height band, which is how the
      navi stack builds its grid today. Kept for comparison: it defines
      obstacles by height, so climbing the ramp puts the robot's own surface
      at 98% occupied. Measured, not hypothetical.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    source = LaunchConfiguration("source")
    threshold = LaunchConfiguration("threshold")
    is_lidar = PythonExpression(["'", source, "' == 'lidar'"])

    trav_cloud = Node(
        package="elevation_mapping_cupy",
        executable="traversability_cloud_node.py",
        name="traversability_cloud",
        output="screen",
        condition=UnlessCondition(is_lidar),
        parameters=[{"use_sim_time": True, "threshold": threshold}],
    )

    # The cloud is already flat and robot-centred, so the band only has to
    # cover z=0. Nothing is excluded by height because nothing carries height.
    trav_octomap = Node(
        package="octomap_server2",
        executable="octomap_server",
        name="octomap_server",
        output="screen",
        condition=UnlessCondition(is_lidar),
        parameters=[{
            "use_sim_time": True,
            "frame_id": "odom",
            "base_frame_id": "base_link",
            "resolution": 0.05,
            # Below the cloud's far_range (6.0) on purpose: a clear bearing's
            # endpoint must land BEYOND max_range so octomap truncates it into
            # a pure free ray. At 8.0 the same point would be inserted as an
            # obstacle at 6 m. Matches the 4.4 m march, so nothing is declared
            # free that was not actually checked.
            "sensor_model/max_range": 4.5,
            "filter_ground": False,
            "occupancy_min_z": -0.10,
            "occupancy_max_z": 0.10,
        }],
        remappings=[("cloud_in", "/traversability/obstacles")],
    )

    # Stairs channel: the main grid keeps stairs blocked (the safe default),
    # and this second, tiny octomap answers WHICH blocked cells are stairs.
    # Nav2's KeepoutFilter can hold /stairs/projected_map closed until the
    # supervisor switches gait; opening the stairs is then one filter toggle.
    # No free-space fan feeds it, so a stair, once seen, stays on the map.
    stairs_cloud = Node(
        package="elevation_mapping_cupy",
        executable="stairs_cloud_node.py",
        name="stairs_cloud",
        output="screen",
        condition=UnlessCondition(is_lidar),
        parameters=[{"use_sim_time": True}],
    )
    stairs_octomap = Node(
        package="octomap_server2",
        executable="octomap_server",
        name="octomap_server",
        namespace="stairs",
        output="screen",
        condition=UnlessCondition(is_lidar),
        parameters=[{
            "use_sim_time": True,
            "frame_id": "odom",
            "base_frame_id": "base_link",
            "resolution": 0.05,
            # Plain occupied insertion, so range only needs to cover the map
            # window -- there is no beyond-range free trick on this channel.
            "sensor_model/max_range": 8.0,
            "filter_ground": False,
            "occupancy_min_z": -0.10,
            "occupancy_max_z": 0.10,
        }],
        remappings=[("cloud_in", "/stairs/cells")],
    )

    lidar_octomap = Node(
        package="octomap_server2",
        executable="octomap_server",
        name="octomap_server",
        output="screen",
        condition=IfCondition(is_lidar),
        parameters=[{
            "use_sim_time": True,
            "frame_id": "odom",
            "base_frame_id": "base_link",
            "resolution": 0.05,
            "sensor_model/max_range": 12.0,
            "filter_ground": False,
            # Obstacles by height in the odom frame: the assumption under test.
            "occupancy_min_z": 0.05,
            "occupancy_max_z": 1.2,
        }],
        remappings=[("cloud_in", "/lidar/points")],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "source",
            default_value="traversability",
            description="traversability (drivability layer) or lidar (raw cloud, height band).",
        ),
        DeclareLaunchArgument(
            "threshold",
            default_value="0.4",
            description="drivability below this becomes an obstacle (traversability source).",
        ),
        trav_cloud,
        trav_octomap,
        stairs_cloud,
        stairs_octomap,
        lidar_octomap,
    ])
