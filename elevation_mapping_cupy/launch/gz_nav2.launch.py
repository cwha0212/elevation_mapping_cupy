"""Nav2 (navi_nav2 dev) over the semantic demo's maps.

    # demo first, then:
    ros2 launch elevation_mapping_cupy gz_nav2.launch.py

Runs navi_nav2's navigation_launch.py with the sim overrides in
config/nav2/gz_nav2_params.yaml: odom as the global frame (the sim has no
localizer and its 3D odometry is ground truth), the traversability octomap
grid as the static map, and no /scan since the sim does not publish one.

What this exists to answer: does a global path thread the stair corridor the
maps now leave open? The planner reads /projected_map, where the flight's
centre is free and the crest's far side is unknown; with allow_unknown the
route over the top is computable before the robot has ever seen beyond it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share_dir = get_package_share_directory("elevation_mapping_cupy")
    params = os.path.join(share_dir, "config", "nav2", "gz_nav2_params.yaml")
    nav2_launch = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch", "navigation_launch.py"
    )
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={
                "params_file": params,
                "use_sim_time": "True",
            }.items(),
        ),
    ])
