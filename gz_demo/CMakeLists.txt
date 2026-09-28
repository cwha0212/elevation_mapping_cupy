cmake_minimum_required(VERSION 3.8)
project(elevation_mapping_gz_demo)

# The Gazebo Fortress bench for elevation_mapping_cupy: worlds, sim configs,
# the stairs/gait channel nodes and the Nav2 sim parameters. Nothing in here
# is needed on the robot; it lives in its own package so the runtime build
# (--packages-select elevation_mapping_cupy elevation_map_msgs) never carries it.
find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/stairs_cloud_node.py
  scripts/nav_grid_fuse_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config rviz gazebo
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
