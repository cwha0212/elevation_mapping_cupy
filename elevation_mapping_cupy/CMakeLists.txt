cmake_minimum_required(VERSION 3.8)
project(elevation_mapping_cupy)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(elevation_map_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)

set(dependencies
  rclpy
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  message_filters
  elevation_map_msgs
  grid_map_msgs
  tf2_ros
  cv_bridge
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/elevation_mapping_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(${dependencies})
ament_package()
