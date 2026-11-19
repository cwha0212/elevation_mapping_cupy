cmake_minimum_required(VERSION 3.8)
project(elevation_mapping_cupy)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
# Only the C++ node needs these at build time; the Python nodes' dependencies are exec_depends in package.xml.
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(tf2_ros REQUIRED)

# The one C++ node here: grid_map's own converter, handing a terrain layer to
# Nav2 whole instead of through the bearing fan. See src/terrain_grid_node.cpp
# for why that is right for a local costmap and wrong for a global map.
add_executable(terrain_grid_node src/terrain_grid_node.cpp)
ament_target_dependencies(terrain_grid_node
  rclcpp grid_map_msgs grid_map_core grid_map_ros nav_msgs tf2_ros geometry_msgs std_msgs)
install(TARGETS terrain_grid_node DESTINATION lib/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(haechi_terrain)   # stairs_fit / stairs_raster / terrain_map / config (plain python, used by the scripts)

install(PROGRAMS
  scripts/elevation_mapping_node.py
  scripts/samtp_node.py
  scripts/terrain_global_node.py
  scripts/terrain_prior_recorder.py
  scripts/terrain_map_merge.py
  scripts/terrain_map_loader.py
  scripts/stairs_scan_filter.py
  scripts/stairs_reclear.py
  scripts/stairs_prior_recorder.py
  scripts/stairs_prior_publisher.py
  scripts/stairs_fit.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
