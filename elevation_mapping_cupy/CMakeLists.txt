cmake_minimum_required(VERSION 3.8)
project(elevation_mapping_cupy)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(elevation_map_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED) # Usually brought in by vendor, but good to be explicit
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

include_directories(include)

add_library(elevation_mapping_ros
  src/elevation_mapping_wrapper.cpp
  src/elevation_mapping_ros.cpp
)
target_include_directories(elevation_mapping_ros PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)
target_link_libraries(elevation_mapping_ros 
  ${OpenCV_LIBRARIES}
  pybind11::embed
  Python3::Python
)
ament_target_dependencies(elevation_mapping_ros
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  geometry_msgs
  visualization_msgs
  elevation_map_msgs
  grid_map_msgs
  grid_map_ros
  image_transport
  pcl_ros
  pcl_conversions
  cv_bridge
)

add_executable(elevation_mapping_node src/elevation_mapping_node.cpp)
target_link_libraries(elevation_mapping_node elevation_mapping_ros)
ament_target_dependencies(elevation_mapping_node rclcpp)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(
  TARGETS elevation_mapping_ros elevation_mapping_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(elevation_mapping_ros)
ament_export_dependencies(
  rclcpp
  elevation_map_msgs
  grid_map_ros
)

ament_package()
