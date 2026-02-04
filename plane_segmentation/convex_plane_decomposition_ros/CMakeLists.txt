cmake_minimum_required(VERSION 3.8)
project(convex_plane_decomposition_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_cv REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(convex_plane_decomposition REQUIRED)
find_package(convex_plane_decomposition_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS imgproc imgcodecs core)
find_package(Eigen3 REQUIRED)
find_package(grid_map_filters_rsl REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}
  src/ConvexPlaneDecompositionRos.cpp
  src/MessageConversion.cpp
  src/ParameterLoading.cpp
  src/RosVisualizations.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  grid_map_core
  grid_map_ros
  grid_map_cv
  grid_map_msgs
  geometry_msgs
  convex_plane_decomposition
  convex_plane_decomposition_msgs
  tf2_ros
  visualization_msgs
  OpenCV
  grid_map_filters_rsl
)
target_link_libraries(${PROJECT_NAME} gmp mpfr ${OpenCV_LIBRARIES})

add_executable(${PROJECT_NAME}_node
  src/ConvexPlaneDecompositionNode.cpp
)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} gmp mpfr ${OpenCV_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME}_node 
  rclcpp 
  grid_map_core 
  grid_map_ros 
  grid_map_msgs
  OpenCV
)

add_executable(${PROJECT_NAME}_add_noise
  src/noiseNode.cpp
)
ament_target_dependencies(${PROJECT_NAME}_add_noise 
  rclcpp 
  grid_map_core 
  grid_map_ros 
  grid_map_msgs
  grid_map_filters_rsl
)

add_executable(${PROJECT_NAME}_save_elevationmap
  src/SaveElevationMapAsImageNode.cpp
)
target_link_libraries(${PROJECT_NAME}_save_elevationmap ${OpenCV_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME}_save_elevationmap 
  rclcpp 
  grid_map_core 
  grid_map_ros 
  grid_map_cv 
  grid_map_msgs 
  OpenCV
)

add_executable(${PROJECT_NAME}_approximation_demo_node
  src/ConvexApproximationDemoNode.cpp
)
target_link_libraries(${PROJECT_NAME}_approximation_demo_node ${PROJECT_NAME} gmp mpfr ${OpenCV_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME}_approximation_demo_node 
  rclcpp 
  grid_map_core 
  grid_map_ros 
  grid_map_msgs
  OpenCV
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  # Add other tests here if needed
  # add_executable(${PROJECT_NAME}_TestShapeGrowing test/TestShapeGrowing.cpp)
endif()

install(TARGETS 
  ${PROJECT_NAME}
  ${PROJECT_NAME}_node
  ${PROJECT_NAME}_add_noise
  ${PROJECT_NAME}_save_elevationmap
  ${PROJECT_NAME}_approximation_demo_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY config data launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
  rclcpp
  convex_plane_decomposition
  convex_plane_decomposition_msgs
)

ament_package()