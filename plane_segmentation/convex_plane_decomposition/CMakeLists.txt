cmake_minimum_required(VERSION 3.8)
project(convex_plane_decomposition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(CGAL REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_cv REQUIRED)
find_package(grid_map_filters_rsl REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS filesystem)

include_directories(include)

add_library(${PROJECT_NAME} SHARED
  src/contour_extraction/ContourExtraction.cpp
  src/contour_extraction/Upsampling.cpp
  src/ransac/RansacPlaneExtractor.cpp
  src/sliding_window_plane_extraction/SlidingWindowPlaneExtractor.cpp
  src/ConvexRegionGrowing.cpp
  src/Draw.cpp
  src/GridMapPreprocessing.cpp
  src/LoadGridmapFromImage.cpp
  src/PlanarRegion.cpp
  src/PlaneDecompositionPipeline.cpp
  src/Postprocessing.cpp
  src/SegmentedPlaneProjection.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)

# CGAL flags
target_compile_options(${PROJECT_NAME} PUBLIC -DCGAL_HAS_THREADS)
target_link_libraries(${PROJECT_NAME} CGAL::CGAL Boost::filesystem)

ament_target_dependencies(${PROJECT_NAME}
  grid_map_core
  grid_map_cv
  grid_map_filters_rsl
  OpenCV
  Eigen3
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_${PROJECT_NAME}
    test/testConvexApproximation.cpp
    test/testPipeline.cpp
    test/testPlanarRegion.cpp
    test/testRegionGrowing.cpp
    test/testUpsampling.cpp
  )
  target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} CGAL::CGAL Boost::filesystem)
  ament_target_dependencies(test_${PROJECT_NAME} grid_map_core grid_map_cv grid_map_filters_rsl OpenCV Eigen3)
endif()

ament_export_dependencies(
  grid_map_core
  grid_map_cv
  grid_map_filters_rsl
  OpenCV
  CGAL
)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_package()
