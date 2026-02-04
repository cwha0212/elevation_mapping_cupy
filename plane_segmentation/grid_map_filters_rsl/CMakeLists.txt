cmake_minimum_required(VERSION 3.8)
project(grid_map_filters_rsl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(grid_map_cv REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME} SHARED
  src/GridMapDerivative.cpp
  src/inpainting.cpp
  src/lookup.cpp
  src/smoothing.cpp
  src/processing.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}
  grid_map_cv
  grid_map_core
  OpenCV
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  
  ament_add_gtest(test_${PROJECT_NAME}
    test/TestDerivativeFilter.cpp
    test/TestFilters.cpp
    test/TestLookup.cpp
  )
  target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})
  ament_target_dependencies(test_${PROJECT_NAME} grid_map_cv grid_map_core OpenCV)
endif()

ament_export_dependencies(
  grid_map_cv
  grid_map_core
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