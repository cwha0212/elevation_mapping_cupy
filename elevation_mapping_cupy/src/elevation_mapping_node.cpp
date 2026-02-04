//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

// Pybind
#include <pybind11/embed.h>  // everything needed for embedding

// ROS
#include <rclcpp/rclcpp.hpp>

#include "elevation_mapping_cupy/elevation_mapping_ros.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  
  py::scoped_interpreter guard{};  // start the interpreter and keep it alive
  
  rclcpp::NodeOptions options;
  // Automatically declare parameters from overrides to handle complex dictionary parameters if we used the generic way, 
  // but we implemented manual parsing. Still good practice.
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<elevation_mapping_cupy::ElevationMappingNode>(options);
  
  py::gil_scoped_release release;

  // Spin
  // Use MultiThreadedExecutor to allow concurrent callback processing if needed (e.g. services and timers)
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  
  rclcpp::shutdown();
  return 0;
}
