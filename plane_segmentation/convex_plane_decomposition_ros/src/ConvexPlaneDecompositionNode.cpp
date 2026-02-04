#include "convex_plane_decomposition_ros/ConvexPlaneDecompositionRos.h"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("convex_plane_decomposition_ros", options);

  double frequency = 10.0;
  // frequency param is not used in the class, effectively.
  // The class uses subscriber.
  // The ROS1 code had a loop with spinOnce and sleep. This implies updating something periodically?
  // But ConvexPlaneExtractionROS only has 'callback' attached to subscriber.
  // So standard spin is fine. The original rate logic might have been a way to not hog CPU or something, but with spin() it blocks.
  
  convex_plane_decomposition::ConvexPlaneExtractionROS convex_plane_decomposition_ros(node.get());

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
