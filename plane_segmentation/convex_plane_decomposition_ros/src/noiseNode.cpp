//
// Created by rgrandia on 25.10.21.
//

#include <rclcpp/rclcpp.hpp>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include <grid_map_filters_rsl/inpainting.hpp>
#include <grid_map_filters_rsl/smoothing.hpp>

namespace convex_plane_decomposition {

class NoiseNode : public rclcpp::Node {
 public:
  NoiseNode() : Node("noise_node") {
    frequency = this->declare_parameter<double>("frequency", 10.0);
    noiseGauss = this->declare_parameter<double>("noiseGauss", 0.0);
    noiseUniform = this->declare_parameter<double>("noiseUniform", 0.0);
    blur = this->declare_parameter<bool>("blur", false);
    outlierPercentage = this->declare_parameter<double>("outlier_percentage", 0.0);
    elevationMapTopicIn = this->declare_parameter<std::string>("elevation_topic_in", "elevation_in");
    elevationMapTopicOut = this->declare_parameter<std::string>("elevation_topic_out", "elevation_out");
    elevationLayer = this->declare_parameter<std::string>("height_layer", "elevation");

    publisher = this->create_publisher<grid_map_msgs::msg::GridMap>(elevationMapTopicOut, 1);
    subscriber = this->create_subscription<grid_map_msgs::msg::GridMap>(
        elevationMapTopicIn, 1, std::bind(&NoiseNode::callback, this, std::placeholders::_1));
  }

 private:
  void createNoise(size_t row, size_t col) {
    grid_map::GridMap::Matrix u1 = 0.5f * grid_map::GridMap::Matrix::Random(row, col).array() + 0.5f;
    grid_map::GridMap::Matrix u2 = 0.5f * grid_map::GridMap::Matrix::Random(row, col).array() + 0.5f;
    grid_map::GridMap::Matrix gauss01 =
        u1.binaryExpr(u2, [&](float v1, float v2) {
            return static_cast<float>(std::sqrt(-2.0 * std::log(static_cast<double>(v1))) * std::cos(2.0 * M_PI * static_cast<double>(v2)));
        });

    noiseLayer = static_cast<float>(noiseUniform) * grid_map::GridMap::Matrix::Random(row, col) + static_cast<float>(noiseGauss) * gauss01;
  }

  void callback(const grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
    grid_map::GridMap messageMap;
    grid_map::GridMapRosConverter::fromMessage(*message, messageMap);

    if (blur) {
      auto originalMap = messageMap.get(elevationLayer);
      grid_map::inpainting::minValues(messageMap, elevationLayer, "i");
      grid_map::smoothing::boxBlur(messageMap, "i", elevationLayer, 3, 1);
      messageMap.get(elevationLayer) = (originalMap.array().isFinite()).select(messageMap.get(elevationLayer), originalMap);
    }

    auto& elevation = messageMap.get(elevationLayer);
    if (noiseLayer.size() != elevation.size()) {
      createNoise(elevation.rows(), elevation.cols());
    }

    elevation += noiseLayer;

    auto messageMapOut = grid_map::GridMapRosConverter::toMessage(messageMap);
    publisher->publish(*messageMapOut);
  }

  double noiseUniform;
  double noiseGauss;
  double outlierPercentage;
  bool blur;
  double frequency;
  std::string elevationMapTopicIn;
  std::string elevationMapTopicOut;
  std::string elevationLayer;
  grid_map::GridMap::Matrix noiseLayer;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr publisher;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr subscriber;
};

}  // namespace convex_plane_decomposition

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<convex_plane_decomposition::NoiseNode>());
  rclcpp::shutdown();
  return 0;
}