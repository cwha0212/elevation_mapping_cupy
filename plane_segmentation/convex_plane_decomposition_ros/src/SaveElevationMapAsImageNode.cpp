//
// Created by rgrandia on 11.06.20.
//

#include <rclcpp/rclcpp.hpp>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_cv/GridMapCvConverter.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include <opencv2/imgcodecs.hpp>

namespace convex_plane_decomposition {

class SaveElevationMapAsImageNode : public rclcpp::Node {
 public:
  SaveElevationMapAsImageNode() : Node("save_elevation_map_to_image") {
    frequency = this->declare_parameter<double>("frequency", 1.0);
    elevationMapTopic = this->declare_parameter<std::string>("elevation_topic", "elevation");
    elevationLayer = this->declare_parameter<std::string>("height_layer", "elevation");
    imageName = this->declare_parameter<std::string>("imageName", "elevation_map");

    subscriber = this->create_subscription<grid_map_msgs::msg::GridMap>(
        elevationMapTopic, 1, std::bind(&SaveElevationMapAsImageNode::callback, this, std::placeholders::_1));
  }

 private:
  void callback(const grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
    grid_map::GridMap messageMap;
    grid_map::GridMapRosConverter::fromMessage(*message, messageMap);

    const auto& data = messageMap[elevationLayer];
    float maxHeight = std::numeric_limits<float>::lowest();
    float minHeight = std::numeric_limits<float>::max();
    for (int i = 0; i < data.rows(); i++) {
      for (int j = 0; j < data.cols(); j++) {
        const auto value = data(i, j);
        if (!std::isnan(value)) {
          maxHeight = std::max(maxHeight, value);
          minHeight = std::min(minHeight, value);
        }
      }
    }

    cv::Mat image;
    grid_map::GridMapCvConverter::toImage<unsigned char, 1>(messageMap, elevationLayer, CV_8UC1, minHeight, maxHeight, image);

    int range = static_cast<int>(100 * (maxHeight - minHeight));
    cv::imwrite(imageName + "_" + std::to_string(count++) + "_" + std::to_string(range) + "cm.png", image);
  }

  int count = 0;
  double frequency;
  std::string elevationMapTopic;
  std::string elevationLayer;
  std::string imageName;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr subscriber;
};

}  // namespace convex_plane_decomposition

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<convex_plane_decomposition::SaveElevationMapAsImageNode>());
  rclcpp::shutdown();
  return 0;
}