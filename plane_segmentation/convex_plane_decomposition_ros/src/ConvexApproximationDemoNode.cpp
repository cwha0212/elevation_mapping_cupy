//
// Created by rgrandia on 24.06.20.
//

#include <mutex>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>

#include <convex_plane_decomposition/ConvexRegionGrowing.h>
#include <convex_plane_decomposition/GeometryUtils.h>
#include <convex_plane_decomposition/SegmentedPlaneProjection.h>

#include <convex_plane_decomposition_msgs/msg/planar_terrain.hpp>

#include "convex_plane_decomposition_ros/MessageConversion.h"
#include "convex_plane_decomposition_ros/RosVisualizations.h"

namespace convex_plane_decomposition {

class ConvexApproximationDemoNode : public rclcpp::Node {
 public:
  ConvexApproximationDemoNode() : Node("convex_approximation_demo_node") {
    frameId = this->declare_parameter<std::string>("frame_id", "odom");

    positionPublisher = this->create_publisher<geometry_msgs::msg::PointStamped>("queryPosition", 1);
    projectionPublisher = this->create_publisher<geometry_msgs::msg::PointStamped>("projectedQueryPosition", 1);
    convexTerrainPublisher = this->create_publisher<geometry_msgs::msg::PolygonStamped>("convex_terrain", 1);
    terrainSubscriber = this->create_subscription<convex_plane_decomposition_msgs::msg::PlanarTerrain>(
        "/convex_plane_decomposition_ros/planar_terrain", 1, std::bind(&ConvexApproximationDemoNode::callback, this, std::placeholders::_1));

    timer = this->create_wall_timer(std::chrono::seconds(1), std::bind(&ConvexApproximationDemoNode::update, this));
  }

 private:
  void callback(const convex_plane_decomposition_msgs::msg::PlanarTerrain::ConstSharedPtr msg) {
    auto newTerrain = std::make_unique<PlanarTerrain>(fromMessage(*msg));

    std::lock_guard<std::mutex> lock(terrainMutex);
    planarTerrainPtr.swap(newTerrain);
  }

  void update() {
    std::lock_guard<std::mutex> lock(terrainMutex);
    if (planarTerrainPtr) {
      const auto& map = planarTerrainPtr->gridMap;

      double maxX = map.getPosition().x() + map.getLength().x() * 0.5;
      double minX = map.getPosition().x() - map.getLength().x() * 0.5;
      double maxY = map.getPosition().y() + map.getLength().y() * 0.5;
      double minY = map.getPosition().y() - map.getLength().y() * 0.5;

      Eigen::Vector3d query{randomFloat(minX, maxX), randomFloat(minY, maxY), randomFloat(0.0, 1.0)};
      auto penaltyFunction = [](const Eigen::Vector3d& projectedPoint) { return 0.0; };

      const auto projection = getBestPlanarRegionAtPositionInWorld(query, planarTerrainPtr->planarRegions, penaltyFunction);

      int numberOfVertices = 16;
      double growthFactor = 1.05;
      const auto convexRegion = growConvexPolygonInsideShape(
          projection.regionPtr->boundaryWithInset.boundary, projection.positionInTerrainFrame, numberOfVertices, growthFactor);

      std_msgs::msg::Header header;
      header.stamp = rclcpp::Time(planarTerrainPtr->gridMap.getTimestamp());
      header.frame_id = frameId;

      auto convexRegionMsg = to3dRosPolygon(convexRegion, projection.regionPtr->transformPlaneToWorld, header);

      convexTerrainPublisher->publish(convexRegionMsg);
      positionPublisher->publish(toMarker(query, header));
      projectionPublisher->publish(toMarker(projection.positionInWorld, header));
    }
  }

  geometry_msgs::msg::PointStamped toMarker(const Eigen::Vector3d& position, const std_msgs::msg::Header& header) {
    geometry_msgs::msg::PointStamped sphere;
    sphere.header = header;
    sphere.point.x = position.x();
    sphere.point.y = position.y();
    sphere.point.z = position.z();
    return sphere;
  }

  float randomFloat(float a, float b) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
  }

  std::string frameId;
  std::mutex terrainMutex;
  std::unique_ptr<PlanarTerrain> planarTerrainPtr;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr positionPublisher;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr projectionPublisher;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr convexTerrainPublisher;
  rclcpp::Subscription<convex_plane_decomposition_msgs::msg::PlanarTerrain>::SharedPtr terrainSubscriber;
  rclcpp::TimerBase::SharedPtr timer;
};

}  // namespace convex_plane_decomposition

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<convex_plane_decomposition::ConvexApproximationDemoNode>());
  rclcpp::shutdown();
  return 0;
}