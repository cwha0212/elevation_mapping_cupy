// Publish a terrain layer straight to Nav2, through grid_map's own converter.
//
// The existing path to a costmap goes through the bearing fan: 720 rays over
// a map of 40,000 judged cells, each ray keeping only the first thing it
// meets, then an octomap to accumulate what survives. That shape is right for
// a GLOBAL map -- it enforces line of sight, so nothing is asserted about
// ground behind an obstacle, and the octree outlives the rolling window.
//
// It is the wrong shape for a LOCAL costmap. A local costmap is rebuilt from
// scratch each cycle and covers exactly the window the elevation map already
// holds, and every cell in that window has been judged. Ray-casting it throws
// away everything except 720 first hits for no gain: there is nothing stale
// to guard against in a map that is rebuilt, and the elevation map only
// contains cells that were measured, so the shadow problem the fan exists to
// solve does not arise here.
//
// So this hands the layer over whole. grid_map_ros does the conversion, which
// is a solved problem in a library we already depend on for the message type.
#include <memory>
#include <string>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>

class TerrainGridNode : public rclcpp::Node
{
public:
  TerrainGridNode()
  : Node("terrain_grid_node")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/elevation_mapping_node/elevation_map_terrain");
    output_topic_ = declare_parameter<std::string>("output_topic", "/terrain/local_grid");
    layer_ = declare_parameter<std::string>("layer", "safety");
    // The layer runs 1 (safe) to 0 (forbidden) and an OccupancyGrid runs 0
    // (free) to 100 (lethal), so the conversion is given the ends the right
    // way round and comes out inverted, which is what a costmap wants.
    data_min_ = declare_parameter<double>("data_min", 1.0);
    data_max_ = declare_parameter<double>("data_max", 0.0);

    publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(output_topic_, 1);
    subscription_ = create_subscription<grid_map_msgs::msg::GridMap>(
      input_topic_, 1,
      std::bind(&TerrainGridNode::onGridMap, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "'%s' layer '%s' -> '%s' (%.2f = free, %.2f = lethal)",
      input_topic_.c_str(), layer_.c_str(), output_topic_.c_str(), data_min_, data_max_);
  }

private:
  void onGridMap(const grid_map_msgs::msg::GridMap::SharedPtr msg)
  {
    grid_map::GridMap map;
    if (!grid_map::GridMapRosConverter::fromMessage(*msg, map)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "could not read the incoming grid map");
      return;
    }
    if (!map.exists(layer_)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "no layer '%s' in the map", layer_.c_str());
      return;
    }
    nav_msgs::msg::OccupancyGrid grid;
    // Cells the chain never judged come out as -1, which is the one thing a
    // costmap must not confuse with open ground.
    grid_map::GridMapRosConverter::toOccupancyGrid(map, layer_, data_min_, data_max_, grid);
    publisher_->publish(grid);
    if (++published_ % 30 == 1) {
      RCLCPP_INFO(
        get_logger(), "published %zu grids (%u x %u at %.3f m)",
        published_, grid.info.width, grid.info.height, grid.info.resolution);
    }
  }

  std::string input_topic_, output_topic_, layer_;
  double data_min_{1.0}, data_max_{0.0};
  size_t published_{0};
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TerrainGridNode>());
  rclcpp::shutdown();
  return 0;
}
