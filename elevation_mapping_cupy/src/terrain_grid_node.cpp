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
#include <cmath>
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
    // Measured on the real bag: the safety layer is bimodal, with 68% of
    // judged cells sitting at exactly 0 (walls, vegetation) and the rest
    // spread thinly. Scaled linearly into 0..100 that hands Nav2 a map that
    // is two thirds lethal with a smear of intermediate cost in between, and
    // StaticLayer's trinary reading then keeps only the cells at exactly 100.
    // So the layer is cut at the same threshold the bearing fan uses, and the
    // costmap receives three values: free, lethal, and unknown. A negative
    // threshold restores the linear scaling.
    threshold_ = declare_parameter<double>("threshold", 0.4);
    // The camera's veto as a cost, not a wall. Where the geometry passes a
    // cell and only the semantic layer fails it, the cell is published at
    // veto_cost (0..99) instead of 100: the planner and MPPI then prefer the
    // path the camera likes but can still cross low grass beside it, which
    // the quadruped walks over without trouble. Measured 2026-09-28 on the
    // real bag: with the veto lethal, a 1 m stepping-stone path narrowed to
    // under the robot's width and the robot's own route was unreachable in
    // most frames. -1 keeps the veto lethal. Needs base_layer to exist and
    // trinary_costmap off downstream, or the value collapses to free.
    base_layer_ = declare_parameter<std::string>("base_layer", "drivability");
    veto_cost_ = declare_parameter<int>("veto_cost", 70);

    publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(output_topic_, 1);
    subscription_ = create_subscription<grid_map_msgs::msg::GridMap>(
      input_topic_, 1,
      std::bind(&TerrainGridNode::onGridMap, this, std::placeholders::_1));

    if (threshold_ >= 0.0) {
      RCLCPP_INFO(
        get_logger(), "'%s' layer '%s' -> '%s' (below %.2f = lethal, else free, NaN = unknown; "
        "camera-only fail -> cost %d over '%s')",
        input_topic_.c_str(), layer_.c_str(), output_topic_.c_str(), threshold_, veto_cost_,
        base_layer_.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(), "'%s' layer '%s' -> '%s' (%.2f = free, %.2f = lethal)",
        input_topic_.c_str(), layer_.c_str(), output_topic_.c_str(), data_min_, data_max_);
    }
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
    if (threshold_ >= 0.0) {
      // 1 where the cell passes, 0 where it fails, NaN where nothing was
      // judged; the converter then maps 1 -> 0 (free) and 0 -> 100 (lethal).
      // A cell the geometry passes and only the camera fails sits in between.
      const grid_map::Matrix & src = map[layer_];
      const bool soft = veto_cost_ >= 0 && veto_cost_ < 100 && map.exists(base_layer_) &&
        base_layer_ != layer_;
      const float veto_value = 1.0f - static_cast<float>(veto_cost_) / 100.0f;
      grid_map::Matrix cut = src;
      const float thr = static_cast<float>(threshold_);
      for (int i = 0; i < cut.size(); ++i) {
        const float v = src(i);
        if (!std::isfinite(v)) {
          continue;
        }
        if (v >= thr) {
          cut(i) = 1.0f;
        } else if (soft && std::isfinite(map[base_layer_](i)) && map[base_layer_](i) >= thr) {
          cut(i) = veto_value;
        } else {
          cut(i) = 0.0f;
        }
      }
      map.add("cut", cut);
      grid_map::GridMapRosConverter::toOccupancyGrid(map, "cut", 1.0, 0.0, grid);
    } else {
      grid_map::GridMapRosConverter::toOccupancyGrid(map, layer_, data_min_, data_max_, grid);
    }
    publisher_->publish(grid);
    if (++published_ % 30 == 1) {
      RCLCPP_INFO(
        get_logger(), "published %zu grids (%u x %u at %.3f m)",
        published_, grid.info.width, grid.info.height, grid.info.resolution);
    }
  }

  std::string input_topic_, output_topic_, layer_, base_layer_;
  double data_min_{1.0}, data_max_{0.0}, threshold_{0.4};
  int veto_cost_{70};
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
