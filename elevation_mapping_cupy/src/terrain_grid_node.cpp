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
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

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
    // Graded cost instead of a wall at the threshold (2026-10-20): a cell scores 0 (free) at safety >= graded_free,
    // 100 (lethal) at safety <= graded_lethal and linear in between, so grass edges and flickering noise become
    // expensive rather than impassable; the camera veto still caps at veto_cost. Needs trinary_costmap off
    // downstream. graded_lethal < 0 keeps the old cut.
    graded_free_ = declare_parameter<double>("graded_free", -1.0);
    graded_lethal_ = declare_parameter<double>("graded_lethal", -1.0);
    // The robot is standing on the cells under its own footprint: they are free by definition, whatever the
    // never-re-measured map says (stale leg returns made the robot's own trail lethal). The polygon is
    // navi_lidar's footprint in the body frame (flat x0 y0 x1 y1 ...), grown by body_margin, placed with TF
    // map -> body_frame at the grid stamp.
    body_polygon_ = declare_parameter<std::vector<double>>("body_polygon", std::vector<double>{});
    body_frame_ = declare_parameter<std::string>("body_frame", "lidar_frame");
    body_margin_ = declare_parameter<double>("body_margin", 0.10);
    body_tail_ = declare_parameter<double>("body_tail", 0.40);
    if (body_polygon_.size() >= 6 && body_polygon_.size() % 2 == 0) {
      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
      RCLCPP_INFO(get_logger(), "body polygon: %zu vertices in %s (+%.2f m margin, %.2f m tail) forced free",
        body_polygon_.size() / 2, body_frame_.c_str(), body_margin_, body_tail_);
    }

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
    // Only the layers this node reads are converted. The terrain message can
    // carry half a dozen layers and the full conversion copied every one of
    // them into Eigen matrices to look at two.
    const bool soft = threshold_ >= 0.0 && veto_cost_ >= 0 && veto_cost_ < 100 &&
      base_layer_ != layer_;
    std::vector<std::string> wanted{layer_};
    if (soft) {
      wanted.push_back(base_layer_);
    }
    grid_map::GridMap map;
    if (!grid_map::GridMapRosConverter::fromMessage(*msg, map, wanted, false, false)) {
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
      // Cut in place: 1 where the cell passes, 0 where it fails, NaN where
      // nothing was judged; the converter then maps 1 -> 0 (free) and
      // 0 -> 100 (lethal). A cell the geometry passes and only the camera
      // fails sits in between. The base layer is looked up once, not per cell.
      grid_map::Matrix & cut = map.get(layer_);
      const grid_map::Matrix * base =
        (soft && map.exists(base_layer_)) ? &map.get(base_layer_) : nullptr;
      const float veto_value = 1.0f - static_cast<float>(veto_cost_) / 100.0f;
      const float thr = static_cast<float>(threshold_);
      const bool graded = graded_lethal_ >= 0.0 && graded_free_ > graded_lethal_;
      const float g_free = static_cast<float>(graded_free_), g_leth = static_cast<float>(graded_lethal_);
      for (int i = 0; i < cut.size(); ++i) {
        const float v = cut(i);
        if (!std::isfinite(v)) {
          continue;
        }
        if (graded) {
          float score = (v - g_leth) / (g_free - g_leth);        // 1 = free, 0 = lethal
          score = std::min(1.0f, std::max(0.0f, score));
          if (base != nullptr && std::isfinite((*base)(i)) && (*base)(i) >= g_free && score < veto_value) {
            score = veto_value;                                 // camera-only fail: never worse than the veto cost
          }
          cut(i) = score;
        } else if (v >= thr) {
          cut(i) = 1.0f;
        } else if (base != nullptr && std::isfinite((*base)(i)) && (*base)(i) >= thr) {
          cut(i) = veto_value;
        } else {
          cut(i) = 0.0f;
        }
      }
      clearBody(map, cut);
      grid_map::GridMapRosConverter::toOccupancyGrid(map, layer_, 1.0, 0.0, grid);
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

  // Force the cells under the body polygon (grown by body_margin, extended body_tail behind) to free.
  void clearBody(grid_map::GridMap & map, grid_map::Matrix & cut)
  {
    if (!tf_buffer_) {
      return;
    }
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform(map.getFrameId(), body_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000, "body polygon: no TF %s -> %s (%s)",
        map.getFrameId().c_str(), body_frame_.c_str(), e.what());
      return;
    }
    const auto & q = tf.transform.rotation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double tx = tf.transform.translation.x, ty = tf.transform.translation.y;
    // grow the polygon: margin on every side, tail behind (-x): done on the body-frame vertices by sign
    std::vector<double> px, py;
    for (size_t k = 0; k + 1 < body_polygon_.size(); k += 2) {
      double x = body_polygon_[k], y = body_polygon_[k + 1];
      x += (x >= 0 ? body_margin_ : -(body_margin_ + body_tail_));
      y += (y >= 0 ? body_margin_ : -body_margin_);
      px.push_back(tx + c * x - s * y);
      py.push_back(ty + s * x + c * y);
    }
    const double x0 = *std::min_element(px.begin(), px.end()), x1 = *std::max_element(px.begin(), px.end());
    const double y0 = *std::min_element(py.begin(), py.end()), y1 = *std::max_element(py.begin(), py.end());
    size_t n = 0;
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
      grid_map::Position pos;
      map.getPosition(*it, pos);
      if (pos.x() < x0 || pos.x() > x1 || pos.y() < y0 || pos.y() > y1) {
        continue;
      }
      bool inside = false;                                   // even-odd, navi_lidar insideFootprint()
      for (size_t i = 0, j = px.size() - 1; i < px.size(); j = i++) {
        if ((py[i] > pos.y()) != (py[j] > pos.y()) &&
          pos.x() < (px[j] - px[i]) * (pos.y() - py[i]) / (py[j] - py[i]) + px[i])
        {
          inside = !inside;
        }
      }
      if (inside) {
        cut((*it)(0), (*it)(1)) = 1.0f;
        ++n;
      }
    }
    if (n && published_ % 60 == 0) {
      RCLCPP_INFO(get_logger(), "body polygon: %zu cells under the footprint forced free", n);
    }
  }

  std::string input_topic_, output_topic_, layer_, base_layer_, body_frame_;
  double data_min_{1.0}, data_max_{0.0}, threshold_{0.4}, graded_free_{-1.0}, graded_lethal_{-1.0};
  double body_margin_{0.1}, body_tail_{0.4};
  std::vector<double> body_polygon_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
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
