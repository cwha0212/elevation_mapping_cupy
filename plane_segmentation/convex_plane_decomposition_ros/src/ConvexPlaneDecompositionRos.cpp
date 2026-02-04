#include "convex_plane_decomposition_ros/ConvexPlaneDecompositionRos.h"

#include <grid_map_core/GridMap.hpp>
#include <grid_map_cv/GridMapCvProcessing.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

#include <convex_plane_decomposition/PlaneDecompositionPipeline.h>
#include <convex_plane_decomposition_msgs/msg/planar_terrain.hpp>

#include "convex_plane_decomposition_ros/MessageConversion.h"
#include "convex_plane_decomposition_ros/ParameterLoading.h"
#include "convex_plane_decomposition_ros/RosVisualizations.h"

namespace convex_plane_decomposition {

ConvexPlaneExtractionROS::ConvexPlaneExtractionROS(rclcpp::Node* node) : node_(node) {
  tfBuffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
  tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_);
  
  bool parametersLoaded = loadParameters(node_);
  
  if (parametersLoaded) {
    using std::placeholders::_1;
    elevationMapSubscriber_ = node_->create_subscription<grid_map_msgs::msg::GridMap>(elevationMapTopic_, 1, std::bind(&ConvexPlaneExtractionROS::callback, this, _1));
    filteredmapPublisher_ = node_->create_publisher<grid_map_msgs::msg::GridMap>("filtered_map", 1);
    boundaryPublisher_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("boundaries", 1);
    insetPublisher_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("insets", 1);
    regionPublisher_ = node_->create_publisher<convex_plane_decomposition_msgs::msg::PlanarTerrain>("planar_terrain", 1);
  }
}

ConvexPlaneExtractionROS::~ConvexPlaneExtractionROS() {
  if (callbackTimer_.getNumTimedIntervals() > 0 && planeDecompositionPipeline_ != nullptr) {
    std::stringstream infoStream;
    infoStream << "\n########################################################################\n";
    infoStream << "The benchmarking is computed over " << callbackTimer_.getNumTimedIntervals() << " iterations. \n";
    infoStream << "PlaneExtraction Benchmarking    : Average time [ms], Max time [ms]\n";
    auto printLine = [](std::string name, const Timer& timer) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2);
      ss << "\t" << name << "\t: " << std::setw(17) << timer.getAverageInMilliseconds() << ", " << std::setw(13)
         << timer.getMaxIntervalInMilliseconds() << "\n";
      return ss.str();
    };
    infoStream << printLine("Pre-process        ", planeDecompositionPipeline_->getPrepocessTimer());
    infoStream << printLine("Sliding window     ", planeDecompositionPipeline_->getSlidingWindowTimer());
    infoStream << printLine("Contour extraction ", planeDecompositionPipeline_->getContourExtractionTimer());
    infoStream << printLine("Post-process       ", planeDecompositionPipeline_->getPostprocessTimer());
    infoStream << printLine("Total callback     ", callbackTimer_);
    std::cerr << infoStream.str() << std::endl;
  }
}

bool ConvexPlaneExtractionROS::loadParameters(const rclcpp::Node* node) {
  // Use ParameterLoading helper style or just get_parameter?
  // Since we already updated ParameterLoading helpers to use node, we can use them for complex structs.
  // For simple params we use getParameter/declare parameter here.
  // But wait, node is const here (from signature in header).
  // We need to cast it or change signature if we want to declare.
  // In ParameterLoading.cpp I casted it.
  
  auto loadParam = [&](const std::string& param, auto& var) -> bool {
      rclcpp::Node* mutable_node = const_cast<rclcpp::Node*>(node);
      try {
          if (!mutable_node->has_parameter(param)) {
             mutable_node->declare_parameter(param, var); // declare with current value as default? Or generic default?
             // Since var is passed by reference, it might have a value.
             // But actually we want to read it.
             // declare_parameter returns the value.
             var = mutable_node->get_parameter(param).get_value<std::remove_reference_t<decltype(var)>>();
          } else {
             var = mutable_node->get_parameter(param).get_value<std::remove_reference_t<decltype(var)>>();
          }
          return true;
      } catch (const std::exception& e) {
          RCLCPP_ERROR_STREAM(node->get_logger(), "Could not read parameter `" << param << "`: " << e.what());
          return false;
      }
  };
  // Wait, declare_parameter template arg is required if not deducible? 
  // And `string` vs `const char*` for param name.
  
  // Let's implement simpler:
  rclcpp::Node* mutable_node = const_cast<rclcpp::Node*>(node);

  elevationMapTopic_ = mutable_node->get_parameter("elevation_topic").as_string();
  
  if (elevationMapTopic_.empty()) {
      RCLCPP_ERROR(node->get_logger(), "[ConvexPlaneExtractionROS] parameter `elevation_topic` is empty or missing.");
      return false;
  }

  targetFrameId_ = mutable_node->get_parameter("target_frame_id").as_string();
  elevationLayer_ = mutable_node->get_parameter("height_layer").as_string();
  subMapWidth_ = mutable_node->get_parameter("submap.width").as_double();
  subMapLength_ = mutable_node->get_parameter("submap.length").as_double();
  publishToController_ = mutable_node->get_parameter("publish_to_controller").as_bool();
  
  if (targetFrameId_.empty() || elevationLayer_.empty()) {
       RCLCPP_ERROR(node->get_logger(), "[ConvexPlaneExtractionROS] Missing required parameters.");
       return false;
  }

  PlaneDecompositionPipeline::Config config;
  config.preprocessingParameters = loadPreprocessingParameters(node, "preprocessing.");
  config.contourExtractionParameters = loadContourExtractionParameters(node, "contour_extraction.");
  config.ransacPlaneExtractorParameters = loadRansacPlaneExtractorParameters(node, "ransac_plane_refinement.");
  config.slidingWindowPlaneExtractorParameters = loadSlidingWindowPlaneExtractorParameters(node, "sliding_window_plane_extractor.");
  config.postprocessingParameters = loadPostprocessingParameters(node, "postprocessing.");

  planeDecompositionPipeline_ = std::make_unique<PlaneDecompositionPipeline>(config);

  return true;
}

void ConvexPlaneExtractionROS::callback(const grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
  callbackTimer_.startTimer();

  // Convert message to map.
  grid_map::GridMap messageMap;
  std::vector<std::string> layers{elevationLayer_};
  grid_map::GridMapRosConverter::fromMessage(*message, messageMap, layers, false, false);
  if (!messageMap.exists(elevationLayer_)) {
      RCLCPP_WARN(node_->get_logger(), "[ConvexPlaneExtractionROS] map does not contain the layer %s", elevationLayer_.c_str());
      callbackTimer_.endTimer();
      return;
  }
  // containsFiniteValue? grid_map API
  if (messageMap[elevationLayer_].hasNaN()) { 
     // hasNaN checks if any NaN. But we want to know if it has ANY finite value?
     // containsFiniteValue implementation in source was using a helper? 
     // "containsFiniteValue(messageMap.get(elevationLayer_))"
     // Checking if it's empty or all NaNs. 
     // Assuming grid_map::Matrix is Eigen matrix.
     if (messageMap[elevationLayer_].size() == 0 || !messageMap[elevationLayer_].allFinite()) { // allFinite checks NO Infs or NaNs.
        // Wait, regular elevation map has many NaNs (unknowns).
        // The original code `containsFiniteValue` probably checked if there is AT LEAST ONE finite value.
        // I will implement a quick check.
        if ((messageMap[elevationLayer_].array().isFinite()).count() == 0) {
            RCLCPP_WARN(node_->get_logger(), "[ConvexPlaneExtractionROS] map does not contain any values");
            callbackTimer_.endTimer();
            return;
        }
     }
  }

  // Transform map if necessary
  if (targetFrameId_ != messageMap.getFrameId()) {
    std::string errorMsg;
    rclcpp::Time timeStamp = rclcpp::Time(0);  // Use Time(0) to get the latest transform.
    using namespace std::chrono_literals;
    if (tfBuffer_->canTransform(targetFrameId_, messageMap.getFrameId(), timeStamp, 1s, &errorMsg)) { // 1.0s timeout
      const auto transform = getTransformToTargetFrame(messageMap.getFrameId(), timeStamp);
      messageMap = messageMap.getTransformedMap(transform, elevationLayer_, targetFrameId_);
    } else {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "[ConvexPlaneExtractionROS] " << errorMsg);
      callbackTimer_.endTimer();
      return;
    }
  }

  // Extract submap
  bool success;
  const grid_map::Position submapPosition = [&]() {
    // The map center might be between cells. Taking the submap there can result in changing submap dimensions.
    // project map center to an index and index to center s.t. we get the location of a cell.
    grid_map::Index centerIndex;
    grid_map::Position centerPosition;
    try {
        messageMap.getIndex(messageMap.getPosition(), centerIndex);
        messageMap.getPosition(centerIndex, centerPosition);
    } catch (...) {
        return messageMap.getPosition();
    }
    return centerPosition;
  }();
  
  grid_map::GridMap elevationMap = messageMap.getSubmap(submapPosition, Eigen::Array2d(subMapLength_, subMapWidth_), success);
  if (!success) {
    RCLCPP_WARN(node_->get_logger(), "[ConvexPlaneExtractionROS] Could not extract submap");
    callbackTimer_.endTimer();
    return;
  }
  const grid_map::Matrix elevationRaw = elevationMap.get(elevationLayer_);

  // Run pipeline.
  planeDecompositionPipeline_->update(std::move(elevationMap), elevationLayer_);
  auto& planarTerrain = planeDecompositionPipeline_->getPlanarTerrain();

  // Publish terrain
  if (publishToController_) {
    regionPublisher_->publish(toMessage(planarTerrain));
  }

  // --- Visualize in Rviz --- Not published to the controller
  // Add raw map
  planarTerrain.gridMap.add("elevation_raw", elevationRaw);

  // Add segmentation
  planarTerrain.gridMap.add("segmentation");
  planeDecompositionPipeline_->getSegmentation(planarTerrain.gridMap.get("segmentation"));

  auto outputMessage = grid_map::GridMapRosConverter::toMessage(planarTerrain.gridMap);
  if (outputMessage) {
      filteredmapPublisher_->publish(*outputMessage);
  }

  const double lineWidth = 0.005;  // [m] RViz marker size
  boundaryPublisher_->publish(convertBoundariesToRosMarkers(planarTerrain.planarRegions, planarTerrain.gridMap.getFrameId(),
                                                           planarTerrain.gridMap.getTimestamp(), lineWidth));
  insetPublisher_->publish(convertInsetsToRosMarkers(planarTerrain.planarRegions, planarTerrain.gridMap.getFrameId(),
                                                    planarTerrain.gridMap.getTimestamp(), lineWidth));

  callbackTimer_.endTimer();
}

Eigen::Isometry3d ConvexPlaneExtractionROS::getTransformToTargetFrame(const std::string& sourceFrame, const rclcpp::Time& time) {
  geometry_msgs::msg::TransformStamped transformStamped;
  try {
    transformStamped = tfBuffer_->lookupTransform(targetFrameId_, sourceFrame, time, rclcpp::Duration::from_seconds(1.0));
  } catch (tf2::TransformException& ex) {
    RCLCPP_ERROR(node_->get_logger(), "[ConvexPlaneExtractionROS] %s", ex.what());
    return Eigen::Isometry3d::Identity();
  }

  Eigen::Isometry3d transformation;

  // Extract translation.
  transformation.translation().x() = transformStamped.transform.translation.x;
  transformation.translation().y() = transformStamped.transform.translation.y;
  transformation.translation().z() = transformStamped.transform.translation.z;

  // Extract rotation.
  Eigen::Quaterniond rotationQuaternion(transformStamped.transform.rotation.w, transformStamped.transform.rotation.x,
                                        transformStamped.transform.rotation.y, transformStamped.transform.rotation.z);
  transformation.linear() = rotationQuaternion.toRotationMatrix();
  return transformation;
}

}  // namespace convex_plane_decomposition
