//
// Created by rgrandia on 10.06.20.
//

#include "convex_plane_decomposition_ros/ParameterLoading.h"
#include <cmath>

namespace convex_plane_decomposition {
namespace {
template <typename T>
bool loadParameter(const rclcpp::Node* node, const std::string& prefix, const std::string& param, T& value) {
  std::string full_param_name = prefix + param;
  // In ROS2 we cannot easily check if a parameter exists without declaring it or having "allow_undeclared_parameters" true.
  // We assume the node allows undeclared parameters or we try to declare it.
  // However, since we are passing a const Node*, we can only call const methods like get_parameter if it was declared.
  // But we need to declare. So we need a non-const Node* or we assume they are declared.
  // The header helper used const Node*.
  // But declare_parameter represents a state change.
  // Let's cast away constness locally or change header signature.
  // I changed header to take `const rclcpp::Node*`?
  // Checking my previous write for ParameterLoading.h:
  // "const rclcpp::Node* node"
  // This is problematic for `declare_parameter`.
  // I should const_cast or better, change the header to `rclcpp::Node*`.
  // But wait, `ConvexPlaneExtractionROS` has `loadParameters(const rclcpp::Node* node)`.
  // I should change that too.
  
  // For now I will use const_cast because logically loading parameters might be considered "reading configuration" 
  // but technically it changes node state (declaring params).
  rclcpp::Node* mutable_node = const_cast<rclcpp::Node*>(node);
  
  try {
    if (!mutable_node->has_parameter(full_param_name)) {
        mutable_node->declare_parameter<T>(full_param_name, value);
    }
    value = mutable_node->get_parameter(full_param_name).get_value<T>();
    return true;
  } catch (const rclcpp::exceptions::InvalidParameterTypeException& e) {
     RCLCPP_ERROR_STREAM(node->get_logger(), "Parameter " << full_param_name << " has wrong type: " << e.what());
     return false;
  } catch (const std::exception& e) {
     RCLCPP_WARN_STREAM(node->get_logger(), "Could not read parameter `" << full_param_name << "`. Using default/initial value: " << value << ". Error: " << e.what());
     // We return true if we fell back to default that we passed in (via declare_parameter default),
     // OR false if we really wanted to warn.
     // In ROS1 code: returns false if not found.
     // But declare_parameter ensures it exists (if we provided default).
     // If the value passed in `value` is the default we want, then declare_parameter uses it.
     return true; 
  }
}
}

PreprocessingParameters loadPreprocessingParameters(const rclcpp::Node* node, const std::string& prefix) {
  PreprocessingParameters preprocessingParameters;
  loadParameter(node, prefix, "resolution", preprocessingParameters.resolution);
  loadParameter(node, prefix, "kernelSize", preprocessingParameters.kernelSize);
  loadParameter(node, prefix, "numberOfRepeats", preprocessingParameters.numberOfRepeats);
  return preprocessingParameters;
}

contour_extraction::ContourExtractionParameters loadContourExtractionParameters(const rclcpp::Node* node,
                                                                                const std::string& prefix) {
  contour_extraction::ContourExtractionParameters contourParams;
  loadParameter(node, prefix, "marginSize", contourParams.marginSize);
  return contourParams;
}

ransac_plane_extractor::RansacPlaneExtractorParameters loadRansacPlaneExtractorParameters(const rclcpp::Node* node,
                                                                                          const std::string& prefix) {
  ransac_plane_extractor::RansacPlaneExtractorParameters ransacParams;
  loadParameter(node, prefix, "probability", ransacParams.probability);
  loadParameter(node, prefix, "min_points", ransacParams.min_points);
  loadParameter(node, prefix, "epsilon", ransacParams.epsilon);
  loadParameter(node, prefix, "cluster_epsilon", ransacParams.cluster_epsilon);
  loadParameter(node, prefix, "normal_threshold", ransacParams.normal_threshold);
  return ransacParams;
}

sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters loadSlidingWindowPlaneExtractorParameters(
    const rclcpp::Node* node, const std::string& prefix) {
  sliding_window_plane_extractor::SlidingWindowPlaneExtractorParameters swParams;
  loadParameter(node, prefix, "kernel_size", swParams.kernel_size);
  loadParameter(node, prefix, "planarity_opening_filter", swParams.planarity_opening_filter);
  
  double inclination_temp = 0.0;
  if (loadParameter(node, prefix, "plane_inclination_threshold_degrees", inclination_temp)) {
     swParams.plane_inclination_threshold = std::cos(inclination_temp * M_PI / 180.0);
  }
  
  double local_inclination_temp = 0.0;
  if (loadParameter(node, prefix, "local_plane_inclination_threshold_degrees", local_inclination_temp)) {
     swParams.local_plane_inclination_threshold = std::cos(local_inclination_temp * M_PI / 180.0);
  }

  loadParameter(node, prefix, "plane_patch_error_threshold", swParams.plane_patch_error_threshold);
  loadParameter(node, prefix, "min_number_points_per_label", swParams.min_number_points_per_label);
  loadParameter(node, prefix, "connectivity", swParams.connectivity);
  loadParameter(node, prefix, "include_ransac_refinement", swParams.include_ransac_refinement);
  loadParameter(node, prefix, "global_plane_fit_distance_error_threshold", swParams.global_plane_fit_distance_error_threshold);
  loadParameter(node, prefix, "global_plane_fit_angle_error_threshold_degrees",
                swParams.global_plane_fit_angle_error_threshold_degrees);
  return swParams;
}

PostprocessingParameters loadPostprocessingParameters(const rclcpp::Node* node, const std::string& prefix) {
  PostprocessingParameters postprocessingParameters;
  loadParameter(node, prefix, "extracted_planes_height_offset", postprocessingParameters.extracted_planes_height_offset);
  loadParameter(node, prefix, "nonplanar_height_offset", postprocessingParameters.nonplanar_height_offset);
  loadParameter(node, prefix, "nonplanar_horizontal_offset", postprocessingParameters.nonplanar_horizontal_offset);
  loadParameter(node, prefix, "smoothing_dilation_size", postprocessingParameters.smoothing_dilation_size);
  loadParameter(node, prefix, "smoothing_box_kernel_size", postprocessingParameters.smoothing_box_kernel_size);
  loadParameter(node, prefix, "smoothing_gauss_kernel_size", postprocessingParameters.smoothing_gauss_kernel_size);
  return postprocessingParameters;
}

}  // namespace convex_plane_decomposition
