//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"

// Pybind
#include <pybind11/eigen.h>

// PCL
#include <pcl/common/projection_matrix.h>

// ROS
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <utility>
#include <vector>
#include <string>

namespace elevation_mapping_cupy {

ElevationMappingWrapper::ElevationMappingWrapper() {}

void ElevationMappingWrapper::initialize(rclcpp::Node* node) {
  // Add the elevation_mapping_cupy path to  py::module::import("elevation_mapping_cupy");
  // However, we check if we can import it.
  
  py::gil_scoped_acquire acquire;
  try {
      auto elevation_mapping = py::module::import("elevation_mapping_cupy.elevation_mapping");
      auto parameter = py::module::import("elevation_mapping_cupy.parameter");
      param_ = parameter.attr("Parameter")();
      setParameters(node);
      map_ = elevation_mapping.attr("ElevationMap")(param_);
  } catch (py::error_already_set& e) {
      RCLCPP_ERROR(node->get_logger(), "Failed to import elevation_mapping_cupy python module: %s", e.what());
      throw;
  }
}

/**
 *  Load ros parameters into Parameter class.
 *  Search for the same name within the name space.
 */
void ElevationMappingWrapper::setParameters(rclcpp::Node* node) {
  // Get all parameters names and types.
  py::list paramNames = param_.attr("get_names")();
  py::list paramTypes = param_.attr("get_types")();
  py::gil_scoped_acquire acquire;

  for (int i = 0; i < paramNames.size(); i++) {
    std::string type = py::cast<std::string>(paramTypes[i]);
    std::string name = py::cast<std::string>(paramNames[i]);

    if (!node->has_parameter(name)) {
      continue;
    }
    
    if (type == "float") {
        double param_val;
        rclcpp::Parameter param = node->get_parameter(name);
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
            param_val = static_cast<double>(param.as_int());
        } else {
            param_val = param.as_double();
        }
        param_.attr("set_value")(name, param_val);
    } else if (type == "str") {
        std::string param_val = node->get_parameter(name).as_string();
        param_.attr("set_value")(name, param_val);
    } else if (type == "bool") {
        bool param_val = node->get_parameter(name).as_bool();
        param_.attr("set_value")(name, param_val);
    } else if (type == "int") {
        int param_val = node->get_parameter(name).as_int();
        param_.attr("set_value")(name, param_val);
    }
  }

  // Subscribers
  py::dict sub_dict;
  // list 'subscribers' parameter prefix
  auto params = node->list_parameters({"subscribers"}, 10);
  std::set<std::string> subscriber_keys;
  for (const auto& name : params.names) {
     auto pos = name.find('.');
     if (pos != std::string::npos && name.substr(0, pos) == "subscribers") {
         auto sub_pos = name.find('.', pos + 1);
         if (sub_pos != std::string::npos) {
             subscriber_keys.insert(name.substr(pos + 1, sub_pos - (pos + 1)));
         } else {
             subscriber_keys.insert(name.substr(pos + 1));
         }
     }
  }
  
  for (const auto& key : subscriber_keys) {
    std::string prefix = "subscribers." + key;
    if (!sub_dict.contains(key.c_str())) {
        sub_dict[key.c_str()] = py::dict();
    }
    
    std::vector<std::string> sub_attributes = {"topic_name", "data_type", "camera_info_topic_name", "channel_info_topic_name"};
    for (const auto& attr : sub_attributes) {
      std::string p_name = prefix + "." + attr;
      if (node->has_parameter(p_name)) {
          sub_dict[key.c_str()][attr.c_str()] = node->get_parameter(p_name).as_string();
      }
    }
    
    if (node->has_parameter(prefix + ".channels")) {
        sub_dict[key.c_str()]["channels"] = node->get_parameter(prefix + ".channels").as_string_array();
    }
  }
  param_.attr("subscriber_cfg") = sub_dict;

  // Pointcloud channel fusion
  py::dict pointcloud_channel_fusion_dict;
  // Handle pointcloud fusion channels from parameters
  auto fusion_params = node->list_parameters({"pointcloud_channel_fusions"}, 10);
  for (const auto& name : fusion_params.names) {
      if (name.find("pointcloud_channel_fusions.") == 0) {
          std::string key = name.substr(27); // length of prefix
          std::string val = node->get_parameter(name).as_string();
      }
  }
  param_.attr("pointcloud_channel_fusions") = pointcloud_channel_fusion_dict;

  // Image channel fusion
  py::dict image_channel_fusion_dict;
   auto img_fusion_params = node->list_parameters({"image_channel_fusions"}, 10);
  for (const auto& name : img_fusion_params.names) {
      if (name.find("image_channel_fusions.") == 0) {
          std::string key = name.substr(22);
          std::string val = node->get_parameter(name).as_string();
          image_channel_fusion_dict[key.c_str()] = val;
      }
  }
  param_.attr("image_channel_fusions") = image_channel_fusion_dict;


  param_.attr("update")();
  resolution_ = py::cast<float>(param_.attr("get_value")("resolution"));
  map_length_ = py::cast<float>(param_.attr("get_value")("true_map_length"));
  map_n_ = py::cast<int>(param_.attr("get_value")("true_cell_n"));
  
  try {
    enable_normal_ = node->declare_parameter<bool>("enable_normal", false);
  } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException&) {
    enable_normal_ = node->get_parameter("enable_normal").as_bool();
  }
   try {
    enable_normal_color_ = node->declare_parameter<bool>("enable_normal_color", false);
  } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException&) {
    enable_normal_color_ = node->get_parameter("enable_normal_color").as_bool();
  }
}

void ElevationMappingWrapper::input(const RowMatrixXd& points, const std::vector<std::string>& channels, const RowMatrixXd& R,
                                    const Eigen::VectorXd& t, const double positionNoise, const double orientationNoise) {
  py::gil_scoped_acquire acquire;
  map_.attr("input_pointcloud")(Eigen::Ref<const RowMatrixXd>(points), channels, Eigen::Ref<const RowMatrixXd>(R),
                     Eigen::Ref<const Eigen::VectorXd>(t), positionNoise, orientationNoise);
}

void ElevationMappingWrapper::input_image(const std::vector<ColMatrixXf>& multichannel_image, const std::vector<std::string>& channels, const RowMatrixXd& R,
                                          const Eigen::VectorXd& t, const RowMatrixXd& cameraMatrix, const Eigen::VectorXd& D, const std::string distortion_model, int height, int width) {
  py::gil_scoped_acquire acquire;
  map_.attr("input_image")(multichannel_image, channels, Eigen::Ref<const RowMatrixXd>(R), Eigen::Ref<const Eigen::VectorXd>(t),
                           Eigen::Ref<const RowMatrixXd>(cameraMatrix), Eigen::Ref<const Eigen::VectorXd>(D), distortion_model, height, width);
}

void ElevationMappingWrapper::move_to(const Eigen::VectorXd& p, const RowMatrixXd& R) {
  py::gil_scoped_acquire acquire;
  map_.attr("move_to")(Eigen::Ref<const Eigen::VectorXd>(p), Eigen::Ref<const RowMatrixXd>(R));
}

void ElevationMappingWrapper::clear() {
  py::gil_scoped_acquire acquire;
  map_.attr("clear")();
}

double ElevationMappingWrapper::get_additive_mean_error() {
  py::gil_scoped_acquire acquire;
  return map_.attr("get_additive_mean_error")().cast<double>();
}

bool ElevationMappingWrapper::exists_layer(const std::string& layerName) {
  py::gil_scoped_acquire acquire;
  return py::cast<bool>(map_.attr("exists_layer")(layerName));
}

void ElevationMappingWrapper::get_layer_data(const std::string& layerName, RowMatrixXf& map) {
  py::gil_scoped_acquire acquire;
  map = RowMatrixXf(map_n_, map_n_);
  map_.attr("get_map_with_name_ref")(layerName, Eigen::Ref<RowMatrixXf>(map));
}

void ElevationMappingWrapper::get_grid_map(grid_map::GridMap& gridMap, const std::vector<std::string>& requestLayerNames) {
  std::vector<std::string> basicLayerNames;
  std::vector<std::string> layerNames = requestLayerNames;
  std::vector<int> selection;
  for (const auto& layerName : layerNames) {
    if (layerName == "elevation") {
      basicLayerNames.push_back("elevation");
    }
  }

  RowMatrixXd pos(1, 3);
  py::gil_scoped_acquire acquire;
  map_.attr("get_position")(Eigen::Ref<RowMatrixXd>(pos));
  grid_map::Position position(pos(0, 0), pos(0, 1));
  grid_map::Length length(map_length_, map_length_);
  gridMap.setGeometry(length, resolution_, position);
  std::vector<Eigen::MatrixXf> maps;

  for (const auto& layerName : layerNames) {
    bool exists = map_.attr("exists_layer")(layerName).cast<bool>();
    if (exists) {
      RowMatrixXf map(map_n_, map_n_);
      map_.attr("get_map_with_name_ref")(layerName, Eigen::Ref<RowMatrixXf>(map));
      gridMap.add(layerName, map);
    }
  }
  if (enable_normal_color_) {
    RowMatrixXf normal_x(map_n_, map_n_);
    RowMatrixXf normal_y(map_n_, map_n_);
    RowMatrixXf normal_z(map_n_, map_n_);
    map_.attr("get_normal_ref")(Eigen::Ref<RowMatrixXf>(normal_x), Eigen::Ref<RowMatrixXf>(normal_y), Eigen::Ref<RowMatrixXf>(normal_z));
    gridMap.add("normal_x", normal_x);
    gridMap.add("normal_y", normal_y);
    gridMap.add("normal_z", normal_z);
  }
  gridMap.setBasicLayers(basicLayerNames);
  if (enable_normal_color_) {
    addNormalColorLayer(gridMap);
  }
}

void ElevationMappingWrapper::get_polygon_traversability(std::vector<Eigen::Vector2d>& polygon, Eigen::Vector3d& result,
                                                         std::vector<Eigen::Vector2d>& untraversable_polygon) {
  if (polygon.size() < 3) {
    return;
  }
  RowMatrixXf polygon_m(polygon.size(), 2);
  int i = 0;
  for (auto& p : polygon) {
    polygon_m(i, 0) = p.x();
    polygon_m(i, 1) = p.y();
    i++;
  }
  py::gil_scoped_acquire acquire;
  const int untraversable_polygon_num =
      map_.attr("get_polygon_traversability")(Eigen::Ref<const RowMatrixXf>(polygon_m), Eigen::Ref<Eigen::VectorXd>(result)).cast<int>();

  untraversable_polygon.clear();
  if (untraversable_polygon_num > 0) {
    RowMatrixXf untraversable_polygon_m(untraversable_polygon_num, 2);
    map_.attr("get_untraversable_polygon")(Eigen::Ref<RowMatrixXf>(untraversable_polygon_m));
    for (int j = 0; j < untraversable_polygon_num; j++) {
      Eigen::Vector2d p;
      p.x() = untraversable_polygon_m(j, 0);
      p.y() = untraversable_polygon_m(j, 1);
      untraversable_polygon.push_back(p);
    }
  }
}

void ElevationMappingWrapper::initializeWithPoints(std::vector<Eigen::Vector3d>& points, std::string method) {
  RowMatrixXd points_m(points.size(), 3);
  int i = 0;
  for (auto& p : points) {
    points_m(i, 0) = p.x();
    points_m(i, 1) = p.y();
    points_m(i, 2) = p.z();
    i++;
  }
  py::gil_scoped_acquire acquire;
  map_.attr("initialize_map")(Eigen::Ref<const RowMatrixXd>(points_m), method);
}

void ElevationMappingWrapper::addNormalColorLayer(grid_map::GridMap& map) {
  const auto& normalX = map["normal_x"];
  const auto& normalY = map["normal_y"];
  const auto& normalZ = map["normal_z"];

  map.add("color");
  auto& color = map["color"];

  // X: -1 to +1 : Red: 0 to 255
  // Y: -1 to +1 : Green: 0 to 255
  // Z:  0 to  1 : Blue: 128 to 255

  // For each cell in map.
  for (size_t i = 0; i < color.size(); ++i) {
    const Eigen::Vector3f colorVector((normalX(i) + 1.0) / 2.0, (normalY(i) + 1.0) / 2.0, (normalZ(i)));
    Eigen::Vector3i intColorVector = (colorVector * 255.0).cast<int>();
    grid_map::colorVectorToValue(intColorVector, color(i));
  }
}

void ElevationMappingWrapper::update_variance() {
  py::gil_scoped_acquire acquire;
  map_.attr("update_variance")();
}

void ElevationMappingWrapper::update_time() {
  py::gil_scoped_acquire acquire;
  map_.attr("update_time")();
}

}  // namespace elevation_mapping_cupy
