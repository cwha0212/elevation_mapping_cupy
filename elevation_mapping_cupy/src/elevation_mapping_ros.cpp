//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include "elevation_mapping_cupy/elevation_mapping_ros.hpp"

// Pybind
#include <pybind11/eigen.h>

// ROS
#include <geometry_msgs/msg/point32.hpp>
#include <std_msgs/msg/empty.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// PCL
#include <pcl/common/projection_matrix.h>

// OpenCV
#include <cv_bridge/cv_bridge.hpp>

namespace elevation_mapping_cupy {

ElevationMappingNode::ElevationMappingNode(const rclcpp::NodeOptions& options)
    : Node("elevation_mapping", options),
      lowpassPosition_(0, 0, 0),
      lowpassOrientation_(0, 0, 0, 1),
      positionError_(0),
      orientationError_(0),
      positionAlpha_(0.1),
      orientationAlpha_(0.1),
      enablePointCloudPublishing_(false),
      isGridmapUpdated_(false) {
  
  // TF2
  tfBuffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  transformListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_);
  tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  std::string pose_topic;
  double recordableFps, updateVarianceFps, timeInterval, updatePoseFps, updateGridMapFps, publishStatisticsFps;
  bool enablePointCloudPublishing(false);

  // Read parameters
  if (!this->has_parameter("initialize_frame_id")) {
    initialize_frame_id_ = this->declare_parameter<std::vector<std::string>>("initialize_frame_id", {"base"});
  } else {
    initialize_frame_id_ = this->get_parameter("initialize_frame_id").as_string_array();
  }
  if (!this->has_parameter("initialize_tf_offset")) {
    initialize_tf_offset_ = this->declare_parameter<std::vector<double>>("initialize_tf_offset", {0.0});
  } else {
    initialize_tf_offset_ = this->get_parameter("initialize_tf_offset").as_double_array();
  }
  pose_topic = this->has_parameter("pose_topic") ? this->get_parameter("pose_topic").as_string() : this->declare_parameter<std::string>("pose_topic", "pose");
  mapFrameId_ = this->has_parameter("map_frame") ? this->get_parameter("map_frame").as_string() : this->declare_parameter<std::string>("map_frame", "map");
  baseFrameId_ = this->has_parameter("base_frame") ? this->get_parameter("base_frame").as_string() : this->declare_parameter<std::string>("base_frame", "base");
  correctedMapFrameId_ = this->has_parameter("corrected_map_frame") ? this->get_parameter("corrected_map_frame").as_string() : this->declare_parameter<std::string>("corrected_map_frame", "corrected_map");
  if (!this->has_parameter("initialize_method")) {
    initializeMethod_ = this->declare_parameter<std::string>("initialize_method", "cubic");
  } else {
    initializeMethod_ = this->get_parameter("initialize_method").as_string();
  }
  positionAlpha_ = this->has_parameter("position_lowpass_alpha") ? this->get_parameter("position_lowpass_alpha").as_double() : this->declare_parameter<double>("position_lowpass_alpha", 0.2);
  orientationAlpha_ = this->has_parameter("orientation_lowpass_alpha") ? this->get_parameter("orientation_lowpass_alpha").as_double() : this->declare_parameter<double>("orientation_lowpass_alpha", 0.2);
  recordableFps = this->has_parameter("recordable_fps") ? this->get_parameter("recordable_fps").as_double() : this->declare_parameter<double>("recordable_fps", 3.0);
  updateVarianceFps = this->has_parameter("update_variance_fps") ? this->get_parameter("update_variance_fps").as_double() : this->declare_parameter<double>("update_variance_fps", 1.0);
  timeInterval = this->has_parameter("time_interval") ? this->get_parameter("time_interval").as_double() : this->declare_parameter<double>("time_interval", 0.1);
  updatePoseFps = this->has_parameter("update_pose_fps") ? this->get_parameter("update_pose_fps").as_double() : this->declare_parameter<double>("update_pose_fps", 10.0);
   if (!this->has_parameter("initialize_tf_grid_size")) {
    initializeTfGridSize_ = this->declare_parameter<double>("initialize_tf_grid_size", 0.5);
  } else {
    initializeTfGridSize_ = this->get_parameter("initialize_tf_grid_size").as_double();
  }
  updateGridMapFps = this->has_parameter("map_acquire_fps") ? this->get_parameter("map_acquire_fps").as_double() : this->declare_parameter<double>("map_acquire_fps", 5.0);
  publishStatisticsFps = this->has_parameter("publish_statistics_fps") ? this->get_parameter("publish_statistics_fps").as_double() : this->declare_parameter<double>("publish_statistics_fps", 1.0);
  enablePointCloudPublishing = this->has_parameter("enable_pointcloud_publishing") ? this->get_parameter("enable_pointcloud_publishing").as_bool() : this->declare_parameter<bool>("enable_pointcloud_publishing", false);
  enableNormalArrowPublishing_ = this->has_parameter("enable_normal_arrow_publishing") ? this->get_parameter("enable_normal_arrow_publishing").as_bool() : this->declare_parameter<bool>("enable_normal_arrow_publishing", false);
  enableDriftCorrectedTFPublishing_ = this->has_parameter("enable_drift_corrected_TF_publishing") ? this->get_parameter("enable_drift_corrected_TF_publishing").as_bool() : this->declare_parameter<bool>("enable_drift_corrected_TF_publishing", false);
  if (!this->has_parameter("use_initializer_at_start")) {
    useInitializerAtStart_ = this->declare_parameter<bool>("use_initializer_at_start", false);
  } else {
    useInitializerAtStart_ = this->get_parameter("use_initializer_at_start").as_bool();
  }
  if (!this->has_parameter("always_clear_with_initializer")) {
    alwaysClearWithInitializer_ = this->declare_parameter<bool>("always_clear_with_initializer", false);
  } else {
    alwaysClearWithInitializer_ = this->get_parameter("always_clear_with_initializer").as_bool();
  }

  enablePointCloudPublishing_ = enablePointCloudPublishing;

  // Handle Subscribers
  std::vector<std::string> subscriber_keys;
  auto sub_params = this->list_parameters({"subscribers"}, 10);
  std::set<std::string> keys_set;
  for (const auto& name : sub_params.names) {
    auto pos = name.find("subscribers.");
    if (pos != std::string::npos) {
      auto rest = name.substr(pos + 12);
      auto dot_pos = rest.find('.');
      if (dot_pos != std::string::npos) {
        keys_set.insert(rest.substr(0, dot_pos));
      } else {
        keys_set.insert(rest);
      }
    }
  }
  subscriber_keys.assign(keys_set.begin(), keys_set.end());
  if (subscriber_keys.empty()) {
    RCLCPP_FATAL(this->get_logger(), "There aren't any subscribers to be configured, the elevation mapping cannot be configured. Exit");
    throw std::runtime_error("No subscribers configured");
  }

  for (const auto& key : subscriber_keys) {
    std::string prefix = "subscribers." + key;
    std::string type;
    try {
      type = this->get_parameter(prefix + ".data_type").as_string();
    } catch (...) {
      RCLCPP_WARN(this->get_logger(), "Subscriber key '%s' has no data_type parameter (or other error). Skipping.", key.c_str());
      continue;
    }
    std::string topic_name = this->get_parameter(prefix + ".topic_name").as_string();

    if (type == "pointcloud") {
      channels_[key].push_back("x");
      channels_[key].push_back("y");
      channels_[key].push_back("z");
      auto sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          topic_name, 10,
          [this, key](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
            this->pointcloudCallback(msg, key);
          });
      pointcloudSubs_.push_back(sub);
      RCLCPP_INFO_STREAM(this->get_logger(), "Subscribed to PointCloud2 topic: " << topic_name);
    } else if (type == "image") {
      std::string camera_info_topic_name;
      try {
          camera_info_topic_name = this->declare_parameter<std::string>(prefix + ".camera_info_topic_name", "");
      } catch (...) {
          camera_info_topic_name = this->get_parameter(prefix + ".camera_info_topic_name").as_string();
      }

      std::string transport_hint = "compressed";
      if (topic_name.find("compressed") == std::string::npos) {
        transport_hint = "raw";
      } else {
         // rudimentary stripping of compressed
         size_t ind = topic_name.find("compressed");
         if (ind != std::string::npos && ind > 0) {
             topic_name = topic_name.substr(0, ind-1); 
         }
      }

      ImageSubscriberPtr image_sub = std::make_shared<ImageSubscriber>(this, topic_name, transport_hint);
      imageSubs_.push_back(image_sub);

      CameraInfoSubscriberPtr cam_info_sub = std::make_shared<CameraInfoSubscriber>(this, camera_info_topic_name);
      cameraInfoSubs_.push_back(cam_info_sub);

      std::string channel_info_topic;
      bool has_channel_info = false;
      try {
          channel_info_topic = this->declare_parameter<std::string>(prefix + ".channel_info_topic_name", "");
          has_channel_info = !channel_info_topic.empty();
      } catch (...) {
          channel_info_topic = this->get_parameter(prefix + ".channel_info_topic_name").as_string();
          has_channel_info = !channel_info_topic.empty();
      }

      if (has_channel_info) {
        ChannelInfoSubscriberPtr channel_info_sub = std::make_shared<ChannelInfoSubscriber>(this, channel_info_topic);
        channelInfoSubs_.push_back(channel_info_sub);
        CameraChannelSyncPtr sync = std::make_shared<CameraChannelSync>(CameraChannelPolicy(10), *image_sub, *cam_info_sub, *channel_info_sub);
        sync->registerCallback(std::bind(&ElevationMappingNode::imageChannelCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        cameraChannelSyncs_.push_back(sync);
        RCLCPP_INFO_STREAM(this->get_logger(), "Subscribed to Image topic: " << topic_name << ", Camera info topic: " << camera_info_topic_name << ", Channel info topic: " << channel_info_topic);
      } else {
        std::vector<std::string> channels_list;
        if (this->has_parameter(prefix + ".channels")) {
          channels_list = this->get_parameter(prefix + ".channels").as_string_array();
        } else {
          channels_list = this->declare_parameter<std::vector<std::string>>(prefix + ".channels", std::vector<std::string>());
        }
        
        if (!channels_list.empty()) {
            channels_[key] = channels_list;
        } else {
            channels_[key].push_back("rgb");
        }
        
        RCLCPP_INFO_STREAM(this->get_logger(), "Subscribed to Image topic: " << topic_name);
        CameraSyncPtr sync = std::make_shared<CameraSync>(CameraPolicy(10), *image_sub, *cam_info_sub);
        sync->registerCallback(std::bind(&ElevationMappingNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2, key));
        cameraSyncs_.push_back(sync);
      }

    } else {
      RCLCPP_WARN_STREAM(this->get_logger(), "Subscriber data_type [" << type << "] Not valid. Supported types: pointcloud, image");
      continue;
    }
  }

  // Initialize Pybind Wrapper
  map_.initialize(this);

  auto pub_params = this->list_parameters({"publishers"}, 10);
  std::set<std::string> publisher_keys;
  for (const auto& name : pub_params.names) {
    auto pos = name.find("publishers.");
    if (pos != std::string::npos) {
      auto rest = name.substr(pos + 11);
      auto dot_pos = rest.find('.');
      if (dot_pos != std::string::npos) {
        publisher_keys.insert(rest.substr(0, dot_pos));
      } else {
        publisher_keys.insert(rest);
      }
    }
  }
  if (publisher_keys.empty()) {
    RCLCPP_FATAL(this->get_logger(), "There aren't any publishers to be configured, the elevation mapping cannot be configured. Exit");
    throw std::runtime_error("No publishers configured");
  }

  for (const auto& key : publisher_keys) {
    std::string prefix = "publishers." + key;
    std::string topic_name = key; 

    std::vector<std::string> layers_list = this->get_parameter(prefix + ".layers").as_string_array();
    std::vector<std::string> basic_layers_list = this->get_parameter(prefix + ".basic_layers").as_string_array();
    double fps = this->get_parameter(prefix + ".fps").as_double();

    if (fps > updateGridMapFps) {
      RCLCPP_WARN(this->get_logger(),
          "[ElevationMappingCupy] fps for topic %s is larger than map_acquire_fps (%f > %f). The topic data will be only updated at %f fps.",
          topic_name.c_str(), fps, updateGridMapFps, updateGridMapFps);
    }
    
     // Make publishers
    auto pub = this->create_publisher<grid_map_msgs::msg::GridMap>(topic_name, 1);
    mapPubs_.push_back(pub);

    // Register map layers
    map_layers_.push_back(layers_list);
    map_basic_layers_.push_back(basic_layers_list);

    // Register map fps
    map_fps_.push_back(fps);
    map_fps_unique_.insert(fps);
  }

  setupMapPublishers();

  pointPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("elevation_map_points", 1);
  alivePub_ = this->create_publisher<std_msgs::msg::Empty>("alive", 1);
  normalPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("normal", 1);
  statisticsPub_ = this->create_publisher<elevation_map_msgs::msg::Statistics>("statistics", 1);

  gridMap_.setFrameId(mapFrameId_);
  
  servicesCallbackGroup_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  
  rawSubmapService_ = this->create_service<grid_map_msgs::srv::GetGridMap>("get_raw_submap", std::bind(&ElevationMappingNode::getSubmap, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, servicesCallbackGroup_);
  clearMapService_ = this->create_service<std_srvs::srv::Empty>("clear_map", std::bind(&ElevationMappingNode::clearMap, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, servicesCallbackGroup_);
  initializeMapService_ = this->create_service<elevation_map_msgs::srv::Initialize>("initialize", std::bind(&ElevationMappingNode::initializeMap, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, servicesCallbackGroup_);
  clearMapWithInitializerService_ = this->create_service<std_srvs::srv::Empty>("clear_map_with_initializer", std::bind(&ElevationMappingNode::clearMapWithInitializer, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, servicesCallbackGroup_);
  setPublishPointService_ = this->create_service<std_srvs::srv::SetBool>("set_publish_points", std::bind(&ElevationMappingNode::setPublishPoint, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, servicesCallbackGroup_);
  checkSafetyService_ = this->create_service<elevation_map_msgs::srv::CheckSafety>("check_safety", std::bind(&ElevationMappingNode::checkSafety, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, servicesCallbackGroup_);

  if (updateVarianceFps > 0) {
    updateVarianceTimer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / (updateVarianceFps + 0.00001)), std::bind(&ElevationMappingNode::updateVariance, this));
  }
  if (timeInterval > 0) {
    updateTimeTimer_ = this->create_wall_timer(std::chrono::duration<double>(timeInterval), std::bind(&ElevationMappingNode::updateTime, this));
  }
  if (updatePoseFps > 0) {
     updatePoseTimer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / (updatePoseFps + 0.00001)), std::bind(&ElevationMappingNode::updatePose, this));
  }
  if (updateGridMapFps > 0) {
     updateGridMapTimer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / (updateGridMapFps + 0.00001)), std::bind(&ElevationMappingNode::updateGridMap, this));
  }
  if (publishStatisticsFps > 0) {
    publishStatisticsTimer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / (publishStatisticsFps + 0.00001)), std::bind(&ElevationMappingNode::publishStatistics, this));
  }
  lastStatisticsPublishedTime_ = this->now();
  RCLCPP_INFO(this->get_logger(), "[ElevationMappingCupy] finish initialization");
  
}

// Setup map publishers
void ElevationMappingNode::setupMapPublishers() {
  float max_fps = -1;
  for (auto fps : map_fps_unique_) {
    std::vector<int> indices;
    if (fps >= max_fps) {
      max_fps = fps;
      map_layers_all_.clear();
    }
    for (int i = 0; i < map_fps_.size(); i++) {
      if (map_fps_[i] == fps) {
        indices.push_back(i);
        if (fps >= max_fps) {
          for (const auto layer : map_layers_[i]) {
            map_layers_all_.insert(layer);
          }
        }
      }
    }
    auto cb = [this, indices]() {
      for (int i : indices) {
        publishMapOfIndex(i);
      }
    };
    mapTimers_.push_back(this->create_wall_timer(std::chrono::duration<double>(1.0 / (fps + 0.00001)), cb));
  }
}

void ElevationMappingNode::publishMapOfIndex(int index) {
  if (!isGridmapUpdated_) {
    return;
  }
  grid_map_msgs::msg::GridMap msg;
  std::vector<std::string> layers;

  {
    std::lock_guard<std::mutex> lock(mapMutex_);
    for (const auto& layer : map_layers_[index]) {
      const bool is_layer_in_all = map_layers_all_.find(layer) != map_layers_all_.end();
      if (is_layer_in_all && gridMap_.exists(layer)) {
        layers.push_back(layer);
      } else if (map_.exists_layer(layer)) {
        ElevationMappingWrapper::RowMatrixXf map_data;
        map_.get_layer_data(layer, map_data);
        gridMap_.add(layer, map_data);
        layers.push_back(layer);
      }
    }
    if (layers.empty()) {
      return;
    }
    std::unique_ptr<grid_map_msgs::msg::GridMap> msg_ptr;
    msg_ptr = grid_map::GridMapRosConverter::toMessage(gridMap_, layers);
    msg = *msg_ptr;
  }

  msg.basic_layers = map_basic_layers_[index];
  mapPubs_[index]->publish(msg);
}

void ElevationMappingNode::pointcloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud, const std::string& key) {
  //  get channels
  auto fields = cloud->fields;
  std::vector<std::string> channels;

  for (int it = 0; it < fields.size(); it++) {
    auto& field = fields[it];
    channels.push_back(field.name);
  }
  inputPointCloud(*cloud, channels);
  pointCloudProcessCounter_++;
}

void ElevationMappingNode::inputPointCloud(const sensor_msgs::msg::PointCloud2& cloud,
                                          const std::vector<std::string>& channels) {
  auto start = this->now();
  auto* pcl_pc = new pcl::PCLPointCloud2;
  pcl::PCLPointCloud2ConstPtr cloudPtr(pcl_pc);
  pcl_conversions::toPCL(cloud, *pcl_pc);

  auto fields = cloud.fields;
  uint array_dim = channels.size();

  RowMatrixXd points = RowMatrixXd(pcl_pc->width * pcl_pc->height, array_dim);

  for (unsigned int i = 0; i < pcl_pc->width * pcl_pc->height; ++i) {
    for (unsigned int j = 0; j < channels.size(); ++j) {
      float temp;
      uint point_idx = i * pcl_pc->point_step + pcl_pc->fields[j].offset;
      memcpy(&temp, &pcl_pc->data[point_idx], sizeof(float));
      points(i, j) = static_cast<double>(temp);
    }
  }
  
  std::string sensorFrameId = cloud.header.frame_id;
  auto timeStamp = cloud.header.stamp;
  Eigen::Affine3d transformationSensorToMap;
  try {
    // tf2 timeout is duration
    geometry_msgs::msg::TransformStamped transformTf = tfBuffer_->lookupTransform(mapFrameId_, sensorFrameId, timeStamp, rclcpp::Duration::from_seconds(1.0));
    transformationSensorToMap = tf2::transformToEigen(transformTf);
  } catch (tf2::TransformException& ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  double positionError{0.0};
  double orientationError{0.0};
  {
    std::lock_guard<std::mutex> lock(errorMutex_);
    positionError = positionError_;
    orientationError = orientationError_;
  }
  map_.input(points, channels, transformationSensorToMap.rotation(), transformationSensorToMap.translation(), positionError,
             orientationError);

  if (enableDriftCorrectedTFPublishing_) {
    publishMapToOdom(map_.get_additive_mean_error());
  }


  RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "ElevationMap processed a point cloud (%i points) in %f sec.", static_cast<int>(points.size()),
                     (this->now() - start).seconds());
}

void ElevationMappingNode::inputImage(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                                      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg,
                                      const std::vector<std::string>& channels) {
  cv::Mat image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;

  if (image_msg->encoding == "bgr8") {
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  } else if (image_msg->encoding == "bgra8") {
    cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
  }

  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> cameraMatrix(&camera_info_msg->k[0]);

  Eigen::VectorXd distortionCoeffs;
  if (!camera_info_msg->d.empty()) {
    distortionCoeffs = Eigen::Map<const Eigen::VectorXd>(camera_info_msg->d.data(), camera_info_msg->d.size());
  } else {
    RCLCPP_WARN(this->get_logger(), "Distortion coefficients are empty.");
    distortionCoeffs = Eigen::VectorXd::Zero(5);
  }

  std::string distortion_model = camera_info_msg->distortion_model;
  
  std::string sensorFrameId = image_msg->header.frame_id;
  auto timeStamp = image_msg->header.stamp;
  Eigen::Affine3d transformationMapToSensor;
  try {
     geometry_msgs::msg::TransformStamped transformTf = tfBuffer_->lookupTransform(sensorFrameId, mapFrameId_, timeStamp, rclcpp::Duration::from_seconds(1.0));
     transformationMapToSensor = tf2::transformToEigen(transformTf);
  } catch (tf2::TransformException& ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  std::vector<cv::Mat> image_split;
  std::vector<ColMatrixXf> multichannel_image;
  cv::split(image, image_split);
  for (auto img : image_split) {
    ColMatrixXf eigen_img;
    cv::cv2eigen(img, eigen_img);
    multichannel_image.push_back(eigen_img);
  }

  int total_channels = 0;
  for (const auto& channel : channels) {
    if (channel == "rgb") {
      total_channels += 3;
    } else {
      total_channels += 1;
    }
  }
  if (total_channels != multichannel_image.size()) {
    RCLCPP_ERROR(this->get_logger(), "Mismatch in the size of multichannel_image (%d), channels (%d). Please check the input.", (int)multichannel_image.size(), (int)channels.size());
    return;
  }

  map_.input_image(multichannel_image, channels, transformationMapToSensor.rotation(), transformationMapToSensor.translation(), cameraMatrix, 
                   distortionCoeffs, distortion_model, image.rows, image.cols);
}

void ElevationMappingNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                                         const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg,
                                         const std::string& key) {
  auto start = this->now();
  inputImage(image_msg, camera_info_msg, channels_[key]);
  RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "ElevationMap processed an image in %f sec.", (this->now() - start).seconds());
}

void ElevationMappingNode::imageChannelCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                                         const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg,
                                         const elevation_map_msgs::msg::ChannelInfo::ConstSharedPtr& channel_info_msg) {
  auto start = this->now();
  std::vector<std::string> channels;
  channels = channel_info_msg->channels;
  inputImage(image_msg, camera_info_msg, channels);
  RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "ElevationMap processed an image in %f sec.", (this->now() - start).seconds());
}

void ElevationMappingNode::updatePose() {
  const auto& timeStamp = this->now();
  Eigen::Affine3d transformationBaseToMap;
  geometry_msgs::msg::TransformStamped transformTf;
  try {
    transformTf = tfBuffer_->lookupTransform(mapFrameId_, baseFrameId_, timeStamp, rclcpp::Duration::from_seconds(1.0));
    transformationBaseToMap = tf2::transformToEigen(transformTf);
  } catch (tf2::TransformException& ex) {
     RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  Eigen::Vector3d position(transformTf.transform.translation.x, transformTf.transform.translation.y, transformTf.transform.translation.z);
  map_.move_to(position, transformationBaseToMap.rotation().transpose());
  Eigen::Vector3d position3 = position;
  Eigen::Vector4d orientation(transformTf.transform.rotation.x, transformTf.transform.rotation.y, transformTf.transform.rotation.z, transformTf.transform.rotation.w);
  lowpassPosition_ = positionAlpha_ * position3 + (1 - positionAlpha_) * lowpassPosition_;
  lowpassOrientation_ = orientationAlpha_ * orientation + (1 - orientationAlpha_) * lowpassOrientation_;
  {
    std::lock_guard<std::mutex> lock(errorMutex_);
    positionError_ = (position3 - lowpassPosition_).norm();
    orientationError_ = (orientation - lowpassOrientation_).norm();
  }

  if (useInitializerAtStart_) {
    RCLCPP_INFO(this->get_logger(), "Clearing map with initializer.");
    initializeWithTF();
    useInitializerAtStart_ = false;
  }
}

void ElevationMappingNode::publishAsPointCloud(const grid_map::GridMap& map) const {
  sensor_msgs::msg::PointCloud2 msg;
  grid_map::GridMapRosConverter::toPointCloud(map, {"elevation"}, "elevation", msg);
  pointPub_->publish(msg);
}

void ElevationMappingNode::getSubmap(const std::shared_ptr<grid_map_msgs::srv::GetGridMap::Request> request, std::shared_ptr<grid_map_msgs::srv::GetGridMap::Response> response) {
  std::string requestedFrameId = request->frame_id;
  Eigen::Isometry3d transformationOdomToMap;
  grid_map::Position requestedSubmapPosition(request->position_x, request->position_y);
  if (requestedFrameId != mapFrameId_) {
    const auto& timeStamp = this->now();
    try {
      geometry_msgs::msg::TransformStamped transformTf = tfBuffer_->lookupTransform(requestedFrameId, mapFrameId_, timeStamp, rclcpp::Duration::from_seconds(1.0));
       transformationOdomToMap = tf2::transformToEigen(transformTf);
    } catch (tf2::TransformException& ex) {
      RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
      return;
    }
    Eigen::Vector3d p(request->position_x, request->position_y, 0);
    Eigen::Vector3d mapP = transformationOdomToMap.inverse() * p;
    requestedSubmapPosition.x() = mapP.x();
    requestedSubmapPosition.y() = mapP.y();
  }
  grid_map::Length requestedSubmapLength(request->length_x, request->length_y);
  RCLCPP_DEBUG(this->get_logger(), "Elevation submap request: Position x=%f, y=%f, Length x=%f, y=%f.", requestedSubmapPosition.x(), requestedSubmapPosition.y(),
            requestedSubmapLength(0), requestedSubmapLength(1));

  bool isSuccess;
  grid_map::Index index;
  grid_map::GridMap subMap;
  {
    std::lock_guard<std::mutex> lock(mapMutex_);
    subMap = gridMap_.getSubmap(requestedSubmapPosition, requestedSubmapLength, isSuccess);
  }
  
  if (requestedFrameId != mapFrameId_) {
    subMap = subMap.getTransformedMap(transformationOdomToMap, "elevation", requestedFrameId);
  }

  std::unique_ptr<grid_map_msgs::msg::GridMap> msg_ptr;
  if (request->layers.empty()) {
     msg_ptr = grid_map::GridMapRosConverter::toMessage(subMap);
  } else {
    std::vector<std::string> layers;
    for (const auto& layer : request->layers) {
      layers.push_back(layer);
    }
     msg_ptr = grid_map::GridMapRosConverter::toMessage(subMap, layers);
  }
  if (msg_ptr) {
      response->map = *msg_ptr;
  }
}

void ElevationMappingNode::clearMap(const std::shared_ptr<std_srvs::srv::Empty::Request> request, std::shared_ptr<std_srvs::srv::Empty::Response> response) {
  (void)request;
  (void)response;
  RCLCPP_INFO(this->get_logger(), "Clearing map.");
  map_.clear();
  if (alwaysClearWithInitializer_) {
    initializeWithTF();
  }
}

void ElevationMappingNode::clearMapWithInitializer(const std::shared_ptr<std_srvs::srv::Empty::Request> request, std::shared_ptr<std_srvs::srv::Empty::Response> response) {
  (void)request;
  (void)response;
  RCLCPP_INFO(this->get_logger(), "Clearing map with initializer.");
  map_.clear();
  initializeWithTF();
}

void ElevationMappingNode::initializeWithTF() {
  std::vector<Eigen::Vector3d> points;
  const auto& timeStamp = this->now();
  int i = 0;
  Eigen::Vector3d p;
  for (const auto& frame_id : initialize_frame_id_) {
    // Get tf from map frame to tf frame
    Eigen::Affine3d transformationBaseToMap;
     geometry_msgs::msg::TransformStamped transformTf;
    try {
      transformTf = tfBuffer_->lookupTransform(mapFrameId_, frame_id, timeStamp, rclcpp::Duration::from_seconds(1.0));
      transformationBaseToMap = tf2::transformToEigen(transformTf);
    } catch (tf2::TransformException& ex) {
      RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
      return;
    }
    p = transformationBaseToMap.translation();
     if(i < initialize_tf_offset_.size())
        p.z() += initialize_tf_offset_[i];
    points.push_back(p);
    i++;
  }
  if (!points.empty() && points.size() < 3) {
    points.emplace_back(p + Eigen::Vector3d(initializeTfGridSize_, initializeTfGridSize_, 0));
    points.emplace_back(p + Eigen::Vector3d(-initializeTfGridSize_, initializeTfGridSize_, 0));
    points.emplace_back(p + Eigen::Vector3d(initializeTfGridSize_, -initializeTfGridSize_, 0));
    points.emplace_back(p + Eigen::Vector3d(-initializeTfGridSize_, -initializeTfGridSize_, 0));
  }
  RCLCPP_INFO_STREAM(this->get_logger(), "Initializing map with points using " << initializeMethod_);
  map_.initializeWithPoints(points, initializeMethod_);
}

void ElevationMappingNode::checkSafety(const std::shared_ptr<elevation_map_msgs::srv::CheckSafety::Request> request,
                                       std::shared_ptr<elevation_map_msgs::srv::CheckSafety::Response> response) {
  for (const auto& polygonstamped : request->polygons) {
    if (polygonstamped.polygon.points.empty()) {
      continue;
    }
    std::vector<Eigen::Vector2d> polygon;
    std::vector<Eigen::Vector2d> untraversable_polygon;
    Eigen::Vector3d result;
    result.setZero();
    const auto& polygonFrameId = polygonstamped.header.frame_id;
    const auto& timeStamp = polygonstamped.header.stamp;
    double polygon_z = polygonstamped.polygon.points[0].z;

    if (mapFrameId_ != polygonFrameId) {
      Eigen::Affine3d transformationBaseToMap;
       geometry_msgs::msg::TransformStamped transformTf;
      try {
        transformTf = tfBuffer_->lookupTransform(mapFrameId_, polygonFrameId, timeStamp, rclcpp::Duration::from_seconds(1.0));
        transformationBaseToMap = tf2::transformToEigen(transformTf);
      } catch (tf2::TransformException& ex) {
        RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
        return;
      }
      for (const auto& p : polygonstamped.polygon.points) {
        const auto& pvector = Eigen::Vector3d(p.x, p.y, p.z);
        const auto transformed_p = transformationBaseToMap * pvector;
        polygon.emplace_back(Eigen::Vector2d(transformed_p.x(), transformed_p.y()));
      }
    } else {
      for (const auto& p : polygonstamped.polygon.points) {
        polygon.emplace_back(Eigen::Vector2d(p.x, p.y));
      }
    }

    map_.get_polygon_traversability(polygon, result, untraversable_polygon);

    geometry_msgs::msg::PolygonStamped untraversable_polygonstamped;
    untraversable_polygonstamped.header.stamp = this->now();
    untraversable_polygonstamped.header.frame_id = mapFrameId_;
    for (const auto& p : untraversable_polygon) {
      geometry_msgs::msg::Point32 point;
      point.x = static_cast<float>(p.x());
      point.y = static_cast<float>(p.y());
      point.z = static_cast<float>(polygon_z);
      untraversable_polygonstamped.polygon.points.push_back(point);
    }
    
    response->is_safe.push_back(bool(result[0] > 0.5));
    response->traversability.push_back(result[1]);
    response->untraversable_polygons.push_back(untraversable_polygonstamped);
  }
}

void ElevationMappingNode::setPublishPoint(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  enablePointCloudPublishing_ = request->data;
  response->success = true;
}

void ElevationMappingNode::updateVariance() {
  map_.update_variance();
}

void ElevationMappingNode::updateTime() {
  map_.update_time();
}

void ElevationMappingNode::publishStatistics() {
  rclcpp::Time now = this->now();
  double dt = (now - lastStatisticsPublishedTime_).seconds();
  lastStatisticsPublishedTime_ = now;
  elevation_map_msgs::msg::Statistics msg;
  msg.header.stamp = now;
  if (dt > 0.0) {
    msg.pointcloud_process_fps = pointCloudProcessCounter_ / dt;
  }
  pointCloudProcessCounter_ = 0;
  statisticsPub_->publish(msg);
}

void ElevationMappingNode::updateGridMap() {
  std::vector<std::string> layers(map_layers_all_.begin(), map_layers_all_.end());
  std::lock_guard<std::mutex> lock(mapMutex_);
  map_.get_grid_map(gridMap_, layers);
  gridMap_.setTimestamp(this->now().nanoseconds());
  alivePub_->publish(std_msgs::msg::Empty());

  if (enablePointCloudPublishing_) {
    publishAsPointCloud(gridMap_);
  }
  if (enableNormalArrowPublishing_) {
    publishNormalAsArrow(gridMap_);
  }
  isGridmapUpdated_ = true;
}

void ElevationMappingNode::initializeMap(const std::shared_ptr<elevation_map_msgs::srv::Initialize::Request> request,
                                         std::shared_ptr<elevation_map_msgs::srv::Initialize::Response> response) {
  if (request->type == request->POINTS) {
    std::vector<Eigen::Vector3d> points;
    for (const auto& point : request->points) {
      const auto& pointFrameId = point.header.frame_id;
      const auto& timeStamp = point.header.stamp;
      const auto& pvector = Eigen::Vector3d(point.point.x, point.point.y, point.point.z);

      if (mapFrameId_ != pointFrameId) {
        Eigen::Affine3d transformationBaseToMap;
         geometry_msgs::msg::TransformStamped transformTf;
        try {
          transformTf = tfBuffer_->lookupTransform(mapFrameId_, pointFrameId, timeStamp, rclcpp::Duration::from_seconds(1.0));
          transformationBaseToMap = tf2::transformToEigen(transformTf);
        } catch (tf2::TransformException& ex) {
          RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
          response->success = false;
          return;
        }
        const auto transformed_p = transformationBaseToMap * pvector;
        points.push_back(transformed_p);
      } else {
        points.push_back(pvector);
      }
    }
    std::string method;
    switch (request->method) {
      case elevation_map_msgs::srv::Initialize::Request::NEAREST:
        method = "nearest";
        break;
      case elevation_map_msgs::srv::Initialize::Request::LINEAR:
        method = "linear";
        break;
      case elevation_map_msgs::srv::Initialize::Request::CUBIC:
        method = "cubic";
        break;
      default:
        method = "cubic";
        break;
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "Initializing map with points using " << method);
    map_.initializeWithPoints(points, method);
  }
  response->success = true;
}

void ElevationMappingNode::publishNormalAsArrow(const grid_map::GridMap& map) const {
  auto startTime = this->now();

  const auto& normalX = map["normal_x"];
  const auto& normalY = map["normal_y"];
  const auto& normalZ = map["normal_z"];
  double scale = 0.1;

  visualization_msgs::msg::MarkerArray markerArray;
  for (grid_map::GridMapIterator iterator(map); !iterator.isPastEnd(); ++iterator) {
    if (!map.isValid(*iterator, "elevation")) {
      continue;
    }
    grid_map::Position3 p;
    map.getPosition3("elevation", *iterator, p);
    Eigen::Vector3d start = p;
    const auto i = iterator.getLinearIndex();
    Eigen::Vector3d normal(normalX(i), normalY(i), normalZ(i));
    Eigen::Vector3d end = start + normal * scale;
    if (normal.norm() < 0.1) {
      continue;
    }
    markerArray.markers.push_back(vectorToArrowMarker(start, end, i));
  }
  normalPub_->publish(markerArray);
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "publish as normal in %f sec.", (this->now() - startTime).seconds());
}

visualization_msgs::msg::Marker ElevationMappingNode::vectorToArrowMarker(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
                                                                     const int id) const {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = mapFrameId_;
  marker.header.stamp = this->now();
  marker.ns = "normal";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::ARROW;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.points.resize(2);
  marker.points[0].x = start.x();
  marker.points[0].y = start.y();
  marker.points[0].z = start.z();
  marker.points[1].x = end.x();
  marker.points[1].y = end.y();
  marker.points[1].z = end.z();
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.01;
  marker.scale.y = 0.02;
  marker.scale.z = 0.0;
  marker.color.a = 1.0;
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  return marker;
}

void ElevationMappingNode::publishMapToOdom(double error) {
  geometry_msgs::msg::TransformStamped transformStamped;
  transformStamped.header.stamp = this->now();
  transformStamped.header.frame_id = correctedMapFrameId_;
  transformStamped.child_frame_id = mapFrameId_;
  transformStamped.transform.translation.x = 0.0;
  transformStamped.transform.translation.y = 0.0;
  transformStamped.transform.translation.z = error;
  transformStamped.transform.rotation.x = 0.0;
  transformStamped.transform.rotation.y = 0.0;
  transformStamped.transform.rotation.z = 0.0;
  transformStamped.transform.rotation.w = 1.0;
  tfBroadcaster_->sendTransform(transformStamped);
}

}  // namespace elevation_mapping_cupy
