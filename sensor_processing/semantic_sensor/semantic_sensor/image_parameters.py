from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable


@dataclass
class FeatureExtractorParameter(Serializable):
    name: str = "DINO"
    interpolation: str = "bilinear"
    model: str = "vit_small"
    patch_size: int = 16
    dim: int = 10
    dropout: bool = False
    dino_feat_type: str = "feat"
    projection_type: str = "nonlinear"
    input_size: list = field(default_factory=lambda: [80, 160])
    pcl: bool = False


@dataclass
class CameraIntrinsics(Serializable):
    """Static camera calibration, used when no CameraInfo topic is available.

    Leave ``k`` empty to keep subscribing to ``ImageParameter.camera_info_topic``
    (the default). Filling ``k`` switches the node to this config instead.
    """

    width: int = 0
    height: int = 0
    # "radtan" keeps the distortion coefficients; the elevation mapping kernel
    # zeroes them for every other model name, including ROS' own "plumb_bob".
    distortion_model: str = "radtan"
    k: list = field(default_factory=list)  # 9 values, row-major
    d: list = field(default_factory=list)  # 5 values, defaults to zeros
    p: list = field(default_factory=list)  # 12 values, derived from k when empty


@dataclass
class ImageParameter(Serializable):
    image_topic: str = "/alphasense_driver_ros/cam4/debayered"
    semantic_segmentation: bool = True
    segmentation_model: str = "lraspp_mobilenet_v3_large"
    show_label_legend: bool = False
    channels: list = field(default_factory=lambda: ["grass", "road", "tree", "sky"])
    publish_topic: str = "semantic_seg"
    publish_image_topic: str = "semantic_seg_img"
    publish_camera_info_topic: str = "semantic_seg_info"
    channel_info_topic: str = "channel_info"
    feature_extractor: bool = False
    feature_config: FeatureExtractorParameter = field(default_factory=FeatureExtractorParameter)
    feature_topic: str = "semantic_seg_feat"
    feat_image_topic: str = "semantic_seg_feat_im"
    feat_channel_info_topic: str = "feat_channel_info"
    resize: float = None
    camera_info_topic: str = "camera_info"
    camera_intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
