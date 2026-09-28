# haechi · elevation 설치와 실행

Orin 보드(`maumai@10.50.7.224`, L4T R36.5.2, ROS 2 Humble)에서 elevation 지형 격자를
NAVI Nav2 로컬 코스트맵에 넣어 주행하기까지. 배경과 이력은 `CHANGELOG.md`.

## 0. 전제

| 항목 | 값 |
|---|---|
| GPU 파이썬 | cupy 13.6, torch 2.11 (네이티브, Docker 불필요) |
| 워크스페이스 | `~/navi_ws/src/{NAVI, navi_nav2, grid_map, elevation_mapping_cupy}` |
| 브랜치 | NAVI `chang_feature` · navi_nav2 `chang_feature` · 이 레포 `chang_feature` |
| 드라이버 | `~/dependencies/{ws_livox, ws_rslidar}` (octomap_ws는 gait 모드만) |
| 환경변수 | `ROBOT_TYPE=haechi` (`~/.bashrc`) |

source 순서 (LIO가 livox 타입서포트를 링크하므로 순서를 지킨다):

```bash
source /opt/ros/humble/setup.bash
source ~/dependencies/ws_livox/install/setup.bash
source ~/dependencies/ws_rslidar/install/setup.bash
source ~/navi_ws/install/setup.bash
```

## 1. 받기

```bash
cd ~/navi_ws/src
git clone https://github.com/cwha0212/elevation_mapping_cupy.git      # 있으면 생략
cd elevation_mapping_cupy && git checkout chang_feature && git pull --ff-only
bash samtp/fetch_assets.sh                                            # SAM-TP ONNX·엔진 → ~/samtp (SHA256 검증)

cd ~/navi_ws/src/NAVI && git fetch origin && git checkout -B chang_feature origin/chang_feature
cd ~/navi_ws/src && [ -d navi_nav2/.git ] || { mv navi_nav2 navi_nav2.stale; git clone -b chang_feature git@github.com-company:MaumAI-Company/navi_nav2.git; }
```

## 2. 빌드

```bash
cd ~/navi_ws
colcon build --packages-select elevation_map_msgs elevation_mapping_cupy navi_lidar nav2_bringup \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/navi_ws/install/setup.bash
```

시뮬 벤치(`elevation_mapping_gz_demo`)는 보드에서 빌드하지 않는다.

## 3. 실행 (터미널 4개)

```bash
# 1 라이다
ros2 launch navi_lidar lidar.launch.py robot:=haechi
# 2 위치추정 (그 뒤 RViz에서 /initialpose)
ros2 launch navi_indoor nx_indoor_localization.launch.py robot:=haechi map_name:=test submap_name:=sub_map use_rviz:=false
# 3 elevation + SAM-TP + 지형 격자
nice -n 10 ros2 launch elevation_mapping_cupy haechi_nav.launch.py          # 버거우면 lite:=true
# 4 Nav2 (elevation 변형 파라미터)
ros2 launch navi_lidar nav2.launch.py robot:=haechi map:=$HOME/map_folder/test/sub_map/map.yaml \
  params_file:=$HOME/navi_ws/install/nav2_bringup/share/nav2_bringup/params/nav2_params_elevation.yaml
```

되돌리기: 4번을 `params_file:=` 없이 다시 띄운다.

## 4. 확인

```bash
ros2 topic hz /points/merged_deskewed /scan /odom_2d                 # 10 Hz
ros2 topic hz /front_cam/samtp_score /terrain/local_grid            # ~4 / 3 Hz
ros2 topic echo --once /terrain/local_grid --field header.frame_id  # map
ros2 param get /local_costmap/local_costmap plugins                 # [terrain_layer, obstacle_layer, inflation_layer]
ros2 param get /local_costmap/local_costmap trinary_costmap         # False
python3 ~/local_costmap_probe.py 60                                 # verdict OK
tegrastats --interval 1000                                          # GR3D 평균 < 92 %
```

## 5. 무엇이 어디에

| 것 | 위치 |
|---|---|
| 실기 런치 | `launch/haechi_nav.launch.py` (→ `haechi.launch.py`) |
| 설정 | `config/setups/haechi/{haechi, plugin_config, plugin_config_gait}.yaml` |
| 격자 노드 | `src/terrain_grid_node.cpp` → `/terrain/local_grid` (0 free · 70 카메라 비용 · 100 lethal · −1 unknown) |
| Nav2 변형 | navi_nav2 `nav2_bringup/params/nav2_params_elevation.yaml` |
| 모델 자산 | `samtp/` (fetch·체크섬·엔진 빌드·ONNX 내보내기), Release `samtp-assets-v1` |
| bag 재생 도구 | 보드 `~/run_samtp.sh`, `~/show_run.sh`, `~/band.py`, `~/kill_runs.sh` |

## 6. 함정 넷

- 카메라는 best-effort. reliable로 구독하면 조용히 0장.
- `/terrain/local_grid`는 volatile. StaticLayer `map_subscribe_transient_local`이 True면 조용히 빈 레이어.
- StaticLayer의 `trinary_costmap`·`lethal_cost_threshold`는 코스트맵 노드 수준 키. 플러그인 아래에 두면 무시.
- NAVI를 패키지 선택 없이 빌드하면 install에서 elevation이 사라진 적 있다. 빌드 후 `ros2 pkg prefix elevation_mapping_cupy` 확인.
