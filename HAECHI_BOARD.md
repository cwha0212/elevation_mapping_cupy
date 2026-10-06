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

## 1. 의존성 (새 보드 한 번만)

```bash
# ROS 쪽 — package.xml 의 depend 를 rosdep 이 깐다 (tf_transformations, cv_bridge, message_filters, grid_map …).
# 시뮬(gz_demo) 의존성은 조건부라 로봇에서는 건너뛴다. 시뮬 머신에서만 ELEVATION_GZ=1 을 앞에 붙인다.
sudo apt install -y python3-rosdep
sudo rosdep init 2>/dev/null; rosdep update
cd ~/navi_ws && rosdep install --from-paths src/elevation_mapping_cupy --ignore-src -r -y

# GPU 파이썬 (JetPack 6.x / CUDA 12.6, 휠은 pypi.jetson-ai-lab.io jp6/cu126)
pip install --user "numpy>=1.23,<2" "cupy-cuda12x<14" ruamel.yaml simple-parsing scipy transforms3d
pip install --user torch-2.11.0-cp310-cp310-linux_aarch64.whl torchvision-0.26.0-cp310-cp310-linux_aarch64.whl

# torch 2.11 휠은 libcudss.so.0 을 링크한다 (cuda 저장소의 cudss, 위 keyring 등록 후).
sudo apt-get install -y cudss
ldconfig -p | grep -q libcudss.so.0 || { dirname "$(dpkg -L libcudss0-cuda-12 | grep 'libcudss.so.0$')" | sudo tee /etc/ld.so.conf.d/cudss.conf; sudo ldconfig; }

# SAM-TP 엔진은 TensorRT 버전에 묶인다. 릴리스 엔진은 Orin(TensorRT 10.7)용이라 버전이 다르면 다시 만든다 (몇 분).
python3 -c "import tensorrt; print(tensorrt.__version__)"          # 10.7 이 아니면 ↓
bash samtp/fetch_assets.sh onnx && bash samtp/build_engine.sh

# 확인 — 셋 다 오류 없이 찍혀야 한다
python3 -c "import cupy as cp; print('cupy', cp.__version__, cp.zeros(3).sum())"
python3 -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
python3 -c "import tensorrt, tf_transformations; print('trt', tensorrt.__version__)"
```

torch 는 SAM-TP(`samtp_node`)만 쓴다. 안 잡히면 `haechi_nav.launch.py use_semantics:=false` 로 elevation 만 먼저 띄울 수 있다.

## 2. 받기

```bash
cd ~/navi_ws/src
git clone https://github.com/cwha0212/elevation_mapping_cupy.git      # 있으면 생략
cd elevation_mapping_cupy && git checkout chang_feature && git pull --ff-only
bash samtp/fetch_assets.sh                                            # SAM-TP ONNX·엔진 → ~/samtp (SHA256 검증)

cd ~/navi_ws/src/NAVI && git fetch origin && git checkout -B chang_feature origin/chang_feature
cd ~/navi_ws/src && [ -d navi_nav2/.git ] || { mv navi_nav2 navi_nav2.stale; git clone -b chang_feature git@github.com-company:MaumAI-Company/navi_nav2.git; }
```

## 3. 빌드

```bash
sudo apt install ros-humble-grid-map*
cd ~/navi_ws
colcon build --packages-select elevation_map_msgs elevation_mapping_cupy navi_lidar nav2_bringup \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
# (전체 빌드를 해도 gz_demo 는 파일만 설치하는 패키지라 실패하지 않는다. 뺄 때는 --packages-skip elevation_mapping_gz_demo)
source ~/navi_ws/install/setup.bash
```

시뮬 벤치(`elevation_mapping_gz_demo`)는 보드에서 빌드하지 않는다.

## 4. 실행 (터미널 4개)

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

되돌리기: 4번 터미널을 `params_file:=` 없이 다시 띄운다.

## 5. 확인

```bash
ros2 topic hz /points/merged_deskewed /scan /odom_2d                 # 10 Hz
ros2 topic hz /front_cam/samtp_score /terrain/local_grid            # ~4 / 10 Hz
ros2 topic echo --once /terrain/local_grid --field header.frame_id  # map
ros2 param get /local_costmap/local_costmap plugins                 # [terrain_layer, inflation_layer]  (/scan 은 전역·collision_monitor 만)
ros2 param get /local_costmap/local_costmap trinary_costmap         # False
python3 ~/local_costmap_probe.py 60                                 # verdict OK
tegrastats --interval 1000                                          # GR3D 평균 < 92 %
```

## 6. 무엇이 어디에

| 것 | 위치 |
|---|---|
| 실기 런치 | `launch/haechi_nav.launch.py` (→ `haechi.launch.py`) |
| 설정 | `config/setups/haechi/{haechi, plugin_config, plugin_config_gait}.yaml` |
| 격자 노드 | `src/terrain_grid_node.cpp` → `/terrain/local_grid` (0 free · 70 카메라 비용 · 100 lethal · −1 unknown) |
| Nav2 변형 | navi_nav2 `nav2_bringup/params/nav2_params_elevation.yaml` (로컬: terrain + inflation, /scan 제외) |
| 모델 자산 | `samtp/` (fetch·체크섬·엔진 빌드·ONNX 내보내기), Release `samtp-assets-v1` |
| bag 재생 도구 | 보드 `~/run_samtp.sh`, `~/show_run.sh`, `~/band.py`, `~/kill_runs.sh` |

## 7. 함정

- **PulseOS 보드(Orin NX, R36.5.0)는 GPU 클럭이 부팅부터 min=max=918 MHz 로 고정돼 있다.** 그 상태에서 SAM-TP(fp16 TensorRT, 4 Hz 간헐 추론)를 돌리면 커널 로그 한 줄 없이 보드가 하드 리셋된다(2026-09-30 실측 6/6, 전원 레일·온도·메모리 정상). GPU 최소 클럭을 풀면 사라진다(SAM-TP 단독 306프레임, 전체 스택 4분 통과).
  ```bash
  echo 306000000 | sudo tee /sys/class/devfreq/17000000.gpu/min_freq     # 재부팅마다 필요. 영구 적용은 oneshot 유닛(HAECHI_BOARD 7절 끝)
  ```
- 카메라는 best-effort. reliable로 구독하면 조용히 0장.
- `/terrain/local_grid`는 volatile. StaticLayer `map_subscribe_transient_local`이 True면 조용히 빈 레이어.
- StaticLayer의 `trinary_costmap`·`lethal_cost_threshold`는 코스트맵 노드 수준 키. 플러그인 아래에 두면 무시.
- `ros-humble-grid-map*` 만 깔고 rosdep 을 건너뛰면 빌드는 되고 실행에서 `No module named 'tf_transformations'` 로 죽는다.
- torch `ImportError: libcudss.so.0` 는 cudss 가 없거나 ld 경로 밖. 1절의 cudss 두 줄.
- samtp_node 가 엔진 역직렬화에서 죽으면 TensorRT 버전 불일치(엔진 헤더에 빌드 버전이 있다). `build_engine.sh` 로 재생성.
- NAVI를 패키지 선택 없이 빌드하면 install에서 elevation이 사라진 적 있다. 빌드 후 `ros2 pkg prefix elevation_mapping_cupy` 확인.

영구 적용 유닛 (`/etc/systemd/system/gpu-dvfs-unpin.service`, `sudo systemctl enable --now gpu-dvfs-unpin`):

```ini
[Unit]
Description=Unpin the GPU clock after boot (PulseOS boots with GPU min=max=918 MHz)
After=nvpmodel.service nvphs.service
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo 306000000 > /sys/class/devfreq/17000000.gpu/min_freq'
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
```
