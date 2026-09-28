# haechi 보드(Orin) 설치·실행 안내

elevation_mapping_cupy 포크를 haechi의 Orin 보드에 올리고, NAVI 스택과 함께
Nav2 로컬 코스트맵에 지형 판정을 넣어 주행하는 절차다. 2026-09-28 기준.

대상 보드: AGX Orin 32 GB, L4T R36.5.2, ROS 2 Humble, `maumai@10.50.7.224`.

## 1. 한 번만 하는 준비

### 1.1 GPU 파이썬 스택

cupy와 torch가 네이티브로 있어야 한다(Docker 불필요). 확인:

```bash
python3 -c "import cupy, torch; print(cupy.__version__, torch.__version__, torch.cuda.is_available())"
# 13.6.0 2.11.0 True
```

### 1.2 워크스페이스 구조

```
~/navi_ws/src/
  NAVI/                  # 모노레포: navi_lidar, navi_indoor, navi_hybrid, navi_interface (브랜치 chang_feature)
  navi_nav2/             # Nav2 포크 (브랜치 chang_feature)  ← 반드시 git clone 이어야 한다
  grid_map/              # grid_map humble 소스 (apt 위에 오버레이)
  elevation_mapping_cupy/ # 이 레포 (브랜치 chang_feature; dev 는 통합, main 은 릴리스)
~/dependencies/
  ws_livox/ ws_rslidar/  # 라이다 드라이버 워크스페이스
  octomap_ws/            # octomap_server2 (gait 모드에서만 필요)
~/samtp/samtp_512_fp16.engine   # SAM-TP 엔진 (samtp/fetch_assets.sh 가 받음)
~/map_folder/<map>/<submap>/map.yaml   # NAVI 저장 맵 (localization 모드용)
```

source 순서는 항상 아래와 같다. navi_indoor의 LIO 바이너리가 livox 타입서포트를
링크하므로 ws_livox를 빼먹으면 exit 127로 즉사한다.

```bash
source /opt/ros/humble/setup.bash
source ~/dependencies/ws_livox/install/setup.bash
source ~/dependencies/ws_rslidar/install/setup.bash
source ~/dependencies/octomap_ws/install/setup.bash    # gait 모드가 아니면 생략 가능
source ~/navi_ws/install/setup.bash
```

`~/.bashrc`에 `ROBOT_TYPE=haechi`가 있어야 한다(없으면 모든 NAVI 런치에 `robot:=haechi`를 명시).

### 1.3 SAM-TP 모델 자산

ONNX(130 MB)와 이 Orin용 fp16 엔진(68 MB)은 레포의 GitHub Release `samtp-assets-v1`에
있다(단일 파일 100 MB 한도 때문에 git 밖). 레포를 받은 뒤 한 번:

```bash
cd ~/navi_ws/src/elevation_mapping_cupy
bash samtp/fetch_assets.sh          # ~/samtp/ 에 받고 SHA256 검증
```

보드가 바뀌었거나(JetPack, GPU) 엔진 로드가 실패하면 ONNX에서 다시 만든다:

```bash
bash samtp/fetch_assets.sh onnx && bash samtp/build_engine.sh
```

자세한 것은 `samtp/README.md`.

## 2. 레포 받기·브랜치 맞추기

```bash
# elevation (이 레포)
cd ~/navi_ws/src
git clone https://github.com/cwha0212/elevation_mapping_cupy.git   # 이미 있으면 생략
cd elevation_mapping_cupy && git fetch origin && git checkout chang_feature && git pull --ff-only

# NAVI: dev로 올리지 않는다. chang_feature 로 맞춘다 (보드에 예전 로컬 편집이 있으면 stash)
cd ~/navi_ws/src/NAVI && git stash push -m "board-local edits" ; git fetch origin && git checkout -B chang_feature origin/chang_feature

# navi_nav2: 보드의 기존 폴더가 git 이 아니면(수동 복사본) 교체한다
cd ~/navi_ws/src && [ -d navi_nav2/.git ] || mv navi_nav2 navi_nav2.stale_$(date +%m%d)
[ -d navi_nav2 ] || git clone -b chang_feature git@github.com-company:MaumAI-Company/navi_nav2.git navi_nav2
ls navi_nav2/nav2_bringup/params/nav2_params_elevation.yaml     # 있어야 한다
```

## 3. 빌드

실기 런타임은 두 패키지만 필요하다. 시뮬 벤치(`elevation_mapping_gz_demo`)는 보드에서 빌드하지 않는다.

```bash
cd ~/navi_ws
colcon build --packages-select elevation_map_msgs elevation_mapping_cupy \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
# NAVI 쪽을 갱신했다면 (nav2_params_elevation.yaml 설치, navi_lidar 프로파일 반영)
colcon build --packages-select navi_lidar nav2_bringup --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/navi_ws/install/setup.bash
ls install/nav2_bringup/share/nav2_bringup/params/nav2_params_elevation.yaml
```

주의: `elevation_map_msgs`의 메시지 목록이 바뀐 뒤 처음 빌드할 때는 옛 빌드 디렉터리를
지워야 한다(`rm -rf build/elevation_map_msgs install/elevation_map_msgs`). 또 NAVI 쪽을
`colcon build`(패키지 선택 없이) 하면 install에서 elevation이 사라진 적이 있다.
NAVI를 빌드한 뒤에는 `ros2 pkg prefix elevation_mapping_cupy`가 되는지 확인한다.

## 4. 실기 실행 (4개 터미널, 모두 위 source 순서 적용)

```bash
# T1 라이다 (병합 클라우드 /points/merged_deskewed, footprint 자기몸 컷 포함)
ros2 launch navi_lidar lidar.launch.py robot:=haechi

# T2 위치추정 (map->odom, map->odom_2d, /scan, /odom_2d). indoor 맵 test/sub_map 예시
ros2 launch navi_indoor nx_indoor_localization.launch.py robot:=haechi \
    map_name:=test submap_name:=sub_map use_rviz:=false
#   -> RViz(노트북)에서 2D Pose Estimate 로 /initialpose 를 준다

# T3 elevation + SAM-TP + 지형 격자
nice -n 10 ros2 launch elevation_mapping_cupy haechi_nav.launch.py
#   보드가 버거우면:  ... haechi_nav.launch.py lite:=true   (8 m 맵, SAM-TP 2 Hz)
#   카메라 없이:      ... haechi_nav.launch.py use_semantics:=false

# T4 Nav2 (elevation 변형 파라미터). map:= 은 LIO 가 쓰는 것과 같은 submap 의 map.yaml
ros2 launch navi_lidar nav2.launch.py robot:=haechi \
    map:=$HOME/map_folder/test/sub_map/map.yaml \
    params_file:=$HOME/navi_ws/install/nav2_bringup/share/nav2_bringup/params/nav2_params_elevation.yaml
```

Nav2 노드 로그는 터미널이 아니라 `~/.ros/log/latest/launch.log`에 쌓인다.

원래 구성으로 되돌리기: T4를 `params_file:=` 없이 다시 띄운다. T3는 꺼도 된다.

### 4.1 haechi.launch.py 인자

| 인자 | 기본 | 뜻 |
|---|---|---|
| `use_semantics` | true | SAM-TP 카메라 가지 |
| `body_filter` | true (haechi_nav에서는 false) | 클라우드에서 footprint 안 점 제거. navi_lidar v0.6.4 이후 bag/실기에선 중복이라 끔 |
| `gait` | false | stairs/ramp/drop 체인 + 옥토맵 경로 추가 |
| `audit` | false | driven_audit 가 읽는 추가 레이어 발행 |
| `geom_grid` | true | 기하만의 격자 `/terrain/local_grid_geom` 도 발행 |
| `grid_threshold` / `veto_cost` | 0.4 / 70 | 격자 컷과 카메라 단독 거부의 비용 |
| `map_length` / `terrain_fps` / `samtp_max_rate` | 10.0 / 3.0 / 4.0 | 부하 조절 |

## 5. 확인 절차

### 5.1 T1~T3 후

```bash
ros2 topic hz /points/merged_deskewed          # 10 Hz
ros2 topic hz /scan /odom_2d                   # 10 Hz
ros2 topic delay /odom_2d                      # < 0.15 s
ros2 run tf2_ros tf2_echo map odom | head -5;      ros2 run tf2_ros tf2_echo map odom_2d | head -5
ros2 run tf2_ros tf2_echo odom lidar_frame | head -5; ros2 run tf2_ros tf2_echo odom_2d base_link | head -5   # 0.23 / -0.105
ros2 topic hz /camera/image_raw/compressed /front_cam/samtp_score     # ~4 Hz
ros2 topic hz /elevation_mapping_node/elevation_map_terrain /terrain/local_grid   # 3 Hz
ros2 topic echo --once /terrain/local_grid --field header.frame_id    # map
ros2 topic echo --once /terrain/local_grid --field info               # 200x200, 0.05
```

### 5.2 T4 후

```bash
ros2 lifecycle get /map_server                                   # active
ros2 param get /local_costmap/local_costmap plugins              # [terrain_layer, obstacle_layer, inflation_layer]
ros2 param get /local_costmap/local_costmap trinary_costmap      # False
ros2 param get /local_costmap/local_costmap resolution           # 0.05
ros2 param get /local_costmap/local_costmap inflation_layer.inflation_radius   # 0.90 (프로파일 재작성)
ros2 topic hz /local_costmap/costmap                             # 5 Hz
grep -E "terrain_layer|Can't update static" ~/.ros/log/latest/launch.log | sort | uniq -c
python3 ~/local_costmap_probe.py 60      # verdict OK: 170~185 비용 셀 존재, 로봇 셀 lethal 아님, terrain lethal 누락 0
```

`StaticLayer: Resizing static layer to 200 X 200`이 로봇이 움직일 때마다 찍히는 것은
정상이다(격자 원점이 움직여서). `Can't update static costmap layer, no map received`가 보이면
격자 토픽이 안 오거나 `map_subscribe_transient_local`이 True 인 것이다.

### 5.3 부하

```bash
tegrastats --interval 1000 | tee ~/load_$(date +%m%d_%H%M).log
```

중단·완화 기준: GR3D 평균 > 92 %, `elevation_map_terrain` < 2.5 Hz, `/odom_2d` 지연 > 0.2 s,
`/scan` < 9.5 Hz, `/local_costmap/costmap` < 4.5 Hz. 순서대로 `lite:=true` →
`use_semantics:=false` → T4 원복.

### 5.4 첫 주행과 기록

포장 5~8 m 직진과 90° 회전 목표부터, 그 다음 디딤돌 길. 사후 채점(driven_audit2,
reach_probe, diff_canvas)을 위해 아래를 녹화한다.

```bash
ros2 bag record -o ~/bags/nav_elev_$(date +%m%d_%H%M) --qos-profile-overrides-path ~/bags/qos_map.yaml \
  /tf /tf_static /odom_2d /aft_mapped_to_init /points/merged_deskewed /points/imu /scan \
  /camera/image_raw/compressed /front_cam/samtp_score /front_cam/samtp_camera_info /front_cam/samtp_channel_info \
  /elevation_mapping_node/elevation_map_terrain /terrain/local_grid /terrain/local_grid_geom \
  /local_costmap/costmap /global_costmap/costmap /map /plan /local_plan /cmd_vel_nav /cmd_vel_smoothed /cmd_vel
```

## 6. bag 재생 회귀 (보드 홈의 도구)

레포 밖, 보드 `~`에 있는 스크립트다. 모든 런은 시작 전에 `clean_stack.sh`로 잔존 프로세스를
0으로 만들고, 잠금으로 동시 실행을 막는다. **kill 패턴이 명령줄에 들어가면 자기 자신을
죽이므로, ssh 명령줄에는 스크립트 이름만 쓴다.**

```bash
LABEL=X AUDIT=true SECS=260 bash ~/run_samtp.sh > /tmp/run_X.out 2>&1   # 실기 bag 재생 + 프로브 전부
bash ~/show_run.sh X                # 요약
python3 ~/band.py G0a G0b G0c X     # 기준 3회와 나란히
bash ~/kill_runs.sh                 # 남은 것 정리
```

프로브: `driven_audit2.py`(걸어간 지면 판정), `reach_probe.py`(4 s 뒤 자기 위치 통과 가능),
`diff_canvas.py`(카메라 거부 위치), `cost_hist.py`, `local_canvas.py`, `real_audit.py`(2D 리포트),
`grid_compare.py`(격자 노드 바이트 비교), `load_probe.sh`(CPU/RSS/GR3D).

## 7. 자주 걸리는 것

- 카메라는 best-effort 다. reliable 로 구독하면 경고 없이 이미지가 0장이다(samtp_node 는 sensor-data QoS).
- `/terrain/local_grid` 발행자는 volatile 이다. StaticLayer 의 `map_subscribe_transient_local`이
  True 면 조용히 빈 레이어가 된다.
- StaticLayer 의 `trinary_costmap`, `lethal_cost_threshold`, `track_unknown_space`는
  **코스트맵 노드 수준** 파라미터다. 플러그인 이름 아래에 두면 무시된다.
- `unknown_cost_value`는 두지 않는다(기본 255 == OccupancyGrid −1). `track_unknown_space`도
  로컬에서는 두지 않는다(발밑 미지 링이 FREE 여야 한다).
- Nav2 `map:=` 기본값(`~/map_folder/empty_map.yaml`)은 보드에 없다. 실제 submap 의 map.yaml 을 준다.
- 실기 bag 에는 TF 가 없다. LIO(nx_indoor_mapping)가 재생 중에 만든다.
