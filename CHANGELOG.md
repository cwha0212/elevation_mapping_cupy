# Changelog

haechi 포크(elevation_mapping_cupy)의 주요 변경사항을 기록한다. 형식은
[Keep a Changelog](https://keepachangelog.com/), 버전은 [SemVer](https://semver.org/)를
따른다. 브랜치 흐름은 NAVI 모노레포와 같다: 작업은 `chang_feature`, 통합은 `dev`
(`Dev vX.Y.Z — … 반영` 머지 커밋), 릴리스는 `main`. 상류(leggedrobotics)의 이력은
`original-ros2`/`ros2` 브랜치에 남아 있다.

## [Unreleased]
### Added
### Changed
- `gz_demo/package.xml`: 시뮬 전용 의존성(ros_gz_*, rviz2, grid_map_rviz_plugin, octomap_server2,
  nav2_bringup, rqt_image_view)에 `condition="$ELEVATION_GZ == 1"`. 로봇에서 `rosdep install` 이
  이들을 건너뛰고, 시뮬 머신은 `ELEVATION_GZ=1` 로 받는다. (ros_gz_sim 은 Humble 바이너리가 없어
  무조건 목록이면 로봇에서 rosdep 이 실패했다.)
- `HAECHI_BOARD.md` 1절 "의존성": rosdep, GPU 파이썬 휠, cudss 링커 경로, TensorRT 버전이 다르면
  엔진 재생성. 새 보드(TensorRT 10.3)에서 실제로 걸린 세 가지.
- `HAECHI_BOARD.md` 함정: PulseOS 보드의 GPU 클럭 고정(min=max=918 MHz)이 SAM-TP 추론에서 하드 리셋을
  일으킨다는 실측과 해제 한 줄, 부팅 oneshot 유닛.
- `HAECHI_BOARD.md`: 로컬 코스트맵 플러그인 기대값을 `[terrain_layer, inflation_layer]`로. navi_nav2
  `nav2_params_elevation.yaml`(chang_feature f793cdc)이 로컬 창에서 `obstacle_layer`(/scan)를 뺐다.
  근거는 실기 bag 실측: /scan의 원천이 로봇이 밟고 간 셀의 75 %를 찍고(z 바닥 40 cm 올려도 69 %),
  elevation 격자는 3.3 %. 전역 코스트맵·collision_monitor의 /scan은 유지.
### Removed
### Fixed

## [v0.2.3] - 2026-09-28

`Dev v0.2.3` — **보드 안내서 축약.** `HAECHI_BOARD.md`를 설치·빌드·실행·확인·위치·함정 여섯 절로 줄였다.

## [v0.2.2] - 2026-09-28

`Dev v0.2.2` — **다리 링: 박스 여유 대신 "이미 잰 지면보다 8 cm 위" 규칙.**

### Changed
- mapper 파라미터 `leg_ring` [front, side, back](기본 [0.50, 0.15, 0.0]), `leg_rise`(0.08),
  `body_filter`(박스 컷 스위치, 박스 자체는 항상 전달). 링 안의 점은 그 셀(또는 미측정이면
  주변 0.1 m 유효 셀 중앙값)의 높이보다 leg_rise 위에 있을 때만 버린다. 지면·경사·riser는
  접근 중 이미 셀에 있어 남는다. `body_margin` 기본 0으로 복귀(실험용).
- 실측(같은 bag, 발밑 프로브): 로봇 셀 lethal 2.2%→1.6%, footprint 안 lethal 3.95%→2.61%,
  걸어간 지면 0–1 m 오판정 4.9%→3.3%, 발밑 unknown 11.9%→11.5%(박스 +0.15 m 방식은 14.3%로
  늘었음). 링만으로는(L1, 앞 0.25 m·유효 셀 전제) 효과가 없었고, 미측정 셀 주변 지면 대체와
  앞 0.5 m가 필요했다(보폭과 첫 관측이 경합).
- 보폭 실측: 박스 밖 다리 높이 점의 폭발은 앞 0~0.25 m에 몰리고 옆은 절반, 뒤는 없음. 0.3 m
  밖은 식생.
- 실기 프리셋(haechi_nav): 박스 컷 off(navi_lidar가 병합 시 컷), 링 on.
- gz_demo: Nav2 파라미터 변형 `gz_nav2_params_haechi.yaml`(프로파일 값 그대로),
  `gz_nav2_params_haechi_tuned.yaml`(cost_scaling 10); `gz_nav2.launch.py params:=`.

## [v0.2.1] - 2026-09-28

`Dev v0.2.1` — **발밑 lethal 귀속 실험 결과 반영: 자기몸 컷에 0.15 m 여유.**

### Changed
- `haechi.launch.py` `body_margin` 기본 0.15(신설 `dedup_voxel` 인자, 기본 0.02).
  실측(같은 bag): 여유 0일 때 로봇 셀 lethal 2.2%(17/776프레임, 15가 보행 중, 항 귀속
  step 10 / slope 6 / rough 1, 그 셀의 step 중앙값 0.20 m)이고 그 순간 박스 바로 바깥
  다리 높이 점이 중앙값 44개(평소 4개). 여유 0.15에서 1.5%, footprint 안 lethal 3.95%→2.0%,
  걸어간 지면 0–1 m 오판정 4.9%→3.3%. 남은 원인은 잔디가 아니라 보폭 중 다리다.
- `haechi_nav.launch.py`: navi_lidar 병합 컷이 있어도 mapper 컷을 여유 0.15로 켬,
  dedup은 끔(navi_lidar가 0.15 m voxel).

## [v0.2.0] - 2026-09-28

`Dev v0.2.0` — **실기 경로만 남긴 리팩토링(haechi-lean)과 SAM-TP 자산 통합.**

### Added
- `launch/haechi_nav.launch.py` — 실기 브링업(옥토맵 없음, body cut 끔, `lite:=true` 프리셋).
- `launch/haechi.launch.py` 인자 `gait`, `audit`, `geom_grid`, `veto_cost`, `map_length`,
  `terrain_fps`, `samtp_max_rate`, `body_margin`.
- `config/setups/haechi/plugin_config_gait.yaml` — stairs/ramp/drop 체인(`gait:=true`로 병합).
- `samtp/` — 모델 자산 fetch/검증(`fetch_assets.sh`, `SHA256SUMS`), 엔진 빌드
  (`build_engine.sh`), ONNX 내보내기(`export_samtp_onnx.py`). 파일은 GitHub Release
  `samtp-assets-v1`(ONNX 130 MB, Orin fp16 엔진 68 MB).
- `gz_demo/` — 새 패키지 `elevation_mapping_gz_demo`: Gazebo Fortress 벤치(월드, 시뮬 설정,
  stairs/nav_grid_fuse 노드, Nav2 시뮬 파라미터). 실기 빌드에서 제외.
- `HAECHI_BOARD.md` — 보드 설치·빌드·실행·확인 절차.
- 동등성 테스트 `tests/test_publish_path_equivalence.py`; `test_stairs_detect` 등록.

### Changed
- 발행 경로: 레이어당 회전 복사 1회 + pinned 호스트 전송 1회(기존 device copy 2, stream 생성 1,
  host copy 3). 바이트 동일.
- `plugin_manager`: 입력을 복사 대신 뷰로, arity·의존성 캐시, 설정 파일 배열 병합.
- `semantic_safety_filter`: 반경 마스크 1회 계산.
- `terrain_grid_node`: 필요한 2레이어만 변환, 제자리 컷, 셀당 해시 조회 제거. 출력 바이트 동일.
- `samtp_node`: heatmap은 구독자 있을 때만, CameraInfo/ChannelInfo 캐시, erosion GPU max-pool.
- 자기몸 컷을 mapper 콜백에 융합(`body_filter_min/max`, `dedup_voxel`); 별도 노드 제거.
- pose 타이머가 `update_pose_fps`를 따름.
- haechi 기본 terrain 발행 레이어 `[drivability, safety]`(audit/gait 모드에서 확장).
- 실측: 같은 bag에서 elevation 스택 CPU 합 97.8%→55.3%, mapper RSS 3.2 GB→1.0 GB,
  GR3D 51%→43%, 보드 RAM 7.1 GB→5.8 GB. 판정값은 기준 3회 밴드 안.

### Removed
- digging 플러그인 체인(inpainting, smooth, erosion, min/max, near_base_height,
  positive_spike, robot_centric)과 core 기본 설정의 그 체인.
- 포인트클라우드 시맨틱 fusion 5종, `custom_semantic_kernels`, 중복 이미지 커널.
- `map_initializer`(노드가 호출하지 않음), `traversability_polygon`, chainer 백엔드.
- save_map/load_map/masked_replace 서비스와 rosbag 코드. `clear_map`은 유지.
- turtlesim/menzi/synthetic/turtle_bot 설정과 런치, semantic_sensor, plane_segmentation,
  docs/benchmarks/docker/CI, 읽고 쓰지 않던 노드 파라미터 20여 개.
- `elevation_map_msgs`: Statistics.msg, CheckSafety.srv, Initialize.srv.

### Fixed
- `semantic_map.shift_map_xy`가 `elements_to_shift`를 새 배열로 굴려 놓고 버리던 버그(제자리 갱신).
- 시뮬 Nav2 파라미터: `trinary_costmap` 등 StaticLayer 값을 코스트맵 노드 수준으로 이동
  (플러그인 아래에서는 무시됨). 이제 무효인 `fatal` 런치 인자 제거.

## [v0.1.0] - 2026-09-28 (`baseline-c`, state-c e12348d)

- haechi 맵 프레임 `map`(LIO의 odom은 몸체 프레임), footprint 자기몸 컷, 한계값
  [45°, 0.30 m, 0.08], 카메라 거부 비용화(veto_cost 70), 지속성 K=3, 카메라 구제.
- 실기 bag 검증: 걸어간 지면 0–1 m 오판정 40%→5.6%, 4 s 뒤 자기 위치 통과 17%→69%.
