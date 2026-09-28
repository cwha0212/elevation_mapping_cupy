# Changelog

haechi 포크(elevation_mapping_cupy)의 주요 변경사항을 기록한다. 형식은
[Keep a Changelog](https://keepachangelog.com/), 버전은 [SemVer](https://semver.org/)를
따른다. 브랜치 흐름은 NAVI 모노레포와 같다: 작업은 `chang_feature`, 통합은 `dev`
(`Dev vX.Y.Z — … 반영` 머지 커밋), 릴리스는 `main`. 상류(leggedrobotics)의 이력은
`original-ros2`/`ros2` 브랜치에 남아 있다.

## [Unreleased]
### Added
### Changed
### Removed
### Fixed

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
