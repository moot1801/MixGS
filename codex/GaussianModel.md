# GaussianModel 클래스 정리

## 1. 클래스 개요
`GaussianModel`은 장면의 3D Gaussian 파라미터(위치, 오프셋, SH 색 특징, 스케일, 회전, opacity)를 보관하고,  
학습 가능한 `nn.Parameter` + `optimizer`를 통해 업데이트하며, PLY 입출력/프루닝/밀집화(densify) 유틸까지 담당하는 코어 클래스다.

정의 파일: `scene/gaussian_model.py`

## 2. 생성 시점 (언제 만들어지고 초기화되는가)

### 2.1 인스턴스 생성 시점
- 학습: `train_mixgs.py:40`
  - `model_config['name']`를 통해 동적으로 생성됨.
  - 기본 config에서는 `name: "GaussianModel"` (`config/*.yaml`).
- 렌더링: `render_mixgs.py:99`
  - 학습과 동일 방식으로 생성.

### 2.2 실제 파라미터 채워지는 시점
- `LargeScene` 생성 과정에서 아래 중 하나 실행:
  - `load_ply(...)`로 기존 포인트클라우드/속성 로드 (`scene/__init__.py:165`, `scene/__init__.py:171`, `scene/__init__.py:174`)
  - `create_from_pcd(...)`로 초기 point cloud에서 새 파라미터 생성 (`scene/__init__.py:210`)

### 2.3 학습 준비 시점
- `training_setup(...)`로 optimizer/스케줄러 세팅 (`train_mixgs.py:54`).
- 체크포인트 재개 시 `restore(...)`로 텐서+옵티마 상태 복구 (`train_mixgs.py:57`).
- 학습 도중 `joint_start_iter`에서 `gaussian_training()` 호출해 Gaussian 파라미터 학습 활성화 (`train_mixgs.py:91`).

## 3. 주요 속성

### 3.1 구조/설정 속성
- `opacity_thr`: opacity 기반 필터 임계값.
- `n_offsets`: 오프셋 슬롯 수(현재 1).
- `active_sh_degree`, `max_sh_degree`: SH degree 현재값/최대값.
- `spatial_lr_scale`, `percent_dense`: 학습 스케일/밀집화 관련 계수.

### 3.2 학습 대상 파라미터 (`nn.Parameter`)
- `_xyz`: Gaussian 중심 좌표.
- `_offset`: 중심 보정 오프셋.
- `_features_dc`, `_features_rest`: SH 색 특징(DC + 나머지 계수).
- `_scaling`: 로그 스케일 파라미터(activation으로 실제 스케일).
- `_rotation`: 회전 파라미터(정규화 후 사용).
- `_opacity`: 로짓 형태 opacity 파라미터(sigmoid 후 사용).

### 3.3 러닝 상태/보조 상태
- `optimizer`: Adam optimizer.
- `xyz_scheduler_args`: xyz 계열 lr 스케줄 함수.
- `max_radii2D`: 화면 공간 반지름 통계.
- `xyz_gradient_accum`, `denom`: densify 통계용 누적값.

### 3.4 활성화 함수 핸들
`setup_functions()`에서 설정:
- `scaling_activation = exp`
- `opacity_activation = sigmoid`
- `rotation_activation = normalize`
- `covariance_activation = build_covariance_from_scaling_rotation`

## 4. 메소드 정리

### 4.1 초기화/복원
- `__init__(sh_degree)`: 빈 텐서/기본 상태 초기화.
- `setup_functions()`: 활성화/변환 함수 바인딩.
- `capture()`: 체크포인트 저장용 상태 패키징.
- `restore(model_args, training_args)`: `capture()` 역복원 + optimizer state 복구.

### 4.2 읽기용 프로퍼티/유틸
- `get_xyz`, `get_offset`, `get_features`, `get_scaling`, `get_rotation`, `get_opacity`
- `get_covariance()`, `get_covariance_frozen()`
- `oneupSHdegree()`: SH degree 증가.

### 4.3 파라미터 생성/학습 세팅
- `create_from_pcd(pcd, spatial_lr_scale)`: 초기 포인트클라우드에서 파라미터 생성.
- `training_setup(training_args)`: optimizer param group(`xyz`, `offset`, `f_dc`, `f_rest`, `opacity`, `scaling`, `rotation`) 구성.
- `update_learning_rate(iteration)`: 현재 코드는 `xyz` group lr만 스케줄 업데이트.
- `gaussian_training()`: Gaussian 파라미터 학습 활성화.
- `gaussian_frozen()`: 대부분 freeze, `offset`만 학습 허용.

### 4.4 저장/로드
- `construct_list_of_attributes()`: PLY attribute 목록 생성.
- `save_ply(path)`: 현재 파라미터를 PLY로 저장.
- `load_ply(path)`: PLY에서 파라미터 복원 + opacity 조건 필터링.
- `reset_opacity()`: opacity 텐서를 optimizer state 유지하며 교체.

### 4.5 optimizer 텐서 교체/토폴로지 변경
- `replace_tensor_to_optimizer(tensor, name)`: 특정 파라미터 텐서 교체.
- `_prune_optimizer(mask)`: mask 기반 파라미터/optimizer state 동시 프루닝.
- `prune_points(mask)`: 실제 멤버 파라미터를 프루닝 결과로 치환.
- `cat_tensors_to_optimizer(tensors_dict)`: 새 텐서를 기존 파라미터 끝에 concat.

### 4.6 densify 관련
- `densification_postfix(...)`
- `densify_and_split(...)`
- `densify_and_clone(...)`
- `densify_and_prune(...)`
- `add_densification_stats(...)`

## 5. 메소드-속성 동작 흐름

### 5.1 학습 시작(신규 학습) 흐름
1. `__init__()`  
2. `LargeScene`에서 `create_from_pcd()` 호출  
3. `training_setup()` 호출  
4. 학습 루프에서 렌더/로스 backward 후 optimizer step  
5. 저장 시 `save_ply()`, 체크포인트 시 `capture()`

핵심 속성 변화:
- `create_from_pcd()`가 `_xyz/_offset/_features/_scaling/_rotation/_opacity`를 실제 텐서로 채움.
- `training_setup()`가 `optimizer`, `xyz_scheduler_args`, 통계 텐서(`xyz_gradient_accum`, `denom`)를 초기화.

### 5.2 재개/파인튜닝 흐름
1. `__init__()`  
2. `LargeScene`에서 `load_ply()` 또는 pretrain PLY 로드  
3. `training_setup()`  
4. 체크포인트 있으면 `restore()`로 optimizer까지 복원  
5. 필요 시 `gaussian_training()`으로 학습 상태 전환

### 5.3 렌더링 시 흐름
1. `__init__()`  
2. `LargeScene(..., load_iteration=...)`에서 `load_ply()`  
3. 렌더러가 `get_xyz/get_offset/get_features/get_scaling/get_rotation/get_opacity`를 읽어 렌더 수행

특히 `render_mix`에서 중심은 `xyz + offset`으로 사용된다 (`gaussian_renderer/__init__.py:84`).

## 6. 현재 MixGS 학습 경로에서의 실제 사용 포인트
- 직접 사용되는 핵심 메소드:
  - `create_from_pcd` 또는 `load_ply`
  - `training_setup`, `gaussian_training`
  - `get_*` 프로퍼티 (렌더 입력)
  - `save_ply`, `capture`, `restore`
- `densify_*`, `add_densification_stats`, `reset_opacity`는 현재 `train_mixgs.py` 경로에서는 직접 호출되지 않는다.

## 7. 참고 메모
- `offset`은 별도 "pool 클래스"가 아니라 `GaussianModel` 내부 파라미터(`_offset`)로 관리된다.
- config의 `model_config.name`을 바꾸면 동일 인터페이스를 가진 다른 구현(예: `GaussianModelLOD`)으로 대체 가능하다.
