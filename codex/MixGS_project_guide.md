# MixGS 프로젝트 지도 (Depth 2~3)

## 1) 최상위 폴더 역할
- `scene`, `gaussian_renderer`, `hashencoder`, `utils`, `arguments`: 학습/렌더링 코어 로직(객체, 렌더러, 인코더, 유틸, 파라미터).
- `config`, `scripts`, `doc`, `tools`, `data`, `output`: 실험 설정/데이터 준비/보조 도구/입출력 경로.
- `submodules`, `SIBR_viewers`, `LargeLightGaussian`, `lpipsPyTorch`, `assets`: 외부 의존/뷰어/참고 구현/메트릭 모듈/정적 리소스.

## 2) 프로젝트 폴더 구조 지도 (Depth 2~3, 핵심 위주)
```text
MixGS/
├── train_mixgs.py
├── render_mixgs.py
├── metrics_mixgs.py
├── viewer.py
├── convert.py
├── convert_cam.py
├── config/
│   ├── rubble_mixgs.yaml
│   ├── building_mixgs.yaml
│   ├── residence_mixgs.yaml
│   └── sciart_mixgs.yaml
├── scene/
│   ├── __init__.py              # Scene, LargeScene
│   ├── gaussian_model.py        # GaussianModel/GaussianModelLOD
│   ├── mixgs_model.py           # MixGSModel (encoder+decoder 래퍼)
│   ├── network.py               # GSEncoder, GSDecoder
│   ├── datasets.py              # GSDataset, CacheDataLoader
│   └── viewer/
│       └── ui/
├── gaussian_renderer/
│   ├── __init__.py              # prefilter_voxel, render_mix, render_viewer
│   └── network_gui.py
├── hashencoder/
│   ├── hashgrid.py
│   └── src/
├── utils/
│   ├── general_utils.py         # parse_cfg, safe_state 등
│   ├── camera_utils.py
│   ├── image_utils.py
│   ├── loss_utils.py
│   └── ...
├── scripts/
│   ├── data_proc_mill19.sh
│   ├── data_proc_mill19_scratch.sh
│   ├── data_proc_us3d.sh
│   └── data_proc_us3d_scratch.sh
├── tools/
│   ├── transform_pt2txt.py
│   ├── copy_images.py
│   └── transform_json2txt_mc*.py
├── submodules/
│   ├── diff-gaussian-rasterization/
│   ├── diff-gaussian-rasterization_filter/
│   └── simple-knn/
├── data/
├── output/
├── SIBR_viewers/
└── LargeLightGaussian/
```

## 3) 핵심 엔트리포인트 파일
- `train_mixgs.py`: 메인 학습 엔트리포인트 (`--config`), 학습 중 주기적 평가/저장 포함.
- `render_mixgs.py`: 학습 결과 렌더링(RGB/depth) 엔트리포인트.
- `metrics_mixgs.py`: 렌더링 결과와 GT의 SSIM/PSNR/LPIPS 계산.
- `viewer.py`: 웹 기반 인터랙티브 뷰어 엔트리포인트.
- `convert.py`, `convert_cam.py`: COLMAP 기반 데이터/카메라 변환 파이프라인 엔트리포인트.

## 4) 데이터 흐름 (학습/렌더링)
1. `config/*.yaml` → `utils/general_utils.py::parse_cfg`로 `lp/model_params`, `op/optim_params`, `pp/pipeline_params` 생성.
2. `train_mixgs.py`에서 `GaussianModel` + `MixGSModel` + `LargeScene` + `GSDataset/CacheDataLoader` 초기화.
3. 배치별로 카메라/GT 로드 후 `prefilter_voxel`로 가시 가우시안 마스크 계산.
4. `MixGSModel.step()`이 hash-encoded spatial feature + pose/scale/rotation 입력으로 decoded Gaussian 속성 생성.
5. `render_mix`가 원본 Gaussian + decoded Gaussian을 혼합 렌더링.
6. L1 + DSSIM(코드상 `fused_ssim`) 손실 역전파 후 `gaussians.optimizer`와 `mixgs.optimizer` 동시 업데이트.
7. iteration 기준으로 `point_cloud/iteration_x/point_cloud.ply` + `decoder/iteration_x/decoder.pth` 저장.

렌더링(`render_mixgs.py`)은 저장된 point cloud/decoder를 로드해 `train|test|custom_test` 뷰에 대해 렌더 결과와 depth를 출력한다.

## 5) 객체 역할과 생성 흐름

### 핵심 객체 역할
- `LargeScene` (`scene/__init__.py`): 데이터셋 구조 인식(COLMAP/Blender), 카메라 목록/스케일, 초기 point cloud 로드.
- `GaussianModel` (`scene/gaussian_model.py`): 3D Gaussian 파라미터(위치/스케일/회전/불투명도/SH feature) 보관 및 최적화.
- `MixGSModel` (`scene/mixgs_model.py`): `GSEncoder` + `GSDecoder` 조합으로 Gaussian 보정량(색/회전/스케일/opacity) 예측.
- `GSDataset`, `CacheDataLoader` (`scene/datasets.py`): 카메라 메타+GT 이미지 공급 및 캐시.
- `prefilter_voxel`, `render_mix` (`gaussian_renderer/__init__.py`): 가시성 필터링 + 혼합 렌더링.

### 생성/초기화 순서 (train)
1. `train_mixgs.py`에서 `gaussians = GaussianModel(...)` 생성.
2. `mixgs = MixGSModel(hash_args, net_args)` 생성 후 `mixgs.train_setting(op)`.
3. `scene = LargeScene(lp, gaussians)` 생성 시 source path에서 카메라/포인트클라우드 로딩.
4. `gs_dataset = GSDataset(scene.getTrainCameras(), ...)` 및 `CacheDataLoader(...)`.
5. `gaussians.training_setup(op)`로 optimizer/lr schedule 구성.
6. 학습 루프에서 `prefilter_voxel` → `mixgs.step` → `render_mix` → loss/backward/step.

## 6) 실험 재현 커맨드

### 환경 준비
```bash
conda create -yn mixgs python=3.10 pip
conda activate mixgs
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install ninja git+https://github.com/hturki/tiny-cuda-nn.git@ht/res-grid#subdirectory=bindings/torch
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization_filter
pip install submodules/simple-knn
```

### 데이터 준비
```bash
# 사전 COLMAP 결과 사용
bash scripts/data_proc_mill19.sh
bash scripts/data_proc_us3d.sh

# scratch 재생성
bash scripts/data_proc_mill19_scratch.sh
bash scripts/data_proc_us3d_scratch.sh
```

### 학습
```bash
python train_mixgs.py --config config/rubble_mixgs.yaml
python train_mixgs.py --config config/building_mixgs.yaml
python train_mixgs.py --config config/residence_mixgs.yaml
python train_mixgs.py --config config/sciart_mixgs.yaml
```

### 렌더링
```bash
python render_mixgs.py --config config/rubble_mixgs.yaml --model_path output/rubble_mixgs/lightning_logs/version_0 --iteration 250000 --skip_train
```

### 정량 평가
```bash
python metrics_mixgs.py \
  --gt_paths output/rubble_mixgs/lightning_logs/version_0/test/ours_250000/gt \
  --render_paths output/rubble_mixgs/lightning_logs/version_0/test/ours_250000/renders
```

### 뷰어
```bash
python viewer.py output/rubble_mixgs/lightning_logs/version_0
```

## 7) 메모 (실행 시 흔한 주의점)
- `render_mixgs.py`의 `--model_path`는 실제 로그 버전(`version_0`, `version_1`, ...)에 맞춰야 한다.
- `config/*.yaml`의 `source_path`/`pretrain_path`가 로컬 데이터 경로와 일치해야 학습이 시작된다.
- CUDA/torch/submodule 빌드 버전이 맞지 않으면 rasterization 관련 import 단계에서 실패할 수 있다.
