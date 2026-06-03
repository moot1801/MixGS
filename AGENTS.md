# Repository Guidelines

## Project Structure & Module Organization
- 루트 실행 진입점은 `train_mixgs.py`, `render_mixgs.py`, `metrics_mixgs.py`입니다.
- 핵심 구현 코드는 `scene/`, `gaussian_renderer/`, `utils/`, `arguments/`, `hashencoder/`에 있습니다.
- 장면별 설정 프리셋은 `config/*.yaml`(예: `config/rubble_mixgs.yaml`)에 위치합니다.
- 데이터 준비 스크립트는 `scripts/`, 카메라/데이터 변환 유틸리티는 `tools/`에 있습니다.
- 대용량 산출물은 `data/`(입력)와 `output/`(체크포인트/렌더 결과)에 두고, PR diff에는 포함하지 마세요.
- `submodules/`, `LargeLightGaussian/`, `SIBR_viewers/`는 외부/보조 구성 요소이므로 필요한 경우에만 수정합니다.

## Build, Test, and Development Commands
- `conda create -yn mixgs python=3.10 pip && conda activate mixgs`: 권장 개발 환경 생성.
- `pip install -r requirements.txt`: Python 의존성 설치.
- `python train_mixgs.py --config config/rubble_mixgs.yaml`: 학습 실행(설정에 따라 렌더링/평가 포함).
- `python render_mixgs.py --model_path output/rubble_mixgs/lightning_logs/version_0 --config config/rubble_mixgs.yaml --iteration 250000 --skip_train`: 체크포인트 기반 테스트 뷰 렌더링.
- `python metrics_mixgs.py --gt_paths <path>/gt --render_paths <path>/renders`: 렌더 결과의 SSIM/PSNR/LPIPS 계산.
- `bash scripts/data_proc_mill19_scratch.sh`: 데이터셋 전처리 파이프라인 예시.

## Coding Style & Naming Conventions
- Python 3.10 기준, 들여쓰기는 공백 4칸을 사용합니다.
- 네이밍은 기존 규칙을 따릅니다: 파일/함수/변수는 `snake_case`, 클래스는 `PascalCase`(예: `MixGSModel`).
- 설정 파일명은 소문자 + 장면 중심 패턴(`\<scene\>_mixgs.yaml`)을 유지합니다.
- 변경은 작고 국소적으로 유지하고, 대규모 리팩터링은 PR 목적이 명확할 때만 진행합니다.

## Testing Guidelines
- 루트에 별도 `tests/` 자동화 스위트가 없으므로 기능 실행 기반으로 검증합니다.
- 모델/렌더러 변경 시 `render_mixgs.py`와 `metrics_mixgs.py`로 최소 1회 체크포인트 렌더링 + 메트릭 평가를 수행하세요.
- PR 본문에는 실행 명령, 사용한 config, 핵심 지표(SSIM/PSNR/LPIPS)를 명시합니다.

## Commit & Pull Request Guidelines
- 최근 이력은 `Update README.md`, `update` 같은 짧은 명령형 메시지를 사용합니다.
- 권장 형식은 더 명확한 `<area>: <imperative summary>`입니다(예: `render: fix depth image save path`).
- PR에는 아래 항목을 포함하세요.
  - 변경 내용과 변경 이유
  - 재현 명령(사용 config 경로 포함)
  - 실행 환경(GPU, CUDA, PyTorch)
  - 출력 변경 시 `output/.../renders` 경로 또는 스크린샷
  
# AGENTS.md — Codex 작업 규칙 (MixGS 연구용 포크)

본 문서는 이 저장소에서 Codex(및 자동화 도구)가 수행하는 작업의 **언어/문서화/깃 워크플로/인증** 규칙을 정의한다.

---

## 1) 기본 언어 정책 (한국어 우선)

### 1.1 적용 범위
Codex가 생성/수정하는 산출물은 기본적으로 **한국어**로 작성한다.
- 프로젝트 문서: README, docs/*, codex/*, AGENTS.md 등
- GitHub 이슈/PR: 제목과 본문
- 커밋 메시지: 가능하면 한국어(단, 타입/스코프 접두어는 영어 허용)
- 릴리즈 노트/체인지로그(작성 시)

### 1.2 예외 (원문 유지)
다음은 번역/변형하지 않고 **원문 그대로 유지**한다.
- 명령어, 경로, 설정 키, CLI 플래그
- 에러 로그/스택트레이스/출력
- 코드(소스) 자체: 변수명/함수명/클래스명/공식 API 명칭
- 고유명사(라이브러리/프로젝트명)는 영문 유지, 필요 시 한국어 설명을 병기

---

## 2) 저장소 운영 방식 (연구용 포크 정책)

이 저장소는 “업스트림 기여” 목적이 아니라 **연구/실험을 위한 포크**로 운영한다.

### 2.1 PR 대상 원칙
- 기본 원칙: PR은 **이 포크 저장소 내부(origin)** 에서만 사용한다.
  - feature 브랜치 → main 으로 PR 생성/리뷰/병합
- 업스트림(upstream) PR은 기본적으로 만들지 않는다.
  - 단, 아래 2.2 예외 조건에서만 허용

### 2.2 PR을 꼭 남기기 위해 main을 revert하지 않는다
이미 main에 반영된 커밋을 “PR 기록용”으로 되돌렸다가 다시 넣는 작업(revert→재적용)은 기본적으로 하지 않는다.
- 예외: 팀 규정/감사 목적 등으로 PR 기록이 절대적으로 필요할 때만 수행

---

## 3) 이슈(Issue)와 PR(Pull Request) 역할 규칙

### 3.1 이슈(Issue)의 목적
- 문제 제기/버그 리포트/개선 요구/토론을 기록하는 공간
- 코드 변경이 없더라도 생성 가능
- “무엇을, 왜”를 명확히 남긴다

### 3.2 PR의 목적
- 실제 변경사항(diff)을 리뷰/검증/병합하기 위한 단위
- “어떻게 고쳤는지(변경)”를 남긴다
- 가능하면 관련 이슈와 연결(Fixes #n)

---

## 참고 자료 우선순위 (MixGS 논문)
- 이론적 배경, 알고리즘 설명, 용어 정의, 단계별 파이프라인 요약을 작성할 때는 저장소에 포함된 논문 PDF를 1차 근거로 사용한다:
  - `Holistic Large-Scale Scene Reconstruction via Mixed Gaussian Splatting.pdf`
- 논문 내용과 추정/개인 지식이 충돌하면 논문을 우선한다.
- 논문에 없는 내용은 “추가 추정/확장”임을 명시하고, 가능한 경우 공식 문서/논문 등 2차 출처를 함께 제시한다.
