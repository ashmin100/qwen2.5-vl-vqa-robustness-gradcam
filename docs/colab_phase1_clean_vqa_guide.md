# Colab Phase 1 Clean VQA 실행 가이드

이 문서는 `image1.png`, `image2.png` 등 여러 도로 이미지를 대상으로 TALE-EP 기반 프롬프트 예산 최적화 Clean VQA 실험을 Google Colab에서 실행하는 절차를 정리한다. 프롬프트 후보와 모델 답변은 영어로 작성한다.

## 1. 사용 파일

프로젝트 기준 경로:

```text
VLLM_Project/
├── data/raw/image1.png
├── experiments/phase1/phase1_clean_vqa.ipynb
├── experiments/phase1/phase1_clean_vqa.py
└── docs/01-plan/features/vlm-adversarial-robustness.plan.md
```

Colab에서 실행할 파일:

```text
experiments/phase1/phase1_clean_vqa.ipynb
```

Colab에 업로드할 이미지:

```text
image1.png
image2.png
image123.png
...
```

지원 확장자:

```text
.png, .jpg, .jpeg, .webp
```

Colab에서는 업로드한 이미지 중 지원 확장자에 해당하는 파일을 모두 인식한다. 로컬 자동 실행에서는 `data/raw/` 아래의 `image<number>` 형식 파일을 인식한다. 예: `image1.png`, `image2.png`, `image123.png`.

## 2. Colab 런타임 설정

Colab 상단 메뉴에서 다음 순서로 설정한다.

```text
Runtime > Change runtime type > Hardware accelerator > T4 GPU
```

권장 환경:

```text
GPU: T4
Model: Qwen/Qwen2.5-VL-3B-Instruct
Quantization: 4bit bitsandbytes
```

GPU가 잡혔는지는 첫 번째 코드 셀에서 확인한다.

```text
device: cuda
```

`device: cpu`가 출력되면 런타임을 GPU로 다시 설정해야 한다.

## 3. 실행 순서

노트북을 위에서 아래로 순서대로 실행한다.

1. `[1] 환경 확인`
   - Colab 여부와 `cuda` 사용 가능 여부를 출력한다.
   - CUDA 메모리 단편화를 줄이기 위해 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 설정한다.

2. `[2] 패키지 설치`
   - Colab에서만 다음 패키지를 설치한다.

```text
transformers
accelerate
bitsandbytes
qwen-vl-utils
```

3. `[3] 이미지 업로드 및 추론용 리사이즈`
   - 업로드 창이 뜨면 프로젝트의 `image1.png`, `image2.png`, `image123.png` 등을 선택한다.
   - 여러 이미지를 한 번에 업로드할 수 있다.
   - 로컬 실행 시에는 `data/raw/` 아래의 `image<number>` 파일을 자동 인식한다.
   - 원본 이미지를 그대로 VLM에 넣으면 T4에서 OOM이 발생할 수 있으므로, 노트북은 각 이미지마다 `{원본이름}_infer.png`를 생성해 긴 변을 896px로 줄인다.

4. `[4] 모델 로드`
   - Colab GPU에서는 기본적으로 `Qwen/Qwen2.5-VL-3B-Instruct`를 4bit로 로드한다.
   - 7B를 꼭 써야 하면 `[4] 모델 로드` 셀의 `MODEL_ID`를 `Qwen/Qwen2.5-VL-7B-Instruct`로 바꾼다.
   - CPU fallback은 가능하지만 매우 느리므로 실험용으로 권장하지 않는다.

5. `[5] 프롬프트/TALE-EP 설정`
   - 프롬프트별 출력 예산을 사전 예측한다.
   - 예측 예산을 프롬프트 내부 제약과 `max_new_tokens`에 동시에 적용한다.
   - 프롬프트 후보를 수정하려면 이 셀의 `PROMPT_CANDIDATES` 값을 바꾼다.
   - 각 후보의 `prompt`는 영어로 작성하고, 출력 형식도 영어로 지정한다.
   - 토큰 예산 후보를 바꾸려면 같은 셀의 `TALE_EP_BUDGET_CANDIDATES` 값을 바꾼다.

6. `[6] 추론 함수`
   - 각 이미지의 `{원본이름}_infer.png`와 프롬프트를 모델 입력으로 변환한다.

7. `[7] 예산별 실행`
   - 각 이미지마다 `PROMPT_CANDIDATES x TALE_EP_BUDGET_CANDIDATES` 조합을 실행한다.
   - 각 결과에 대해 입력 토큰 수, 예상 출력 토큰 수, 실제 출력 토큰 수, 총 토큰 수, latency를 출력한다.

8. `[8] 결과 저장`
   - Colab에서는 결과 JSON을 자동 다운로드한다.

## 4. 결과 파일

Colab 다운로드 파일:

```text
phase1_prompt_optimization_results.json
```

로컬에서 같은 노트북을 실행할 경우 저장 경로:

```text
experiments/results/phase1_prompt_optimization_results.json
```

결과 JSON 구조:

```json
{
  "images": [
    {
      "original": "image1.png",
      "infer": "image1_infer.png",
      "stem": "image1",
      "size": [896, 562]
    }
  ],
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "device": "cuda",
  "method": "TALE-EP prompt candidate optimization",
  "prompt_candidates": [],
  "budget_candidates": [64, 80, 96, 120, 144],
  "manual_label": {
    "label": "",
    "safety_objects": [],
    "notes": ""
  },
  "results": {
    "image1": {
      "prompt_01_budget_64": {
        "output": "...",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_sec": 0,
        "predicted_output_tokens": 0,
        "token_budget_error": 0,
        "prompt": "...",
        "final_prompt": "...",
        "quality": {
          "has_decision": true,
          "has_reason": true,
          "has_impact": true,
          "object_mentions": [],
          "hallucination_risk": false,
          "has_non_english": false,
          "quality_score": 6
        }
      }
    }
  }
}
```

## 5. 수동 라벨 기록

노트북 마지막 셀의 `manual` 값을 실험 결과 확인 후 채운다.

```python
manual = {
    "label": "Proceed",
    "safety_objects": ["green traffic light", "vehicles moving forward"],
    "notes": "image1 may be interpreted as a proceed-straight scene because a green signal and moving vehicles are visible",
}
```

연구 계획의 공격 목표가 `직진 불가 -> 직진 가능`이면 `image1`이 적절한지 다시 확인해야 한다. `image1`은 야간 도심 장면에서 초록 신호가 보이므로 `직진 가능` 레이블이 될 가능성이 있다.

## 6. 문제 해결

`bitsandbytes` 관련 오류:

```text
Runtime > Restart runtime
```

후 설치 셀부터 다시 실행한다.

CUDA 메모리 부족:

```text
Runtime > Disconnect and delete runtime
Runtime > Change runtime type > T4 GPU
```

이후 `[1] 환경 확인` 셀부터 다시 실행한다. 기존 런타임에서 OOM이 한 번 발생하면 GPU 메모리가 깨끗하게 회수되지 않을 수 있으므로, 단순히 해당 셀만 재실행하지 말고 런타임을 재시작한다.

계속 OOM이 나면 다음 순서로 낮춘다.

```text
[3] MAX_IMAGE_SIDE = 768
[5] TALE_EP_BUDGET_CANDIDATES 값을 [48, 64, 80]으로 축소
```

7B 모델에서 OOM:

```text
[4] MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
```

T4에서는 3B 4bit를 기본값으로 사용한다.

업로드 파일명 오류:

```text
FileNotFoundError: No supported image files found. Use names like image1.png, image2.png, or image123.png.
```

지원 확장자(`.png`, `.jpg`, `.jpeg`, `.webp`)의 이미지 파일을 업로드한 뒤 이미지 업로드 셀부터 다시 실행한다.

## 7. 다음 단계

Phase 1 결과에서 clean 판단과 수동 라벨이 일치하는지 확인한다. 이후 Phase 2에서는 같은 이미지 또는 재선정한 `직진 불가` 이미지에 대해 CLIP surrogate 기반 FGSM/PGD 공격을 생성하고, 공격 이미지 입력 시 VLM 판단 변화와 토큰 예산 변화를 비교한다.
