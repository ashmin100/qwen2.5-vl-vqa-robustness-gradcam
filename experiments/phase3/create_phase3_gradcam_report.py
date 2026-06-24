import argparse
import csv
import os
import unicodedata
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vllm_project_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from PIL import Image


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory containing both data/ and experiments/."""
    for candidate in [start, *start.parents]:
        if (candidate / "data").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    return start


ORDERED_NAMES = [
    "clean",
    "weather_fog_mild",
    "weather_fog_dense",
    "weather_rain_streaks",
    "weather_snow_particles",
    "weather_dust_haze",
    "illumination_sun_glare",
    "illumination_night_low_light",
    "camera_motion_blur",
    "camera_defocus_blur",
    "camera_windshield_droplets",
    "camera_jpeg_q45",
    "camera_resolution_drop_070",
    "camera_low_light_sensor_noise",
]

CATEGORY_NAMES = {
    "clean": "Clean baseline",
    "weather": "Weather perturbations",
    "illumination": "Illumination perturbations",
    "camera": "Camera perturbations",
}

DISPLAY_NAMES = {
    "clean": "Clean",
    "weather_fog_mild": "Mild fog",
    "weather_fog_dense": "Dense fog",
    "weather_rain_streaks": "Rain streaks",
    "weather_snow_particles": "Snow particles",
    "weather_dust_haze": "Dust haze",
    "illumination_sun_glare": "Sun glare",
    "illumination_night_low_light": "Night low light",
    "camera_motion_blur": "Motion blur",
    "camera_defocus_blur": "Defocus blur",
    "camera_windshield_droplets": "Windshield droplets",
    "camera_jpeg_q45": "JPEG quality 45",
    "camera_resolution_drop_070": "Resolution drop 0.70",
    "camera_low_light_sensor_noise": "Low-light sensor noise",
}


def configure_font():
    candidates = [
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Arial Unicode MS",
        "Malgun Gothic",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_metrics(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    metrics = {}
    for row in rows:
        name = row["perturbation"]
        metrics[name] = {
            "perturbation": name,
            "category": row["category"],
            "target_text": row["target_text"],
            "target_logprob": float(row["target_logprob"]),
            "drift": float(row["drift"]) if row["drift"] else 0.0,
            "soar": float(row["soar"]),
            "soar_signal": float(row["soar_signal"]),
            "soar_pedestrian": float(row["soar_pedestrian"]),
            "soar_crosswalk": float(row["soar_crosswalk"]),
            "decision": row["decision"],
            "raw_output": row["raw_output"],
            "grid": row["grid"],
            "cam_valid": row["cam_valid"],
            "cam_valid_reason": row["cam_valid_reason"],
        }
    return metrics


def read_peak_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["rank"] = int(row["rank"])
        row["x_frac"] = float(row["x_frac"])
        row["y_frac"] = float(row["y_frac"])
        row["value"] = float(row["value"])
    return rows


def ordered_metrics(metrics: dict) -> list[dict]:
    return [metrics[name] for name in ORDERED_NAMES if name in metrics]


def display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def split_long_token(token: str, width: int) -> list[str]:
    chunks = []
    current = []
    current_width = 0
    for char in token:
        char_width = 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        if current and current_width + char_width > width:
            chunks.append("".join(current))
            current = [char]
            current_width = char_width
        else:
            current.append(char)
            current_width += char_width
    if current:
        chunks.append("".join(current))
    return chunks


def wrap_text(text: str, width: int = 92) -> list[str]:
    tokens = []
    for word in text.split():
        if display_width(word) > width:
            tokens.extend(split_long_token(word, width))
        else:
            tokens.append(word)
    lines = []
    current = []
    current_len = 0
    for word in tokens:
        word_len = display_width(word)
        next_len = current_len + word_len + (1 if current else 0)
        if current and next_len > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len = next_len
    if current:
        lines.append(" ".join(current))
    return lines


def pdf_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]):
    page_no = 1
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = 0.94
    ax.text(0.06, y, title, fontsize=21, weight="bold", va="top")
    y -= 0.075

    def new_page():
        nonlocal fig, ax, y, page_no
        pdf.savefig(fig)
        plt.close(fig)
        page_no += 1
        fig = plt.figure(figsize=(11.0, 8.5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = 0.94
        ax.text(0.06, y, f"{title} ({page_no})", fontsize=21, weight="bold", va="top")
        y -= 0.075

    for heading, body in sections:
        body_lines = wrap_text(body, 100)
        needed = 0.034 + len(body_lines) * 0.027 + 0.022
        if y - needed < 0.07:
            new_page()
        ax.text(0.06, y, heading, fontsize=13.5, weight="bold", va="top")
        y -= 0.034
        for line in body_lines:
            if y < 0.07:
                new_page()
            ax.text(0.075, y, line, fontsize=10.5, va="top")
            y -= 0.027
        y -= 0.022
    pdf.savefig(fig)
    plt.close(fig)


def pdf_image_page(pdf: PdfPages, image_path: Path, title: str, caption: str, image_height: float = 0.70):
    if not image_path.exists():
        return
    image = Image.open(image_path).convert("RGB")
    fig = plt.figure(figsize=(11.0, 8.5))
    title_ax = fig.add_axes([0.04, 0.92, 0.92, 0.055])
    title_ax.axis("off")
    title_ax.text(0, 0.55, title, fontsize=15.5, weight="bold", va="center")
    img_ax = fig.add_axes([0.04, 0.20, 0.92, image_height])
    img_ax.imshow(image)
    img_ax.axis("off")
    caption_ax = fig.add_axes([0.04, 0.04, 0.92, 0.145])
    caption_ax.axis("off")
    y = 0.96
    for line in wrap_text(caption, 128):
        caption_ax.text(0, y, line, fontsize=9.5, va="top")
        y -= 0.23
    pdf.savefig(fig)
    plt.close(fig)


def pdf_table_page(pdf: PdfPages, title: str, columns: list[str], rows: list[list[str]], footnote: str):
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.84])
    ax.axis("off")
    ax.text(0, 1.04, title, fontsize=15.5, weight="bold", va="bottom")
    table = ax.table(cellText=rows, colLabels=columns, loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.34)
    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row_idx == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f9fafb")
    ax.text(0, -0.055, footnote, fontsize=9.0, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def metric_rows(metrics: dict) -> list[list[str]]:
    rows = []
    for item in ordered_metrics(metrics):
        rows.append(
            [
                DISPLAY_NAMES.get(item["perturbation"], item["perturbation"]),
                CATEGORY_NAMES.get(item["category"], item["category"]),
                item["decision"],
                f"{item['drift']:.3f}",
                f"{item['soar']:.3f}",
                f"{item['soar_signal']:.3f}",
                f"{item['soar_pedestrian']:.3f}",
                f"{item['soar_crosswalk']:.3f}",
                f"{item['target_logprob']:.3f}",
                item["cam_valid_reason"],
            ]
        )
    return rows


def peak_summary_rows(metrics: dict, peak_rows: list[dict]) -> list[list[str]]:
    by_name = {}
    for row in peak_rows:
        if row["rank"] == 1:
            by_name[row["perturbation"]] = row
    rows = []
    for item in ordered_metrics(metrics):
        peak = by_name.get(item["perturbation"])
        if peak is None:
            rows.append([DISPLAY_NAMES.get(item["perturbation"], item["perturbation"]), "-", "-", "-", "-"])
            continue
        rows.append(
            [
                DISPLAY_NAMES.get(item["perturbation"], item["perturbation"]),
                f"{peak['x_frac']:.3f}",
                f"{peak['y_frac']:.3f}",
                f"{peak['value']:.3f}",
                "lower image" if peak["y_frac"] > 0.62 else "mid/upper image",
            ]
        )
    return rows


def build_findings(metrics: dict) -> list[tuple[str, str]]:
    perturbations = [m for m in ordered_metrics(metrics) if m["perturbation"] != "clean"]
    high_drift = sorted(perturbations, key=lambda m: m["drift"], reverse=True)[:3]
    low_soar = sorted(perturbations, key=lambda m: m["soar"])[:3]
    uncertain = [m for m in perturbations if m["decision"] == "cannot_determine"]
    clean = metrics["clean"]
    sections = [
        (
            "Clean baseline 결과",
            f"Clean 이미지의 decision은 {clean['decision']}이다. Clean SOAR는 {clean['soar']:.3f}이며, 이는 정규화된 Grad-CAM energy 중 약 {clean['soar'] * 100:.1f}%가 사람이 정의한 stop-cue box 내부에 위치한다는 뜻이다.",
        ),
        (
            "CAM drift가 가장 큰 사례",
            ", ".join(f"{DISPLAY_NAMES[m['perturbation']]} ({m['drift']:.3f})" for m in high_drift)
            + ". 이 perturbation들은 clean baseline 대비 decision-targeted heatmap 위치를 가장 크게 이동시킨 사례다.",
        ),
        (
            "SOAR가 가장 낮은 사례",
            ", ".join(f"{DISPLAY_NAMES[m['perturbation']]} ({m['soar']:.3f})" for m in low_soar)
            + ". 이 사례들은 heatmap이 직접적인 stop evidence에 가장 적게 집중된 경우다.",
        ),
        (
            "Decision uncertainty 사례",
            (
                "모델은 다음 조건에서 Cannot determine를 반환했다: "
                + ", ".join(DISPLAY_NAMES[m["perturbation"]] for m in uncertain)
                + ". 이는 Proceed로 잘못 판단한 실패는 아니지만, 시각적 판단 확신이 낮아졌다는 신호다."
                if uncertain
                else "Cannot determine를 만든 perturbation은 없었다. 전체 perturbation set에서 decision label이 안정적으로 유지되었다."
            ),
        ),
        (
            "종합 메트릭 해석",
            "가장 우려되는 경우는 high drift와 low SOAR가 동시에 나타나는 사례다. 이 경우 모델이 같은 safety decision을 유지하더라도 시각적 근거가 stop cue 밖으로 이동했을 가능성이 크다. 반대로 drift가 높아도 SOAR가 유지된다면 설명 위치는 이동했지만 여전히 관련 근거 안에 남아 있는 것으로 해석할 수 있다.",
        ),
    ]
    return sections


def metric_interpretation_sections() -> list[tuple[str, str]]:
    return [
        (
            "그래프 색상의 의미",
            "정량 그래프에서 색상은 perturbation category를 구분한다. 검정색은 clean baseline, 파란색은 weather 계열, 주황색은 illumination 계열, 보라색은 camera 계열이다. 색상은 좋고 나쁨을 의미하지 않고, 어떤 perturbation 그룹에 속하는지 보여주는 범주 표시다.",
        ),
        (
            "x축 위치의 의미",
            "정량 그래프의 x축은 서로 다른 이미지 조건을 의미한다. 왼쪽부터 clean, weather perturbation 5종, illumination perturbation 2종, camera perturbation 6종 순서로 배치된다. 따라서 각 막대는 하나의 이미지 결과이고, clean을 기준으로 나머지 13개 perturbation을 비교한다.",
        ),
        (
            "점선과 X marker의 의미",
            "SOAR 그래프의 점선은 clean image의 SOAR 기준선이다. perturbation SOAR가 이 점선보다 낮으면 clean보다 stop-cue 집중도가 낮아졌다는 뜻이다. 그래프 위의 검은 X marker는 해당 이미지에서 모델 decision이 Do not proceed가 아니라 Cannot determine로 바뀐 경우를 표시한다.",
        ),
        (
            "SOAR는 왜 필요한가",
            "Grad-CAM heatmap은 빨간 영역이 어디에 있는지만 보여주기 때문에, 설명이 실제 안전 근거에 놓였는지 수치로 판단하기 어렵다. SOAR는 전체 CAM energy 중 stop-cue box 내부에 들어간 비율을 계산하여, heatmap이 빨간 신호등, 보행자, 횡단보도에 얼마나 정렬되어 있는지 정량화한다.",
        ),
        (
            "SOAR 해석 시 주의점",
            "SOAR가 높다는 것은 설명이 stop cue와 공간적으로 잘 겹친다는 뜻이지, 모델이 그 객체를 인과적으로 사용했다는 완전한 증명은 아니다. 또한 box를 사람이 정의했기 때문에 bbox 좌표 품질이 SOAR 값에 직접 영향을 준다. 그래서 bbox debug image와 qualitative overlay를 함께 확인해야 한다.",
        ),
        (
            "CAM drift는 왜 clean 기준인가",
            "Phase 3의 질문은 perturbation이 들어갔을 때 같은 장면에 대한 설명 위치가 얼마나 흔들리는지다. 따라서 clean image의 Grad-CAM을 기준점으로 두고, 각 perturbation CAM이 clean CAM에서 얼마나 멀어졌는지를 drift로 측정한다.",
        ),
        (
            "Drift가 높으면 항상 실패인가",
            "아니다. 조명이나 날씨가 바뀌면 heatmap 위치가 어느 정도 바뀔 수 있다. 중요한 것은 drift가 커진 뒤에도 heatmap이 stop cue 안에 남아 있는지다. 따라서 drift는 SOAR와 함께 봐야 한다. high drift + high SOAR는 설명 위치가 이동했지만 관련 근거에 남은 경우이고, high drift + low SOAR는 설명이 stop cue 밖으로 이탈했을 가능성이 큰 경우다.",
        ),
    ]


def diagnostic_interpretation_sections() -> list[tuple[str, str]]:
    return [
        (
            "Target Contrast Sanity의 목적",
            "이 검사는 본 실험의 주 결과가 아니라 Grad-CAM 방법이 target-specific하게 작동하는지 확인하기 위한 sanity check다. 기본 분석은 'Decision: Do not proceed' target으로 수행하지만, 반대 target인 'Decision: Proceed'에 대해서도 clean image CAM을 계산하여 두 heatmap이 실제로 달라지는지 확인한다.",
        ),
        (
            "Proceed target CAM은 무엇을 의미하는가",
            "Proceed target CAM은 모델이 '가도 된다'라는 답변을 만들 때 어떤 시각 위치가 해당 target score에 기여하는지 보는 검증용 map이다. 현재 도로 장면에서는 실제 안전 판단상 proceed가 적절하지 않으므로, 이 map은 최종 해석 결과가 아니라 Do not proceed CAM과 대비하기 위한 control 조건으로 사용한다.",
        ),
        (
            "두 target CAM이 같으면 왜 문제인가",
            "Do not proceed CAM과 Proceed CAM이 거의 동일하다면 heatmap이 특정 decision target의 근거를 설명하지 못하고 단순 saliency처럼 동작했을 가능성이 있다. 이 경우 hook layer, target score 설정, gradient 계산 방식, normalization 방식 등을 다시 점검해야 한다.",
        ),
        (
            "두 target CAM이 다르면 무엇을 확인한 것인가",
            "두 CAM이 다르게 나타나면 Grad-CAM이 적어도 target 변화에 반응하고 있음을 확인할 수 있다. 이는 이후 clean과 perturbation의 Do not proceed CAM을 비교할 때, heatmap이 특정 decision target에 대한 설명이라는 기본 전제를 보강한다.",
        ),
        (
            "Change map의 의미",
            "Change map은 perturbation CAM에서 clean CAM을 뺀 시각화다. 빨간 영역은 clean 대비 CAM energy가 증가한 위치이고, 파란 영역은 감소한 위치다. 단, 증가한 위치가 반드시 좋은 근거라는 뜻은 아니다. 그래서 change map은 SOAR, drift, input-vs-overlay pair와 함께 읽어야 한다.",
        ),
        (
            "CAM peak debug의 의미",
            "Peak debug는 Grad-CAM에서 가장 강한 좌표가 어디인지 기록한다. aggregate SOAR가 괜찮아 보여도 peak가 반복적으로 이미지 경계, 검은 letterbox, 도로 하단, blur artifact 주변에 찍히면 heatmap 품질에 문제가 있을 수 있다. 반대로 peak가 stop cue 주변에 있으면 정성 해석을 보강한다.",
        ),
        (
            "Cannot determine 해석",
            "Cannot determine은 Proceed로 잘못 판단한 것은 아니지만, perturbation으로 인해 모델의 시각적 판단 확신이 낮아졌다는 신호다. 안전 관점에서는 decision flip만 보는 것이 부족하며, Cannot determine도 robustness degradation으로 기록해야 한다.",
        ),
    ]


def qualitative_interpretation_sections() -> list[tuple[str, str]]:
    return [
        (
            "Input-vs-overlay pair를 읽는 법",
            "각 pair의 왼쪽은 실제 입력 이미지이고 오른쪽은 같은 이미지 위에 Grad-CAM을 overlay한 결과다. 오른쪽에서 붉은색에 가까운 영역일수록 고정 target인 'Decision: Do not proceed' score에 더 강하게 기여한 영역으로 해석한다.",
        ),
        (
            "좋은 heatmap의 조건",
            "이 장면에서 좋은 heatmap은 빨간 신호등, 횡단 중인 보행자, 좌측 보행자, 횡단보도 주변에 집중되어야 한다. 이들은 차량이 멈춰야 한다는 직접 근거이기 때문이다.",
        ),
        (
            "주의해야 할 heatmap의 조건",
            "heatmap이 도로 하단 질감, 건물 배경, 이미지 경계, black letterbox, blur나 noise artifact에 집중되면 설명 신뢰성이 낮다. 이런 경우 답변이 Do not proceed로 맞더라도 모델이 올바른 근거를 사용했다고 보기 어렵다.",
        ),
        (
            "정성 결과와 정량 결과의 관계",
            "정성 overlay는 heatmap이 실제로 어디에 있는지 눈으로 확인하게 해주고, SOAR와 drift는 그 위치를 수치화한다. 따라서 최종 해석은 이미지와 그래프 중 하나만 보고 내리지 말고 두 결과를 함께 사용해야 한다.",
        ),
    ]


def pair_caption(item: dict) -> str:
    return (
        f"Decision={item['decision']}; CAM drift={item['drift']:.3f}; SOAR={item['soar']:.3f}; "
        f"signal={item['soar_signal']:.3f}, pedestrian={item['soar_pedestrian']:.3f}, crosswalk={item['soar_crosswalk']:.3f}. "
        "왼쪽은 input image, 오른쪽은 고정 target 'Decision: Do not proceed'에 대한 Grad-CAM overlay다. 붉은색에 가까울수록 해당 target에 대한 CAM energy가 강한 영역이다."
    )


def create_report(results_dir: Path, output_path: Path):
    metrics = read_metrics(results_dir / "phase3_gradcam_metrics.csv")
    peaks = read_peak_rows(results_dir / "cam_peak_debug.csv")

    with PdfPages(output_path) as pdf:
        pdf_text_page(
            pdf,
            "Phase 3 Grad-CAM 분석 보고서",
            [
                (
                    "연구 질문",
                    "Phase 3의 목적은 동일한 도로 장면에 semantic perturbation을 적용했을 때, VLM의 안전 판단 근거가 실제 멈춤 근거에 유지되는지 확인하는 것이다. 분석 대상은 clean 이미지 1장과 Phase 2에서 생성한 perturbation 이미지 13장이다.",
                ),
                (
                    "실험 설계",
                    "각 이미지에 대해 모델 답변을 수집하고, 고정 target인 'Decision: Do not proceed'에 대한 decision-targeted Grad-CAM을 생성했다. target을 고정한 이유는 clean 이미지와 perturbation 이미지의 heatmap을 같은 안전 판단 기준에서 비교하기 위해서다.",
                ),
                (
                    "결과를 읽는 기준",
                    "좋은 결과라면 빨간 heatmap 영역이 빨간 신호등, 보행자, 횡단보도 주변에 유지되어야 한다. 반대로 모델 답변은 Do not proceed로 유지되더라도 heatmap이 도로 질감, 이미지 경계, blur artifact, 무관한 배경으로 이동하면 설명 신뢰성은 낮게 해석해야 한다.",
                ),
            ],
        )
        pdf_text_page(
            pdf,
            "본 보고서에서 사용한 메트릭",
            [
                (
                    "SOAR score",
                    "SOAR는 Stop-cue Object Attention Ratio의 약자다. 전체 Grad-CAM energy 중 사람이 정의한 stop-cue box 내부에 들어간 비율을 의미한다. 여기서 stop-cue는 차량이 멈춰야 한다는 직접 근거인 빨간 신호등, 보행자, 횡단보도만 포함한다. SOAR가 높을수록 모델 설명이 실제 안전 판단 근거와 더 잘 정렬되어 있다고 해석한다.",
                ),
                (
                    "SOAR group decomposition",
                    "SOAR energy를 signal, pedestrian, crosswalk 그룹으로 나누어 본 지표다. 이를 통해 모델 설명이 빨간 신호등, 보행자, 횡단보도 중 어떤 근거 유형에 주로 의존하는지 확인할 수 있다.",
                ),
                (
                    "CAM drift",
                    "CAM drift는 각 perturbation Grad-CAM과 clean Grad-CAM 사이의 공간적 차이를 의미한다. 값이 높을수록 perturbation으로 인해 모델의 시각적 판단 근거 위치가 많이 이동했다는 뜻이다.",
                ),
                (
                    "Decision consistency and CAM validity",
                    "Decision consistency는 perturbation 이후에도 모델 답변이 Do not proceed로 유지되는지, 또는 Cannot determine로 바뀌는지 확인하는 항목이다. CAM validity는 all-zero, uniform, NaN, Inf heatmap을 해석 전에 걸러내기 위한 유효성 검사다.",
                ),
                (
                    "Peak location",
                    "Peak location은 Grad-CAM에서 가장 강한 좌표를 기록한 것이다. SOAR가 어느 정도 높게 나와도 peak가 반복적으로 이미지 경계나 무관한 하단 영역에 찍히면 heatmap 품질을 의심해야 한다.",
                ),
            ],
        )
        pdf_text_page(pdf, "메트릭 해석 방법", metric_interpretation_sections())
        pdf_image_page(
            pdf,
            results_dir / "bbox_debug_clean.png",
            "SOAR 계산을 위한 Stop-Cue Evidence Box",
            "이 box들은 일반 object detection box가 아니라 SOAR 계산을 위한 stop-cue evidence box다. 즉 차량이 멈춰야 하는 직접 근거인 빨간 신호등, 보행자, 횡단보도만 포함한다. 버스, 일반 차량 흐름, 배경 건물은 직접적인 stop cue가 아니므로 SOAR 계산에서 제외했다.",
        )
        pdf_table_page(
            pdf,
            "Clean + Semantic Perturbation 13종 메트릭 결과",
            ["Image", "Category", "Decision", "Drift", "SOAR", "Signal", "Ped.", "Crosswalk", "LogProb", "CAM"],
            metric_rows(metrics),
            "Drift는 clean 기준 상대 이동량이다. SOAR와 group column은 Grad-CAM energy 비율이며, 값이 높을수록 heatmap이 stop-cue evidence에 더 잘 정렬되어 있음을 의미한다.",
        )
        pdf_image_page(
            pdf,
            results_dir / "phase3_gradcam_metric_bars.png",
            "메트릭 결과: CAM Drift와 SOAR",
            "색상은 perturbation category를 의미한다: 검정색은 clean, 파란색은 weather, 주황색은 illumination, 보라색은 camera 계열이다. x축의 각 막대는 clean과 13개 perturbation 조건을 순서대로 나타낸다. 위 그래프는 각 perturbation heatmap이 clean에서 얼마나 이동했는지 보여주고, 아래 그래프는 heatmap energy가 stop-cue box 안에 얼마나 남아 있는지 보여준다. 아래 그래프의 점선은 clean SOAR 기준선이며, 검은 X marker는 decision이 Cannot determine로 바뀐 사례다. drift가 높고 SOAR가 낮은 경우가 가장 주의해서 해석해야 할 사례다.",
        )
        pdf_image_page(
            pdf,
            results_dir / "phase3_gradcam_soar_grouped_bars.png",
            "메트릭 결과: Stop-Cue 유형별 SOAR",
            "x축의 각 막대는 clean과 13개 perturbation 조건을 의미한다. 이 그래프는 SOAR를 red signal, pedestrian, crosswalk evidence로 나눈 것이다. 막대 안의 구성 비율을 보면 모델 설명이 횡단보도, 사람, 신호등 중 어느 근거에 주로 집중되는지 확인할 수 있다. 특정 perturbation에서 전체 SOAR가 낮거나 특정 stop-cue group이 급격히 줄어들면, 해당 조건에서 모델 설명이 중요한 안전 근거를 놓쳤을 가능성이 있다.",
        )
        pdf_image_page(
            pdf,
            results_dir / "phase3_gradcam_drift_vs_soar.png",
            "메트릭 결과: Drift와 Stop-Cue 유지 정도",
            "각 점은 하나의 perturbation 조건을 의미하고, 색상은 perturbation category를 의미한다. x축은 clean 대비 CAM drift이고, y축은 SOAR다. 오른쪽 아래에 위치한 점은 heatmap이 clean에서 많이 이동했고 stop-cue 집중도도 낮아진 경우이므로 가장 우려되는 패턴이다. 오른쪽 위의 점은 heatmap이 이동했지만 여전히 관련 stop cue 안에 남아 있는 경우로 해석한다.",
        )
        pdf_table_page(
            pdf,
            "CAM Peak Debug 요약",
            ["Image", "Top peak x", "Top peak y", "Peak value", "Coarse location"],
            peak_summary_rows(metrics, peaks),
            "좌표는 이미지 크기 기준으로 정규화된 비율이다. 이 표는 Grad-CAM peak가 반복적으로 이미지 경계나 무관한 영역에 몰리는지 확인하기 위한 debug summary다.",
        )
        pdf_text_page(pdf, "Diagnostic Check 해석 방법", diagnostic_interpretation_sections())
        pdf_text_page(pdf, "현재 결과 해석", build_findings(metrics))
        pdf_text_page(pdf, "정성 결과 읽는 법", qualitative_interpretation_sections())
        pdf_image_page(
            pdf,
            results_dir / "pairs/clean_input_vs_gradcam.png",
            "정성 결과: Clean Baseline",
            pair_caption(metrics["clean"]),
        )
        pdf_image_page(
            pdf,
            results_dir / "phase3_gradcam_weather_perturbations.png",
            "정성 요약: Weather Perturbations",
            "Weather perturbation에 대한 category-level Grad-CAM overlay다. 개별 input-vs-overlay pair를 보기 전에 날씨 변화가 전체적으로 heatmap 위치에 어떤 영향을 주는지 비교하기 위한 페이지다.",
            image_height=0.74,
        )
        pdf_image_page(
            pdf,
            results_dir / "phase3_gradcam_illumination_perturbations.png",
            "정성 요약: Illumination Perturbations",
            "Illumination perturbation은 glare나 low light가 판단 근거를 바꾸는지 확인하기 위한 조건이다. 특히 low-light 계열은 모델의 판단 확신을 낮출 수 있으므로 주의해서 확인해야 한다.",
            image_height=0.74,
        )
        pdf_image_page(
            pdf,
            results_dir / "phase3_gradcam_camera_perturbations.png",
            "정성 요약: Camera Perturbations",
            "Camera perturbation은 blur, droplets, compression, resolution loss, sensor noise가 heatmap에 미치는 영향을 확인하기 위한 조건이다. 이런 변화는 Grad-CAM을 stop cue가 아니라 artifact나 texture 쪽으로 이동시킬 수 있다.",
            image_height=0.74,
        )
        pdf_image_page(
            pdf,
            results_dir / "phase3_gradcam_top5_drift_comparison.png",
            "정성 요약: Drift Top-5 Perturbations",
            "clean과 drift가 가장 큰 perturbation 5개를 함께 배치한 페이지다. 설명 위치가 가장 크게 흔들린 사례를 빠르게 확인하기 위한 요약 시각화다.",
            image_height=0.74,
        )
        pdf_image_page(
            pdf,
            results_dir / "phase3_gradcam_change_maps_vs_clean.png",
            "Diagnostic 결과: Clean 대비 Grad-CAM Change Map",
            "빨간 영역은 clean 대비 Grad-CAM energy가 증가한 곳이고, 파란 영역은 감소한 곳이다. 변화량만으로 새 근거가 올바른지 판단할 수 없으므로 SOAR와 함께 해석해야 한다.",
            image_height=0.74,
        )
        sanity_dir = results_dir / "target_contrast_sanity"
        pdf_image_page(
            pdf,
            sanity_dir / "clean_do_not_proceed_overlay.png",
            "Target Contrast Sanity: Do Not Proceed Target",
            "clean 이미지에서 safety target에 대한 Grad-CAM이다. Proceed target 결과와 비교하여 target 선택에 따라 explanation이 실제로 달라지는지 확인한다.",
        )
        pdf_image_page(
            pdf,
            sanity_dir / "clean_proceed_overlay.png",
            "Target Contrast Sanity: Proceed Target",
            "clean 이미지에서 반대 target에 대한 Grad-CAM이다. 이 결과가 Do Not Proceed map과 거의 동일하다면 target-specific explanation의 신뢰성이 낮다고 봐야 한다.",
        )
        pdf_text_page(
            pdf,
            "Appendix: 전체 Input-vs-Grad-CAM Pair",
            [
                (
                    "Appendix 읽는 법",
                    "이후 각 페이지는 Phase 3 이미지 한 쌍을 보여준다. 왼쪽은 input image이고 오른쪽은 고정 target인 Do not proceed에 대한 Grad-CAM overlay다. clean 이미지와 semantic perturbation 13장이 모두 포함된다.",
                )
            ],
        )
        for name in ORDERED_NAMES:
            if name not in metrics:
                continue
            item = metrics[name]
            pdf_image_page(
                pdf,
                results_dir / "pairs" / f"{name}_input_vs_gradcam.png",
                f"Appendix Pair: {DISPLAY_NAMES.get(name, name)}",
                pair_caption(item),
            )
        pdf_text_page(
            pdf,
            "결론",
            [
                (
                    "핵심 해석",
                    "최종 decision label과 Grad-CAM explanation은 분리해서 해석해야 한다. Do not proceed라는 답변이 맞더라도 SOAR가 높고 CAM peak가 stop cue 근처에 남아 있을 때 설명 신뢰성이 더 높다.",
                ),
                (
                    "보고해야 할 실패 패턴",
                    "가장 중요한 explanation failure pattern은 높은 CAM drift와 낮은 SOAR가 동시에 나타나거나, decision이 Cannot determine로 바뀌는 경우다. 이는 perturbation이 시각적 근거를 stop cue 밖으로 이동시켰거나 모델의 판단 확신을 낮췄다는 의미다.",
                ),
                (
                    "해석 범위",
                    "이 실험은 single-scene perturbation study다. 따라서 Phase 3의 정성적, metric 기반 explanation analysis로 해석해야 하며, dataset-level statistical robustness claim으로 과장해서 설명하면 안 된다.",
                ),
            ],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Create a PDF report from saved Phase 3 Grad-CAM outputs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=find_repo_root(Path(__file__).resolve()) / "experiments/results/phase3_gradcam",
        help="Directory containing Phase 3 Grad-CAM outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to <results-dir>/phase3_gradcam_report.pdf.",
    )
    return parser.parse_args()


def main():
    configure_font()
    args = parse_args()
    output = args.output or args.results_dir / "phase3_gradcam_report.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    create_report(args.results_dir, output)
    print(f"Saved PDF report: {output}")


if __name__ == "__main__":
    main()
