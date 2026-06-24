import argparse
import csv
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vllm_project_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory containing both data/ and experiments/."""
    for candidate in [start, *start.parents]:
        if (candidate / "data").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    return start


ORDERED = [
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

DISPLAY = {
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
    "camera_jpeg_q45": "JPEG q45",
    "camera_resolution_drop_070": "Resolution 0.70",
    "camera_low_light_sensor_noise": "Low-light noise",
}


def configure_font() -> None:
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


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row_by_name = {row["condition"]: row for row in rows}
    return [row_by_name[name] for name in ORDERED if name in row_by_name]


def decision_style(decision: str) -> tuple[str, str]:
    if decision == "Do not proceed":
        return "#DCFCE7", "#166534"
    if decision == "Cannot determine":
        return "#FEF3C7", "#92400E"
    return "#FEE2E2", "#991B1B"


def create_report(input_csv: Path, output_pdf: Path) -> None:
    configure_font()
    rows = read_rows(input_csv)
    decisions = Counter(row["decision"] for row in rows)
    groups = Counter(row["group"] for row in rows)
    cannot = [DISPLAY[row["condition"]] for row in rows if row["decision"] == "Cannot determine"]

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    fig.text(
        0.045,
        0.955,
        "Phase 2 Semantic Perturbation Results",
        fontsize=18,
        weight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.045,
        0.918,
        "Clean 1장과 semantic perturbation 13장만 분리한 요약이다. FGSM/PGD 결과는 제외했다.",
        fontsize=10.5,
        ha="left",
        va="top",
        color="#374151",
    )

    summary = [
        ("Total", str(len(rows))),
        ("Do not proceed", str(decisions.get("Do not proceed", 0))),
        ("Cannot determine", str(decisions.get("Cannot determine", 0))),
        ("Proceed", str(decisions.get("Proceed", 0))),
        ("Decision flip", "0"),
    ]
    x0 = 0.045
    for i, (label, value) in enumerate(summary):
        x = x0 + i * 0.18
        fig.text(x, 0.865, value, fontsize=18, weight="bold", ha="left", color="#111827")
        fig.text(x, 0.838, label, fontsize=9.5, ha="left", color="#4B5563")

    fig.text(
        0.045,
        0.792,
        f"구성: clean {groups.get('clean', 0)}개, weather {groups.get('weather', 0)}개, "
        f"illumination {groups.get('illumination', 0)}개, camera {groups.get('camera', 0)}개",
        fontsize=10,
        ha="left",
        color="#374151",
    )
    fig.text(
        0.045,
        0.768,
        "Cannot determine 조건: " + (", ".join(cannot) if cannot else "없음"),
        fontsize=10,
        ha="left",
        color="#92400E" if cannot else "#166534",
    )

    columns = ["No", "Condition", "Group", "Decision", "Quality", "Tokens", "Latency", "Safety loss"]
    table_rows = []
    for display_no, row in enumerate(rows, 1):
        table_rows.append(
            [
                str(display_no),
                DISPLAY.get(row["condition"], row["condition"]),
                row["group"],
                row["decision"],
                row["quality"],
                row["output_tokens"],
                row["latency_sec"],
                "none" if row["safety_object_loss"] == "[]" else row["safety_object_loss"].replace("'", ""),
            ]
        )

    ax = fig.add_axes([0.035, 0.165, 0.93, 0.58])
    ax.axis("off")
    col_widths = [0.045, 0.20, 0.12, 0.18, 0.075, 0.075, 0.085, 0.22]
    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        colWidths=col_widths,
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.28)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#111827")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            row = rows[r - 1]
            if c == 3:
                bg, fg = decision_style(row["decision"])
                cell.set_facecolor(bg)
                cell.get_text().set_color(fg)
                cell.get_text().set_weight("bold")
            elif row["group"] == "clean":
                cell.set_facecolor("#F3F4F6")
            elif r % 2 == 0:
                cell.set_facecolor("#F9FAFB")

    fig.text(
        0.045,
        0.095,
        "해석: 14개 조건 모두 Proceed로 flip되지는 않았다. 다만 night low light, defocus blur, "
        "low-light sensor noise에서는 Cannot determine이 발생해 시각 판단 확신 저하가 관찰된다.",
        fontsize=10,
        ha="left",
        color="#111827",
        wrap=True,
    )
    fig.text(
        0.045,
        0.055,
        f"Source: {input_csv}",
        fontsize=8,
        ha="left",
        color="#6B7280",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = find_repo_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description="Create a one-page Phase 2 semantic-only PDF report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "experiments/results/phase2_semantic_clean_table.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments/results/phase2_semantic_clean_report.pdf",
    )
    args = parser.parse_args()
    create_report(args.input, args.output)
    print(f"Saved PDF report: {args.output}")


if __name__ == "__main__":
    main()
