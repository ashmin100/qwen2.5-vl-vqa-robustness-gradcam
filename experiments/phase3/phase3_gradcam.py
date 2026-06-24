import argparse
import csv
import gc
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vllm_project_matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_IMAGE_SIDE = 896
CMAP = plt.cm.jet
OVERLAY_ALPHA = 0.45

BEST_PROMPT = (
    "Look at the road image and judge only whether the vehicle may proceed straight.\n\n"
    "Output:\n"
    "Decision: [Proceed / Do not proceed / Cannot determine]\n"
    "Evidence: Observation -> Impact\n\n"
    "Rules:\n"
    "- Use only what is visible in the image.\n"
    "- Avoid long explanations.\n"
    "- If uncertain, choose Cannot determine.\n\n"
)

SEMANTIC_PERTURBATIONS_13 = {
    "weather_fog_mild": "image2_weather_fog_mild.png",
    "weather_fog_dense": "image2_weather_fog_dense.png",
    "weather_rain_streaks": "image2_weather_rain_streaks.png",
    "weather_snow_particles": "image2_weather_snow_particles.png",
    "weather_dust_haze": "image2_weather_dust_haze.png",
    "illumination_sun_glare": "image2_illumination_sun_glare.png",
    "illumination_night_low_light": "image2_illumination_night_low_light.png",
    "camera_motion_blur": "image2_camera_motion_blur.png",
    "camera_defocus_blur": "image2_camera_defocus_blur.png",
    "camera_windshield_droplets": "image2_camera_windshield_droplets.png",
    "camera_jpeg_q45": "image2_camera_jpeg_q45.png",
    "camera_resolution_drop_070": "image2_camera_resolution_drop_070.png",
    "camera_low_light_sensor_noise": "image2_camera_low_light_sensor_noise.png",
}

CATEGORIES = {
    "weather": [
        "weather_fog_mild",
        "weather_fog_dense",
        "weather_rain_streaks",
        "weather_snow_particles",
        "weather_dust_haze",
    ],
    "illumination": [
        "illumination_sun_glare",
        "illumination_night_low_light",
    ],
    "camera": [
        "camera_motion_blur",
        "camera_defocus_blur",
        "camera_windshield_droplets",
        "camera_jpeg_q45",
        "camera_resolution_drop_070",
        "camera_low_light_sensor_noise",
    ],
}

SAFETY_EVIDENCE_BOXES = [
    {
        "id": "E1",
        "label": "left red traffic light",
        "group": "signal",
        "bbox": [0.304, 0.300, 0.345, 0.340],
    },
    {
        "id": "E2",
        "label": "center red traffic lights",
        "group": "signal",
        "bbox": [0.515, 0.382, 0.610, 0.425],
    },
    {
        "id": "E3",
        "label": "crossing pedestrian",
        "group": "pedestrian",
        "bbox": [0.495, 0.500, 0.585, 0.710],
        "label_pos": "bottom",
    },
    {
        "id": "E4",
        "label": "left pedestrian",
        "group": "pedestrian",
        "bbox": [0.305, 0.470, 0.365, 0.625],
    },
    {
        "id": "E5",
        "label": "crosswalk",
        "group": "crosswalk",
        "bbox": [0.340, 0.565, 0.820, 0.760],
    },
]

SAFETY_BBOX_FRACS = [box["bbox"] for box in SAFETY_EVIDENCE_BOXES]

EVIDENCE_COLORS = {
    "signal": "#f59e0b",
    "pedestrian": "#22c55e",
    "crosswalk": "#38bdf8",
}

CATEGORY_COLORS = {
    "clean": "#111827",
    "weather": "#2563eb",
    "illumination": "#f59e0b",
    "camera": "#7c3aed",
}

DECISION_TEXT = {
    "do_not_proceed": "Decision: Do not proceed",
    "cannot_determine": "Decision: Cannot determine",
    "proceed": "Decision: Proceed",
}

PEAK_TOP_K = 10


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested, but torch.backends.mps.is_available() is False in this Python environment."
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def normalize_map(cam: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    cam = np.nan_to_num(cam.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        valid = cam[mask > 0]
    else:
        valid = cam.reshape(-1)
    if valid.size == 0:
        return np.zeros_like(cam, dtype=np.float32)
    lo = float(valid.min())
    hi = float(valid.max())
    out = (cam - lo) / (hi - lo + 1e-8)
    out = np.clip(out, 0.0, 1.0)
    if mask is not None:
        out = out * (mask > 0)
    return out.astype(np.float32)


def artifact_mask(image_pil: Image.Image) -> np.ndarray:
    """Mask uniform black letterbox rows/cols without masking real dark road content."""
    arr = np.asarray(image_pil.convert("RGB")).astype(np.float32)
    gray = arr.mean(axis=2)
    h, w = gray.shape
    row_valid = np.ones(h, dtype=bool)
    col_valid = np.ones(w, dtype=bool)

    row_dark_uniform = (gray.mean(axis=1) < 12.0) & (gray.std(axis=1) < 8.0)
    col_dark_uniform = (gray.mean(axis=0) < 12.0) & (gray.std(axis=0) < 8.0)
    row_valid[row_dark_uniform] = False
    col_valid[col_dark_uniform] = False

    mask = np.outer(row_valid, col_valid).astype(np.float32)
    return mask


def resize_map(cam: np.ndarray, size: tuple[int, int], resample=Image.Resampling.BILINEAR) -> np.ndarray:
    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(cam_u8).resize(size, resample)).astype(np.float32) / 255.0


def overlay_heatmap(image_pil: Image.Image, cam: np.ndarray, mask_artifacts: bool) -> tuple[np.ndarray, np.ndarray]:
    w, h = image_pil.size
    mask = artifact_mask(image_pil) if mask_artifacts else None
    cam_norm = normalize_map(cam)
    cam_up = resize_map(cam_norm, (w, h))
    if mask is not None:
        cam_up = normalize_map(cam_up, mask)
    img = np.asarray(image_pil).astype(np.float32) / 255.0
    heat = CMAP(cam_up)[:, :, :3]
    overlay = np.clip((1.0 - OVERLAY_ALPHA) * img + OVERLAY_ALPHA * heat, 0.0, 1.0)
    if mask is not None:
        overlay = overlay * mask[:, :, None] + img * (1.0 - mask[:, :, None])
    return overlay, cam_up


def validate_cam(cam: np.ndarray) -> tuple[bool, str]:
    if not np.isfinite(cam).all():
        return False, "cam_has_nan_or_inf"
    cam_max = float(cam.max()) if cam.size else 0.0
    cam_min = float(cam.min()) if cam.size else 0.0
    if cam.size == 0:
        return False, "cam_empty"
    if cam_max <= 1e-12:
        return False, "cam_all_zero"
    if abs(cam_max - cam_min) <= 1e-12:
        return False, "cam_uniform"
    return True, "ok"


def parse_decision(text: str) -> str:
    lower = text.lower()
    if re.search(r"\bdo\s+not\s+proceed\b", lower):
        return "do_not_proceed"
    if re.search(r"\bcannot\s+determine\b", lower):
        return "cannot_determine"
    if re.search(r"\bproceed\b", lower):
        return "proceed"
    return "parse_fail"


def write_bbox_debug(image_pil: Image.Image, out_dir: Path):
    write_bbox_debug_plain(image_pil, out_dir)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.imshow(image_pil)
    w, h = image_pil.size
    legend_handles = {}
    for box in SAFETY_EVIDENCE_BOXES:
        x0f, y0f, x1f, y1f = box["bbox"]
        x0 = x0f * w
        y0 = y0f * h
        bw = (x1f - x0f) * w
        bh = (y1f - y0f) * h
        color = EVIDENCE_COLORS[box["group"]]
        rect = patches.Rectangle(
            (x0, y0),
            bw,
            bh,
            linewidth=1.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
            joinstyle="round",
        )
        ax.add_patch(rect)
        border = patches.Rectangle(
            (x0, y0),
            bw,
            bh,
            linewidth=1.8,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(border)
        label_y = y0 + bh + 12 if box.get("label_pos") == "bottom" else max(10, y0 - 4)
        ax.text(
            x0 + 3,
            min(h - 8, label_y),
            f"{box['id']}  {box['label']}",
            color="white",
            fontsize=8,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )
        if box["group"] not in legend_handles:
            legend_handles[box["group"]] = patches.Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.35,
                label=box["group"],
            )
    ax.legend(
        handles=list(legend_handles.values()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=9,
    )
    ax.set_title("Safety Evidence Boxes for SOAR (clean image)", fontsize=13, weight="bold", pad=10)
    ax.axis("off")
    out_path = out_dir / "bbox_debug_clean.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def write_bbox_debug_plain(image_pil: Image.Image, out_dir: Path):
    fig, ax = plt.subplots(figsize=(image_pil.size[0] / 100, image_pil.size[1] / 100), dpi=100)
    ax.imshow(image_pil)
    w, h = image_pil.size
    for box in SAFETY_EVIDENCE_BOXES:
        x0f, y0f, x1f, y1f = box["bbox"]
        x0 = x0f * w
        y0 = y0f * h
        bw = (x1f - x0f) * w
        bh = (y1f - y0f) * h
        color = EVIDENCE_COLORS[box["group"]]
        rect = patches.Rectangle((x0, y0), bw, bh, linewidth=2.0, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x0 + 2,
            max(10, y0 - 3),
            box["id"],
            color="white",
            fontsize=8,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": color, "edgecolor": "none", "alpha": 0.95},
        )
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    out_path = out_dir / "bbox_debug_clean_plain.png"
    plt.savefig(out_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}")


def extract_cam_peaks(name: str, image_pil: Image.Image, cam_up: np.ndarray, top_k: int) -> list[dict]:
    flat = cam_up.reshape(-1)
    if flat.size == 0:
        return []
    top_k = min(top_k, flat.size)
    indices = np.argpartition(-flat, top_k - 1)[:top_k]
    indices = indices[np.argsort(-flat[indices])]
    h, w = cam_up.shape
    rows = []
    for rank, idx in enumerate(indices, start=1):
        y = int(idx // w)
        x = int(idx % w)
        rows.append(
            {
                "perturbation": name,
                "rank": rank,
                "x_px": x,
                "y_px": y,
                "x_frac": round(x / max(w - 1, 1), 6),
                "y_frac": round(y / max(h - 1, 1), 6),
                "value": round(float(flat[idx]), 6),
                "image_width": image_pil.size[0],
                "image_height": image_pil.size[1],
            }
        )
    return rows


def write_peak_debug(peak_rows: list[dict], out_dir: Path):
    csv_path = out_dir / "cam_peak_debug.csv"
    fieldnames = ["perturbation", "rank", "x_px", "y_px", "x_frac", "y_frac", "value", "image_width", "image_height"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(peak_rows)


def write_method_notes(out_dir: Path, layer_name: str, mask_artifacts: bool, target_mode: str):
    notes_path = out_dir / "phase3_gradcam_method_notes.md"
    notes_path.write_text(
        "\n".join(
            [
                "# Phase 3 Grad-CAM Method Notes",
                "",
                "- Method: Grad-CAM-style decision-targeted visual explanation.",
                f"- Hook layer: `{layer_name}`.",
                "- Coordinate alignment: Qwen visual `pooler_output` is used by default because it is produced after window-token reverse indexing.",
                f"- Target mode: `{target_mode}`.",
                "- Default target: `Decision: Do not proceed`, fixed across clean and semantic perturbations for comparable drift scores.",
                f"- Artifact masking: `{mask_artifacts}` for uniform black letterbox rows/columns.",
                "- Outputs for PDF: category grids, clean-vs-top5 comparison, change maps, bbox debug image, peak CSV, validity flags.",
                "- Interpretation limit: this is Grad-CAM, not Attention Rollout; it is a post-hoc explanation for the selected decision target.",
                "",
                "Recommended quality checks before using figures:",
                "1. Inspect `bbox_debug_clean.png` to verify SOAR safety boxes.",
                "2. Inspect `overlays/clean_overlay.png` and `cam_peak_debug.csv` to verify peak locations are not dominated by borders.",
                "3. Run `--target-contrast-sanity` at least once and confirm `Proceed` and `Do not proceed` maps are not identical.",
                "4. Check `cam_valid` and `cam_valid_reason` in `phase3_gradcam_metrics.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved: {notes_path}")


def project_root() -> Path:
    # Walk up from the current working dir, then from this file, until a directory
    # containing both data/ and experiments/ is found, so the script works from anywhere.
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        for candidate in [start, *start.parents]:
            if (candidate / "data").is_dir() and (candidate / "experiments").is_dir():
                return candidate
    return Path.cwd().resolve()


def local_model_path(model_id: str) -> str:
    cache_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
    snapshots = sorted(cache_root.glob("*/")) if cache_root.exists() else []
    return str(snapshots[0]) if snapshots and model_id == MODEL_ID else model_id


def load_model(model_id: str, device: torch.device, local_files_only: bool):
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    path = local_model_path(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=local_files_only,
    ).to(device)
    processor = AutoProcessor.from_pretrained(path, local_files_only=local_files_only)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def prepare_inputs(processor, image_pil: Image.Image, prompt: str, device: torch.device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def infer_decision(model, processor, image_pil: Image.Image, prompt: str, device: torch.device) -> str:
    inputs = prepare_inputs(processor, image_pil, prompt, device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
    out_ids = generated[0][inputs["input_ids"].shape[1] :]
    return processor.decode(out_ids, skip_special_tokens=True).strip()


def get_visual_module(model):
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise RuntimeError("Could not find Qwen visual module on model.")


def select_gradcam_layer(model, layer_index: int):
    visual = get_visual_module(model)
    if layer_index == 0:
        return visual, "visual.pooler_output", 0
    if not hasattr(visual, "blocks"):
        raise RuntimeError("Qwen visual module does not expose .blocks; cannot register Grad-CAM hook.")
    blocks = visual.blocks
    if layer_index < 0:
        layer_index = len(blocks) + layer_index
    if layer_index < 0 or layer_index >= len(blocks):
        raise ValueError(f"Invalid layer_index={layer_index}; visual encoder has {len(blocks)} blocks.")
    return blocks[layer_index], f"visual.blocks[{layer_index}]", len(blocks)


def target_ids_for_text(processor, target_text: str, device: torch.device) -> torch.Tensor:
    tokenized = processor.tokenizer(target_text, add_special_tokens=False, return_tensors="pt")
    ids = tokenized["input_ids"][0]
    if ids.numel() == 0:
        raise ValueError(f"Target text produced no tokens: {target_text!r}")
    return ids.to(device)


def teacher_forced_logprob_score(model, base_inputs: dict, target_ids: torch.Tensor) -> torch.Tensor:
    input_ids = base_inputs["input_ids"]
    bsz = input_ids.shape[0]
    if bsz != 1:
        raise ValueError("This Grad-CAM script expects batch size 1.")

    target = target_ids.unsqueeze(0)
    full_input_ids = torch.cat([input_ids, target], dim=1)
    target_mask = torch.ones_like(target, device=input_ids.device)
    full_attention_mask = torch.cat([base_inputs["attention_mask"], target_mask], dim=1)

    forward_inputs = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "pixel_values": base_inputs["pixel_values"],
        "image_grid_thw": base_inputs["image_grid_thw"],
        "use_cache": False,
    }
    if "pixel_values_videos" in base_inputs:
        forward_inputs["pixel_values_videos"] = base_inputs["pixel_values_videos"]
    if "video_grid_thw" in base_inputs:
        forward_inputs["video_grid_thw"] = base_inputs["video_grid_thw"]

    outputs = model(**forward_inputs)
    prompt_len = int(input_ids.shape[1])
    logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + target_ids.numel(), :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    selected = log_probs[0, torch.arange(target_ids.numel(), device=target_ids.device), target_ids]
    return selected.mean()


class ActivationHook:
    def __init__(self, module):
        self.activation = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        if hasattr(output, "pooler_output"):
            tensor = output.pooler_output
        elif isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        self.activation = tensor
        tensor.retain_grad()

    def close(self):
        self.handle.remove()


def infer_spatial_shape(seq_len: int, grid_thw: torch.Tensor) -> tuple[int, int, int]:
    t = int(grid_thw[0, 0])
    h = int(grid_thw[0, 1])
    w = int(grid_thw[0, 2])
    raw = t * h * w
    if seq_len == raw:
        return t, h, w

    for merge in (2, 4):
        mh = h // merge
        mw = w // merge
        if h % merge == 0 and w % merge == 0 and seq_len == t * mh * mw:
            return t, mh, mw

    raise RuntimeError(
        f"Cannot reshape Grad-CAM tokens: seq_len={seq_len}, image_grid_thw={(t, h, w)}. "
        "Try a different --layer-index."
    )


def cam_path(out_dir: Path, name: str) -> Path:
    return out_dir / "cams" / f"{name}_gradcam.npy"


def overlay_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    overlay_dir = out_dir / "overlays"
    return overlay_dir / f"{name}_overlay.png", overlay_dir / f"{name}_raw_gradcam.png"


def compute_gradcam(
    model,
    processor,
    image_pil: Image.Image,
    prompt: str,
    target_text: str,
    layer,
    device: torch.device,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    inputs = prepare_inputs(processor, image_pil, prompt, device)
    inputs["pixel_values"] = inputs["pixel_values"].detach().clone()
    inputs["pixel_values"].requires_grad_(True)
    target_ids = target_ids_for_text(processor, target_text, device)

    model.zero_grad(set_to_none=True)
    hook = ActivationHook(layer)
    try:
        with torch.enable_grad():
            score = teacher_forced_logprob_score(model, inputs, target_ids)
            score.backward()

        activation = hook.activation
        if activation is None or activation.grad is None:
            raise RuntimeError("Grad-CAM hook did not capture activation gradients.")
        act = activation.detach().float()
        grad = activation.grad.detach().float()
    finally:
        hook.close()

    if act.dim() == 3 and act.shape[0] == 1:
        act = act[0]
        grad = grad[0]
    if act.dim() != 2:
        raise RuntimeError(f"Unexpected activation shape for Grad-CAM: {tuple(act.shape)}")

    weights = grad.mean(dim=0)
    cam_tokens = torch.relu((act * weights).sum(dim=-1))
    seq_len = int(cam_tokens.numel())
    t, h, w = infer_spatial_shape(seq_len, inputs["image_grid_thw"])
    cam = cam_tokens[: t * h * w].view(t, h, w).mean(dim=0)
    cam_np = cam.detach().cpu().numpy().astype(np.float32)
    return cam_np, (t, h, w), float(score.detach().cpu())


def attention_drift(clean_cam: np.ndarray, pert_cam: np.ndarray) -> float:
    clean = normalize_map(clean_cam)
    pert = normalize_map(pert_cam)
    if clean.shape != pert.shape:
        pert = resize_map(pert, (clean.shape[1], clean.shape[0]))
    return float(np.sqrt(((clean - pert) ** 2).mean()))


def soar(cam: np.ndarray, bboxes: list[list[float]]) -> float:
    cam_norm = normalize_map(cam)
    h, w = cam_norm.shape
    mask = np.zeros((h, w), dtype=bool)
    for x0f, y0f, x1f, y1f in bboxes:
        x0 = max(0, min(w, int(round(x0f * w))))
        x1 = max(0, min(w, int(round(x1f * w))))
        y0 = max(0, min(h, int(round(y0f * h))))
        y1 = max(0, min(h, int(round(y1f * h))))
        mask[y0:y1, x0:x1] = True
    total = float(cam_norm.sum()) + 1e-8
    return float(cam_norm[mask].sum() / total)


def soar_by_group(cam: np.ndarray) -> dict[str, float]:
    values = {}
    for group in EVIDENCE_COLORS:
        bboxes = [box["bbox"] for box in SAFETY_EVIDENCE_BOXES if box["group"] == group]
        values[f"soar_{group}"] = soar(cam, bboxes)
    return values


def make_category_figures(results: dict, metrics: dict, out_dir: Path, mask_artifacts: bool):
    for category, names in CATEGORIES.items():
        fig, axes = plt.subplots(len(names), 2, figsize=(13.6, 4.6 * len(names)))
        if len(names) == 1:
            axes = axes.reshape(1, 2)
        for row, name in enumerate(names):
            item = results[name]
            m = metrics[name]
            overlay, _ = overlay_heatmap(item["image_pil"], item["cam"], mask_artifacts)
            axes[row, 0].imshow(item["image_pil"])
            axes[row, 0].set_title(f"{name} - input", fontsize=11, weight="bold", pad=4)
            axes[row, 0].axis("off")
            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title(
                f"Grad-CAM overlay | Drift={m['drift']:.3f} | SOAR={m['soar']:.3f} | {m['decision']}",
                fontsize=10,
            )
            axes[row, 1].axis("off")

        fig.suptitle(
            f"Phase 3 Grad-CAM - {category.upper()} semantic perturbations\n"
            "Target: teacher-forced safety decision log-prob",
            fontsize=13,
        )
        plt.tight_layout(pad=0.35, rect=[0, 0, 1, 0.965])
        out_path = out_dir / f"phase3_gradcam_{category}_perturbations.png"
        plt.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        print(f"Saved: {out_path}")


def make_summary_figures(results: dict, metrics: dict, out_dir: Path, mask_artifacts: bool):
    sorted_names = sorted(
        [name for name in metrics if name != "clean"],
        key=lambda n: metrics[n]["drift"],
        reverse=True,
    )
    selected = ["clean"] + sorted_names[:5]
    fig, axes = plt.subplots(len(selected), 2, figsize=(13.6, 4.6 * len(selected)))
    for row, name in enumerate(selected):
        item = results[name]
        m = metrics[name]
        overlay, cam_up = overlay_heatmap(item["image_pil"], item["cam"], mask_artifacts)
        axes[row, 0].imshow(item["image_pil"])
        axes[row, 0].set_title(
            "[clean] input" if name == "clean" else f"{name} input",
            fontsize=11,
            weight="bold",
            pad=4,
        )
        axes[row, 0].axis("off")
        axes[row, 1].imshow(overlay)
        label = f"SOAR={m['soar']:.3f}" if name == "clean" else f"Drift={m['drift']:.3f} | SOAR={m['soar']:.3f}"
        axes[row, 1].set_title(f"Grad-CAM overlay | {label} | {m['decision']}", fontsize=10, pad=4)
        axes[row, 1].axis("off")
    fig.suptitle("Phase 3 Grad-CAM - clean vs top-5 drifted semantic perturbations", fontsize=13)
    plt.tight_layout(pad=0.35, rect=[0, 0, 1, 0.965])
    out_path = out_dir / "phase3_gradcam_top5_drift_comparison.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out_path}")


def ordered_metric_names(metrics: dict) -> list[str]:
    names = []
    for category in ("clean", "weather", "illumination", "camera"):
        if category == "clean" and "clean" in metrics:
            names.append("clean")
        elif category in CATEGORIES:
            names.extend([name for name in CATEGORIES[category] if name in metrics])
    return names


def short_label(name: str) -> str:
    replacements = {
        "weather_": "w_",
        "illumination_": "illum_",
        "camera_": "cam_",
        "_low_light": "_low",
        "_sensor_noise": "_noise",
        "_windshield_droplets": "_droplets",
        "_resolution_drop_070": "_resdrop",
        "_rain_streaks": "_rain",
        "_snow_particles": "_snow",
        "_dust_haze": "_dust",
    }
    label = name
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def decision_marker(decision: str) -> str:
    if decision == "do_not_proceed":
        return "o"
    if decision == "cannot_determine":
        return "X"
    if decision == "proceed":
        return "^"
    return "s"


def make_quantitative_figures(metrics: dict, out_dir: Path):
    names = ordered_metric_names(metrics)
    x = np.arange(len(names))
    colors = [CATEGORY_COLORS[metrics[name]["category"]] for name in names]
    labels = [short_label(name) for name in names]

    fig, axes = plt.subplots(2, 1, figsize=(13.6, 8.2), sharex=True)
    drift_values = [metrics[name]["drift"] if metrics[name]["drift"] is not None else 0.0 for name in names]
    soar_values = [metrics[name]["soar"] for name in names]
    axes[0].bar(x, drift_values, color=colors, alpha=0.86, edgecolor="#111827", linewidth=0.4)
    axes[0].set_ylabel("Grad-CAM Drift vs clean", fontsize=11)
    axes[0].set_title("Decision-targeted Grad-CAM drift under semantic perturbations", fontsize=13, weight="bold")
    axes[0].grid(axis="y", alpha=0.25)
    clean_soar = metrics["clean"]["soar"] if "clean" in metrics else None
    axes[1].bar(x, soar_values, color=colors, alpha=0.86, edgecolor="#111827", linewidth=0.4)
    if clean_soar is not None:
        axes[1].axhline(clean_soar, color="#111827", linestyle="--", linewidth=1.2, label="clean SOAR")
        axes[1].legend(frameon=False, loc="upper left")
    axes[1].set_ylabel("SOAR on stop cues", fontsize=11)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
    for ax in axes:
        for idx, name in enumerate(names):
            if metrics[name]["decision"] != "do_not_proceed":
                value = drift_values[idx] if ax is axes[0] else soar_values[idx]
                ymax = ax.get_ylim()[1]
                ax.scatter(
                    idx,
                    min(ymax * 0.96, value + ymax * 0.035),
                    marker=decision_marker(metrics[name]["decision"]),
                    color="black",
                    s=44,
                    zorder=5,
                )
    plt.tight_layout()
    out_path = out_dir / "phase3_gradcam_metric_bars.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out_path}")

    group_keys = ["soar_signal", "soar_pedestrian", "soar_crosswalk"]
    group_labels = ["signal", "pedestrian", "crosswalk"]
    group_colors = [EVIDENCE_COLORS["signal"], EVIDENCE_COLORS["pedestrian"], EVIDENCE_COLORS["crosswalk"]]
    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    bottom = np.zeros(len(names), dtype=np.float32)
    for key, label, color in zip(group_keys, group_labels, group_colors):
        vals = np.array([metrics[name][key] for name in names], dtype=np.float32)
        ax.bar(x, vals, bottom=bottom, color=color, alpha=0.82, edgecolor="#111827", linewidth=0.35, label=label)
        bottom += vals
    ax.set_title("SOAR decomposition by direct stop-cue group", fontsize=13, weight="bold")
    ax.set_ylabel("Attention ratio within stop-cue boxes", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    plt.tight_layout()
    out_path = out_dir / "phase3_gradcam_soar_grouped_bars.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out_path}")

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    for name in names:
        if name == "clean":
            continue
        m = metrics[name]
        ax.scatter(
            m["drift"],
            m["soar"],
            s=95,
            color=CATEGORY_COLORS[m["category"]],
            marker=decision_marker(m["decision"]),
            edgecolor="#111827",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.text(m["drift"] + 0.002, m["soar"], short_label(name), fontsize=8, va="center")
    if clean_soar is not None:
        ax.axhline(clean_soar, color="#111827", linestyle="--", linewidth=1.1, label="clean SOAR")
    ax.set_xlabel("Grad-CAM Drift vs clean", fontsize=11)
    ax.set_ylabel("SOAR on stop cues", fontsize=11)
    ax.set_title("Attention drift vs stop-cue attention retention", fontsize=13, weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    out_path = out_dir / "phase3_gradcam_drift_vs_soar.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out_path}")


def top_metric_names(metrics: dict, key: str, n: int = 3, reverse: bool = True) -> list[str]:
    names = [name for name in metrics if name != "clean"]
    return sorted(names, key=lambda name: metrics[name][key], reverse=reverse)[:n]


def wrapped_lines(text: str, width: int = 96) -> list[str]:
    words = text.split()
    lines = []
    line = []
    size = 0
    for word in words:
        if size + len(word) + len(line) > width and line:
            lines.append(" ".join(line))
            line = [word]
            size = len(word)
        else:
            line.append(word)
            size += len(word)
    if line:
        lines.append(" ".join(line))
    return lines


def add_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]):
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = 0.94
    ax.text(0.06, y, title, fontsize=20, weight="bold", va="top")
    y -= 0.08
    for heading, body in sections:
        ax.text(0.06, y, heading, fontsize=13, weight="bold", va="top")
        y -= 0.035
        for line in wrapped_lines(body):
            ax.text(0.075, y, line, fontsize=10.5, va="top")
            y -= 0.026
        y -= 0.025
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, title: str, caption: str):
    image = Image.open(image_path).convert("RGB")
    fig = plt.figure(figsize=(11.0, 8.5))
    ax_title = fig.add_axes([0.04, 0.91, 0.92, 0.06])
    ax_title.axis("off")
    ax_title.text(0.0, 0.5, title, fontsize=15, weight="bold", va="center")
    ax_img = fig.add_axes([0.04, 0.18, 0.92, 0.72])
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_cap = fig.add_axes([0.04, 0.04, 0.92, 0.12])
    ax_cap.axis("off")
    y = 0.95
    for line in wrapped_lines(caption, width=130):
        ax_cap.text(0, y, line, fontsize=9.5, va="top")
        y -= 0.25
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_phase3_pdf(metrics: dict, out_dir: Path):
    pdf_path = out_dir / "phase3_gradcam_report.pdf"
    clean_soar = metrics["clean"]["soar"]
    top_drift = top_metric_names(metrics, "drift", 3, reverse=True)
    low_soar = top_metric_names(metrics, "soar", 3, reverse=False)
    uncertain = [name for name, m in metrics.items() if name != "clean" and m["decision"] == "cannot_determine"]

    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Phase 3 Grad-CAM Analysis",
            [
                (
                    "Objective",
                    "This phase visualizes where Qwen2.5-VL places decision-targeted visual evidence under semantic perturbations. "
                    "The target is fixed to 'Decision: Do not proceed' so clean and perturbed images are compared under the same safety decision concept.",
                ),
                (
                    "Stop-cue annotation",
                    "SOAR is computed only over direct stop cues: red traffic lights, pedestrians, and the crosswalk. "
                    "Vehicle flow and bus context are intentionally excluded because they are contextual evidence rather than direct stop cues.",
                ),
                (
                    "Quality controls",
                    "The Grad-CAM hook uses Qwen visual.pooler_output to preserve spatial alignment after window-token reordering. "
                    "Every CAM is checked for NaN, all-zero, and uniform maps. A clean target-contrast sanity check compares 'Do not proceed' and 'Proceed' CAMs.",
                ),
            ],
        )
        add_image_page(
            pdf,
            out_dir / "bbox_debug_clean.png",
            "Safety Evidence Boxes",
            "Manual stop-cue boxes used for SOAR. These boxes define the quantitative region for signal, pedestrian, and crosswalk attention ratios.",
        )
        add_image_page(
            pdf,
            out_dir / "pairs/clean_input_vs_gradcam.png",
            "Clean Baseline Grad-CAM",
            f"Clean prediction is {metrics['clean']['decision']}. Clean SOAR is {clean_soar:.3f}. "
            "The overlay indicates the baseline decision evidence for 'Do not proceed'.",
        )
        add_image_page(
            pdf,
            out_dir / "phase3_gradcam_metric_bars.png",
            "Drift and SOAR Summary",
            "Top panel shows Grad-CAM drift from clean. Bottom panel shows stop-cue SOAR. Black X markers indicate cases where the generated decision became 'Cannot determine'.",
        )
        add_image_page(
            pdf,
            out_dir / "phase3_gradcam_soar_grouped_bars.png",
            "SOAR by Stop-Cue Group",
            "Stacked bars decompose SOAR into red-signal, pedestrian, and crosswalk contributions. This shows which stop cue type retains or loses visual evidence under perturbation.",
        )
        add_image_page(
            pdf,
            out_dir / "phase3_gradcam_drift_vs_soar.png",
            "Drift vs Stop-Cue Retention",
            "Upper-right points indicate large attention drift with retained stop-cue attention. Lower-right points are more concerning because attention shifts away from stop cues.",
        )
        add_text_page(
            pdf,
            "Key Findings",
            [
                (
                    "Highest drift cases",
                    ", ".join(f"{name} (drift={metrics[name]['drift']:.3f}, SOAR={metrics[name]['soar']:.3f})" for name in top_drift)
                    + ". These perturbations cause the largest spatial shift in the decision-targeted Grad-CAM relative to clean.",
                ),
                (
                    "Lowest SOAR cases",
                    ", ".join(f"{name} (SOAR={metrics[name]['soar']:.3f}, drift={metrics[name]['drift']:.3f})" for name in low_soar)
                    + ". These are the cases where Grad-CAM evidence is least concentrated on direct stop cues.",
                ),
                (
                    "Decision uncertainty",
                    ("No perturbation changed the decision to Proceed. " if not uncertain else "")
                    + (
                        "The model returned Cannot determine for: " + ", ".join(uncertain) + ". "
                        if uncertain
                        else "No perturbation produced Cannot determine. "
                    )
                    + "These cases should be interpreted as safety-relevant degradation even without a Proceed flip, because the model loses decision certainty.",
                ),
                (
                    "Interpretation",
                    "The main robustness signal is not only whether the final answer flips, but whether Grad-CAM remains concentrated on direct stop cues. "
                    "Perturbations with high drift and low SOAR are the most concerning for this single-image safety scenario.",
                ),
            ],
        )
        for name in top_drift:
            add_image_page(
                pdf,
                out_dir / f"pairs/{name}_input_vs_gradcam.png",
                f"Representative High-Drift Case: {name}",
                f"Decision={metrics[name]['decision']}, drift={metrics[name]['drift']:.3f}, SOAR={metrics[name]['soar']:.3f}. "
                "Use this pair to inspect whether the visual evidence moved away from the annotated stop cues.",
            )
        add_image_page(
            pdf,
            out_dir / "phase3_gradcam_top5_drift_comparison.png",
            "Top-5 Drift Comparison",
            "Clean and the five most drifted perturbations are shown as input/overlay pairs. This page is intended as a visual summary of the strongest attention shifts.",
        )
        add_image_page(
            pdf,
            out_dir / "phase3_gradcam_change_maps_vs_clean.png",
            "Grad-CAM Change Maps",
            "Red regions gained decision-targeted Grad-CAM relative to clean; blue regions lost it. This is a diagnostic visualization and should be read together with SOAR.",
        )
        add_text_page(
            pdf,
            "Limitations",
            [
                (
                    "Method limit",
                    "This is Grad-CAM, not Attention Rollout. It is a post-hoc decision-targeted visual explanation and should not be treated as a full causal proof of model reasoning.",
                ),
                (
                    "Spatial resolution",
                    "Qwen visual tokens are spatially merged, producing a 20x32 Grad-CAM grid for this image. The overlay is therefore coarse and should be interpreted at region level.",
                ),
                (
                    "Single-image scope",
                    "This experiment uses one road scene with semantic perturbations. Results support qualitative robustness analysis for this scene, not dataset-level statistical claims.",
                ),
            ],
        )
    print(f"Saved: {pdf_path}")
def make_change_map_figure(results: dict, metrics: dict, out_dir: Path, mask_artifacts: bool):
    sorted_names = sorted(
        [name for name in metrics if name != "clean"],
        key=lambda n: metrics[n]["drift"],
        reverse=True,
    )
    clean_img = results["clean"]["image_pil"]
    _, clean_up = overlay_heatmap(clean_img, results["clean"]["cam"], mask_artifacts)
    w, h = clean_img.size
    top_names = sorted_names[:5]
    fig, axes = plt.subplots(2, len(top_names), figsize=(3.8 * len(top_names), 8))
    if len(top_names) == 1:
        axes = axes.reshape(2, 1)
    for col, name in enumerate(top_names):
        item = results[name]
        m = metrics[name]
        pert_overlay, pert_up = overlay_heatmap(item["image_pil"], item["cam"], mask_artifacts)
        pert_up_clean_size = resize_map(pert_up, (w, h))
        diff = pert_up_clean_size - clean_up
        axes[0, col].imshow(pert_overlay)
        axes[0, col].set_title(f"{name}\nSOAR={m['soar']:.3f}", fontsize=8)
        axes[0, col].axis("off")
        vmax = max(abs(float(diff.min())), abs(float(diff.max())), 0.01)
        axes[1, col].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, col].set_title(f"Grad-CAM diff\nDrift={m['drift']:.3f}", fontsize=8)
        axes[1, col].axis("off")
    fig.suptitle("Phase 3 Grad-CAM difference maps: perturbed - clean", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / "phase3_gradcam_change_maps_vs_clean.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out_path}")


def write_metrics(metrics: dict, out_dir: Path):
    csv_path = out_dir / "phase3_gradcam_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "perturbation",
                "category",
                "target_text",
                "target_logprob",
                "drift",
                "soar",
                "soar_signal",
                "soar_pedestrian",
                "soar_crosswalk",
                "decision",
                "raw_output",
                "grid",
                "cam_valid",
                "cam_valid_reason",
            ],
        )
        writer.writeheader()
        for name, m in metrics.items():
            writer.writerow(
                {
                    "perturbation": name,
                    "category": m["category"],
                    "target_text": m["target_text"],
                    "target_logprob": round(m["target_logprob"], 6),
                    "drift": round(m["drift"], 6),
                    "soar": round(m["soar"], 6),
                    "soar_signal": round(m["soar_signal"], 6),
                    "soar_pedestrian": round(m["soar_pedestrian"], 6),
                    "soar_crosswalk": round(m["soar_crosswalk"], 6),
                    "decision": m["decision"],
                    "raw_output": m["raw_output"],
                    "grid": "x".join(str(v) for v in m["grid"]),
                    "cam_valid": m["cam_valid"],
                    "cam_valid_reason": m["cam_valid_reason"],
                }
            )
    print(f"Saved: {csv_path}")


def write_partial_metrics(metrics: dict, out_dir: Path):
    csv_path = out_dir / "phase3_gradcam_metrics_partial.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "perturbation",
                "category",
                "target_text",
                "target_logprob",
                "drift",
                "soar",
                "soar_signal",
                "soar_pedestrian",
                "soar_crosswalk",
                "decision",
                "raw_output",
                "grid",
                "cam_valid",
                "cam_valid_reason",
            ],
        )
        writer.writeheader()
        for name, m in metrics.items():
            writer.writerow(
                {
                    "perturbation": name,
                    "category": m["category"],
                    "target_text": m["target_text"],
                    "target_logprob": round(m["target_logprob"], 6),
                    "drift": "" if m["drift"] is None else round(m["drift"], 6),
                    "soar": round(m["soar"], 6),
                    "soar_signal": round(m["soar_signal"], 6),
                    "soar_pedestrian": round(m["soar_pedestrian"], 6),
                    "soar_crosswalk": round(m["soar_crosswalk"], 6),
                    "decision": m["decision"],
                    "raw_output": m["raw_output"],
                    "grid": "x".join(str(v) for v in m["grid"]),
                    "cam_valid": m["cam_valid"],
                    "cam_valid_reason": m["cam_valid_reason"],
                }
            )


def build_metric(name: str, item: dict, clean_cam: np.ndarray | None) -> dict:
    category = next((cat for cat, keys in CATEGORIES.items() if name in keys), "clean")
    drift = None
    if name == "clean":
        drift = 0.0
    elif clean_cam is not None:
        drift = attention_drift(clean_cam, item["cam"])
    group_soar = soar_by_group(item["cam"])
    return {
        "category": category,
        "target_text": item["target_text"],
        "target_logprob": item["target_logprob"],
        "drift": drift,
        "soar": soar(item["cam"], SAFETY_BBOX_FRACS),
        **group_soar,
        "decision": item["decision"],
        "raw_output": item["raw_output"].replace("\n", " "),
        "grid": item["grid"],
        "cam_valid": item["cam_valid"],
        "cam_valid_reason": item["cam_valid_reason"],
    }


def save_intermediate(name: str, item: dict, metrics: dict, out_dir: Path, mask_artifacts: bool):
    cam_dir = out_dir / "cams"
    overlay_dir = out_dir / "overlays"
    pair_dir = out_dir / "pairs"
    cam_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    np.save(cam_path(out_dir, name), item["cam"])
    overlay, cam_up = overlay_heatmap(item["image_pil"], item["cam"], mask_artifacts)
    overlay_path, raw_path = overlay_paths(out_dir, name)
    plt.imsave(overlay_path, overlay)
    plt.imsave(raw_path, cam_up, cmap="hot", vmin=0, vmax=1)
    m = metrics[name]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    axes[0].imshow(item["image_pil"])
    axes[0].set_title("Input", fontsize=12, weight="bold", pad=4)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM overlay", fontsize=12, weight="bold", pad=4)
    axes[1].axis("off")
    subtitle = (
        f"{name} | decision={m['decision']} | SOAR={m['soar']:.3f} | "
        f"valid={m['cam_valid']}"
    )
    if m["drift"] is not None:
        subtitle += f" | drift={m['drift']:.3f}"
    fig.suptitle(subtitle, fontsize=11, y=0.985)
    plt.tight_layout(pad=0.35, rect=[0, 0, 1, 0.955])
    pair_path = pair_dir / f"{name}_input_vs_gradcam.png"
    plt.savefig(pair_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    write_partial_metrics(metrics, out_dir)


def load_resumed_item(name: str, path: Path, out_dir: Path) -> dict | None:
    path_cam = cam_path(out_dir, name)
    if not path_cam.exists():
        return None
    image_pil = Image.open(path).convert("RGB")
    cam = np.load(path_cam)
    cam_valid, cam_valid_reason = validate_cam(cam)
    return {
        "image_pil": image_pil,
        "path": path,
        "cam": cam,
        "grid": (1, int(cam.shape[0]), int(cam.shape[1])),
        "raw_output": "[resumed_existing_cam]",
        "decision": "resumed",
        "target_text": DECISION_TEXT["do_not_proceed"],
        "target_logprob": float("nan"),
        "cam_valid": cam_valid,
        "cam_valid_reason": cam_valid_reason,
    }


def parse_args():
    root = project_root()
    parser = argparse.ArgumentParser(description="Phase 3 Grad-CAM for Qwen2.5-VL semantic perturbations.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--clean-image", type=Path, default=root / "experiments/image2_infer.png")
    parser.add_argument("--semantic-dir", type=Path, default=root / "experiments/semantic_perturbations")
    parser.add_argument("--output-dir", type=Path, default=root / "experiments/results/phase3_gradcam")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help=(
            "0 hooks Qwen visual.pooler_output after window-token reverse indexing, "
            "which keeps heatmap coordinates aligned. Non-zero block hooks are for debugging."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=["do_not_proceed", "generated_decision"],
        default="do_not_proceed",
        help="Use a fixed safety target for comparable CAMs, or target each generated decision.",
    )
    parser.add_argument("--no-mask-artifacts", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only N perturbations after clean; useful for tests.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing cams/*.npy and continue interrupted runs.")
    parser.add_argument(
        "--target-contrast-sanity",
        action="store_true",
        help="Also save clean Proceed-target CAM for target sensitivity sanity checking.",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate figures from existing cams/*.npy without loading the model.",
    )
    return parser.parse_args()


def load_metrics_csv(csv_path: Path) -> dict:
    metrics = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["perturbation"]
            metrics[name] = {
                "category": row["category"],
                "target_text": row["target_text"],
                "target_logprob": float(row["target_logprob"]),
                "drift": float(row["drift"]),
                "soar": float(row["soar"]),
                "soar_signal": float(row["soar_signal"]),
                "soar_pedestrian": float(row["soar_pedestrian"]),
                "soar_crosswalk": float(row["soar_crosswalk"]),
                "decision": row["decision"],
                "raw_output": row["raw_output"],
                "grid": tuple(int(v) for v in row["grid"].split("x")),
                "cam_valid": row["cam_valid"] == "True",
                "cam_valid_reason": row["cam_valid_reason"],
            }
    return metrics


def load_results_from_cams(image_paths: dict, out_dir: Path) -> dict:
    results = {}
    for name, path in image_paths.items():
        path_cam = cam_path(out_dir, name)
        if not path_cam.exists():
            raise FileNotFoundError(f"Missing CAM for postprocess: {path_cam}")
        image_pil = Image.open(path).convert("RGB")
        cam = np.load(path_cam)
        results[name] = {
            "image_pil": image_pil,
            "path": path,
            "cam": cam,
            "grid": (1, int(cam.shape[0]), int(cam.shape[1])),
        }
    return results


def main():
    args = parse_args()
    mask_artifacts = not args.no_mask_artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = {"clean": args.clean_image}
    image_paths.update({name: args.semantic_dir / filename for name, filename in SEMANTIC_PERTURBATIONS_13.items()})
    if args.limit > 0:
        limited = {"clean": image_paths["clean"]}
        for name in list(SEMANTIC_PERTURBATIONS_13)[: args.limit]:
            limited[name] = image_paths[name]
        image_paths = limited

    for name, path in image_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing image for {name}: {path}")

    if args.postprocess_only:
        metrics_path = args.output_dir / "phase3_gradcam_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for postprocess: {metrics_path}")
        print(f"postprocess_only=True")
        print(f"output_dir={args.output_dir}")
        metrics = load_metrics_csv(metrics_path)
        results = load_results_from_cams(image_paths, args.output_dir)
        mask_artifacts = not args.no_mask_artifacts
        make_category_figures(results, metrics, args.output_dir, mask_artifacts)
        make_summary_figures(results, metrics, args.output_dir, mask_artifacts)
        make_quantitative_figures(metrics, args.output_dir)
        make_change_map_figure(results, metrics, args.output_dir, mask_artifacts)
        print(f"Postprocess outputs regenerated in: {args.output_dir}")
        return

    device = select_device(args.device)
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA/MPS device is available in this Python environment. "
            "Refusing CPU execution because full Phase 3 would be too slow. "
            "Use a Python environment where torch.backends.mps.is_available() is True, "
            "or pass --allow-cpu for a small debug run only."
        )

    item_count = len(image_paths)
    if device.type == "mps":
        estimate = f"~{max(5, item_count * 2)}-{max(10, item_count * 5)} min"
    elif device.type == "cuda":
        estimate = f"~{max(3, item_count)}-{max(6, item_count * 3)} min"
    else:
        estimate = f"debug only; full run may take several hours ({item_count} images)"
    print(f"device={device}")
    print(f"estimated_runtime={estimate}")
    print(f"output_dir={args.output_dir}")
    model, processor = load_model(args.model_id, device, local_files_only=not args.allow_download)
    layer, layer_name, total_layers = select_gradcam_layer(model, args.layer_index)
    print(f"Grad-CAM layer: {layer_name} / visual_blocks={total_layers}")
    write_method_notes(args.output_dir, layer_name, mask_artifacts, args.target_mode)

    results = {}
    metrics = {}
    peak_rows = []
    clean_cam = None
    for idx, (name, path) in enumerate(image_paths.items(), start=1):
        progress = 100.0 * (idx - 1) / len(image_paths)
        t0 = time.time()
        print(f"\n[progress {progress:5.1f}%] starting {idx}/{len(image_paths)}: {name}", flush=True)

        resumed = load_resumed_item(name, path, args.output_dir) if args.resume else None
        if resumed is not None:
            print("  resume: loaded existing CAM", flush=True)
            results[name] = resumed
            if name == "clean":
                clean_cam = resumed["cam"]
        else:
            print("  step 1/3: inference", flush=True)
            image_pil = Image.open(path).convert("RGB")
            raw_output = infer_decision(model, processor, image_pil, BEST_PROMPT, device)
            parsed = parse_decision(raw_output)
            if args.target_mode == "generated_decision" and parsed in DECISION_TEXT:
                target_text = DECISION_TEXT[parsed]
            else:
                target_text = DECISION_TEXT["do_not_proceed"]

            print(f"  step 2/3: Grad-CAM target={target_text!r}", flush=True)
            cam, grid, target_logprob = compute_gradcam(
                model=model,
                processor=processor,
                image_pil=image_pil,
                prompt=BEST_PROMPT,
                target_text=target_text,
                layer=layer,
                device=device,
            )
            cam_valid, cam_valid_reason = validate_cam(cam)
            if not cam_valid:
                print(f"  WARNING: invalid CAM for {name}: {cam_valid_reason}", flush=True)
            results[name] = {
                "image_pil": image_pil,
                "path": path,
                "cam": cam,
                "grid": grid,
                "raw_output": raw_output,
                "decision": parsed,
                "target_text": target_text,
                "target_logprob": target_logprob,
                "cam_valid": cam_valid,
                "cam_valid_reason": cam_valid_reason,
            }
            if name == "clean":
                clean_cam = cam
                write_bbox_debug(image_pil, args.output_dir)
                if args.target_contrast_sanity:
                    print("  sanity: computing clean Proceed-target Grad-CAM", flush=True)
                    proceed_cam, _, proceed_score = compute_gradcam(
                        model=model,
                        processor=processor,
                        image_pil=image_pil,
                        prompt=BEST_PROMPT,
                        target_text=DECISION_TEXT["proceed"],
                        layer=layer,
                        device=device,
                    )
                    contrast_dir = args.output_dir / "target_contrast_sanity"
                    contrast_dir.mkdir(parents=True, exist_ok=True)
                    proceed_overlay, proceed_up = overlay_heatmap(image_pil, proceed_cam, mask_artifacts)
                    default_overlay, default_up = overlay_heatmap(image_pil, cam, mask_artifacts)
                    plt.imsave(contrast_dir / "clean_do_not_proceed_overlay.png", default_overlay)
                    plt.imsave(contrast_dir / "clean_proceed_overlay.png", proceed_overlay)
                    plt.imsave(contrast_dir / "clean_do_not_proceed_raw_gradcam.png", default_up, cmap="hot", vmin=0, vmax=1)
                    plt.imsave(contrast_dir / "clean_proceed_raw_gradcam.png", proceed_up, cmap="hot", vmin=0, vmax=1)
                    diff = float(np.sqrt(((default_up - proceed_up) ** 2).mean()))
                    with (contrast_dir / "target_contrast_sanity.txt").open("w") as f:
                        f.write(f"do_not_proceed_score={target_logprob:.6f}\n")
                        f.write(f"proceed_score={proceed_score:.6f}\n")
                        f.write(f"raw_gradcam_l2_diff={diff:.6f}\n")
                    print(f"  sanity: target contrast raw CAM L2 diff={diff:.4f}", flush=True)

        metrics[name] = build_metric(name, results[name], clean_cam)
        print("  step 3/3: saving intermediate outputs", flush=True)
        save_intermediate(name, results[name], metrics, args.output_dir, mask_artifacts)
        _, cam_up = overlay_heatmap(results[name]["image_pil"], results[name]["cam"], mask_artifacts)
        peak_rows = [row for row in peak_rows if row["perturbation"] != name]
        peak_rows.extend(extract_cam_peaks(name, results[name]["image_pil"], cam_up, PEAK_TOP_K))
        write_peak_debug(peak_rows, args.output_dir)
        cleanup(device)
        progress_done = 100.0 * idx / len(image_paths)
        current_grid = results[name]["grid"]
        print(
            f"[progress {progress_done:5.1f}%] done {idx}/{len(image_paths)} {name:<35} "
            f"grid={current_grid[1]}x{current_grid[2]} "
            f"target={results[name]['target_text']!r} score={results[name]['target_logprob']:.4f} "
            f"decision={results[name]['decision']} cam_valid={results[name]['cam_valid']} "
            f"elapsed={time.time() - t0:.1f}s"
        )

    np.savez_compressed(
        args.output_dir / "phase3_gradcam_cams.npz",
        **{name: item["cam"] for name, item in results.items()},
    )
    write_metrics(metrics, args.output_dir)
    make_category_figures(results, metrics, args.output_dir, mask_artifacts)
    make_summary_figures(results, metrics, args.output_dir, mask_artifacts)
    make_quantitative_figures(metrics, args.output_dir)
    make_change_map_figure(results, metrics, args.output_dir, mask_artifacts)
    print(f"All Grad-CAM outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
