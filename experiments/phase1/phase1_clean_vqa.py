import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory containing both data/ and experiments/."""
    for candidate in [start, *start.parents]:
        if (candidate / "data").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    return start


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_IMAGE_SIDE = 896
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TALE_EP_BUDGET_CANDIDATES = [64, 80, 96, 120, 144]
MIN_GOOD_QUALITY = 6
COMPLETION_MARGIN = 0.90
MAX_WORSE_STREAK = 2

PROMPT_CANDIDATES = [
    {
        "id": "prompt_01",
        "style": "concise",
        "prompt": (
            "You are an autonomous-driving assistance evaluator. Use only information "
            "directly visible in the image to decide whether the vehicle may proceed straight.\n\n"
            "Output format:\n"
            "Decision: [Proceed / Do not proceed / Cannot determine]\n"
            "Evidence:\n"
            "- Observation: ... -> Impact: ...\n"
            "- Observation: ... -> Impact: ...\n\n"
            "Rules:\n"
            "- Use only traffic lights, signs, pedestrians, vehicle flow, and obstacles as evidence.\n"
            "- Do not infer situations that are not visible in the image.\n"
            "- Use at most 3 evidence items.\n"
            "- If uncertain, choose 'Cannot determine'."
        ),
        "expected_strength": "Compact structure that encourages token-efficient safety judgment.",
        "risk": "May under-explain complex scenes because of strong compression.",
    },
    {
        "id": "prompt_02",
        "style": "safety_focused",
        "prompt": (
            "Look at the forward road image and decide from a safety perspective whether "
            "the vehicle may proceed straight.\n\n"
            "Output:\n"
            "Decision: [Proceed / Do not proceed / Cannot determine]\n"
            "Safety evidence:\n"
            "1. Observed element: ...\n"
            "   Impact on decision: ...\n"
            "2. Observed element: ...\n"
            "   Impact on decision: ...\n\n"
            "Rules:\n"
            "- Use only visible objects.\n"
            "- If a visible risk signal exists, judge conservatively.\n"
            "- If visibility or evidence is insufficient, choose 'Cannot determine'.\n"
            "- Avoid unnecessary explanation."
        ),
        "expected_strength": "Strongly encourages safety-first reasoning and risk detection.",
        "risk": "May become overly conservative in scenes where proceeding is allowed.",
    },
    {
        "id": "prompt_03",
        "style": "evidence_strict",
        "prompt": (
            "Decide whether the vehicle may proceed straight using only evidence directly "
            "visible inside the image.\n\n"
            "Output format:\n"
            "{\n"
            '  "decision": "Proceed | Do not proceed | Cannot determine",\n'
            '  "evidence": [\n'
            "    {\n"
            '      "observation": "...",\n'
            '      "impact": "..."\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- No guessing or filling gaps with general knowledge.\n"
            "- Do not mention objects that are not visually observable.\n"
            "- If the evidence is weak, choose 'Cannot determine'."
        ),
        "expected_strength": "Good for evidence grounding and hallucination control.",
        "risk": "May produce more 'Cannot determine' answers than necessary.",
    },
    {
        "id": "prompt_04",
        "style": "uncertainty_aware",
        "prompt": (
            "Given one road image, decide whether the vehicle may proceed straight. If visibility, "
            "occlusion, or resolution makes the judgment uncertain, defer the decision.\n\n"
            "Output:\n"
            "Decision: [Proceed / Do not proceed / Cannot determine]\n"
            "Confidence: [High / Medium / Low]\n"
            "Evidence:\n"
            "- Observation: ... -> Impact: ...\n\n"
            "Rules:\n"
            "- Use only visible information.\n"
            "- Provide at most 2 evidence items.\n"
            "- Do not force a decision."
        ),
        "expected_strength": "Captures uncertainty and is useful for comparing robust early-exit behavior.",
        "risk": "Confidence reporting may increase token usage slightly.",
    },
    {
        "id": "prompt_05",
        "style": "token_efficient",
        "prompt": (
            "Analyze the forward-driving image.\n\n"
            "Format:\n"
            "Decision: [Proceed / Do not proceed / Cannot determine]\n"
            "Evidence 1: Observation -> Impact\n"
            "Evidence 2: Observation -> Impact\n\n"
            "Rules:\n"
            "- Around 40 tokens if possible.\n"
            "- Use only directly visible objects.\n"
            "- Do not guess."
        ),
        "expected_strength": "Very short; useful for token and latency optimization.",
        "risk": "May not provide enough reasoning detail.",
    },
    {
        "id": "prompt_06",
        "style": "evidence_strict",
        "prompt": (
            "Judge whether the vehicle may proceed straight based only on visual evidence in the image.\n\n"
            "Strict format:\n"
            "Decision: ...\n"
            "Evidence:\n"
            "- [Object] observed state / impact on decision\n\n"
            "Allowed objects:\n"
            "traffic lights, signs, pedestrians, vehicles, obstacles, lane state\n\n"
            "Forbidden:\n"
            "- Predicting future events\n"
            "- Inferring driver intent\n"
            "- Deciding from general traffic rules alone"
        ),
        "expected_strength": "Limits evidence scope and improves grounding.",
        "risk": "May ignore useful cues outside the allowed object list.",
    },
    {
        "id": "prompt_07",
        "style": "safety_focused",
        "prompt": (
            "You are a conservative autonomous-driving safety evaluator. If a visible risk for "
            "proceeding straight exists, choose 'Do not proceed' or 'Cannot determine'.\n\n"
            "Output:\n"
            "Decision: [Proceed / Do not proceed / Cannot determine]\n"
            "Risk evidence:\n"
            "- Observation: ...\n"
            "  Impact: ...\n\n"
            "Conditions:\n"
            "- Mention only visible risks.\n"
            "- Choose Proceed only when no risk factor is visible.\n"
            "- Keep the answer short."
        ),
        "expected_strength": "Helps reduce false negatives in safety-critical scenes.",
        "risk": "May increase false positives.",
    },
    {
        "id": "prompt_08",
        "style": "uncertainty_aware",
        "prompt": (
            "Determine whether the vehicle may proceed straight from the forward road image.\n\n"
            "Output format:\n"
            "Conclusion: [Proceed / Do not proceed / Cannot determine]\n"
            "Evidence:\n"
            "1. Observed fact\n"
            "2. How that fact affects the straight-driving decision\n\n"
            "Decision rules:\n"
            "- Do not use information that is not directly visible.\n"
            "- If evidence is insufficient, choose Cannot determine.\n"
            "- Make the evidence and conclusion logically connected."
        ),
        "expected_strength": "Strongly enforces connection between observation and judgment.",
        "risk": "May use more tokens because of the structured reasoning requirement.",
    },
    {
        "id": "prompt_09",
        "style": "token_efficient",
        "prompt": (
            "Look at the road image and judge only whether the vehicle may proceed straight.\n\n"
            "Output:\n"
            "Decision: [Proceed / Do not proceed / Cannot determine]\n"
            "Evidence: Observation -> Impact (max 2)\n\n"
            "Rules:\n"
            "- Use only what is visible in the image.\n"
            "- Avoid long explanations.\n"
            "- If uncertain, choose Cannot determine."
        ),
        "expected_strength": "Concise while preserving a minimal reasoning trace.",
        "risk": "May be too brief for complex intersections.",
    },
    {
        "id": "prompt_10",
        "style": "concise",
        "prompt": (
            "Perform an autonomous-driving judgment.\n\n"
            "First select only the key objects relevant to proceeding straight. Then decide whether "
            "proceeding straight is allowed.\n\n"
            "Output format:\n"
            "Key observations:\n"
            "- ...\n"
            "Decision: [Proceed / Do not proceed / Cannot determine]\n"
            "Reason: ...\n\n"
            "Constraints:\n"
            "- At most 3 key observations.\n"
            "- Use only directly observed information.\n"
            "- Do not speculate."
        ),
        "expected_strength": "Compresses attention onto key objects before the final judgment.",
        "risk": "The initial object selection may omit an important cue.",
    },
]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_images(image_dir: Path) -> list[Path]:
    image_name_pattern = re.compile(r"^image\d+$", re.IGNORECASE)
    return [
        p
        for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS and image_name_pattern.match(p.stem)
    ]


def prepare_images(image_paths: list[Path], output_dir: Path, max_side: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        infer_path = output_dir / f"{image_path.stem}_infer.png"
        img.save(infer_path)
        images.append(
            {
                "original": str(image_path),
                "infer": str(infer_path),
                "stem": image_path.stem,
                "size": list(img.size),
            }
        )
    return images


def load_model(model_id: str, device: torch.device):
    if device.type == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        dtype = torch.float16 if device.type == "mps" else torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cpu",
            attn_implementation="eager",
        ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=224 * 224,
        max_pixels=MAX_IMAGE_SIDE * MAX_IMAGE_SIDE,
    )
    return model, processor


def build_tale_ep_prompt(prompt: str, budget: int) -> str:
    constraint = (
        f"\n\nTALE-EP output budget: at most {budget} tokens.\n"
        "Answer in English only. Compress the key safety objects and decision rationale, "
        "but include both observation and impact."
    )
    return prompt + constraint


def quality_check(output: str) -> dict:
    lower = output.lower()
    has_decision = any(label in lower for label in ["proceed", "do not proceed", "cannot determine"])
    has_reason = any(term in lower for term in ["evidence", "reason", "observation", "observed", "impact"])
    has_impact = any(
        term in lower
        for term in ["impact", "because", "therefore", "risk", "allows", "prevents", "indicates"]
    )
    visible_object_terms = [
        "traffic light",
        "sign",
        "pedestrian",
        "obstacle",
        "vehicle",
        "lane",
        "green",
        "red",
        "intersection",
    ]
    object_mentions = [term for term in visible_object_terms if term in lower]
    hallucination_risk_terms = ["probably", "seems like", "not visible but", "generally", "guess", "assume"]
    hallucination_risk = any(term in lower for term in hallucination_risk_terms)
    non_english_markers = ["판단", "근거", "관찰", "영향", "직진"]
    has_non_english = any(term in output for term in non_english_markers)
    return {
        "has_decision": has_decision,
        "has_reason": has_reason,
        "has_impact": has_impact,
        "object_mentions": object_mentions,
        "hallucination_risk": hallucination_risk,
        "has_non_english": has_non_english,
        "quality_score": (
            int(has_decision)
            + int(has_reason)
            + int(has_impact)
            + min(len(object_mentions), 3)
            - int(hallucination_risk)
            - int(has_non_english)
        ),
    }


def optimization_score(result: dict) -> float:
    quality = result["quality"]["quality_score"]
    input_penalty = result["input_tokens"] / 1000
    output_penalty = result["output_tokens"] / 100
    latency_penalty = result["latency_sec"] / 20
    return round(quality - input_penalty - output_penalty - latency_penalty, 4)


def looks_truncated(output: str, output_tokens: int, budget: int) -> bool:
    stripped = output.strip()
    if output_tokens >= budget:
        return True
    if not stripped:
        return True
    if stripped[-1] not in ".]}":
        return True
    dangling_endings = (" to", " of", " and", " the", " a", " an", " Ign", " before", " with")
    return stripped.endswith(dangling_endings)


def infer(
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    model,
    processor,
    device: torch.device,
) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[1])
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    latency = round(time.time() - t0, 2)

    out_ids = generated[0][inputs["input_ids"].shape[1] :]
    output = processor.decode(out_ids, skip_special_tokens=True).strip()
    output_tokens = int(len(out_ids))
    return {
        "output": output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_sec": latency,
    }


def run_experiment(images: list[dict], model, processor, device: torch.device) -> tuple[dict, dict]:
    results = {}
    best = None

    for image_item in images:
        image_key = image_item["stem"]
        results[image_key] = {}
        print("\n" + "#" * 70)
        print(f"[IMAGE] {image_key} ({image_item['infer']})")

        for candidate in PROMPT_CANDIDATES:
            candidate_best = None
            worse_streak = 0

            for budget in TALE_EP_BUDGET_CANDIDATES:
                name = f"{candidate['id']}_budget_{budget}"
                print(f"\n{'-' * 50}\n[{image_key} / {name}] style={candidate['style']}")
                final_prompt = build_tale_ep_prompt(candidate["prompt"], budget)
                r = infer(image_item["infer"], final_prompt, budget, model, processor, device)
                r.update(
                    {
                        "image_original": image_item["original"],
                        "image_infer": image_item["infer"],
                        "candidate_id": candidate["id"],
                        "style": candidate["style"],
                        "prompt": candidate["prompt"],
                        "final_prompt": final_prompt,
                        "expected_strength": candidate["expected_strength"],
                        "risk": candidate["risk"],
                        "predicted_output_tokens": budget,
                        "token_budget_error": r["output_tokens"] - budget,
                        "quality": quality_check(r["output"]),
                    }
                )
                r["optimization_score"] = optimization_score(r)
                r["truncated"] = looks_truncated(r["output"], r["output_tokens"], budget)
                r["pruned_after_this"] = False
                r["prune_reason"] = ""
                results[image_key][name] = r

                overall_candidate = {"image": image_key, "name": name, **r}
                if best is None or r["optimization_score"] > best["optimization_score"]:
                    best = overall_candidate

                if candidate_best is None or r["optimization_score"] > candidate_best["optimization_score"]:
                    candidate_best = {"image": image_key, "name": name, **r}
                    worse_streak = 0
                else:
                    worse_streak += 1

                print(r["output"])
                print(
                    f"\nscore={r['optimization_score']}  "
                    f"quality={r['quality']['quality_score']}  "
                    f"input={r['input_tokens']}  "
                    f"output={r['output_tokens']}/{budget}  "
                    f"total={r['total_tokens']}  "
                    f"truncated={r['truncated']}  "
                    f"latency={r['latency_sec']}s"
                )

                good_complete_answer = (
                    r["quality"]["quality_score"] >= MIN_GOOD_QUALITY
                    and not r["truncated"]
                    and r["output_tokens"] <= budget * COMPLETION_MARGIN
                )
                dominated_by_smaller_budget = (
                    candidate_best is not None
                    and candidate_best["name"] != name
                    and r["quality"]["quality_score"] <= candidate_best["quality"]["quality_score"]
                    and r["optimization_score"] < candidate_best["optimization_score"]
                    and r["output_tokens"] >= candidate_best["output_tokens"]
                )

                if good_complete_answer:
                    r["pruned_after_this"] = True
                    r["prune_reason"] = "good_complete_answer"
                    print(
                        f"[PRUNE] Stop larger budgets for {candidate['id']}: "
                        f"good complete answer at budget={budget}."
                    )
                    break

                if dominated_by_smaller_budget and worse_streak >= MAX_WORSE_STREAK:
                    r["pruned_after_this"] = True
                    r["prune_reason"] = "dominated_by_smaller_budget"
                    print(
                        f"[PRUNE] Stop larger budgets for {candidate['id']}: "
                        "larger budgets are dominated by smaller budget result."
                    )
                    break

    if best:
        print("\n" + "=" * 50)
        print("[BEST OVERALL]")
        print("image:", best["image"])
        print("name:", best["name"])
        print("style:", best["style"])
        print("budget:", best["predicted_output_tokens"])
        print("input_tokens:", best["input_tokens"])
        print("output_tokens:", best["output_tokens"])
        print("total_tokens:", best["total_tokens"])
        print("score:", best["optimization_score"])
        print("prompt:\n", best["prompt"])
        print("output:\n", best["output"])

    return results, best


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    default_image_dir = repo_root / "data/raw"
    default_work_dir = repo_root / "experiments/work"
    default_output = repo_root / "experiments/results/phase1_prompt_optimization_results.json"
    parser = argparse.ArgumentParser(description="Phase 1 TALE-EP-style prompt optimization for VLM VQA.")
    parser.add_argument("--image-dir", type=Path, default=default_image_dir)
    parser.add_argument("--work-dir", type=Path, default=default_work_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-image-side", type=int, default=MAX_IMAGE_SIDE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files found in {args.image_dir}. "
            "Use names like image1.png, image2.png, or image123.png."
        )

    images = prepare_images(image_paths, args.work_dir, args.max_image_side)
    print("Images ready:", [item["stem"] for item in images])

    device = select_device()
    print("device:", device)
    model, processor = load_model(args.model_id, device)
    print(f"Model loaded: {args.model_id}")

    results, best = run_experiment(images, model, processor, device)

    out = {
        "images": images,
        "model": args.model_id,
        "device": str(device),
        "method": "TALE-EP prompt candidate optimization",
        "prompt_candidates": PROMPT_CANDIDATES,
        "budget_candidates": TALE_EP_BUDGET_CANDIDATES,
        "best": best,
        "manual_label": {
            "label": "",
            "safety_objects": [],
            "notes": "",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
