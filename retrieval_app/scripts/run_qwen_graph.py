"""Run a fixed zero/few-shot Qwen floorplan image-to-graph experiment.

The output is intentionally a small semantic graph JSON rather than SVG. This
measures direct semantic/topological reasoning independently of vectorisation.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from retrieval_app.vlm_graph.prompts import SYSTEM_PROMPT, support_instruction, target_instruction
from retrieval_app.vlm_graph.schema import extract_json, validate_graph


def resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_support(path: Path | None, corpus_root: Path) -> list[dict[str, Any]]:
    if path is None:
        return []
    supports: list[dict[str, Any]] = []
    for item in read_jsonl(path):
        if not isinstance(item.get("graph"), dict):
            raise ValueError("Every few-shot support record needs an inline 'graph' object.")
        supports.append({"image_path": resolve(str(item["image_path"]), corpus_root), "graph": validate_graph(item["graph"])})
    if not supports:
        raise ValueError("Few-shot mode needs at least one support record.")
    return supports


def make_messages(supports: list[dict[str, Any]], target: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for support in supports:
        content.extend([
            {"type": "image", "image": str(support["image_path"])},
            {"type": "text", "text": support_instruction(json.dumps(support["graph"], separators=(",", ":")))},
        ])
    content.extend([{"type": "image", "image": str(target)}, {"type": "text", "text": target_instruction()}])
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen zero/few-shot floorplan image-to-graph experiment.")
    parser.add_argument("--manifest", type=Path, required=True, help="CubiCasa image/SVG JSONL manifest.")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL experiment output.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mode", choices=("zero", "few"), default="zero")
    parser.add_argument("--support-manifest", type=Path, help="JSONL with image_path and verified inline graph objects.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024, help="Bound vision tokens; use same value across conditions.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "few" and args.support_manifest is None:
        parser.error("--support-manifest is required in few-shot mode.")
    if args.mode == "zero" and args.support_manifest is not None:
        parser.error("Do not pass --support-manifest in zero-shot mode.")

    # Must be configured before Transformers/Hugging Face import.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this 8B VLM experiment.")

    corpus_root = args.corpus_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    supports = load_support(args.support_manifest.expanduser().resolve() if args.support_manifest else None, corpus_root)
    if args.mode == "few" and len(supports) > 2:
        raise ValueError("Use at most two support examples so few-shot remains a controlled condition.")

    existing: set[str] = set()
    if output.exists() and not args.overwrite:
        existing = {str(row["plan_id"]) for row in read_jsonl(output)}
    elif output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    rows = read_jsonl(manifest)
    if args.limit:
        rows = rows[: args.limit]
    with output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(rows, start=1):
            plan_id = str(record["plan_id"])
            if plan_id in existing:
                print(f"[{index}/{len(rows)}] {plan_id}: skipped (already present)")
                continue
            image_path = resolve(str(record["image_path"]), corpus_root)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {plan_id}: {image_path}")
            messages = make_messages(supports, image_path)
            # qwen-vl-utils loads local image paths and creates correctly ordered vision tensors.
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", images=image_inputs, videos=video_inputs,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            raw_output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            result: dict[str, Any] = {
                "plan_id": plan_id,
                "image_path": record["image_path"],
                "mode": args.mode,
                "support_count": len(supports),
                "model": args.model,
                "max_pixels": args.max_pixels,
                "max_new_tokens": args.max_new_tokens,
                "decoded_at": datetime.now(timezone.utc).isoformat(),
                "raw_output": raw_output,
            }
            try:
                result["graph"] = validate_graph(extract_json(raw_output))
                result["valid"] = True
                result["error"] = None
            except (ValueError, json.JSONDecodeError) as error:
                result["graph"] = None
                result["valid"] = False
                result["error"] = str(error)
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {plan_id}: {'valid' if result['valid'] else 'INVALID'}")


if __name__ == "__main__":
    main()
