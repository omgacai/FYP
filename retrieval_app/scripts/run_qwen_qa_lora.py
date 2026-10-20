#!/usr/bin/env python3
"""Evaluate an image-only Qwen LoRA QA adapter on a frozen question CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT_VERSION = "qa_lora_image_only_v1"
DIRECTIONS = {"top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right", "above", "below"}
QUESTION_NOTES = {
    "shortest_access_path": "Shortest path is the minimum number of direct room-to-room transitions. Count each direct-access edge as one transition.",
}


def load_questions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"question_id", "plan_id", "question", "gold_answer", "answer_format", "answerability"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"{path}: QA CSV requires {sorted(required)}")
    return rows


def make_messages(row: dict[str, str], image_path: Path) -> list[dict[str, Any]]:
    system = (
        "Answer residential floor-plan questions using the supplied floor-plan image. "
        "Base the answer only on visible labels, room boundaries, doors, openings, and spatial layout. "
        "Return exactly one JSON object, with no Markdown, with fields answer, evidence, and reasoning. "
        "evidence must be a JSON array of short visual observations. "
        "reasoning must be one concise sentence explaining how the visual evidence supports the answer."
    )
    question_note = QUESTION_NOTES.get(row.get("category", ""), "")
    text = "\n".join(part for part in (
        f"Question: {row['question']}", question_note,
        f"Required answer format: {row['answer_format']}. Use yes/no/unknown for yes_no_unknown; a base-10 integer for integer; "
        "top/bottom/left/right/top_left/top_right/bottom_left/bottom_right for direction; or a JSON array for room_id_list.",
        'Example: {"answer":"top_right","evidence":["Bedroom 2 is above and right of Entry 2."],"reasoning":"Bedroom 2 is in the top-right position relative to Entry 2."}',
    ) if part)
    return [{"role": "system", "content": system}, {"role": "user", "content": [
        {"type": "image", "image": str(image_path)}, {"type": "text", "text": text},
    ]}]


def parse_response(raw: str, answer_format: str) -> tuple[Any | None, Any | None, str | None, str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, None, None, f"invalid_json:{error.msg}"
    if not isinstance(value, dict) or set(value) != {"answer", "evidence", "reasoning"}:
        return None, None, None, "schema_must_be_answer_evidence_reasoning"
    evidence, reasoning, answer = value["evidence"], value["reasoning"], value["answer"]
    if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
        return None, None, None, "evidence_must_be_nonempty_strings"
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, evidence, None, "reasoning_must_be_nonempty_string"
    if answer_format == "integer" and ((isinstance(answer, int) and not isinstance(answer, bool)) or (isinstance(answer, str) and answer.strip().lstrip("-").isdigit())):
        return str(int(answer)), evidence, reasoning, None
    if answer_format == "yes_no_unknown" and isinstance(answer, str) and answer.strip().lower() in {"yes", "no", "unknown"}:
        return answer.strip().lower(), evidence, reasoning, None
    if answer_format == "direction" and isinstance(answer, str) and answer.strip().lower() in DIRECTIONS:
        normalised = {"top": "above", "bottom": "below"}.get(answer.strip().lower(), answer.strip().lower())
        return normalised, evidence, reasoning, None
    if answer_format == "room_id_list" and isinstance(answer, list) and all(isinstance(item, str) for item in answer):
        return sorted(answer), evidence, reasoning, None
    return None, evidence, reasoning, f"answer_does_not_match_{answer_format}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard arguments")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Qwen QA inference.")
    adapter = args.adapter.expanduser().resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    rows = [row for index, row in enumerate(load_questions(args.questions.expanduser().resolve())) if index % args.num_shards == args.shard_index]
    images, output = args.images_dir.expanduser().resolve(), args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    base = Qwen3VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter)).eval()

    def generate(messages: list[dict[str, Any]]) -> str:
        image_inputs, video_inputs = process_vision_info(messages)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=image_inputs or None, videos=video_inputs or None, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        return processor.batch_decode([out[len(inp):] for inp, out in zip(inputs.input_ids, generated)], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            image_path = images / f"{row['plan_id']}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            messages = make_messages(row, image_path)
            raw = generate(messages)
            answer, evidence, reasoning, error = parse_response(raw, row["answer_format"])
            handle.write(json.dumps({"question_id": row["question_id"], "plan_id": row["plan_id"], "condition": "image_only",
                                     "model": args.model, "adapter": str(adapter), "gold_answer": row["gold_answer"], "answer_format": row["answer_format"],
                                     "raw_output": raw, "parsed_answer": answer, "valid_output": error is None, "parse_error": error,
                                     "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens, "temperature": 0,
                                     "prompt_version": PROMPT_VERSION, "evidence": evidence, "reasoning": reasoning,
                                     "prompt": messages, "decoded_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {row['question_id']}: {'valid' if error is None else error}")


if __name__ == "__main__":
    main()
