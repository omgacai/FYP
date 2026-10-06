#!/usr/bin/env python3
"""Run one frozen Qwen QA condition; raw/invalid outputs are retained."""
from __future__ import annotations

import argparse, csv, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def questions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"question_id", "plan_id", "question", "gold_answer", "answer_format", "answerability"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"QA CSV requires {sorted(required)}")
    return rows


def graph(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    nodes = [{key: node.get(key) for key in ("id", "label", "type", "x", "y")} for node in raw.get("nodes", [])]
    ids = {node["id"] for node in nodes if isinstance(node.get("id"), str)}
    if len(ids) != len(nodes) or not all(isinstance(node.get("label"), str) for node in nodes):
        raise ValueError(f"{path}: every node needs unique id and label")
    edges = []
    for edge in raw.get("edges", []):
        if edge.get("a") in ids and edge.get("b") in ids and isinstance(edge.get("relation"), str):
            relation = "direct_access" if edge["relation"] in {"connected_by_door", "open_connected"} else edge["relation"]
            if relation in {"direct_access", "adjacent_to", "uncertain"}:
                edges.append({"a": edge["a"], "b": edge["b"], "relation": relation})
    return {"plan_id": raw.get("plan_id"), "nodes": nodes, "edges": edges}


def parse(raw: str, answer_format: str) -> tuple[Any | None, str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, f"invalid_json:{error.msg}"
    if not isinstance(value, dict) or set(value) != {"answer"}:
        return None, "schema_must_be_exactly_answer"
    answer = value["answer"]
    if answer_format == "integer":
        if isinstance(answer, int) and not isinstance(answer, bool): return str(answer), None
        if isinstance(answer, str) and answer.strip().lstrip("-").isdigit(): return str(int(answer.strip())), None
    if answer_format == "yes_no_unknown" and isinstance(answer, str) and answer.strip().lower() in {"yes", "no", "unknown"}:
        return answer.strip().lower(), None
    if answer_format == "direction" and isinstance(answer, str) and answer.strip().lower() in {"left", "right", "above", "below"}:
        return answer.strip().lower(), None
    if answer_format == "room_id_list" and isinstance(answer, list) and all(isinstance(item, str) for item in answer):
        return sorted(answer), None
    return None, f"answer_does_not_match_{answer_format}"


def make_messages(row: dict[str, str], condition: str, image_path: Path, graph_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    system = ("Answer residential floor-plan questions from supplied evidence only. Do not infer an absent graph edge as a negative fact. "
              "Return exactly one JSON object, no Markdown, with exactly one field: answer.")
    content: list[dict[str, Any]] = []
    if condition in {"image_only", "image_graph"}:
        content.append({"type": "image", "image": str(image_path)})
    if condition in {"graph_only", "image_graph"}:
        assert graph_data is not None
        content.append({"type": "text", "text": "Floor-plan graph; direct_access merges doors and open connections:\n" + json.dumps(graph_data, separators=(",", ":"))})
    content.append({"type": "text", "text": (
        f"Question: {row['question']}\nRequired answer format: {row['answer_format']}. "
        "Use yes/no/unknown for yes_no_unknown; a base-10 integer for integer; left/right/above/below for direction; "
        "or a JSON array for room_id_list. Example: {\"answer\":\"yes\"}."
    )})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen Qwen floor-plan QA input condition.")
    parser.add_argument("--questions", type=Path, required=True); parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=("image_only", "graph_only", "image_graph"), required=True)
    parser.add_argument("--graph-dir", type=Path); parser.add_argument("--graph-source", default="gold_manual")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct"); parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--max-new-tokens", type=int, default=96); parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards: parser.error("invalid shard arguments")
    if args.condition != "image_only" and args.graph_dir is None: args.graph_dir = args.annotations_dir
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available(): raise RuntimeError("A CUDA GPU is required for Qwen QA inference.")
    rows = [row for i, row in enumerate(questions(args.questions.expanduser().resolve())) if i % args.num_shards == args.shard_index]
    images, graphs, output = args.images_dir.expanduser().resolve(), args.graph_dir.expanduser().resolve() if args.graph_dir else None, args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if output.exists() and not args.overwrite: done = {json.loads(line)["question_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    elif output.exists(): output.unlink()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    def generate(messages: list[dict[str, Any]]) -> str:
        image_inputs, video_inputs = process_vision_info(messages)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=image_inputs or None, videos=video_inputs or None, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode(): generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        return processor.batch_decode([out[len(inp):] for inp, out in zip(inputs.input_ids, generated)], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    with output.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(rows, 1):
            if row["question_id"] in done: continue
            image_path = images / f"{row['plan_id']}.png"
            if args.condition in {"image_only", "image_graph"} and not image_path.is_file(): raise FileNotFoundError(image_path)
            graph_data = graph(graphs / f"{row['plan_id']}.graph.json") if graphs else None
            messages = make_messages(row, args.condition, image_path, graph_data)
            raw = generate(messages); answer, error = parse(raw, row["answer_format"])
            record = {"question_id": row["question_id"], "plan_id": row["plan_id"], "condition": args.condition,
                      "graph_source": args.graph_source if graph_data else None, "model": args.model, "answer_format": row["answer_format"],
                      "gold_answer": row["gold_answer"], "raw_output": raw, "parsed_answer": answer, "valid_output": error is None,
                      "parse_error": error, "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens, "temperature": 0,
                      "prompt": messages, "decoded_at": datetime.now(timezone.utc).isoformat()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
            print(f"[{i}/{len(rows)}] {row['question_id']}: {'valid' if error is None else error}")


if __name__ == "__main__": main()
