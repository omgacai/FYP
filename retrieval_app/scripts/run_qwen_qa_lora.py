#!/usr/bin/env python3
"""Evaluate a Qwen LoRA QA adapter with image, graph, or fused inputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval_app.scripts.run_qwen_qa import make_messages as condition_prompt_messages

PROMPT_VERSION = "qa_lora_condition_ablation_v1"
CONDITIONS = ("image_only", "graph_only", "image_graph")
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


def edge_id(a: str, b: str, relation: str) -> str:
    return "--".join(sorted((a, b))) + f":{relation}"


def load_graph(path: Path) -> dict[str, Any]:
    """Serialize a reviewed graph with compact IDs so citations remain short."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    source_nodes = [{key: node.get(key) for key in ("id", "label", "type", "x", "y")} for node in raw.get("nodes", [])]
    ids = {node["id"] for node in source_nodes if isinstance(node.get("id"), str)}
    if len(ids) != len(source_nodes) or not all(isinstance(node.get("label"), str) for node in source_nodes):
        raise ValueError(f"{path}: every node needs a unique string id and label")
    aliases = {node["id"]: f"R{index}" for index, node in enumerate(source_nodes, start=1)}
    nodes = [{**node, "id": aliases[node["id"]]} for node in source_nodes]
    edges = []
    for edge in raw.get("edges", []):
        if edge.get("a") not in ids or edge.get("b") not in ids or not isinstance(edge.get("relation"), str):
            continue
        relation = "direct_access" if edge["relation"] in {"connected_by_door", "open_connected"} else edge["relation"]
        if relation in {"direct_access", "adjacent_to", "uncertain"}:
            a, b = aliases[edge["a"]], aliases[edge["b"]]
            edges.append({"a": a, "b": b, "relation": relation, "edge_id": f"E{len(edges) + 1}"})
    return {"plan_id": raw.get("plan_id"), "nodes": nodes, "edges": edges}


def make_messages(row: dict[str, str], condition: str, image_path: Path | None, graph_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    # Graph-conditioned SFT data uses this shared prompt contract. Reuse it at
    # inference so the image+graph adapter sees the same structure and wording.
    if condition != "image_only":
        return condition_prompt_messages(row, condition, image_path or Path("unused.png"), graph_data)
    if condition == "image_only":
        system = (
            "Answer residential floor-plan questions using the supplied floor-plan image. "
            "Base the answer only on visible labels, room boundaries, doors, openings, and spatial layout. "
            "Return exactly one JSON object, with no Markdown, with fields answer, evidence, and reasoning. "
            "evidence must be a JSON array of short visual observations. "
            "reasoning must be one concise sentence explaining how the visual evidence supports the answer."
        )
        evidence_example = '["Bedroom 2 is above and right of Entry 2."]'
    else:
        input_description = "the supplied room graph only" if condition == "graph_only" else "the supplied floor-plan image and reviewed room graph"
        system = (
            f"Answer residential floor-plan questions using {input_description}. "
            "Use graph edges only as explicit facts: an absent edge is not a negative fact. "
            "A direct_access edge is symmetric and proves direct access in either direction. "
            "Return exactly one JSON object, with no Markdown, with fields answer, evidence, and reasoning. "
            "evidence must be an object with node_ids and edge_ids arrays, citing only supplied R-number and E-number IDs. "
            "reasoning must be one concise sentence grounded in the cited evidence."
        )
        evidence_example = '{"node_ids":["R1","R2"],"edge_ids":["E1"]}'
    question_note = QUESTION_NOTES.get(row.get("category", ""), "")
    text = "\n".join(part for part in (
        f"Question: {row['question']}", question_note,
        f"Required answer format: {row['answer_format']}. Use yes/no/unknown for yes_no_unknown; a base-10 integer for integer; "
        "top/bottom/left/right/top_left/top_right/bottom_left/bottom_right for direction; or a JSON array for room_id_list.",
        f'Example: {{"answer":"top_right","evidence":{evidence_example},"reasoning":"Bedroom 2 is in the top-right position relative to Entry 2."}}',
    ) if part)
    content: list[dict[str, Any]] = []
    if condition in {"image_only", "image_graph"}:
        assert image_path is not None
        content.append({"type": "image", "image": str(image_path)})
    if condition in {"graph_only", "image_graph"}:
        assert graph_data is not None
        content.append({"type": "text", "text": "Floor-plan graph; direct_access merges doors and open connections:\n" + json.dumps(graph_data, separators=(",", ":"))})
    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def parse_response(raw: str, answer_format: str, condition: str, legacy_cardinal_labels: bool = False) -> tuple[Any | None, Any | None, str | None, str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, None, None, f"invalid_json:{error.msg}"
    if not isinstance(value, dict) or set(value) != {"answer", "evidence", "reasoning"}:
        return None, None, None, "schema_must_be_answer_evidence_reasoning"
    evidence, reasoning, answer = value["evidence"], value["reasoning"], value["answer"]
    if condition == "image_only":
        if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
            return None, None, None, "evidence_must_be_nonempty_visual_strings"
    elif not isinstance(evidence, dict) or set(evidence) != {"node_ids", "edge_ids"} or not all(
        isinstance(evidence[key], list) and all(isinstance(item, str) for item in evidence[key])
        for key in ("node_ids", "edge_ids")
    ):
        return None, None, None, "evidence_must_have_node_ids_and_edge_ids"
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, evidence, None, "reasoning_must_be_nonempty_string"
    if answer_format == "integer" and ((isinstance(answer, int) and not isinstance(answer, bool)) or (isinstance(answer, str) and answer.strip().lstrip("-").isdigit())):
        return str(int(answer)), evidence, reasoning, None
    if answer_format == "yes_no_unknown" and isinstance(answer, str) and answer.strip().lower() in {"yes", "no", "unknown"}:
        return answer.strip().lower(), evidence, reasoning, None
    if answer_format == "direction" and isinstance(answer, str) and answer.strip().lower() in DIRECTIONS:
        normalised = answer.strip().lower()
        if legacy_cardinal_labels:
            normalised = {"top": "above", "bottom": "below"}.get(normalised, normalised)
        return normalised, evidence, reasoning, None
    if answer_format == "room_id_list" and isinstance(answer, list) and all(isinstance(item, str) for item in answer):
        return sorted(answer), evidence, reasoning, None
    return None, evidence, reasoning, f"answer_does_not_match_{answer_format}"


def validate_evidence(evidence: Any | None, graph_data: dict[str, Any] | None) -> list[str] | None:
    if graph_data is None or not isinstance(evidence, dict):
        return None
    node_ids = {node["id"] for node in graph_data["nodes"]}
    edge_ids = {edge["edge_id"] for edge in graph_data["edges"]}
    errors = [f"unknown_node_id:{item}" for item in evidence["node_ids"] if item not in node_ids]
    errors.extend(f"unknown_edge_id:{item}" for item in evidence["edge_ids"] if item not in edge_ids)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="image_only")
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--graph-source", default="gold_manual")
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
    if args.condition != "image_only" and args.graph_dir is None:
        parser.error("--graph-dir is required for graph_only and image_graph")
    images, output = args.images_dir.expanduser().resolve(), args.output.expanduser().resolve()
    graphs = args.graph_dir.expanduser().resolve() if args.graph_dir else None
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
            if args.condition in {"image_only", "image_graph"} and not image_path.is_file():
                raise FileNotFoundError(image_path)
            graph_data = load_graph(graphs / f"{row['plan_id']}.graph.json") if graphs else None
            messages = make_messages(row, args.condition, image_path if args.condition != "graph_only" else None, graph_data)
            raw = generate(messages)
            answer, evidence, reasoning, error = parse_response(
                raw, row["answer_format"], args.condition,
                legacy_cardinal_labels=row["answer_format"] == "direction" and row["gold_answer"].strip().lower() in {"above", "below"},
            )
            evidence_errors = validate_evidence(evidence, graph_data)
            handle.write(json.dumps({"question_id": row["question_id"], "plan_id": row["plan_id"], "condition": args.condition,
                                     "model": args.model, "adapter": str(adapter), "gold_answer": row["gold_answer"], "answer_format": row["answer_format"],
                                     "raw_output": raw, "parsed_answer": answer, "valid_output": error is None, "parse_error": error,
                                     "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens, "temperature": 0,
                                     "prompt_version": PROMPT_VERSION, "evidence": evidence, "reasoning": reasoning,
                                     "graph_source": args.graph_source if graph_data else None, "evidence_valid": None if evidence_errors is None else not evidence_errors,
                                     "evidence_errors": evidence_errors,
                                     "prompt": messages, "decoded_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {row['question_id']}: {'valid' if error is None else error}")


if __name__ == "__main__":
    main()
