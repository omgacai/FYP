#!/usr/bin/env python3
"""Run one frozen Qwen QA condition; raw/invalid outputs are retained."""
from __future__ import annotations

import argparse, csv, json, os
from dataclasses import dataclass
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


def graph(path: Path, compact_evidence_ids: bool = False) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    source_nodes = [{key: node.get(key) for key in ("id", "label", "type", "x", "y")} for node in raw.get("nodes", [])]
    ids = {node["id"] for node in source_nodes if isinstance(node.get("id"), str)}
    if len(ids) != len(source_nodes) or not all(isinstance(node.get("label"), str) for node in source_nodes):
        raise ValueError(f"{path}: every node needs unique id and label")
    aliases = {node["id"]: f"R{index}" for index, node in enumerate(source_nodes, start=1)}
    nodes = [{**node, "id": aliases[node["id"]]} if compact_evidence_ids else node for node in source_nodes]
    edges = []
    for edge in raw.get("edges", []):
        if edge.get("a") in ids and edge.get("b") in ids and isinstance(edge.get("relation"), str):
            relation = "direct_access" if edge["relation"] in {"connected_by_door", "open_connected"} else edge["relation"]
            if relation in {"direct_access", "adjacent_to", "uncertain"}:
                a, b = (aliases[edge["a"]], aliases[edge["b"]]) if compact_evidence_ids else (edge["a"], edge["b"])
                edges.append({"a": a, "b": b, "relation": relation,
                              "edge_id": f"E{len(edges) + 1}" if compact_evidence_ids else edge_id(a, b, relation)})
    return {"plan_id": raw.get("plan_id"), "nodes": nodes, "edges": edges}


PROMPT_VERSION = "condition_specific_evidence_v1"
CONDITIONS = ("image_only", "graph_only", "image_graph")


@dataclass(frozen=True)
class PromptSpec:
    system: str
    includes_image: bool
    includes_graph: bool
    evidence_kind: str
    example: str


PROMPT_SPECS = {
    "image_only": PromptSpec(
        system=(
            "Answer residential floor-plan questions using the supplied floor-plan image. "
            "Base the answer only on visible labels, room boundaries, doors, openings, and spatial layout. "
            "Return exactly one JSON object, with no Markdown, with fields answer, evidence, and reasoning. "
            "evidence must be a JSON array of short visual observations. "
            "reasoning must be one concise sentence explaining how the visual evidence supports the answer."
        ),
        includes_image=True,
        includes_graph=False,
        evidence_kind="visual_observations",
        example=(
            '{"answer":"above","evidence":["Bedroom 2 is drawn higher on the plan than Entry 2."],'
            '"reasoning":"Bedroom 2 is vertically above Entry 2 in the floor-plan layout."}'
        ),
    ),
    "graph_only": PromptSpec(
        system=(
            "Answer residential floor-plan questions using the supplied room graph only. "
            "Do not infer an absent graph edge as a negative fact. "
            "A direct_access edge is symmetric: it proves direct access in either direction, and only an explicit "
            "direct_access edge proves it. "
            "Return exactly one JSON object, with no Markdown, with fields answer, evidence, and reasoning. "
            "evidence must be {\"node_ids\":[...],\"edge_ids\":[...]}. Cite only supplied graph IDs. "
            "Graph node IDs use R-number aliases and relation IDs use E-number aliases; cite those aliases exactly. "
            "Use [] when no node or edge ID is needed to support the answer. "
            "reasoning must be one concise sentence grounded in the cited graph evidence."
        ),
        includes_image=False,
        includes_graph=True,
        evidence_kind="graph_citations",
        example=(
            '{"answer":"yes","evidence":{"node_ids":["R1","R8"],"edge_ids":["E3"]},'
            '"reasoning":"E3 is a direct_access edge connecting Bedroom 1 (R1) and Corridor 1 (R8)."}'
        ),
    ),
    "image_graph": PromptSpec(
        system=(
            "Answer residential floor-plan questions using the supplied floor-plan image and room graph. "
            "Use the graph for room identities and explicit relations; use the image for visible layout evidence. "
            "Do not infer an absent graph edge as a negative fact. "
            "A direct_access edge is symmetric: it proves direct access in either direction, and only an explicit "
            "direct_access edge proves it. "
            "Return exactly one JSON object, with no Markdown, with fields answer, evidence, and reasoning. "
            "evidence must be {\"node_ids\":[...],\"edge_ids\":[...]}. Cite only supplied graph IDs. "
            "Graph node IDs use R-number aliases and relation IDs use E-number aliases; cite those aliases exactly. "
            "Use [] when no graph ID is needed. "
            "reasoning must be one concise sentence grounded in the supplied evidence."
        ),
        includes_image=True,
        includes_graph=True,
        evidence_kind="graph_citations",
        example=(
            '{"answer":"1","evidence":{"node_ids":["R6"],"edge_ids":[]},'
            '"reasoning":"R6 is the only graph node labelled Bath, consistent with the single bathroom visible in the plan."}'
        ),
    ),
}

QUESTION_NOTES = {
    "shortest_access_path": (
        "For this question, shortest path means the minimum number of room-to-room transitions needed to travel "
        "from the start node to the end node. Count every direct_access edge in the path: each edge represents "
        "one transition through either a door or a direct open connection. For example, A to B to C has shortest "
        "path length 2 from A to C. Count transitions, not rooms visited."
    ),
}


def edge_id(a: str, b: str, relation: str) -> str:
    return "--".join(sorted((a, b))) + f":{relation}"


def parse_response(raw: str, answer_format: str, condition: str) -> tuple[Any | None, Any | None, str | None, str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, None, None, f"invalid_json:{error.msg}"
    spec = PROMPT_SPECS[condition]
    expected = {"answer", "evidence", "reasoning"}
    if not isinstance(value, dict) or set(value) != expected:
        return None, None, None, f"schema_must_be_exactly_{'_'.join(sorted(expected))}"
    evidence = value["evidence"]
    reasoning = value["reasoning"]
    if spec.evidence_kind == "visual_observations":
        if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
            return None, None, None, "evidence_must_be_nonempty_visual_observation_strings"
    elif not isinstance(evidence, dict) or set(evidence) != {"node_ids", "edge_ids"} or not all(
        isinstance(evidence[key], list) and all(isinstance(item, str) for item in evidence[key])
        for key in ("node_ids", "edge_ids")
    ):
        return None, None, None, "evidence_must_have_string_node_ids_and_edge_ids"
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, None, None, "reasoning_must_be_nonempty_string"
    answer = value["answer"]
    if answer_format == "integer":
        if isinstance(answer, int) and not isinstance(answer, bool): return str(answer), evidence, reasoning, None
        if isinstance(answer, str) and answer.strip().lstrip("-").isdigit(): return str(int(answer.strip())), evidence, reasoning, None
    if answer_format == "yes_no_unknown" and isinstance(answer, str) and answer.strip().lower() in {"yes", "no", "unknown"}:
        return answer.strip().lower(), evidence, reasoning, None
    if answer_format == "direction" and isinstance(answer, str) and answer.strip().lower() in {"left", "right", "above", "below"}:
        return answer.strip().lower(), evidence, reasoning, None
    if answer_format == "room_id_list" and isinstance(answer, list) and all(isinstance(item, str) for item in answer):
        return sorted(answer), evidence, reasoning, None
    return None, evidence, reasoning, f"answer_does_not_match_{answer_format}"


def parse(raw: str, answer_format: str, condition: str = "image_only") -> tuple[Any | None, str | None]:
    """Parse the scored answer; detailed evidence stays available in run records."""
    answer, _, _, error = parse_response(raw, answer_format, condition)
    return answer, error


def validate_evidence(evidence: Any | None, graph_data: dict[str, Any] | None) -> list[str] | None:
    if not isinstance(evidence, dict) or graph_data is None:
        return None
    node_ids = {node["id"] for node in graph_data["nodes"]}
    edge_ids = {edge.get("edge_id", edge_id(edge["a"], edge["b"], edge["relation"])) for edge in graph_data["edges"]}
    errors = [f"unknown_node_id:{item}" for item in evidence["node_ids"] if item not in node_ids]
    errors.extend(f"unknown_edge_id:{item}" for item in evidence["edge_ids"] if item not in edge_ids)
    return errors


def make_messages(row: dict[str, str], condition: str, image_path: Path, graph_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    spec = PROMPT_SPECS[condition]
    content: list[dict[str, Any]] = []
    if spec.includes_image:
        content.append({"type": "image", "image": str(image_path)})
    if spec.includes_graph:
        assert graph_data is not None
        content.append({"type": "text", "text": "Floor-plan graph; direct_access merges doors and open connections:\n" + json.dumps(graph_data, separators=(",", ":"))})
    question_note = QUESTION_NOTES.get(row.get("category", ""), "")
    content.append({"type": "text", "text": "\n".join(part for part in (
        f"Question: {row['question']}",
        question_note,
        f"Required answer format: {row['answer_format']}. "
        "Use yes/no/unknown for yes_no_unknown; a base-10 integer for integer; left/right/above/below for direction; "
        "or a JSON array for room_id_list.",
        f"Example: {spec.example}",
    ) if part)})
    return [{"role": "system", "content": spec.system}, {"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen Qwen floor-plan QA input condition.")
    parser.add_argument("--questions", type=Path, required=True); parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
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
            graph_data = graph(graphs / f"{row['plan_id']}.graph.json", compact_evidence_ids=True) if graphs else None
            messages = make_messages(row, args.condition, image_path, graph_data)
            raw = generate(messages); answer, evidence, reasoning, error = parse_response(raw, row["answer_format"], args.condition)
            evidence_errors = validate_evidence(evidence, graph_data)
            record = {"question_id": row["question_id"], "plan_id": row["plan_id"], "condition": args.condition,
                      "graph_source": args.graph_source if graph_data else None, "model": args.model, "answer_format": row["answer_format"],
                      "gold_answer": row["gold_answer"], "raw_output": raw, "parsed_answer": answer, "valid_output": error is None,
                      "parse_error": error, "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens, "temperature": 0,
                      "prompt_version": PROMPT_VERSION, "evidence": evidence, "reasoning": reasoning,
                      "evidence_valid": None if evidence_errors is None else not evidence_errors, "evidence_errors": evidence_errors,
                      "prompt": messages, "decoded_at": datetime.now(timezone.utc).isoformat()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
            print(f"[{i}/{len(rows)}] {row['question_id']}: {'valid' if error is None else error}")


if __name__ == "__main__": main()
