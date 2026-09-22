"""Run a fixed zero/few-shot Qwen floorplan image-to-graph experiment.

The output is intentionally a small semantic graph JSON rather than SVG. This
measures direct semantic/topological reasoning independently of vectorisation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval_app.scripts.build_qwen_silver_support_from_bundle import normalise, rooms_from_relation_svg
from retrieval_app.scripts.build_qwen_spatial_support import qwen_type
from retrieval_app.vlm_graph.prompts import (
    correction_system_prompt, correction_target_instruction, silver_correction_instruction,
    support_instruction, system_prompt, target_instruction,
)
from retrieval_app.vlm_graph.schema import (
    cubigraph_adjacency, extract_json, graph_from_fixed_candidate,
    validate_correction_patch, validate_graph,
)


RELATIONS = {1: "adjacent_to", 2: "connected_by_door", 3: "open_connected"}


def resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_support(path: Path | None, corpus_root: Path, require_spatial: bool) -> list[dict[str, Any]]:
    if path is None:
        return []
    supports: list[dict[str, Any]] = []
    for item in read_jsonl(path):
        if not isinstance(item.get("graph"), dict):
            raise ValueError("Every few-shot support record needs an inline 'graph' object.")
        supports.append({"image_path": resolve(str(item["image_path"]), corpus_root), "graph": validate_graph(item["graph"], require_spatial=require_spatial)})
    if not supports:
        raise ValueError("Few-shot mode needs at least one support record.")
    return supports


def candidate_from_cubigraph(graph_path: Path, adjacency: dict[str, Any]) -> dict[str, Any]:
    """Bind silver edges to immutable source-SVG room geometry, not VLM boxes."""
    relation_svg = graph_path.with_name(f"{graph_path.stem}_relations.svg")
    if not relation_svg.exists():
        raise FileNotFoundError(f"Missing CubiGraph relation SVG: {relation_svg}")
    width, height, source_rooms = rooms_from_relation_svg(relation_svg)
    rooms = []
    for room_id, bbox in source_rooms:
        scaled = normalise(bbox, width, height)
        rooms.append({
            "id": room_id,
            "type": qwen_type(re.sub(r"[_-]\\d+$", "", room_id)),
            "bbox": scaled,
            "centroid": [round((scaled[0] + scaled[2]) / 2, 2), round((scaled[1] + scaled[3]) / 2, 2)],
        })
    room_ids = {room["id"] for room in rooms}
    edges, seen = [], set()
    for source, neighbours in adjacency.items():
        if source not in room_ids or not isinstance(neighbours, dict):
            continue
        for target, code in neighbours.items():
            key = tuple(sorted((source, target)))
            if target not in room_ids or key in seen or code not in RELATIONS:
                continue
            seen.add(key)
            edges.append({"source": key[0], "target": key[1], "type": RELATIONS[code], "confidence": 1.0})
    return validate_graph({"canvas": {"width": 1000, "height": 1000}, "rooms": rooms, "edges": edges}, require_spatial=True)


def make_messages(supports: list[dict[str, Any]], target: Path, spatial: bool, prompt_version: str, silver_graph: dict[str, Any] | None = None, task: str = "direct") -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for support in supports:
        content.extend([
            {"type": "image", "image": str(support["image_path"])},
            {"type": "text", "text": support_instruction(json.dumps(support["graph"], separators=(",", ":")))},
        ])
    content.append({"type": "image", "image": str(target)})
    if silver_graph is not None:
        content.append({"type": "text", "text": silver_correction_instruction(json.dumps(silver_graph, separators=(",", ":")))})
    content.append({"type": "text", "text": correction_target_instruction() if task == "fixed_node_correction" else target_instruction(spatial=spatial, prompt_version=prompt_version)})
    prompt = correction_system_prompt() if task == "fixed_node_correction" else system_prompt(spatial=spatial, prompt_version=prompt_version)
    return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen zero/few-shot floorplan image-to-graph experiment.")
    parser.add_argument("--manifest", type=Path, required=True, help="CubiCasa image/SVG JSONL manifest.")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL experiment output.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mode", choices=("zero", "few"), default="zero")
    parser.add_argument("--representation", choices=("semantic", "spatial"), default="semantic")
    parser.add_argument("--task", choices=("direct", "fixed_node_correction"), default="direct")
    parser.add_argument("--prompt-version", choices=("baseline", "cubicasa_fewshot_v1", "edge_recall_v1", "edge_recall_json_v1"), default="baseline")
    parser.add_argument("--graph-context", choices=("none", "silver"), default="none", help="Whether to give Qwen the CubiGraph candidate as a correction input.")
    parser.add_argument("--support-manifest", type=Path, help="JSONL with image_path and verified inline graph objects.")
    parser.add_argument("--exclude-plan-id", action="append", default=[], help="Repeatable plan ID exclusion, e.g. for few-shot support plans.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard number for resumable corpus jobs.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of deterministic manifest shards.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024, help="Bound vision tokens; use same value across conditions.")
    parser.add_argument("--json-repair-attempts", type=int, default=1, help="Bounded text-only retries when Qwen returns malformed JSON.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "few" and args.support_manifest is None:
        parser.error("--support-manifest is required in few-shot mode.")
    if args.mode == "zero" and args.support_manifest is not None:
        parser.error("Do not pass --support-manifest in zero-shot mode.")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards).")
    if args.prompt_version == "cubicasa_fewshot_v1" and args.mode != "few":
        parser.error("cubicasa_fewshot_v1 requires --mode few and verified support examples.")
    if args.task == "fixed_node_correction" and args.graph_context != "silver":
        parser.error("fixed_node_correction requires --graph-context silver.")
    if args.task == "fixed_node_correction" and args.representation != "spatial":
        parser.error("fixed_node_correction requires --representation spatial.")
    if args.json_repair_attempts < 0 or args.json_repair_attempts > 2:
        parser.error("--json-repair-attempts must be between 0 and 2.")

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
    spatial = args.representation == "spatial"
    supports = load_support(args.support_manifest.expanduser().resolve() if args.support_manifest else None, corpus_root, require_spatial=spatial)
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

    def generate(messages: list[dict[str, Any]]) -> str:
        """Generate once; repair calls intentionally have no image to save GPU time."""
        image_inputs, video_inputs = process_vision_info(messages)
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt_text], images=image_inputs or None, videos=video_inputs or None,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    rows = read_jsonl(manifest)
    if args.limit:
        rows = rows[: args.limit]
    excluded_plan_ids = set(args.exclude_plan_id)
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index and str(row["plan_id"]) not in excluded_plan_ids]
    with output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(rows, start=1):
            plan_id = str(record["plan_id"])
            if plan_id in existing:
                print(f"[{index}/{len(rows)}] {plan_id}: skipped (already present)")
                continue
            image_path = resolve(str(record["image_path"]), corpus_root)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {plan_id}: {image_path}")
            silver_graph = None
            candidate_graph = None
            graph_path = None
            if args.graph_context == "silver":
                graph_path_value = record.get("graph_path")
                if not graph_path_value:
                    raise ValueError(f"Silver-correction mode requires graph_path for {plan_id}.")
                graph_path = resolve(str(graph_path_value), corpus_root)
                if not graph_path.exists():
                    raise FileNotFoundError(f"Missing CubiGraph JSON for {plan_id}: {graph_path}")
                silver_graph = json.loads(graph_path.read_text(encoding="utf-8"))
                if args.task == "fixed_node_correction":
                    candidate_graph = candidate_from_cubigraph(graph_path, silver_graph)
            candidate_for_prompt = candidate_graph if candidate_graph is not None else silver_graph
            messages = make_messages(supports, image_path, spatial=spatial, prompt_version=args.prompt_version, silver_graph=candidate_for_prompt, task=args.task)
            raw_output = generate(messages)
            result: dict[str, Any] = {
                "plan_id": plan_id,
                "image_path": record["image_path"],
                "mode": args.mode,
                "task": args.task,
                "prompt_version": args.prompt_version,
                "graph_context": args.graph_context,
                "graph_context_provenance": "silver" if silver_graph is not None else None,
                "representation": args.representation,
                "support_count": len(supports),
                "model": args.model,
                "max_pixels": args.max_pixels,
                "max_new_tokens": args.max_new_tokens,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                # Persist the exact prompt ingredients so a run can be
                # inspected without reconstructing a moving codebase.
                "prompt_system": messages[0]["content"],
                "prompt_messages": messages,
                "decoded_at": datetime.now(timezone.utc).isoformat(),
                "raw_output": raw_output,
                "json_repair_outputs": [],
            }
            candidate_output = raw_output
            error: ValueError | json.JSONDecodeError | None = None
            for attempt in range(args.json_repair_attempts + 1):
                try:
                    parsed = extract_json(candidate_output)
                    if args.task == "fixed_node_correction":
                        assert candidate_graph is not None
                        patch = validate_correction_patch(parsed, {room["id"] for room in candidate_graph["rooms"]})
                        result["correction_patch"] = patch
                        result["candidate_graph"] = candidate_graph
                        result["graph"] = graph_from_fixed_candidate(candidate_graph, patch)
                        result["cubigraph_adjacency"] = cubigraph_adjacency(result["graph"])
                    else:
                        result["graph"] = validate_graph(parsed, require_spatial=spatial)
                    result["valid"] = True
                    result["error"] = None
                    result["valid_output"] = candidate_output
                    result["json_repair_count"] = attempt
                    break
                except (ValueError, json.JSONDecodeError) as caught:
                    error = caught
                    if attempt == args.json_repair_attempts:
                        continue
                    required_schema = correction_system_prompt() if args.task == "fixed_node_correction" else system_prompt(
                        spatial=spatial, prompt_version=args.prompt_version
                    )
                    repair_messages = [{
                        "role": "system",
                        "content": required_schema + "\n\nYour previous answer was rejected. Return only a corrected object in this exact schema; preserve its intended factual claims but remove Markdown, prose, and unsupported fields.",
                    }, {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Your previous output failed validation: {caught}\n\nPrevious output:\n{candidate_output}"}],
                    }]
                    candidate_output = generate(repair_messages)
                    result["json_repair_outputs"].append(candidate_output)
            if not result.get("valid"):
                result["graph"] = None
                result["valid"] = False
                result["error"] = str(error or "Unknown JSON validation error")
                result["json_repair_count"] = args.json_repair_attempts
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {plan_id}: {'valid' if result['valid'] else 'INVALID'}")


if __name__ == "__main__":
    main()
