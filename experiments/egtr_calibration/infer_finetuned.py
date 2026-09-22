"""Run a validation-selected CubiCasa-fine-tuned EGTR checkpoint on manual gold.

This is image-only inference: manual graph files are read only by the separate
evaluator, never by this program. Predictions retain their raw tensors and a
thresholded graph for transparent post-hoc inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Immutable manual-20 run directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fine-tuned best.pt checkpoint.")
    parser.add_argument("--model-config", type=Path, required=True, help="Fine-tuned model_config directory.")
    parser.add_argument("--object-threshold", type=float, default=0.30)
    parser.add_argument("--relation-threshold", type=float, default=0.01)
    parser.add_argument("--max-objects", type=int, default=100)
    args = parser.parse_args()
    if platform.machine() != "x86_64":
        raise RuntimeError("Use the verified x86_64 SOC GPU runtime.")
    if not 0 <= args.object_threshold <= 1 or not 0 <= args.relation_threshold <= 1:
        parser.error("thresholds must be in [0, 1]")
    import torch
    from PIL import Image
    from transformers import file_utils
    from transformers.image_transforms import center_to_corners_format
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for fine-tuned EGTR inference.")

    root = args.root.expanduser().resolve()
    checkpoint_path, config_path = args.checkpoint.expanduser().resolve(), args.model_config.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2] / "third_party/egtr"
    legacy = importlib.import_module("transformers.models.detr.feature_extraction_detr")
    legacy.center_to_corners_format = center_to_corners_format
    original_cuda_check = file_utils.is_torch_cuda_available
    file_utils.is_torch_cuda_available = lambda: False
    sys.path.insert(0, str(repo))
    try:
        from model import deformable_detr
        from model.deformable_detr import DeformableDetrFeatureExtractor
        from model.egtr import DetrForSceneGraphGeneration
    finally:
        file_utils.is_torch_cuda_available = original_cuda_check

    # Keep the same PyTorch CUDA attention fallback used during fine-tuning.
    def quiet_attention(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return deformable_detr.ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    deformable_detr.MultiScaleDeformableAttentionFunction.apply = staticmethod(quiet_attention)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes, relations = payload["metadata"]["classes"], payload["metadata"]["relations"]
    config = deformable_detr.DeformableDetrConfig.from_pretrained(str(config_path), local_files_only=True)
    if config.num_labels != len(classes) or config.num_rel_labels != len(relations):
        raise ValueError("Checkpoint metadata and saved model config disagree on vocabulary dimensions")
    original_create_model = deformable_detr.create_model
    def create_without_download(*positional, **keywords):
        keywords["pretrained"] = False
        return original_create_model(*positional, **keywords)
    deformable_detr.create_model = create_without_download
    try:
        model = DetrForSceneGraphGeneration(config)
    finally:
        deformable_detr.create_model = original_create_model
    model.load_state_dict(payload["model_state"], strict=True)
    model.cuda().eval()
    extractor = DeformableDetrFeatureExtractor(size=800, max_size=1333)

    raw_dir, prediction_dir = root / "raw", root / "predictions" / "egtr_finetuned"
    raw_dir.mkdir(exist_ok=True); prediction_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": digest(checkpoint_path),
                  "model_config": str(config_path), "model_config_sha256": digest(config_path / "config.json"),
                  "classes": classes, "relations": relations, "device": torch.cuda.get_device_name(0),
                  "torch": torch.__version__, "object_threshold": args.object_threshold,
                  "relation_threshold": args.relation_threshold, "max_objects": args.max_objects,
                  "inference": "image_only_fine_tuned_egtr"}
    evidence = root / "provenance" / "infer_finetuned"; evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "model_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for row in json.loads((root / "manifest.json").read_text()):
        plan_id = str(row["plan_id"])
        raw_path, prediction_path = raw_dir / f"{plan_id}.json", prediction_dir / f"{plan_id}.json"
        if raw_path.exists() or prediction_path.exists():
            raise FileExistsError(f"Refusing to overwrite inference artifacts for {plan_id}")
        raw = {"plan_id": plan_id, "provenance": provenance}
        try:
            image_path = root / row["image_path"]
            if digest(image_path) != row["image_sha256"]:
                raise ValueError("Manual image hash mismatch")
            with Image.open(image_path) as image:
                encoded = extractor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {key: value.cuda() for key, value in encoded.items()}
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode():
                output = model(**inputs, output_attentions=False, output_attention_states=True, output_hidden_states=True)
            torch.cuda.synchronize()
            tensors = {name: output[name].detach().cpu() for name in ("logits", "pred_boxes", "pred_rel", "pred_connectivity")}
            torch.save(tensors, raw_dir / f"{plan_id}.pt")
            scores, class_ids = tensors["logits"][0].softmax(-1)[:, :len(classes)].max(-1)
            selected = [index for index, score in enumerate(scores) if float(score) >= args.object_threshold]
            selected = sorted(selected, key=lambda index: float(scores[index]), reverse=True)[:args.max_objects]
            nodes = []
            for query in selected:
                cx, cy, width, height = tensors["pred_boxes"][0, query].tolist()
                box = [max(0.0, cx - width / 2), max(0.0, cy - height / 2), min(1.0, cx + width / 2), min(1.0, cy + height / 2)]
                if box[0] < box[2] and box[1] < box[3]:
                    nodes.append({"id": f"q{query}", "type": classes[int(class_ids[query])], "bbox_xyxy": box,
                                  "confidence": float(scores[query])})
            selected = [int(node["id"][1:]) for node in nodes]
            edges: dict[tuple[int, int], dict] = {}
            for left in selected:
                for right in selected:
                    if left == right:
                        continue
                    for relation_index, relation in enumerate(relations):
                        score = (float(tensors["pred_rel"][0, left, right, relation_index]) *
                                 float(tensors["pred_connectivity"][0, left, right, 0]) *
                                 float(scores[left]) * float(scores[right]))
                        if score < args.relation_threshold:
                            continue
                        key = tuple(sorted((left, right)))
                        candidate = {"a": f"q{key[0]}", "b": f"q{key[1]}", "relation": relation, "confidence": score}
                        if key not in edges or score > edges[key]["confidence"]:
                            edges[key] = candidate
            raw.update({"status": "ok", "seconds_preprocess_and_forward": time.perf_counter() - started,
                        "objects": [{"query": index, "label": classes[int(class_ids[index])], "score": float(scores[index]),
                                     "bbox_cxcywh": tensors["pred_boxes"][0, index].tolist()} for index in range(len(scores))]})
            prediction = {"plan_id": plan_id, "valid": True, "status": "ok", "provenance": provenance,
                          "graph": {"nodes": nodes, "edges": list(edges.values())}}
        except Exception as error:
            raw.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            prediction = {"plan_id": plan_id, "valid": False, "error": raw["error"]}
        raw_path.write_text(json.dumps(raw, indent=2) + "\n")
        prediction_path.write_text(json.dumps(prediction, indent=2) + "\n")
        print(plan_id, raw["status"], flush=True)


if __name__ == "__main__":
    main()
