#!/usr/bin/env python3
"""Report teacher-forced QA validation metrics separately for each question category.

This uses the SFT validation split only.  It intentionally does not load
Manual20 or perform free generation: the output is directly comparable with
the per-epoch validation loss/token-accuracy recorded during LoRA training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from retrieval_app.scripts.train_qwen_graph_lora import (
    RasterSftCollator,
    preprocess_logits_for_metrics,
    read_jsonl,
    resolve_image,
    token_accuracy,
)


def run_evaluation(*, rows: list[dict[str, Any]], model: Any, processor: Any,
                   corpus_root: Path, output_dir: Path) -> dict[str, float]:
    """Evaluate one category without changing model parameters."""
    from transformers import Trainer, TrainingArguments

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_eval_batch_size=1,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        ),
        eval_dataset=rows,
        data_collator=RasterSftCollator(processor=processor, corpus_root=corpus_root),
        compute_metrics=token_accuracy,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    return {key: float(value) for key, value in trainer.evaluate().items()
            if isinstance(value, (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=501_760)
    args = parser.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Run validation evaluation on an allocated CUDA GPU node.")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    corpus_root = args.corpus_root.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json under {adapter_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    val_rows = read_jsonl(dataset_dir / "val.jsonl")
    for row in val_rows:
        resolve_image(row, corpus_root)
    categories = sorted({str(row.get("category")) for row in val_rows})
    if "None" in categories:
        raise ValueError("Every validation row must have a category")

    output_dir.mkdir(parents=True)
    config = PeftConfig.from_pretrained(str(adapter_dir))
    processor = AutoProcessor.from_pretrained(str(adapter_dir), max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        config.base_model_name_or_path, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    model.config.use_cache = False

    report: dict[str, Any] = {
        "schema_version": "qa-validation-by-category/1",
        "metric_note": (
            "Teacher-forced target-token metrics on the held-out SFT validation split. "
            "They are not free-generation exact-match metrics and do not use Manual20."
        ),
        "adapter_dir": str(adapter_dir),
        "dataset_dir": str(dataset_dir),
        "validation_records": len(val_rows),
        "category_counts": dict(sorted(Counter(str(row["category"]) for row in val_rows).items())),
        "overall": run_evaluation(rows=val_rows, model=model, processor=processor,
                                  corpus_root=corpus_root, output_dir=output_dir / "overall"),
        "by_category": {},
    }
    for category in categories:
        rows = [row for row in val_rows if row["category"] == category]
        report["by_category"][category] = {
            "questions": len(rows),
            **run_evaluation(rows=rows, model=model, processor=processor,
                             corpus_root=corpus_root, output_dir=output_dir / category),
        }
    (output_dir / "validation_by_category.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
