#!/usr/bin/env python3
"""Evaluate a saved Qwen graph LoRA adapter on the held-out SFT validation set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support both ``python -m`` and direct invocation from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval_app.scripts.train_qwen_graph_lora import (
    GraphSftCollator,
    preprocess_logits_for_metrics,
    read_jsonl,
    resolve_image,
    token_accuracy,
)


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
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("Run adapter evaluation on an allocated CUDA GPU node.")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    corpus_root = args.corpus_root.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    val_rows = read_jsonl(dataset_dir / "val.jsonl")
    for record in val_rows:
        resolve_image(record, corpus_root)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json under {adapter_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    config = PeftConfig.from_pretrained(str(adapter_dir))
    base_model = config.base_model_name_or_path
    processor = AutoProcessor.from_pretrained(str(adapter_dir), max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir), per_device_eval_batch_size=1, bf16=True,
            report_to="none", remove_unused_columns=False, dataloader_num_workers=0,
        ),
        eval_dataset=val_rows,
        data_collator=GraphSftCollator(processor=processor, corpus_root=corpus_root),
        compute_metrics=token_accuracy,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(json.dumps({
        "adapter_dir": str(adapter_dir), "base_model": base_model, "validation_records": len(val_rows),
        **metrics,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
