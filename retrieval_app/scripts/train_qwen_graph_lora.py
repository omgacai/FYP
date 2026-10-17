#!/usr/bin/env python3
"""LoRA SFT smoke trainer for raster-only Qwen floor-plan graph extraction.

Targets must be produced by build_qwen_graph_sft_dataset.py. The source SVG and
CubiGraph outputs are labels only; the model receives the floor-plan raster and
the fixed parsing instruction.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no SFT records")
    return rows


def resolve_image(record: dict[str, Any], corpus_root: Path) -> Path:
    value = record.get("image_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{record.get('plan_id')}: missing image_path")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else corpus_root / path
    if not path.is_file():
        raise FileNotFoundError(f"{record.get('plan_id')}: missing image {path}")
    return path


@dataclass
class GraphSftCollator:
    processor: Any
    corpus_root: Path

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Batch-size one keeps the multimodal masking logic explicit and is
        # appropriate for the initial A100 smoke run. Gradient accumulation
        # controls the effective batch size.
        if len(features) != 1:
            raise ValueError("GraphSftCollator requires per-device batch size 1")
        record = features[0]
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3 or messages[-1].get("role") != "assistant":
            raise ValueError(f"{record.get('plan_id')}: expected system/user/assistant messages")
        assistant = messages[-1].get("content")
        if not isinstance(assistant, str):
            raise ValueError(f"{record.get('plan_id')}: assistant target must be text")
        prompt_messages = messages[:-1]
        image_path = resolve_image(record, self.corpus_root)
        image_messages = [*prompt_messages]
        # Dataset paths are relative to a different machine's corpus root, so
        # inject the locally resolved path before qwen_vl_utils opens it.
        user_content = list(image_messages[-1]["content"])
        user_content[0] = {**user_content[0], "image": str(image_path)}
        image_messages[-1] = {**image_messages[-1], "content": user_content}

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(image_messages)
        prompt_text = self.processor.apply_chat_template(image_messages, tokenize=False, add_generation_prompt=True)
        full_messages = [*image_messages, {"role": "assistant", "content": assistant}]
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_inputs = self.processor(text=[prompt_text], images=image_inputs or None, videos=video_inputs or None, return_tensors="pt")
        full_inputs = self.processor(text=[full_text], images=image_inputs or None, videos=video_inputs or None, return_tensors="pt")
        prompt_length = prompt_inputs.input_ids.shape[1]
        labels = full_inputs.input_ids.clone()
        labels[:, :prompt_length] = -100
        full_inputs["labels"] = labels
        return full_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--corpus-root", type=Path, required=True, help="Root resolving relative image_path values")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=20, help="Positive value caps a smoke run; -1 uses --epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=501_760)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be positive or -1")
    if args.epochs <= 0 or args.gradient_accumulation_steps < 1:
        parser.error("--epochs and --gradient-accumulation-steps must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Run LoRA training on an allocated CUDA GPU node, not a login node.")
    dataset_dir, corpus_root, output_dir = (args.dataset_dir.expanduser().resolve(), args.corpus_root.expanduser().resolve(), args.output_dir.expanduser().resolve())
    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    val_rows = read_jsonl(dataset_dir / "val.jsonl")
    for record in [*train_rows, *val_rows]:
        resolve_image(record, corpus_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_config.json").write_text(json.dumps({
        "model": args.model, "dataset_dir": str(dataset_dir), "corpus_root": str(corpus_root),
        "train_records": len(train_rows), "val_records": len(val_rows), "epochs": args.epochs,
        "max_steps": args.max_steps, "learning_rate": args.learning_rate, "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
        "gradient_accumulation_steps": args.gradient_accumulation_steps, "max_pixels": args.max_pixels,
        "seed": args.seed, "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0),
    }, indent=2) + "\n", encoding="utf-8")
    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output_dir), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        num_train_epochs=args.epochs, max_steps=args.max_steps, bf16=True, fp16=False,
        logging_steps=1, eval_strategy="steps", eval_steps=5, save_strategy="steps", save_steps=5,
        save_total_limit=2, report_to="none", remove_unused_columns=False, seed=args.seed,
        dataloader_num_workers=0, optim="adamw_torch",
    )
    collator = GraphSftCollator(processor=processor, corpus_root=corpus_root)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_rows, eval_dataset=val_rows, data_collator=collator)
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "adapter"))
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
