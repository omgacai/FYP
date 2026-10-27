#!/usr/bin/env python3
"""LoRA SFT trainer for raster and graph-conditioned Qwen floor-plan tasks.

The graph and QA dataset builders both emit image-plus-chat records. Source SVG
and CubiGraph data may be labels only or explicit serialized prompt context,
depending on the dataset's input-condition provenance.
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


def preprocess_logits_for_metrics(logits: Any, _labels: Any) -> Any:
    """Keep only token IDs so validation metrics do not gather vocabulary logits."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def token_accuracy(eval_prediction: Any) -> dict[str, float]:
    """Teacher-forced target-token accuracy, ignoring the image/prompt tokens."""
    import numpy as np

    predictions, labels = (np.asarray(value) for value in eval_prediction)
    # Causal-LM logits at position t predict the token at t + 1.
    predictions, labels = predictions[:, :-1], labels[:, 1:]
    mask = labels != -100
    if not np.any(mask):
        return {"token_accuracy": 0.0}
    return {"token_accuracy": float(np.mean(predictions[mask] == labels[mask]))}


def tensorboard_callback(log_dir: Path, epoch_metrics_path: Path) -> Any:
    """Use a direct callback for compatibility with the SOC Transformers build."""
    from torch.utils.tensorboard import SummaryWriter
    from transformers import TrainerCallback

    writer = SummaryWriter(log_dir=str(log_dir))

    class MetricsCallback(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_kwargs):
            numeric = {key: float(value) for key, value in (logs or {}).items()
                       if isinstance(value, (int, float))}
            for key, value in numeric.items():
                writer.add_scalar(key, value, state.global_step)
            if numeric:
                print("METRICS " + json.dumps({"step": state.global_step, **numeric}, sort_keys=True), flush=True)

        def on_evaluate(self, _args, state, _control, metrics=None, **_kwargs):
            """Keep a compact, auditable validation record for every epoch."""
            record = {
                "epoch": state.epoch,
                "step": state.global_step,
                **{key: float(value) for key, value in (metrics or {}).items()
                   if isinstance(value, (int, float))},
            }
            with epoch_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        def on_train_end(self, _args, _state, _control, **_kwargs):
            writer.flush()
            writer.close()

    return MetricsCallback()


@dataclass
class RasterSftCollator:
    processor: Any
    corpus_root: Path

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate one or more image/chat records with right padding.

        Qwen vision tensors are variable length because input images have
        different aspect ratios. Token sequences are padded; vision patches are
        concatenated, while ``image_grid_thw`` retains each image boundary.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        rows: list[dict[str, Any]] = []
        for record in features:
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) != 3 or messages[-1].get("role") != "assistant":
                raise ValueError(f"{record.get('plan_id')}: expected system/user/assistant messages")
            assistant = messages[-1].get("content")
            if not isinstance(assistant, str):
                raise ValueError(f"{record.get('plan_id')}: assistant target must be text")
            image_messages = [*messages[:-1]]
            user_content = list(image_messages[-1]["content"])
            image_indices = [index for index, item in enumerate(user_content) if item.get("type") == "image"]
            if image_indices:
                image_path = resolve_image(record, self.corpus_root)
                for index in image_indices:
                    user_content[index] = {**user_content[index], "image": str(image_path)}
            image_messages[-1] = {**image_messages[-1], "content": user_content}
            image_inputs, video_inputs = process_vision_info(image_messages)
            prompt_text = self.processor.apply_chat_template(image_messages, tokenize=False, add_generation_prompt=True)
            full_messages = [*image_messages, {"role": "assistant", "content": assistant}]
            full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            prompt_inputs = self.processor(text=[prompt_text], images=image_inputs or None, videos=video_inputs or None, return_tensors="pt")
            full_inputs = self.processor(text=[full_text], images=image_inputs or None, videos=video_inputs or None, return_tensors="pt")
            labels = full_inputs.input_ids.clone()
            labels[:, :prompt_inputs.input_ids.shape[1]] = -100
            full_inputs["labels"] = labels
            rows.append(full_inputs)

        max_length = max(row["input_ids"].shape[1] for row in rows)
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        batch: dict[str, Any] = {}
        for key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
            padded = []
            for row in rows:
                value = row[key]
                pad_width = max_length - value.shape[1]
                if pad_width:
                    fill = -100 if key == "labels" else (0 if key in {"attention_mask", "mm_token_type_ids"} else pad_token_id)
                    value = torch.nn.functional.pad(value, (0, pad_width), value=fill)
                padded.append(value)
            batch[key] = torch.cat(padded, dim=0)
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [row[key] for row in rows if key in row]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--corpus-root", type=Path, required=True, help="Root resolving relative image_path values")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path,
                        help="Optional existing LoRA adapter to continue training; output must be a fresh directory.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=20, help="Positive value caps a smoke run; -1 uses --epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--eval-strategy", choices=("steps", "epoch"), default="steps")
    parser.add_argument("--save-strategy", choices=("steps", "epoch"), default="steps")
    parser.add_argument("--max-pixels", type=int, default=501_760)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Recompute activations during backward pass to reduce GPU memory use.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--task-name", default="graph_extraction", help="Auditable label for this SFT run.")
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be positive or -1")
    if args.epochs <= 0 or args.per_device_train_batch_size < 1 or args.gradient_accumulation_steps < 1 or args.eval_steps < 1 or args.save_steps < 1:
        parser.error("--epochs, --per-device-train-batch-size, --gradient-accumulation-steps, --eval-steps, and --save-steps must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Run LoRA training on an allocated CUDA GPU node, not a login node.")
    dataset_dir, corpus_root, output_dir = (args.dataset_dir.expanduser().resolve(), args.corpus_root.expanduser().resolve(), args.output_dir.expanduser().resolve())
    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    val_rows = read_jsonl(dataset_dir / "val.jsonl")
    for record in [*train_rows, *val_rows]:
        messages = record.get("messages", [])
        user_content = messages[1].get("content", []) if isinstance(messages, list) and len(messages) > 1 else []
        if any(isinstance(item, dict) and item.get("type") == "image" for item in user_content):
            resolve_image(record, corpus_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_config.json").write_text(json.dumps({
        "task_name": args.task_name, "model": args.model, "dataset_dir": str(dataset_dir), "corpus_root": str(corpus_root),
        "train_records": len(train_rows), "val_records": len(val_rows), "epochs": args.epochs,
        "max_steps": args.max_steps, "learning_rate": args.learning_rate, "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout, "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps, "max_pixels": args.max_pixels,
        "gradient_checkpointing": args.gradient_checkpointing,
        "eval_strategy": args.eval_strategy, "save_strategy": args.save_strategy,
        "eval_steps": args.eval_steps, "save_steps": args.save_steps,
        "init_adapter": str(args.init_adapter.expanduser().resolve()) if args.init_adapter else None,
        "tensorboard_dir": str(output_dir / "tensorboard"),
        "seed": args.seed, "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0),
    }, indent=2) + "\n", encoding="utf-8")
    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    if args.init_adapter:
        init_adapter = args.init_adapter.expanduser().resolve()
        if not (init_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"--init-adapter must contain adapter_config.json: {init_adapter}")
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    else:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    if args.gradient_checkpointing:
        # LoRA leaves input embeddings frozen; checkpointed layers still need
        # their inputs to carry gradients for adapter updates.
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output_dir), per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        num_train_epochs=args.epochs, max_steps=args.max_steps, bf16=True, fp16=False,
        logging_steps=1,
        eval_strategy=args.eval_strategy, eval_steps=args.eval_steps,
        save_strategy=args.save_strategy, save_steps=args.save_steps,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_token_accuracy", greater_is_better=True,
        report_to="none", remove_unused_columns=False, seed=args.seed,
        dataloader_num_workers=0, optim="adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
    )
    collator = RasterSftCollator(processor=processor, corpus_root=corpus_root)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_rows, eval_dataset=val_rows,
        data_collator=collator, compute_metrics=token_accuracy,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[tensorboard_callback(output_dir / "tensorboard", output_dir / "epoch_metrics.jsonl")],
    )
    trainer.train()
    (output_dir / "model_selection.json").write_text(json.dumps({
        "selection_split": "validation",
        "selection_metric": "eval_token_accuracy",
        "greater_is_better": True,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_token_accuracy": trainer.state.best_metric,
        "epoch_metrics_file": "epoch_metrics.jsonl",
        "note": "Teacher-forced target-token accuracy; this is not a free-generation or Manual20 evaluation metric.",
    }, indent=2) + "\n", encoding="utf-8")
    trainer.save_model(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "adapter"))
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
