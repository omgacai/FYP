#!/usr/bin/env python3
"""Evaluate QA JSONL runs with exact matching and plan-level bootstrap CIs."""
from __future__ import annotations

import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold(row: dict[str, str]) -> Any:
    value = row["gold_answer"].strip()
    if row["answer_format"] == "integer": return str(int(value))
    if row["answer_format"] in {"yes_no_unknown", "direction"}: return value.lower()
    if row["answer_format"] == "room_id_list": return sorted(json.loads(value))
    return value


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def class_f1(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    per, observed = {}, []
    for label in labels:
        item = prf(sum(r["gold"] == label and r["pred"] == label for r in rows),
                   sum(r["gold"] != label and r["pred"] == label for r in rows),
                   sum(r["gold"] == label and r["pred"] != label for r in rows))
        per[label] = item
        if item["f1"] is not None and any(r["gold"] == label for r in rows): observed.append(item["f1"])
    return {"macro_f1_observed_gold_labels": sum(observed) / len(observed) if observed else None,
            "gold_labels_observed": sorted({str(r["gold"]) for r in rows}), "per_label": per}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_plan[row["plan_id"]].append(row)
    plan_accuracy = {key: sum(r["correct"] for r in value) / len(value) for key, value in by_plan.items()}
    result: dict[str, Any] = {"questions": len(rows), "correct": sum(r["correct"] for r in rows),
        "micro_exact_accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
        "valid_output_rate": sum(r["valid"] for r in rows) / len(rows) if rows else None,
        "plans": len(plan_accuracy), "plan_macro_accuracy": sum(plan_accuracy.values()) / len(plan_accuracy) if plan_accuracy else None,
        "per_plan_accuracy": plan_accuracy}
    for key in ("difficulty", "category", "answer_format", "answerability"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows: groups[str(row[key])].append(row)
        result[f"by_{key}"] = {name: {"questions": len(items), "exact_accuracy": sum(i["correct"] for i in items) / len(items),
            "valid_output_rate": sum(i["valid"] for i in items) / len(items)} for name, items in sorted(groups.items())}
    integer = [r for r in rows if r["answer_format"] == "integer"]; parsed = [r for r in integer if r["valid"]]
    result["integer"] = {"questions": len(integer), "exact_accuracy": sum(r["correct"] for r in integer) / len(integer) if integer else None,
        "parse_rate": len(parsed) / len(integer) if integer else None,
        "mae_on_valid_integer_outputs": sum(abs(int(r["pred"]) - int(r["gold"])) for r in parsed) / len(parsed) if parsed else None}
    ynu = [r for r in rows if r["answer_format"] == "yes_no_unknown"]; direction = [r for r in rows if r["answer_format"] == "direction"]
    result["yes_no_unknown"] = class_f1(ynu, ["yes", "no", "unknown"])
    result["direction"] = class_f1(direction, ["left", "right", "above", "below"])
    unknown = [r for r in rows if r["gold"] == "unknown"]; tp = sum(r["gold"] == "unknown" and r["pred"] == "unknown" for r in rows)
    unknown_metrics = prf(tp, sum(r["pred"] == "unknown" for r in rows) - tp, len(unknown) - tp)
    result["unknown"] = {**unknown_metrics, "gold_unknown_questions": len(unknown),
        "unsupported_answer_rate": sum(r["pred"] != "unknown" for r in unknown) / len(unknown) if unknown else None,
        "warning": None if unknown else "No gold unknown rows: unknown metrics are not estimable."}
    return result


def bootstrap_delta(base: list[dict[str, Any]], candidate: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    a, b = {r["question_id"]: r for r in base}, {r["question_id"]: r for r in candidate}
    grouped: dict[str, list[str]] = defaultdict(list)
    for question_id in a.keys() & b.keys(): grouped[a[question_id]["plan_id"]].append(question_id)
    plans = sorted(grouped)
    if not plans: return {"questions": 0, "point_estimate_pp": None, "ci95_pp": None}
    def score(sample: list[str]) -> float:
        diffs = [(sum(b[q]["correct"] for q in grouped[p]) - sum(a[q]["correct"] for q in grouped[p])) / len(grouped[p]) for p in sample]
        return 100 * sum(diffs) / len(diffs)
    point = score(plans); rng = random.Random(seed); draws = sorted(score([rng.choice(plans) for _ in plans]) for _ in range(samples))
    return {"questions": sum(map(len, grouped.values())), "plans": len(plans), "point_estimate_pp": point,
            "ci95_pp": [draws[int(.025 * (samples - 1))], draws[int(.975 * (samples - 1))]], "bootstrap_samples": samples, "unit": "plan"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one or more frozen QA runs.")
    parser.add_argument("--questions", type=Path, required=True); parser.add_argument("--predictions", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--baseline-condition", default="image_only")
    parser.add_argument("--bootstrap-samples", type=int, default=10000); parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    with args.questions.open(newline="", encoding="utf-8") as handle: questions = {r["question_id"]: r for r in csv.DictReader(handle)}
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list); all_rows: list[dict[str, Any]] = []
    prediction_paths = [path for group in args.predictions for path in group]
    for path in prediction_paths:
        for prediction in read_jsonl(path):
            question = questions.get(prediction.get("question_id"))
            if not question: raise ValueError(f"{path}: unknown question {prediction.get('question_id')}")
            condition = str(prediction.get("condition") or path.stem); source = prediction.get("graph_source")
            label = condition if not source else f"{condition}:{source}"; pred = prediction.get("parsed_answer") if prediction.get("valid_output") else None
            row = {"question_id": question["question_id"], "plan_id": question["plan_id"], "difficulty": question["difficulty"], "category": question["category"],
                "answer_format": question["answer_format"], "answerability": question["answerability"], "gold": gold(question), "pred": pred,
                "valid": bool(prediction.get("valid_output")), "condition": condition, "label": label, "model": prediction.get("model")}
            row["correct"] = row["gold"] == row["pred"]; runs[label].append(row); all_rows.append(row)
    for name, rows in runs.items():
        if len({r["question_id"] for r in rows}) != len(rows): raise ValueError(f"{name}: duplicate question records")
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": "floorplan-qa-evaluation/1", "runs": {name: summary(rows) for name, rows in runs.items()}, "comparisons": {}}
    bases = [(name, rows) for name, rows in runs.items() if rows and rows[0]["condition"] == args.baseline_condition]
    for base_name, base_rows in bases:
        for name, rows in runs.items():
            if name != base_name and rows and rows[0]["model"] == base_rows[0]["model"]:
                report["comparisons"][f"{name} minus {base_name}"] = bootstrap_delta(base_rows, rows, args.bootstrap_samples, args.seed)
    (output / "qa_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["label", "question_id", "plan_id", "difficulty", "category", "answer_format", "answerability", "gold", "pred", "valid", "correct"]
    with (output / "qa_question_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({f: row.get(f) for f in fields} for row in all_rows)
    print(json.dumps({"runs": {k: {f: v[f] for f in ("questions", "plan_macro_accuracy", "valid_output_rate")} for k, v in report["runs"].items()}, "comparisons": report["comparisons"]}, indent=2))


if __name__ == "__main__": main()
