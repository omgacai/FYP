# Two-stage graph-conditioned QA LoRA

Manual20 identities are excluded from every training and validation corpus.
They remain evaluation-only.

## Stage 1: CubiGraph door-only silver data

Use the explicit `door_only` policy. It serializes room nodes, adjacency, and
observed `connected_by_door` edges, but emits only these QA categories:

- `count`
- `relative_position`
- positive `direct_access` supported by a listed threshold-door edge

It never emits access negatives, `compound_access`, or
`shortest_access_path`: the door-only topology is incomplete because absent
open passages are unknown.

On SOC, create a fresh SFT dataset before training:

```bash
cd ~/vlm/code/FYP
git pull --ff-only

export CANONICAL_CORPUS=~/vlm/data/cubicasa5k/derived_full_v1/canonical.jsonl
export QA_SFT_OUT=~/vlm/data/qwen_qa_sft/cubigraph_v2_door_only_3q_image_graph_v1
export QA_INPUT_CONDITION=image_graph
export QA_TOPOLOGY_POLICY=door_only

sbatch --partition=gpu --gres=gpu:h100-96:1 slurm/build_qwen_qa_sft_full.sbatch
```

After it succeeds, inspect `dataset_manifest.json`; it must report
`"topology_policy": "door_only"`. Train into a new adapter directory—do not
overwrite the previous image-only or image+graph adapters.

```bash
export CUBICASA_ROOT=~/vlm/data/cubicasa5k
export QA_SFT_DATASET_DIR=~/vlm/data/qwen_qa_sft/cubigraph_v2_door_only_3q_image_graph_v1
export QA_RUN_NAME=cubigraph_v2_door_only_image_graph_batch4_3epochs_v1
export QA_OUTPUT_DIR=~/vlm/outputs/qwen_qa_lora/$QA_RUN_NAME
export QA_INPUT_CONDITION=image_graph
export QA_EPOCHS=3
export QA_PER_DEVICE_TRAIN_BATCH_SIZE=4
export QA_GRADIENT_ACCUMULATION_STEPS=4
sbatch --partition=gpu --gres=gpu:h100-96:1 slurm/qwen_qa_lora.sbatch
```

## Stage 2: reviewed Codex topology data

The 100-plan Codex corpus must be plan-disjoint across train/validation and
retain the raw proposal status on every canonical record:

```json
{
  "topology_coverage": "partial",
  "label_tier": "codex_silver",
  "review_status": "unreviewed",
  "reviewed_edges": [
    {"room_a": "living_1", "room_b": "dining_1", "predicate": "open_connected"}
  ]
}
```

Retain door versus open mechanisms and per-edge provenance in the source
annotation. Initially build this corpus with `positive_access_silver`: it
permits explicit door/open direct-access positives, but no absence-based,
compound, or path claims. A record may declare `reviewed_complete` only after
its topology has been checked sufficiently to make absent-access and path
questions.

Build it with `--topology-policy reviewed_complete` only from that reviewed
canonical corpus:

```bash
export CANONICAL_CORPUS=~/vlm/data/cubicasa5k/codex_topology_100/canonical.jsonl
export QA_SFT_OUT=~/vlm/data/qwen_qa_sft/codex_topology_100_complete_image_graph_v1
export QA_INPUT_CONDITION=image_graph
export QA_TOPOLOGY_POLICY=positive_access_silver
export QA_EDGE_FIELD=reviewed_edges
sbatch --partition=gpu --gres=gpu:h100-96:1 slurm/build_qwen_qa_sft_full.sbatch
```

Do not use the Stage-1 CubiGraph corpus with `reviewed_complete`.
Stage 2 should continue from the Stage-1 adapter at a lower learning rate and
include a sampled 20–30% Stage-1 replay set; preserve plan-disjoint validation.
The launcher supports continuation through `QA_INIT_ADAPTER`; always write the
continued adapter to a fresh `QA_OUTPUT_DIR`.
