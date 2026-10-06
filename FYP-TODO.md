# FYP to-do list: floorplan question answering

Created: 28 September 2026

## Main objective

Test whether explicit room graphs improve floorplan QA, which question types benefit, and how graph quality affects answer accuracy. Use those results to decide whether further scene graph generation (SGG) work is worthwhile.

All unchecked items are planned work, not verified completed work. Check them off when the stated output exists.

## Start here — next three priorities

- [ ] **Define the QA task and graph schema.** Write a short specification covering room IDs/types, wall adjacency versus doorway access, question categories, answer formats, and when the answer must be unknown. Finish when every proposed question has clearly defined required evidence.
- [ ] **Prepare a small, reviewed development QA set.** Use plans outside Manual20; start with about 5 development plans and 20–30 questions spanning the categories below. Store gold answers and supporting rooms/edges/paths. Finish when every answer has been checked against the image and annotations.
- [ ] **Run the first baseline comparison.** Evaluate image only, gold graph only, and image + gold graph on identical questions. Record model/version, prompts, tool access, settings, outputs, and accuracy by category. Finish with one comparison table and examples of failures.

## 1. Define the benchmark

- [ ] Include room identification/counting as simple controls.
- [ ] Include single-hop questions about direct room access.
- [ ] Include multi-hop questions about paths and minimum doorway counts.
- [ ] Include constrained paths, such as reaching a bathroom without crossing another bedroom.
- [ ] Include questions where incomplete annotations require an unknown answer.
- [ ] Define how questions identify individual rooms when several have the same type.
- [ ] Confirm that access annotations exist before generating navigation questions; wall adjacency alone does not establish access.
- [ ] Define the evaluation split by floorplan identity and check near duplicates. Keep Manual20 out of training and prompt/template development. If it becomes a development pilot, reserve fresh plans for final testing.
- [ ] Balance answer types where practical so that guessing yes/no or the most common count is not a strong shortcut.

**Output:** QA specification, split manifest, and reviewed question set.

## 2. Create deterministic reference answers

- [ ] Store question ID, plan ID, category, question text, gold answer, supporting room IDs/edges/path, and answerability for each question.
- [ ] Implement graph operations for counting, direct access, shortest paths, and constrained reachability.
- [ ] Check generated answers against manual annotations and the floorplan image.
- [ ] Freeze reference answers before graph-quality experiments. Never regenerate gold answers from degraded or predicted graphs.
- [ ] Handle incomplete relations explicitly: a missing annotation is not automatically evidence that no connection exists.

**Output:** A reproducible reference-answer generator and reviewed QA records.

## 3. Compare graph quality and QA performance

- [ ] Establish image-only, gold-graph-only, and image + gold-graph results first.
- [ ] Run a deterministic graph solver as a diagnostic, separately from natural-language QA, to isolate extraction errors from question interpretation/reasoning errors.
- [ ] Test existing EGTR predictions, both graph only and image + graph.
- [ ] Create controlled graph errors separately: missing rooms, wrong room types, missing access edges, and spurious access edges.
- [ ] Test increasing error levels, then selected combinations. Save corruption settings and random seeds.
- [ ] Use the same questions and frozen reference answers across conditions.
- [ ] Report answer accuracy by question category and path length, evidence/path validity, and handling of unknown answers.
- [ ] Report node/edge F1 alongside QA accuracy; inspect which errors actually change answers.
- [ ] Record failures under perception, graph extraction, question interpretation, and graph reasoning.
- [ ] Treat Manual20 results as small-sample evidence; quantify uncertainty across plans rather than treating all questions as independent plans.

**Output:** Graph-quality versus QA results table/plot and error analysis.

## 4. Analyse the Codex workflow

- [ ] Compare direct image answering with an explicit extract-graph-then-answer workflow on the same development questions.
- [ ] Save inputs, prompts, available tools, answers, explicit graphs, tool calls, and cited evidence.
- [ ] Check whether the intermediate graph and supporting evidence are correct.
- [ ] Identify useful, reproducible steps and recurring failure modes. Do not treat written explanations as proof of hidden internal reasoning.

**Output:** A short workflow comparison with representative successes and failures.

## 5. Decide which component to improve

- [ ] Review whether gold graphs improve QA over image-only inputs, especially for relational questions.
- [ ] If gold graphs help but predicted graphs lose that benefit, prioritise graph extraction improvements.
- [ ] Before longer EGTR training, run the proposed tiny-training-subset fitting check and inspect whether rooms and relations can be learned.
- [ ] If QA fails with correct gold graphs, prioritise question interpretation or graph reasoning.
- [ ] If image-based QA has a clear visual failure mode, define a targeted fine-tuning experiment for the image-based method.
- [ ] Clarify what “open-source BE” means, its pipeline role, and its training objective before planning its fine-tuning.
- [ ] For any fine-tuning experiment, define training/validation data, a baseline, compute budget, and success criterion before starting. Keep test plans excluded.
- [ ] If graphs show little benefit, revisit question coverage and the evidence before committing more time to SGG.

**Output:** A written decision identifying the next component to improve and the QA evidence supporting it.

## 6. Document the findings

- [ ] Keep a results log with dataset versions, model/checkpoint identifiers, prompts, configurations, seeds, and output locations.
- [ ] Prepare a supervisor update covering the QA task, baseline findings, graph-quality effects, and next decision.
- [ ] Separate observed results from hypotheses and planned experiments in the report.

## Existing context — reported in the previous discussion

- Training: 3,789 plans; validation: 764 plans; manual gold test: 20 plans.
- Manual20 EGTR continuation: node F1 24.3%, edge F1 3.4%.
- Manual20 EGTR heads-only: node F1 24.4%, edge F1 3.0%.
- Raising detection thresholds did not resolve the extraction problem.
- These numbers are carried over from the supplied summary, not revalidated when creating this checklist.

## Working rule

Complete the first QA comparison before committing to substantial new training. Keep at most three tasks active at once; add dates once workload and deadlines are known.
