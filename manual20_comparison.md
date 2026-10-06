# Manual-20 comparison

| Metric | Qwen3-VL-8B original baseline | Codex-assisted | CubiCasa CNN + CubiGraph | EGTR epoch-3 baseline | EGTR continuation 873168 | EGTR original pretrained, heads-only 873306 |
|---|---|---|---|---|---|---|
| Evaluated plans | 9/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| Node precision | 54.0% | 94.8% | 77.7% | — | 17.9% | 19.8% |
| Node recall | 32.9% | 96.1% | 77.3% | 0.0% | 37.7% | 31.9% |
| Node F1 | 40.9% | 95.4% | 77.5% | 0.0% | 24.3% | 24.4% |
| Edge precision | 15.0% | 80.4% | 43.5% | — | 24.0% | 55.6% |
| Edge recall | 4.8% | 78.0% | 41.9% | 0.0% | 1.8% | 1.5% |
| Edge F1 | 7.3% | 79.2% | 42.7% | 0.0% | 3.4% | 3.0% |
| direct_access F1 | 7.1% | 78.1% | 47.2% | 0.0% | 5.0% | 4.4% |
| adjacent_to F1 | 7.7% | 81.0% | 37.6% | 0.0% | 0.0% | 0.0% |
| Matched type accuracy (macro) | 69.8% | 89.5% | 84.7% | — | 68.7% | 68.5% |
| Mean matched IoU | 0.485 | 0.884 | 0.700 | — | 0.720 | 0.683 |
| Mean aligned edit cost | 11.33 | 5.05 | 10.25 | 10.35 | 29.05 | 23.50 |
| Excluded pairs | 1 | 22 | 19 | 0 | 0 | 0 |

All use class-blind matching IoU >=0.30 and merged direct_access relations.

Qwen scores cover only 9 valid plans; all other methods cover 20.

Codex is assisted visual interpretation with programmatic JSON assembly.

EGTR 873168 continues from three CubiCasa epochs with frozen backbone and trainable transformer; 873306 starts from original pretrained EGTR with backbone and transformer frozen. Neither is trained from scratch.

EGTR object threshold 0.30 and relation threshold 0.01. Undefined precision is blank, not zero.
