# Floor-plan retrieval and graph-extraction FYP

This repository keeps buyer-query retrieval (`retrieval_app/`) separate from the
image → parser → SVG → graph inspector (`floorplan_app/`).  For the CubiCasa5K/EGTR
extension, read [the validation specification](CUBICASA_EGTR_VALIDATION_SPEC.md):
source SVG produces targets, while CNN/VLM graphs are predictions to evaluate.

The full corpus remains on SOC storage. See [SOC_CLUSTER_TRAINING_GUIDE.md](SOC_CLUSTER_TRAINING_GUIDE.md).
