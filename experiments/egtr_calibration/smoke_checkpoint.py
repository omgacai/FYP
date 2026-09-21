"""Run one annotated raster through the frozen checkpoint; no package installs.

This is a runtime smoke test, not graph evaluation. See run_smoke_soc.sh.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root / 'cubicasa_eval/manual20_v1')
    parser.add_argument('--artifact', type=Path, default=Path.home() / 'aigc-storage/fyp-model-cache/egtr/vg/artifact')
    parser.add_argument('--checkpoint', type=Path, help='Choose explicitly if artifact contains several checkpoints')
    parser.add_argument('--outputs', type=Path, default=Path.home() / 'vlm/outputs/egtr')
    args = parser.parse_args()
    if platform.machine() != 'x86_64':
        raise SystemExit('Use an x86_64 GPU node such as xgph1; the existing Torch is x86_64.')
    rows = json.loads((args.data / 'manifest.json').read_text())
    row = rows[0]
    image_path = args.data / row['image_path']
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if image_hash != row['image_sha256']:
        raise ValueError('Annotated image hash mismatch')
    checkpoints = [args.checkpoint] if args.checkpoint else sorted(args.artifact.rglob('*.ckpt'))
    if len(checkpoints) != 1:
        raise ValueError(f'Expected one checkpoint, found {len(checkpoints)}. Set --checkpoint.')
    checkpoint = checkpoints[0].resolve()
    config_dir = checkpoint.parent.parent

    import torch
    import transformers
    from PIL import Image
    from transformers import DetrImageProcessor, file_utils
    from transformers.image_transforms import center_to_corners_format

    print('Torch:', torch.__version__, 'Transformers:', transformers.__version__, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run inside your x86_64 Slurm GPU allocation.')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    # Bridge the legacy EGTR import for the verified Transformers 4.44.2 overlay.
    legacy = importlib.import_module('transformers.models.detr.feature_extraction_detr')
    legacy.center_to_corners_format = center_to_corners_format
    original_cuda_check = file_utils.is_torch_cuda_available
    file_utils.is_torch_cuda_available = lambda: False
    sys.path.insert(0, str(root / 'third_party/egtr'))
    try:
        from model import deformable_detr
        from model.egtr import DetrForSceneGraphGeneration
    finally:
        file_utils.is_torch_cuda_available = original_cuda_check
    # Skip the backbone download: strict loading below supplies all model weights.
    original_create_model = deformable_detr.create_model

    def create_without_download(*positional, **kwargs):
        kwargs['pretrained'] = False
        return original_create_model(*positional, **kwargs)

    deformable_detr.create_model = create_without_download
    config = deformable_detr.DeformableDetrConfig.from_pretrained(str(config_dir), local_files_only=True)
    config.logit_adjustment = False
    try:
        model = DetrForSceneGraphGeneration(config)
    finally:
        deformable_detr.create_model = original_create_model
    print('Loading checkpoint:', checkpoint, flush=True)
    # Only use the official, trusted downloaded checkpoint (Lightning pickle).
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict({k.removeprefix('model.'): v for k, v in state.items()}, strict=True)
    del state
    model.cuda().eval()
    processor = DetrImageProcessor(size={'shortest_edge': 800, 'longest_edge': 1333})
    with Image.open(image_path) as raster:
        inputs = processor(images=raster.convert('RGB'), return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}
    print('Running:', row['plan_id'], {k: tuple(v.shape) for k, v in inputs.items()}, flush=True)
    with torch.inference_mode():
        output = model(**inputs, output_attentions=False, output_attention_states=True, output_hidden_states=True)
    tensors = {k: output[k].detach().cpu() for k in ('logits', 'pred_boxes', 'pred_rel', 'pred_connectivity')}
    for key, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite output: {key}')
        print(key, tuple(tensor.shape), flush=True)
    destination = args.outputs / datetime.now(timezone.utc).strftime('smoke_%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True, exist_ok=False)
    torch.save(tensors, destination / 'raw.pt')
    info = dict(plan_id=row['plan_id'], image_path=str(image_path.resolve()), image_sha256=image_hash,
                checkpoint=str(checkpoint), torch=torch.__version__, transformers=transformers.__version__,
                device=torch.cuda.get_device_name(0), purpose='Runtime smoke test; not graph metrics',
                backend='Upstream PyTorch attention fallback; CUDA tensors; custom extension compilation skipped',
                preprocessing=processor.to_dict())
    (destination / 'info.json').write_text(json.dumps(info, indent=2) + '\n')
    print('FIRST IMAGE: PASS', destination, flush=True)


if __name__ == '__main__':
    main()
