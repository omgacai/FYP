"""One GPU forward/backward/update on the CubiCasa EGTR adapter; no Torch install."""
import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Released EGTR checkpoint config.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-queries', type=int, default=200)
    args = parser.parse_args()
    if platform.machine() != 'x86_64':
        raise SystemExit('Use the verified x86_64 GPU runtime, not an ARM node.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run this inside an x86_64 Slurm GPU allocation.')
    try:
        import pycocotools  # noqa: F401 -- required by torchvision CocoDetection
        import scipy  # noqa: F401 -- required by EGTR Hungarian matching
    except ImportError as error:
        raise RuntimeError(f'Missing small EGTR dependency: {error.name}. Do not reinstall Torch.') from error
    from transformers import file_utils
    from transformers.image_transforms import center_to_corners_format
    legacy = importlib.import_module('transformers.models.detr.feature_extraction_detr')
    legacy.center_to_corners_format = center_to_corners_format
    original_cuda_check = file_utils.is_torch_cuda_available
    file_utils.is_torch_cuda_available = lambda: False
    sys.path.insert(0, str(root / 'third_party/egtr'))
    try:
        from data.visual_genome import VGDataset
        from model import deformable_detr
        from model.deformable_detr import DeformableDetrFeatureExtractor
        from model.egtr import DetrForSceneGraphGeneration
    finally:
        file_utils.is_torch_cuda_available = original_cuda_check
    train_json = json.loads((args.data / 'train.json').read_text())
    relation_json = json.loads((args.data / 'rel.json').read_text())
    categories = train_json['categories']
    relation_categories = relation_json['rel_categories'][1:]
    if [item['id'] for item in categories] != list(range(1, len(categories) + 1)):
        raise ValueError('Room category IDs must be contiguous and one-based.')
    if relation_categories != ['adjacent_to', 'direct_access']:
        raise ValueError(f'Unexpected primary relation labels: {relation_categories}')
    class CubiCasaVGDataset(VGDataset):
        """Adapter-aware replacement for EGTR's upstream hard-coded 50 channels."""
        def _get_rel_tensor(self, rel_tensor):
            relation_count = len(self.rel_categories)
            rel = torch.zeros([self.num_object_queries, self.num_object_queries, relation_count])
            if rel_tensor.size == 0:
                return rel
            indices = torch.as_tensor(rel_tensor, dtype=torch.long).T
            indices[-1, :] -= 1  # COCO adapter relation IDs include no_relation=0.
            if indices.shape[0] != 3 or indices.min() < 0 or indices[2].max() >= relation_count:
                raise ValueError(f'Invalid adapter relation indices: {indices}')
            rel[indices[0, :], indices[1, :], indices[2, :]] = 1.0
            return rel

    extractor = DeformableDetrFeatureExtractor(size=800, max_size=1333)
    dataset = CubiCasaVGDataset(str(args.data), extractor, 'train', num_object_queries=args.num_queries)
    pixels, target = dataset[0]
    # Transformers 4.44 no longer exposes the old feature extractor's
    # pad_and_create_pixel_mask helper. The smoke test is intentionally batch
    # size one, so the resized tensor needs no padding and its full mask is 1.
    encoding = {'pixel_values': pixels.unsqueeze(0),
                'pixel_mask': torch.ones((1, pixels.shape[-2], pixels.shape[-1]), dtype=torch.long)}
    labels = [{key: value.cuda() for key, value in target.items()}]
    target_rel = labels[0]['rel']
    nonzero_relations = target_rel.nonzero()
    if target_rel.shape != (args.num_queries, args.num_queries, len(relation_categories)):
        raise ValueError(f'Unexpected target relation tensor shape: {tuple(target_rel.shape)}')
    if nonzero_relations.numel() and int(nonzero_relations[:, 2].max()) >= len(relation_categories):
        raise ValueError(f'Relation channel exceeds vocabulary: {nonzero_relations[:, 2].unique().tolist()}')
    original_create_model = deformable_detr.create_model
    def create_without_download(*positional, **kwargs):
        kwargs['pretrained'] = False
        return original_create_model(*positional, **kwargs)
    deformable_detr.create_model = create_without_download
    config = deformable_detr.DeformableDetrConfig.from_pretrained(str(args.config.parent), local_files_only=True)
    config.num_labels = len(categories)
    config.num_rel_labels = len(relation_categories)
    config.num_queries = args.num_queries
    config.use_freq_bias = False
    config.logit_adjustment = False
    try:
        model = DetrForSceneGraphGeneration(config)
    finally:
        deformable_detr.create_model = original_create_model
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    checkpoint = {key.removeprefix('model.'): value for key, value in checkpoint.items()}
    target_state = model.state_dict()
    transferable = {key: value for key, value in checkpoint.items() if key in target_state and target_state[key].shape == value.shape}
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.inference_mode():
        preview = model(pixel_values=encoding['pixel_values'].cuda(), pixel_mask=encoding['pixel_mask'].cuda(),
                        output_attentions=False, output_attention_states=True, output_hidden_states=True)
    if preview.pred_rel.shape[-1] != len(relation_categories):
        raise ValueError(f'Model relation channels do not match adapter: {tuple(preview.pred_rel.shape)}')
    print(json.dumps({'target_relation_shape': list(target_rel.shape),
                      'target_relation_channels': nonzero_relations[:, 2].unique().tolist() if nonzero_relations.numel() else [],
                      'model_relation_shape': list(preview.pred_rel.shape)}, indent=2), flush=True)
    del preview
    output = model(pixel_values=encoding['pixel_values'].cuda(), pixel_mask=encoding['pixel_mask'].cuda(), labels=labels,
                   output_attentions=False, output_attention_states=True, output_hidden_states=True)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError(f'Invalid training loss: {output.loss}')
    output.loss.backward()
    gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if not gradients_finite:
        raise RuntimeError('Non-finite gradient')
    optimizer.step()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'PASS', 'created_at': datetime.now(timezone.utc).isoformat(), 'device': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'train_examples': len(dataset), 'classes': [item['name'] for item in categories],
              'relations': relation_categories, 'loss': float(output.loss.detach().cpu()),
              'loss_dict': {key: float(value.detach().cpu()) for key, value in output.loss_dict.items()},
              'checkpoint_tensors_loaded': len(transferable), 'checkpoint_tensors_not_loaded': len(missing),
              'unexpected_checkpoint_tensors': len(unexpected), 'target_room_count': int(labels[0]['class_labels'].numel()),
              'target_relation_tensor_shape': list(labels[0]['rel'].shape)}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
