"""Fine-tune EGTR on CubiCasa silver supervision with validation-only selection.

This intentionally avoids the upstream Lightning runner: its old API and its
hard-coded 50-predicate Visual Genome loader do not match this two-predicate
CubiCasa adapter. Manual-20 gold data is never an input to this program.
"""
import argparse
import hashlib
import importlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collate(batch):
    pixels, labels = zip(*batch)
    height = max(value.shape[-2] for value in pixels)
    width = max(value.shape[-1] for value in pixels)
    values = pixels[0].new_zeros((len(pixels), 3, height, width))
    mask = __import__('torch').zeros((len(pixels), height, width), dtype=__import__('torch').long)
    for index, pixel in enumerate(pixels):
        h, w = pixel.shape[-2:]
        values[index, :, :h, :w] = pixel
        mask[index, :h, :w] = 1
    return {'pixel_values': values, 'pixel_mask': mask, 'labels': list(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-train-batches', type=int, default=0, help='0 means all batches')
    parser.add_argument('--max-val-batches', type=int, default=0, help='0 means all batches')
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--num-queries', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if platform.machine() != 'x86_64':
        raise SystemExit('Use an x86_64 GPU node; the confirmed Torch runtime is x86_64.')
    if args.epochs < 1 or args.batch_size < 1 or args.accumulate < 1:
        raise ValueError('epochs, batch-size and accumulate must be positive')
    import torch
    from torch.utils.data import DataLoader
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run in an x86_64 Slurm GPU allocation.')
    try:
        import pycocotools  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as error:
        raise RuntimeError(f'Missing small EGTR dependency: {error.name}. Do not reinstall Torch.') from error
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    from transformers import file_utils
    from transformers.image_transforms import center_to_corners_format
    legacy = importlib.import_module('transformers.models.detr.feature_extraction_detr')
    legacy.center_to_corners_format = center_to_corners_format
    original_cuda_check = file_utils.is_torch_cuda_available
    file_utils.is_torch_cuda_available = lambda: False
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'third_party/egtr'))
    try:
        from data.visual_genome import VGDataset
        from model import deformable_detr
        from model.deformable_detr import DeformableDetrFeatureExtractor
        from model.egtr import DetrForSceneGraphGeneration
    finally:
        file_utils.is_torch_cuda_available = original_cuda_check

    # Use the upstream PyTorch attention fallback directly. It executes on the
    # CUDA tensors but avoids EGTR's repeated, misleading "running on cpu" log.
    def quiet_attention_fallback(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return deformable_detr.ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    deformable_detr.MultiScaleDeformableAttentionFunction.apply = staticmethod(quiet_attention_fallback)

    class CubiCasaVGDataset(VGDataset):
        def _get_rel_tensor(self, rel_tensor):
            channels = len(self.rel_categories)
            relation = torch.zeros([self.num_object_queries, self.num_object_queries, channels])
            if rel_tensor.size == 0:
                return relation
            indices = torch.as_tensor(rel_tensor, dtype=torch.long).T
            indices[-1, :] -= 1
            if indices.shape[0] != 3 or indices.min() < 0 or indices[2].max() >= channels:
                raise ValueError(f'Invalid adapter relation indices: {indices}')
            relation[indices[0, :], indices[1, :], indices[2, :]] = 1.0
            return relation

    train_json = json.loads((args.data / 'train.json').read_text())
    rel_json = json.loads((args.data / 'rel.json').read_text())
    categories, predicates = train_json['categories'], rel_json['rel_categories'][1:]
    if predicates != ['adjacent_to', 'direct_access']:
        raise ValueError(f'Expected primary adapter predicates, received {predicates}')
    if [item['id'] for item in categories] != list(range(1, len(categories) + 1)):
        raise ValueError('Room category IDs must be contiguous and one-based')
    extractor = DeformableDetrFeatureExtractor(size=800, max_size=1333)
    train_data = CubiCasaVGDataset(str(args.data), extractor, 'train', num_object_queries=args.num_queries)
    val_data = CubiCasaVGDataset(str(args.data), extractor, 'val', num_object_queries=args.num_queries)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate)
    original_create_model = deformable_detr.create_model
    def create_without_download(*positional, **kwargs):
        kwargs['pretrained'] = False
        return original_create_model(*positional, **kwargs)
    deformable_detr.create_model = create_without_download
    config = deformable_detr.DeformableDetrConfig.from_pretrained(str(args.config.parent), local_files_only=True)
    config.num_labels, config.num_rel_labels, config.num_queries = len(categories), len(predicates), args.num_queries
    config.use_freq_bias, config.logit_adjustment = False, False
    try:
        model = DetrForSceneGraphGeneration(config)
    finally:
        deformable_detr.create_model = original_create_model
    initial = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    initial = {key.removeprefix('model.'): value for key, value in initial.items()}
    model_state = model.state_dict()
    transferable = {key: value for key, value in initial.items() if key in model_state and model_state[key].shape == value.shape}
    model.load_state_dict(transferable, strict=False)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_validation = 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        start_epoch, best_validation = saved['epoch'] + 1, saved['best_validation_loss']
    args.output.mkdir(parents=True, exist_ok=False)
    config.save_pretrained(args.output / 'model_config')
    metadata = {'created_at': datetime.now(timezone.utc).isoformat(), 'purpose': 'CubiCasa silver-supervision fine-tuning',
                'manual_gold_used': False, 'data': str(args.data.resolve()), 'data_rel_sha256': digest(args.data / 'rel.json'),
                'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest(args.checkpoint),
                'classes': [item['name'] for item in categories], 'relations': predicates, 'args': vars(args),
                'initial_checkpoint_tensors_loaded': len(transferable), 'train_examples': len(train_data), 'val_examples': len(val_data)}
    (args.output / 'provenance.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    metrics_file = (args.output / 'metrics.jsonl').open('x')

    def move(batch):
        return batch['pixel_values'].cuda(non_blocking=True), batch['pixel_mask'].cuda(non_blocking=True), [{key: value.cuda(non_blocking=True) for key, value in target.items()} for target in batch['labels']]

    for epoch in range(start_epoch, args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(train_loader, start=1):
            pixels, mask, labels = move(batch)
            output = model(pixel_values=pixels, pixel_mask=mask, labels=labels, output_attentions=False,
                           output_attention_states=True, output_hidden_states=True)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError(f'Non-finite training loss at epoch={epoch} step={step}: {output.loss}')
            (output.loss / args.accumulate).backward()
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(output.loss.detach().cpu()))
            if step % args.log_every == 0:
                print(json.dumps({'epoch': epoch, 'phase': 'train', 'batch': step,
                                  'mean_loss_so_far': sum(losses) / len(losses)}), flush=True)
            if args.max_train_batches and step >= args.max_train_batches:
                break
        if len(losses) % args.accumulate:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        model.eval(); validation = []
        with torch.inference_mode():
            for step, batch in enumerate(val_loader, start=1):
                pixels, mask, labels = move(batch)
                output = model(pixel_values=pixels, pixel_mask=mask, labels=labels, output_attentions=False,
                               output_attention_states=True, output_hidden_states=True)
                if output.loss is None or not torch.isfinite(output.loss):
                    raise RuntimeError(f'Non-finite validation loss at epoch={epoch} step={step}: {output.loss}')
                validation.append(float(output.loss.detach().cpu()))
                if step % args.log_every == 0:
                    print(json.dumps({'epoch': epoch, 'phase': 'validation', 'batch': step,
                                      'mean_loss_so_far': sum(validation) / len(validation)}), flush=True)
                if args.max_val_batches and step >= args.max_val_batches:
                    break
        record = {'epoch': epoch, 'train_loss': sum(losses) / len(losses), 'validation_loss': sum(validation) / len(validation),
                  'train_batches': len(losses), 'validation_batches': len(validation)}
        print(json.dumps(record), flush=True); metrics_file.write(json.dumps(record) + '\n'); metrics_file.flush()
        best_validation = min(best_validation, record['validation_loss'])
        payload = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                   'best_validation_loss': best_validation, 'record': record, 'metadata': metadata}
        torch.save(payload, args.output / 'last.pt')
        if record['validation_loss'] <= best_validation:
            torch.save(payload, args.output / 'best.pt')
    metrics_file.close()
    print(json.dumps({'status': 'PASS', 'best_validation_loss': best_validation, 'output': str(args.output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
