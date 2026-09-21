"""Raw frozen EGTR inference. Run in the upstream EGTR GPU environment."""
import argparse
import hashlib
import json
import subprocess
import shutil
import sys
import time
import importlib
import platform
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True, help='Directory containing config.json')
    p.add_argument('--checkpoint', type=Path, required=True, help='Explicit official EGTR .ckpt')
    p.add_argument('--labels', type=Path, required=True, help='JSON: objects and relations arrays, in model index order; no background entry')
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2] / 'third_party/egtr')
    p.add_argument('--architecture', default='SenseTime/deformable-detr')
    args = p.parse_args()
    if platform.machine() != 'x86_64':
        raise RuntimeError('Use the verified x86_64 SOC GPU runtime; ARM nodes cannot load this Torch build')
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from PIL import Image
    from transformers import DetrImageProcessor, file_utils
    from transformers.image_transforms import center_to_corners_format
    if not torch.cuda.is_available():
        raise RuntimeError('This runner requires the upstream CUDA environment')
    # EGTR targets an older Transformers API. Supply its moved helper and
    # temporarily suppress optional CUDA-extension compilation at import time.
    legacy = importlib.import_module('transformers.models.detr.feature_extraction_detr')
    legacy.center_to_corners_format = center_to_corners_format
    original_cuda_check = file_utils.is_torch_cuda_available
    file_utils.is_torch_cuda_available = lambda: False
    try:
        from model import deformable_detr
        from model.egtr import DetrForSceneGraphGeneration
    finally:
        file_utils.is_torch_cuda_available = original_cuda_check
    labels = json.loads(args.labels.read_text())
    config = deformable_detr.DeformableDetrConfig.from_pretrained(str(args.artifact), local_files_only=True)
    config.logit_adjustment = False
    if len(labels['objects']) != config.num_labels or len(labels['relations']) != config.num_rel_labels:
        raise ValueError('Vocabulary dimensions do not match checkpoint config')
    # Strict checkpoint loading supplies backbone weights, so avoid an unrelated
    # network download from timm during construction.
    original_create_model = deformable_detr.create_model
    def create_without_download(*positional, **kwargs):
        kwargs['pretrained'] = False
        return original_create_model(*positional, **kwargs)
    deformable_detr.create_model = create_without_download
    try:
        model = DetrForSceneGraphGeneration(config)
    finally:
        deformable_detr.create_model = original_create_model
    state = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)['state_dict']
    state = {k[6:] if k.startswith('model.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    extractor = DetrImageProcessor(size={'shortest_edge': 800, 'longest_edge': 1333})
    raw_dir = args.root / 'raw'
    raw_dir.mkdir(exist_ok=True)
    upstream = subprocess.run(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    provenance = {'egtr_commit': upstream, 'checkpoint_sha256': digest(args.checkpoint), 'config_sha256': digest(args.artifact / 'config.json'),
                  'labels': labels, 'labels_sha256': digest(args.labels), 'feature_extractor': extractor.to_dict(),
                  'upstream_source_hashes': {str(p.relative_to(args.repo)): digest(p) for p in args.repo.rglob('*.py')},
                  'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(0),
                  'preprocessing': {'size': 800, 'max_size': 1333}, 'logit_adjustment': False}
    evidence = args.root / 'provenance/infer'
    evidence.mkdir(parents=True, exist_ok=True)
    for source, name in [(args.artifact / 'config.json', 'model_config.json'), (args.labels, 'labels.json')]:
        target = evidence / name
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source, target)
    (evidence / 'model_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    failed = 0
    for row in json.loads((args.root / 'manifest.json').read_text()):
        target = raw_dir / f"{row['plan_id']}.json"
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        record = {'plan_id': row['plan_id'], 'provenance': provenance, 'image_sha256': row['image_sha256']}
        try:
            image_path = args.root / row['image_path']
            if digest(image_path) != row['image_sha256']:
                raise ValueError('Input image hash mismatch')
            torch.cuda.synchronize()
            start = time.perf_counter()
            with Image.open(image_path) as image:
                inputs = extractor(images=image.convert('RGB'), return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs, output_attentions=False, output_attention_states=True, output_hidden_states=True)
            torch.cuda.synchronize()
            record['seconds_preprocess_and_forward'] = time.perf_counter() - start
            # Keep full tensors for later threshold selection. pred_rel/connectivity already use sigmoid.
            tensors = {k: output[k].detach().cpu() for k in ('logits', 'pred_boxes', 'pred_rel', 'pred_connectivity')}
            torch.save(tensors, raw_dir / f"{row['plan_id']}.pt")
            scores, classes = tensors['logits'][0].softmax(-1)[:, :config.num_labels].max(-1)
            record['objects'] = [{'query': i, 'label': labels['objects'][int(c)], 'score': float(s), 'bbox_cxcywh': b.tolist()}
                                 for i, (c, s, b) in enumerate(zip(classes, scores, tensors['pred_boxes'][0]))]
            # Human-readable ranked triplets supplement the complete tensor archive.
            relations = tensors['pred_rel'][0] * tensors['pred_connectivity'][0]
            triplet_scores = relations * torch.outer(scores, scores).unsqueeze(-1)
            indices = torch.arange(len(scores))
            triplet_scores[indices, indices] = -1
            count = min(100, (len(scores) * (len(scores)-1)) * config.num_rel_labels)
            values, flat_indices = triplet_scores.flatten().topk(count)
            record['top_relations'] = []
            for value, flat in zip(values, flat_indices):
                pair, r = divmod(int(flat), config.num_rel_labels)
                a, b = divmod(pair, len(scores))
                record['top_relations'].append({'subject_query': a, 'object_query': b,
                                                'predicate': labels['relations'][r], 'triplet_score': float(value)})
            record['status'] = 'ok'
        except Exception as exc:
            failed += 1
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        target.write_text(json.dumps(record, indent=2) + '\n')
        print(row['plan_id'], record['status'], flush=True)
    if failed:
        raise SystemExit(f'{failed} images failed; records preserved in {raw_dir}')


if __name__ == '__main__':
    main()
