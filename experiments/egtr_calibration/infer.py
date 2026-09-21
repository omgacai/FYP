"""Raw frozen EGTR inference. Run in the upstream EGTR GPU environment."""
import argparse
import hashlib
import json
import sys
import time
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
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from PIL import Image
    from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
    from model.egtr import DetrForSceneGraphGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('This runner requires the upstream CUDA environment')
    labels = json.loads(args.labels.read_text())
    config = DeformableDetrConfig.from_pretrained(str(args.artifact))
    config.logit_adjustment = False
    if len(labels['objects']) != config.num_labels or len(labels['relations']) != config.num_rel_labels:
        raise ValueError('Vocabulary dimensions do not match checkpoint config')
    model = DetrForSceneGraphGeneration.from_pretrained(args.architecture, config=config, ignore_mismatched_sizes=True)
    state = torch.load(str(args.checkpoint), map_location='cpu')['state_dict']
    state = {k[6:] if k.startswith('model.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    extractor = DeformableDetrFeatureExtractor.from_pretrained(args.architecture, size=800, max_size=1333)
    raw_dir = args.root / 'raw'
    raw_dir.mkdir(exist_ok=True)
    provenance = {'checkpoint_sha256': digest(args.checkpoint), 'config_sha256': digest(args.artifact / 'config.json'),
                  'labels': labels, 'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(0),
                  'preprocessing': {'size': 800, 'max_size': 1333}, 'logit_adjustment': False}
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
            with torch.no_grad():
                output = model(**inputs, output_attentions=False, output_attention_states=True, output_hidden_states=True)
            torch.cuda.synchronize()
            record['seconds_preprocess_and_forward'] = time.perf_counter() - start
            # Keep full tensors for later threshold selection. pred_rel/connectivity already use sigmoid.
            tensors = {k: output[k].detach().cpu() for k in ('logits', 'pred_boxes', 'pred_rel', 'pred_connectivity')}
            torch.save(tensors, raw_dir / f"{row['plan_id']}.pt")
            scores, classes = tensors['logits'][0].softmax(-1)[:, :config.num_labels].max(-1)
            record['objects'] = [{'query': i, 'label': labels['objects'][int(c)], 'score': float(s), 'bbox_cxcywh': b.tolist()}
                                 for i, (c, s, b) in enumerate(zip(classes, scores, tensors['pred_boxes'][0]))]
            record['status'] = 'ok'
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        target.write_text(json.dumps(record, indent=2) + '\n')
        print(row['plan_id'], record['status'], flush=True)


if __name__ == '__main__':
    main()
