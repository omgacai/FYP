"""Export only explicitly justified ontology mappings; preserve unsupported coverage."""
import argparse
import json
from pathlib import Path

RELATIONS = {'connected_by_door', 'open_connected', 'adjacent_to'}
ROOMS = {'Bedroom', 'Bath', 'Kitchen', 'LivingRoom', 'Dining', 'Corridor', 'Storage', 'Entry', 'Garage', 'Outdoor', 'Other'}


def adapt(raw, tensors, mapping, object_threshold, relation_threshold):
    if raw['status'] != 'ok':
        return {'plan_id': raw['plan_id'], 'valid': False, 'error': raw.get('error', 'Inference failed')}
    for section, allowed in [('objects', ROOMS), ('relations', RELATIONS)]:
        for source, spec in mapping[section].items():
            if spec['target'] not in allowed or not spec.get('justification', '').strip():
                raise ValueError(f'Invalid or unjustified mapping: {source}')
    nodes, supported, excluded = [], {}, []
    for obj in raw['objects']:
        if obj['score'] < object_threshold:
            continue
        spec = mapping['objects'].get(obj['label'])
        if not spec:
            excluded.append(obj)
            continue
        cx, cy, w, h = obj['bbox_cxcywh']
        box = [max(0, cx-w/2), max(0, cy-h/2), min(1, cx+w/2), min(1, cy+h/2)]
        if box[0] >= box[2] or box[1] >= box[3]:
            excluded.append({**obj, 'reason': 'degenerate box'})
            continue
        q = obj['query']
        supported[q] = obj['score']
        nodes.append({'id': f'q{q}', 'type': spec['target'], 'bbox_xyxy': box, 'confidence': obj['score']})
    labels = raw['provenance']['labels']['relations']
    pairs = {}
    rel = tensors['pred_rel'][0]
    connectivity = tensors['pred_connectivity'][0]
    for a, sa in supported.items():
        for b, sb in supported.items():
            if a == b:
                continue
            for r, name in enumerate(labels):
                spec = mapping['relations'].get(name)
                if not spec:
                    continue
                score = float(rel[a, b, r]) * float(connectivity[a, b].reshape(-1)[0]) * sa * sb
                if score < relation_threshold:
                    continue
                key = tuple(sorted((a, b)))
                if key not in pairs or score > pairs[key]['confidence']:
                    pairs[key] = {'a': f'q{key[0]}', 'b': f'q{key[1]}', 'relation': spec['target'], 'confidence': score}
    ontology_supported = bool(mapping['objects']) and bool(mapping['relations'])
    return {'plan_id': raw['plan_id'], 'valid': ontology_supported,
            'error': None if ontology_supported else 'Room or relation ontology unsupported; do not score as an ordinary empty graph',
            'graph': {'nodes': nodes, 'edges': list(pairs.values())},
            'coverage': {'retained_nodes': len(nodes), 'unsupported_objects': excluded,
                         'unsupported_relation_labels': [r for r in labels if r not in mapping['relations']]},
            'adapter': {'object_threshold': object_threshold, 'relation_threshold': relation_threshold,
                        'relation_score': 'object_a * object_b * relation * connectivity',
                        'symmetric_reduction': 'maximum score over directions and supported types', 'mapping': mapping}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--mapping', required=True, type=Path)
    p.add_argument('--object-threshold', type=float, default=0.3)
    p.add_argument('--relation-threshold', type=float, default=0.01)
    args = p.parse_args()
    import torch
    if not 0 <= args.object_threshold <= 1 or not 0 <= args.relation_threshold <= 1:
        p.error('Thresholds must be in [0,1]')
    mapping = json.loads(args.mapping.read_text())
    directory = args.root / 'predictions/egtr_frozen'
    directory.mkdir(parents=True, exist_ok=True)
    for row in json.loads((args.root / 'manifest.json').read_text()):
        plan = row['plan_id']
        target = directory / f'{plan}.json'
        if target.exists():
            raise FileExistsError(target)
        raw = json.loads((args.root / 'raw' / f'{plan}.json').read_text())
        tensors = torch.load(args.root / 'raw' / f'{plan}.pt', map_location='cpu') if raw['status'] == 'ok' else None
        result = adapt(raw, tensors, mapping, args.object_threshold, args.relation_threshold)
        target.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
