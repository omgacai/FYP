"""Select three non-excluded plans and generate source-SVG/CubiGraph silver references."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    parts = Path(path).parts
    return '/'.join(parts[-2:])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True, help='Directory containing colorful/ and high_quality/')
    p.add_argument('--exclusions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', default='egtr-calibration-2026-09-21')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; use a new versioned directory')
    from PIL import Image
    from floorplan_app.core.models import SVGResult
    from floorplan_app.pipeline.graph_extractor import extract_graph
    from retrieval_app.scripts.build_cubicasa_egtr_corpus import source_rooms, canonical_edges
    excluded = {identity(x) for x in json.loads(args.exclusions.read_text())}
    candidates = [s.parent for category in ('colorful', 'high_quality')
                  for s in (args.dataset / category).glob('*/model.svg')
                  if identity(s.parent) not in excluded and (s.parent / 'F1_scaled.png').exists()]
    candidates.sort(key=lambda d: hashlib.sha256(f'{args.seed}:{identity(d)}'.encode()).hexdigest())
    if len(candidates) < 3:
        raise ValueError('Fewer than three eligible plans')
    selected = candidates[:3]
    for name in ('images', 'annotations', 'raw', 'predictions', 'results', 'source'):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2] / 'third_party/CubiGraph5K'
    manifest, questions = [], []
    for directory in selected:
        key = identity(directory)
        plan = '_'.join(reversed(key.split('/')))
        image, svg = directory / 'F1_scaled.png', directory / 'model.svg'
        with Image.open(image) as im:
            width, height = im.size
        rooms = source_rooms(svg, (width, height), repo, svg_coordinate_size=(width, height))
        graph = extract_graph(SVGResult(svg_text=svg.read_text(), path=svg, diagnostics={}), repo,
                              args.output / 'source' / f'{plan}_relations.svg')
        edges = canonical_edges(graph.adjacency, {r['room_id'] for r in rooms})
        nodes = [{'id': r['room_id'], 'type': r['category_name'],
                  'bbox_xyxy': [v / (width if i % 2 == 0 else height) for i, v in enumerate(r['bbox_xyxy'])]}
                 for r in rooms]
        reference = {'plan_id': plan, 'status': 'pipeline_reference', 'provenance': 'source_svg_rooms_and_cubigraph_silver_edges',
                     'all_pairs_reviewed': False,
                     'unsupported_reference_relations': ['open_connected'],
                     'image': {'width': width, 'height': height, 'sha256': sha(image), 'local_path': f'images/{plan}.png'},
                     'nodes': nodes, 'edges': [{'a': e['room_a'], 'b': e['room_b'], 'relation': e['predicate']} for e in edges],
                     'source': {'coordinate_policy': 'CubiCasa SVG points are F1_scaled pixels; no viewport rescaling', 'plan_identity': key, 'svg_sha256': sha(svg), 'diagnostics': graph.diagnostics,
                                'extractor_sha256': sha(Path(__file__).resolve().parents[2] / 'floorplan_app/pipeline/graph_extractor.py')}}
        annotation = args.output / 'annotations' / f'{plan}.graph.json'
        annotation.write_text(json.dumps(reference, indent=2) + '\n')
        shutil.copy2(image, args.output / 'images' / f'{plan}.png')
        shutil.copy2(svg, args.output / 'source' / f'{plan}.svg')
        manifest.append({'plan_id': plan, 'source_identity': key, 'split': 'calibration', 'image_path': f'images/{plan}.png',
                         'image_sha256': sha(image), 'annotation_sha256': sha(annotation)})
        # Draft questions precede model predictions; review visually before QA execution.
        for room_type, wording in [('Bedroom', 'bedrooms'), ('Kitchen', 'kitchen zones'), ('Bath', 'bathrooms')]:
            evidence = [n['id'] for n in nodes if n['type'] == room_type]
            questions.append({'question_id': f'{plan}-count-{room_type}', 'plan_id': plan, 'category': 'count',
                              'question': f'How many {wording} are shown? Answer with an integer.',
                              'accepted_answers': [str(len(evidence))], 'supporting_nodes': evidence,
                              'review_status': 'draft_pipeline_answer'})
        print(plan, len(nodes), 'rooms;', len(edges), 'silver edges')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'selection.json').write_text(json.dumps({'seed': args.seed, 'excluded_identities': sorted(excluded),
                                                          'selected_identities': [identity(d) for d in selected]}, indent=2) + '\n')
    (args.output / 'questions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in questions))


if __name__ == '__main__':
    main()
