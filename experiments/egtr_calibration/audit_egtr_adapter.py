"""Validate the exported EGTR training adapter before an expensive GPU run."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='cubicasa_visual_genome_v1 adapter directory')
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--exclusions', type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    rel = json.loads((data / 'rel.json').read_text())
    expected_rel = ['no_relation', 'adjacent_to', 'direct_access']
    if rel.get('rel_categories') != expected_rel:
        raise ValueError(f'Unexpected relation vocabulary: {rel.get("rel_categories")}')
    excluded = {item.strip('/') for item in json.loads(args.exclusions.read_text())}
    canonical = read_jsonl(args.canonical)
    leaked = [row['source_identity'] for row in canonical if row.get('source_identity') in excluded]
    if leaked:
        raise ValueError(f'Gold identity leaked into canonical data: {leaked[:3]}')
    summaries, categories, predicate_counts = {}, None, Counter()
    for split in ('train', 'val', 'test'):
        coco = json.loads((data / f'{split}.json').read_text())
        category_ids = {item['id'] for item in coco['categories']}
        if categories is None:
            categories = coco['categories']
        elif coco['categories'] != categories:
            raise ValueError(f'{split}.json has a different category map')
        by_image = defaultdict(list)
        images = {item['id']: item for item in coco['images']}
        if len(images) != len(coco['images']):
            raise ValueError(f'{split}: duplicate image ID')
        for annotation in coco['annotations']:
            image = images.get(annotation['image_id'])
            if image is None or annotation['category_id'] not in category_ids:
                raise ValueError(f'{split}: invalid annotation reference')
            x, y, width, height = annotation['bbox']
            if not (width > 0 and height > 0 and x >= 0 and y >= 0 and x + width <= image['width'] + 1e-4 and y + height <= image['height'] + 1e-4):
                raise ValueError(f'{split}: invalid bounding box for image {image["id"]}')
            by_image[image['id']].append(annotation)
        if set(rel[split]) != {str(image_id) for image_id in images}:
            raise ValueError(f'{split}: relation image IDs do not match COCO image IDs')
        for image_id, triples in rel[split].items():
            count = len(by_image[int(image_id)])
            for subject, obj, predicate in triples:
                if not (0 <= subject < count and 0 <= obj < count and subject != obj and predicate in (1, 2)):
                    raise ValueError(f'{split}: invalid relation triple for image {image_id}: {[subject, obj, predicate]}')
                predicate_counts[expected_rel[predicate]] += 1
        missing = [image['file_name'] for image in images.values() if not (data / 'images' / image['file_name']).is_file()]
        if missing:
            raise FileNotFoundError(f'{split}: linked source images missing, e.g. {missing[0]}')
        summaries[split] = {'plans': len(images), 'objects': len(coco['annotations']), 'directed_relation_triples': sum(len(value) for value in rel[split].values())}
    print(json.dumps({'adapter': str(data), 'canonical_plans': len(canonical), 'excluded_gold_identities': len(excluded),
                      'splits': summaries, 'categories': [item['name'] for item in categories],
                      'directed_relation_counts': dict(predicate_counts), 'status': 'PASS'}, indent=2))


if __name__ == '__main__':
    main()
