"""Extract exact EGTR/RelTR VG label ordering from its official annotation zip."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        names=z.namelist()
        objects_files=[n for n in names if Path(n).name=='train.json']
        relation_files=[n for n in names if Path(n).name=='rel.json']
        if len(objects_files)!=1 or len(relation_files)!=1:
            raise ValueError(f'Expected one train.json and rel.json; found {names[:30]}')
        categories=json.loads(z.read(objects_files[0]))['categories']
        by_id={int(c['id']): c['name'] for c in categories}
        if len(categories)!=150 or set(by_id)!=set(range(1,151)):
            raise ValueError('Expected VG object category IDs 1..150')
        relations=json.loads(z.read(relation_files[0]))['rel_categories']
        if len(relations)!=51 or relations[0] not in ('no_relation','__background__','background'):
            raise ValueError(f'Unexpected relation vocabulary: {relations}')
        labels={'objects':[by_id[i] for i in range(1,151)],'relations':relations[1:],
                'provenance':{'source':'Official EGTR-linked RelTR VG annotation archive',
                              'archive_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                              'objects_member':objects_files[0],'relations_member':relation_files[0],
                              'indexing':'objects=category_id-1; predicates=rel_categories[1:]'}}
    if args.output.exists():
        if json.loads(args.output.read_text())!=labels:
            raise FileExistsError('Existing labels differ; choose a new output path')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as out: json.dump(labels,out,indent=2)
    print('Saved:',args.output)
    print('Objects:',len(labels['objects']),labels['objects'][:10])
    print('Relations:',len(labels['relations']),labels['relations'])


if __name__=='__main__': main()
