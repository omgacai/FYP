"""Package selected manual references with their exact annotated raster bytes."""
import argparse
import json
import shutil
from pathlib import Path
from runs import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('cubicasa_eval'))
    p.add_argument('--selection',type=Path,default=Path('experiments/egtr_calibration/exclusions.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    selected=json.loads(args.selection.read_text())
    records=[]
    for identity in selected:
        category,number=Path(identity).parts[-2:]
        plan=f'{number}_{category}'
        annotation=args.source/'annotations'/f'{plan}.graph.json'
        g=json.loads(annotation.read_text())
        image=args.source/'images'/Path(g['image']['local_path']).name
        if g['plan_id']!=plan or g['status']!='manually_reviewed': raise ValueError(f'{plan}: identity/status mismatch')
        if digest(image)!=g['image']['sha256']: raise ValueError(f'{plan}: image mismatch')
        records.append((identity,plan,annotation,image))
    if len({r[1] for r in records})!=len(records): raise ValueError('Duplicate plans')
    (args.output/'annotations').mkdir(parents=True)
    (args.output/'images').mkdir()
    manifest=[]
    for identity,plan,annotation,image in records:
        image_relative=f'images/{plan}{image.suffix}'
        shutil.copy2(image,args.output/image_relative)
        shutil.copy2(annotation,args.output/'annotations'/annotation.name)
        manifest.append({'plan_id':plan,'source_identity':identity,'split':'manual_evaluation',
                         'image_path':image_relative,'image_sha256':digest(image),'annotation_sha256':digest(annotation)})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'selection.json').write_text(json.dumps({'purpose':'User-selected 20-plan manual evaluation; excluded from training',
        'selected_identities':selected,'reference':'manual_from_raster','relation_mode':'access',
        'note':'Exploratory inspection/tuning on these plans disqualifies them as untouched final test data.'},indent=2)+'\n')
    print(f'Packaged {len(records)} exact annotated rasters at {args.output}')


if __name__=='__main__': main()
