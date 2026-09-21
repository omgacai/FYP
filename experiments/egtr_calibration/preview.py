"""Save separate reference, raw EGTR, and adapted graph overlays for inspection."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def render(image_path, nodes, edges, title, output):
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    image.thumbnail((1400, 1400))
    width, height = image.size
    canvas = Image.new('RGB', (width, height+40), 'white')
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 12), title, fill='black')
    points = {}
    for n in nodes:
        x0, y0, x1, y1 = n['bbox_xyxy']
        x0, x1, y0, y1 = x0*width, x1*width, y0*height+40, y1*height+40
        draw.rectangle((x0,y0,x1,y1), outline='#d00000', width=2)
        label = f"{n['id']}: {n['type']}"
        draw.text((x0+2, y0+2), label, fill='#a00000', stroke_width=1, stroke_fill='white')
        points[n['id']] = ((x0+x1)/2,(y0+y1)/2)
    for edge in edges:
        a,b = points[edge['a']],points[edge['b']]
        relation = edge['relation']
        color = '#1453e0' if relation in ('direct_access','connected_by_door','open_connected') else '#915500'
        draw.line([a,b], fill=color, width=3)
        draw.text(((a[0]+b[0])/2,(a[1]+b[1])/2), relation, fill=color, stroke_width=1, stroke_fill='white')
    if output.exists():
        raise FileExistsError(output)
    canvas.save(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    args=p.parse_args()
    directory=args.root/'overlays'
    directory.mkdir(exist_ok=True)
    for row in json.loads((args.root/'manifest.json').read_text()):
        plan=row['plan_id']
        reference=json.loads((args.root/'annotations'/f'{plan}.graph.json').read_text())
        image=args.root/row['image_path']
        render(image,reference['nodes'],reference['edges'],f'{plan}: {reference["status"]}',directory/f'{plan}_reference.png')
        raw_path=args.root/'raw'/f'{plan}.json'
        if raw_path.exists():
            raw=json.loads(raw_path.read_text())
            if raw['status']=='ok':
                nodes=[]
                for obj in sorted(raw['objects'],key=lambda o:-o['score'])[:20]:
                    if obj['score']<.3:
                        continue
                    x,y,w,h=obj['bbox_cxcywh']
                    box=[max(0,x-w/2),max(0,y-h/2),min(1,x+w/2),min(1,y+h/2)]
                    if box[0]>=box[2] or box[1]>=box[3]:
                        continue
                    nodes.append({'id':f'q{obj["query"]}', 'type':f'{obj["label"]} {obj["score"]:.2f}', 'bbox_xyxy':box})
                render(image,nodes,[],f'{plan}: raw EGTR objects (top 20, score >= .3; display only)',directory/f'{plan}_raw.png')
        prediction_path=args.root/'predictions/egtr_frozen'/f'{plan}.json'
        if prediction_path.exists():
            prediction=json.loads(prediction_path.read_text())
            if prediction.get('valid'):
                graph=prediction['graph']
                render(image,graph['nodes'],graph['edges'],f'{plan}: adapted EGTR graph',directory/f'{plan}_egtr.png')
    print(f'Wrote overlays to {directory}')


if __name__=='__main__':
    main()
