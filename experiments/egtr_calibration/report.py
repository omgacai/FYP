"""Build a self-contained interactive HTML report; no server or dependencies."""
import argparse
import base64
import json
import mimetypes
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prediction-dir',default='predictions/egtr_frozen')
    args=p.parse_args()
    root=args.root
    rows=json.loads((root/'manifest.json').read_text())
    def read(path): return json.loads(path.read_text()) if path.exists() else None
    metrics=read(root/'results/graph_metrics.json')
    reports={r['plan_id']:r for r in (metrics or {}).get('per_plan',[])}
    plans=[]
    for row in rows:
        image=root/row['image_path']
        plans.append({'id':row['plan_id'],'image':'data:'+ (mimetypes.guess_type(image.name)[0] or 'image/png')+';base64,'+base64.b64encode(image.read_bytes()).decode(),
            'reference':read(root/'annotations'/f"{row['plan_id']}.graph.json"),
            'raw':read(root/'raw'/f"{row['plan_id']}.json"),
            'prediction':read(root/args.prediction_dir/f"{row['plan_id']}.json"),
            'evaluation':reports.get(row['plan_id'])})
    template=Path(__file__).with_name('report_template.html').read_text()
    prediction_name=Path(args.prediction_dir).name
    method_title = ('CubiCasa CNN + CubiGraph evaluation' if prediction_name == 'cubicasa_cnn_cubigraph'
                    else 'EGTR experiment review')
    payload=json.dumps({'plans':plans,'metrics':metrics,'run':read(root/'run.json'),'status':read(root/'status.json'),
                        'method_title': method_title}).replace('<','\\u003c')
    output=root/'report.html'
    output.write_text(template.replace('__PAYLOAD__',payload))
    print(f'Report: {output.resolve()} ({len(plans)} plans; prediction-free plans explicitly marked)')


if __name__=='__main__': main()
