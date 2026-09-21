"""Paired Qwen3-VL QA. Requires reviewed questions; keeps unavailable arms explicit."""
import argparse
import hashlib
import json
import time
from pathlib import Path

SYSTEM = ('Answer the floor-plan question using the image and, when supplied, the graph. '
          'The graph may contain errors. connected_by_door and open_connected mean direct access; '
          'adjacent_to means shared boundary without direct access. '
          'Return only the requested integer, yes, no, or unknown. Use unknown when the evidence is ambiguous.')


def graph_context(record):
    g = record.get('graph', record)
    return {'nodes': [{k: n[k] for k in ('id', 'type', 'bbox_xyxy')} for n in g['nodes']],
            'edges': [{k: e[k] for k in ('a', 'b', 'relation')} for e in g['edges']]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', required=True, help='Fixed Qwen3-VL model ID or local path')
    p.add_argument('--revision', required=True, help='Pinned model revision, or local snapshot identity')
    args = p.parse_args()
    questions_path = args.root / 'questions.jsonl'
    questions = [json.loads(x) for x in questions_path.read_text().splitlines() if x.strip()]
    manifest = {r['plan_id']: r for r in json.loads((args.root / 'manifest.json').read_text())}
    if not questions or len({q['question_id'] for q in questions}) != len(questions):
        raise ValueError('Need nonempty questions with unique IDs')
    for q in questions:
        if q.get('review_status') != 'human_reviewed' or q['plan_id'] not in manifest:
            raise ValueError('Every question must be human reviewed and belong to calibration')
        if not q.get('accepted_answers') or not q.get('category'):
            raise ValueError('Questions need accepted_answers and category')
    output = args.root / 'results/qa.jsonl'
    if output.exists():
        raise FileExistsError(output)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    revision = {} if Path(args.model).exists() else {'revision': args.revision}
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=1280*28*28, **revision)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto', **revision).eval()
    output.parent.mkdir(exist_ok=True)
    with output.open('x') as handle:
        for q in questions:
            plan = q['plan_id']
            image = args.root / manifest[plan]['image_path']
            if hashlib.sha256(image.read_bytes()).hexdigest() != manifest[plan]['image_sha256']:
                raise ValueError(f'{plan}: image hash mismatch')
            manual = json.loads((args.root / 'annotations' / f'{plan}.graph.json').read_text())
            prediction_path = args.root / 'predictions/egtr_frozen' / f'{plan}.json'
            prediction = json.loads(prediction_path.read_text()) if prediction_path.exists() else None
            for arm, record in [('image_only', None), ('reference_graph', manual), ('egtr_graph', prediction)]:
                result = {'question_id': q['question_id'], 'plan_id': plan, 'category': q['category'], 'arm': arm,
                          'model': args.model, 'revision': args.revision, 'question_sha256': hashlib.sha256(questions_path.read_bytes()).hexdigest()}
                if arm == 'egtr_graph' and (record is None or record.get('valid') is False):
                    result.update(status='unavailable', reason='Missing prediction or unsupported/invalid ontology')
                else:
                    context = graph_context(record) if record is not None else None
                    prompt = q['question'] + ('\nGraph: ' + json.dumps(context, separators=(',', ':')) if context is not None else '')
                    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [
                        {'type': 'image', 'image': str(image.resolve())}, {'type': 'text', 'text': prompt}]}]
                    result.update(prompt=prompt, system_prompt=SYSTEM, max_new_tokens=32, do_sample=False)
                    start = time.perf_counter()
                    try:
                        images, videos = process_vision_info(messages)
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to(model.device)
                        with torch.inference_mode():
                            generated = model.generate(**inputs, do_sample=False, max_new_tokens=32)
                        answer = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                        normalized = answer.strip().lower().rstrip('.')
                        result.update(status='ok', answer=answer, correct=normalized in [str(a).strip().lower().rstrip('.') for a in q['accepted_answers']])
                    except Exception as exc:
                        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    result['seconds'] = time.perf_counter() - start
                handle.write(json.dumps(result) + '\n')
                handle.flush()
                print(q['question_id'], arm, result['status'], flush=True)


if __name__ == '__main__':
    main()
