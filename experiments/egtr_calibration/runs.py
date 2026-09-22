"""Local experiment tracking: immutable inputs, one folder per run, streamed stage logs."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def git(*args, directory=REPO):
    result = subprocess.run(['git', '-C', str(directory), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def verify_inputs(root):
    config = json.loads((root / 'run.json').read_text())
    for relative, expected in config['input_hashes'].items():
        if digest(root / relative) != expected:
            raise ValueError(f'Frozen input changed: {relative}. Create a new run.')
    return config


def initialize(source, runs, name, note):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,70}', name):
        raise ValueError('Name must use letters, digits, hyphens or underscores')
    source = source.resolve()
    rows = json.loads((source / 'manifest.json').read_text())
    if not rows or len({r['plan_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain unique plan IDs and at least one plan')
    inputs = {'manifest.json': source / 'manifest.json'}
    for row in rows:
        plan = row['plan_id']
        if Path(plan).name != plan or plan in ('.', '..'):
            raise ValueError('Invalid plan ID')
        for relative, expected in [(row['image_path'], row['image_sha256']),
                                   (f'annotations/{plan}.graph.json', row['annotation_sha256'])]:
            item = (source / relative).resolve()
            if source not in item.parents or digest(item) != expected:
                raise ValueError(f'Input outside snapshot or hash mismatch: {relative}')
            inputs[relative] = item
    for optional in ('questions.jsonl', 'selection.json'):
        if (source / optional).exists():
            inputs[optional] = source / optional
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + name
    root = runs.resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    try:
        for relative, source_file in inputs.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
        for directory in ('raw', 'predictions', 'results', 'logs', 'provenance', 'overlays'):
            (root / directory).mkdir(exist_ok=True)
        write_json(root / 'run.json', {
            'schema_version': 'egtr-run/1', 'run_id': run_id, 'name': name, 'created_at': now(),
            'experiment': 'frozen_egtr_transfer', 'note': note, 'source_snapshot': str(source),
            'relation_mode': 'access', 'plan_ids': [r['plan_id'] for r in rows],
            'input_hashes': {relative: digest(root / relative) for relative in inputs}})
        write_json(root / 'status.json', {'state': 'prepared', 'updated_at': now(), 'stages': {}})
    except BaseException:
        # Retain the partial folder for diagnosis; it is not registered as a valid run without run.json.
        raise
    return root


def snapshot_code(root, stage):
    destination = root / 'provenance' / stage / 'code'
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.mjs')) + list(HERE.glob('*.html')) + [REPO / 'review_react/src/graphEvaluation.js', REPO / 'retrieval_app/scripts/build_cubicasa_egtr_corpus.py', REPO / 'retrieval_app/scripts/run_qwen_graph.py', REPO / 'retrieval_app/scripts/convert_qwen_to_manual_graph.py', REPO / 'floorplan_app/pipeline/graph_extractor.py']
    hashes = {}
    for source in files:
        relative = source.relative_to(REPO)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = digest(source)
    return {'git_commit': git('rev-parse', 'HEAD'), 'source_sha256': hashes}


def stage_command(args, root):
    stage = args.stage
    if stage == 'infer':
        if not all((args.artifact, args.checkpoint, args.labels)):
            raise ValueError('infer needs --artifact, --checkpoint and --labels')
        return [sys.executable, str(HERE / 'infer.py'), '--root', str(root), '--artifact', str(args.artifact.resolve()),
                '--checkpoint', str(args.checkpoint.resolve()), '--labels', str(args.labels.resolve())]
    if stage == 'adapt':
        if not args.mapping:
            raise ValueError('adapt needs --mapping with reviewed ontology mappings')
        target = root / 'provenance/adapt/mapping.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.mapping, target)
        return [sys.executable, str(HERE / 'adapt.py'), '--root', str(root), '--mapping', str(target),
                '--object-threshold', str(args.object_threshold), '--relation-threshold', str(args.relation_threshold)]
    if stage == 'evaluate':
        return ['node', str(HERE / 'evaluate.mjs'), str(root), '--min-iou', str(args.min_iou)]
    if stage == 'qa':
        if not args.model or not args.revision:
            raise ValueError('qa needs --model and --revision')
        return [sys.executable, str(HERE / 'qa.py'), '--root', str(root), '--model', args.model, '--revision', args.revision]
    if stage == 'report':
        return [sys.executable, str(HERE / 'report.py'), '--root', str(root)]
    if stage == 'preview':
        return [sys.executable, str(HERE / 'preview.py'), '--root', str(root)]
    raise ValueError(stage)


def execute(args):
    root = args.run.resolve()
    verify_inputs(root)
    lock = root / '.stage-lock'
    lock.mkdir()  # Prevent two workers from writing the same run.
    status_path = root / 'status.json'
    status = json.loads(status_path.read_text())
    started = False
    child = None
    try:
        if args.stage in status['stages']:
            raise ValueError('Stage already attempted; preserve this run and initialize a new one')
        command = stage_command(args, root)
        code = snapshot_code(root, args.stage)
        evidence = root / 'provenance' / args.stage
        environment = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
        (evidence / 'packages.txt').write_text(environment.stdout + environment.stderr)
        write_json(evidence / 'execution.json', {
            'command': command, 'cwd': str(REPO), 'python': sys.version, 'platform': platform.platform(),
            'hostname': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_node': os.environ.get('SLURMD_NODENAME'), 'started_at': now(), **code})
        entry = {'state': 'running', 'started_at': now(), 'command': command, 'log': f'logs/{args.stage}.log'}
        status['stages'][args.stage] = entry
        status.update(state=f'{args.stage}_running', updated_at=now())
        write_json(status_path, status)
        started = True
        with (root / entry['log']).open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        entry.update(state='completed' if code == 0 else 'failed', exit_code=code, finished_at=now())
        status.update(state=f'{args.stage}_{entry["state"]}', updated_at=now())
        write_json(status_path, status)
        refresh_summary(root)
        return code
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        if started:
            status['stages'][args.stage].update(state='failed', error=str(exc), finished_at=now())
            status.update(state=f'{args.stage}_failed', updated_at=now())
            write_json(status_path, status)
        raise
    finally:
        if child is not None and child.stdout is not None:
            child.stdout.close()
        lock.rmdir()


def refresh_summary(root):
    config = json.loads((root / 'run.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    def setting(stage, flag):
        command = status['stages'].get(stage, {}).get('command', [])
        return command[command.index(flag)+1] if flag in command else None
    raw = [json.loads(p.read_text()) for p in sorted((root / 'raw').glob('*.json'))]
    predictions = [json.loads(p.read_text()) for p in sorted((root / 'predictions/egtr_frozen').glob('*.json'))]
    summary = {'run_id': config['run_id'], 'name': config['name'], 'state': status['state'],
               'plans': len(config['plan_ids']), 'note': config['note'], 'relation_mode': config['relation_mode'],
               'inference_state': status['stages'].get('infer', {}).get('state', 'not_run'),
               'checkpoint': setting('infer', '--checkpoint'),
               'checkpoint_sha256': raw[0].get('provenance', {}).get('checkpoint_sha256') if raw else None,
               'object_threshold': setting('adapt', '--object-threshold'),
               'relation_threshold': setting('adapt', '--relation-threshold'),
               'min_iou': setting('evaluate', '--min-iou'),
               'manifest_sha256': config['input_hashes']['manifest.json'], 'raw_ok': sum(r.get('status') == 'ok' for r in raw),
               'raw_failed': sum(r.get('status') == 'failed' for r in raw),
               'graph_valid': sum(r.get('valid') is True for r in predictions),
               'graph_unavailable': sum(r.get('valid') is False for r in predictions)}
    metrics_path = root / 'results/graph_metrics.json'
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        summary.update(graph_evaluated=metrics['coverage']['evaluated'], reference_kind=metrics['reference_kind'],
                       node_f1=metrics['micro_nodes']['f1'], edge_f1=metrics['micro_edges']['f1'],
                       mean_aligned_edit_cost=metrics['mean_aligned_edit_cost'])
    qa_path = root / 'results/qa.jsonl'
    if qa_path.exists():
        answers = [json.loads(line) for line in qa_path.read_text().splitlines() if line.strip()]
        arms = ('image_only', 'reference_graph', 'egtr_graph')
        qa = {}
        for arm in arms:
            rows = [r for r in answers if r['arm'] == arm]
            ok = [r for r in rows if r['status'] == 'ok']
            qa[arm] = {'total': len(rows), 'answered': len(ok), 'unavailable_or_failed': len(rows)-len(ok),
                       'accuracy_answered': sum(r['correct'] for r in ok)/len(ok) if ok else None}
        maps = {arm: {r['question_id']: r for r in answers if r['arm'] == arm and r['status'] == 'ok'} for arm in arms}
        paired = set.intersection(*(set(m) for m in maps.values()))
        qa['paired_questions'] = len(paired)
        if paired:
            acc = {arm: sum(m[q]['correct'] for q in paired)/len(paired) for arm, m in maps.items()}
            qa.update(paired_accuracy=acc, qa_gap_pp=100*(acc['reference_graph']-acc['egtr_graph']),
                      qa_gain_pp=100*(acc['egtr_graph']-acc['image_only']))
        summary['qa'] = qa
    write_json(root / 'summary.json', summary)
    return summary


def compare(runs):
    summaries = [refresh_summary(p.parent) for p in sorted(runs.glob('*/run.json'))]
    columns = ['run_id', 'name', 'state', 'plans', 'raw_ok', 'raw_failed', 'graph_valid', 'graph_unavailable',
               'graph_evaluated', 'reference_kind', 'node_f1', 'edge_f1', 'mean_aligned_edit_cost',
               'inference_state', 'checkpoint', 'checkpoint_sha256', 'manifest_sha256', 'relation_mode',
               'object_threshold', 'relation_threshold', 'min_iou', 'note']
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / 'experiments.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))
    print(f'Comparison CSV: {runs / "experiments.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('--data', type=Path, required=True)
    init.add_argument('--runs', type=Path, default=Path('egtr_runs'))
    init.add_argument('--name', required=True)
    init.add_argument('--note', default='')
    stage = commands.add_parser('stage')
    stage.add_argument('stage', choices=['infer', 'adapt', 'evaluate', 'preview', 'report', 'qa'])
    stage.add_argument('--run', type=Path, required=True)
    for key in ('artifact', 'checkpoint', 'labels', 'mapping'):
        stage.add_argument('--' + key, type=Path)
    stage.add_argument('--object-threshold', type=float, default=.3)
    stage.add_argument('--relation-threshold', type=float, default=.01)
    stage.add_argument('--min-iou', type=float, default=.3)
    stage.add_argument('--model')
    stage.add_argument('--revision')
    listing = commands.add_parser('list')
    listing.add_argument('--runs', type=Path, default=Path('egtr_runs'))
    args = parser.parse_args()
    if args.command == 'init':
        print(initialize(args.data, args.runs, args.name, args.note))
    elif args.command == 'stage':
        sys.exit(execute(args))
    else:
        compare(args.runs)


if __name__ == '__main__':
    main()
