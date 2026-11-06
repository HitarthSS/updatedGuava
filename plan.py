"""Deterministic, released-task-only 120-case subset and frozen input hashes."""
import hashlib
import json
from pathlib import Path


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def hashes(package):
    paths = [p for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.toml')
             and '__pycache__' not in p.parts]
    return {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def build(root):
    package = root / 'package'
    manifest = json.loads((package / 'manifest.json').read_text())
    tasks = manifest['tasks']
    attempt = json.loads((root/'RUN_ATTEMPT.json').read_text()) if (root/'RUN_ATTEMPT.json').exists() else {}
    final_seed = attempt.get('final_seed_base', 42)
    if type(final_seed) is not int or final_seed < 0:
        raise ValueError('Final seed base must be a nonnegative integer')
    run_directory = attempt.get('directory', 'runs')
    if (not isinstance(run_directory, str) or run_directory in ('', '.', '..')
            or Path(run_directory).name != run_directory or '/' in run_directory or '\\' in run_directory):
        raise ValueError('Run attempt must use one workspace subdirectory')
    unseen = [t for t in tasks if t['unseen']]
    seen = [t for t in tasks if not t['unseen']]
    if len(tasks) != 15 or len(unseen) != 8 or len(seen) != 7:
        raise ValueError('Reviewed task split changed')
    cases = [{'task': t['name'], 'episode_idx': i, 'seed': final_seed + tasks.index(t) * 15 + i,
              'turn_cap': t['cap']} for t in unseen for i in range(15)]
    plan = {'label': '120-case unseen subset, not full paper evaluation', 'cases_per_scored_arm': 120,
            'scored_arms': {'baseline': 'authors released Guava harness via adapter; see reproduction status',
                            'candidate': 'my harness / '+manifest.get('candidate', {}).get('method', 'guava-feedback-v1')},
            'control_arm': 'null', 'physical_final_episodes': 360,
            'seed_status': f'chosen seed {final_seed} using released run_study spacing; paper seeds not established',
            'source_commit': manifest['source']['commit'], 'model': manifest['model'],
            'cases': cases, 'horizon': {'semantic_turns': '25/30 per task, including automatic recovery tools',
            'native_control_calls': 500000, 'multiplier': 1.0,
            'physical_source': 'released task factories and active robosuite termination; not a LIBERO horizon'},
            'cost': {'external_model_api_calls': 0, 'additional_models': 0}}
    plan['run_attempt'] = attempt or {'directory': 'runs'}
    if (root/'RUN_MODE.json').exists():
        plan['reproduction'] = json.loads((root/'RUN_MODE.json').read_text())
    write(root / 'PLAN120.json', plan)
    groups = []
    smoke_seed = attempt.get('smoke_seed_base', 9000)
    development_seed = attempt.get('development_seed_base', 4200)
    if any(type(value) is not int or value < 0 for value in (smoke_seed, development_seed)):
        raise ValueError('Development and smoke seeds must be nonnegative integers')
    for phase, selected, episodes, seed in [('smoke', [seen[0]], 1, smoke_seed),
                                           ('development', seen, 2, development_seed), ('final', unseen, 15, final_seed)]:
        for number, task in enumerate(selected):
            base = seed + tasks.index(task) * episodes
            group = root / run_directory / phase / task['name']
            config = {'server': {'timeout': 600}, 'benchmarks': [{
                'benchmark': 'hs_guava.benchmark:GuavaBenchmark', 'mode': 'sync',
                'episodes_per_task': episodes, 'params': {'task_names': [task['name']],
                'seed_bases': {task['name']: base}, 'output_dir': str(group/'native-audit')}}]}
            write(group / 'benchmark.json', config)
            groups.append({'phase': phase, 'task': task['name'], 'episodes': episodes,
                           'seed_base': base, 'directory': str(group),
                           'arms': ['baseline', 'null', 'candidate'] if number % 2 == 0
                                   else ['candidate', 'null', 'baseline']})
    write(root / 'GROUPS.json', groups)
    return plan, groups
