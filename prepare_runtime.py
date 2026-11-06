#!/usr/bin/env python3
"""Isolated, resumable installation of the authors' locked Guava environments."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

GUAVA = '8eeed908bef4fb63c8bfc7c4e2fa2f5684eb067f'
STUDIO = '785515ff63cacf70617a1bc884cf8e118ce75b85'
MODEL = 'AIcell/guava-v13b-qwen3.5-4b'
REVISION = '19a8c77544cf7c161966f49bbb2654b85c6b2fa3'


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--group', choices=['eval', 'sam3', 'serve', 'all'], default='all')
    args = parser.parse_args()
    root = args.workspace.resolve()
    source = root / 'source'
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    uv = root / '.tools/bin/uv'
    state = {'pid': os.getpid(), 'phase': 'starting', 'state': 'running', 'groups': {},
             'source_commit': GUAVA, 'harnessstudio_commit': STUDIO, 'external_model_calls': 0}
    receipt = root / 'RUNTIME_SETUP.json'
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    state['groups'] = previous.get('groups', {})

    def status(phase, **fields):
        state.update(phase=phase, updated_unix=time.time(), **fields)
        write(receipt, state)
        print(phase, flush=True)

    def command(name, argv, cwd, env):
        status(name)
        with (logs / (name + '.log')).open('a') as log:
            subprocess.run([str(item) for item in argv], cwd=cwd, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    env = dict(os.environ)
    # Only model downloading can consult an existing HF login; no LLM API keys are needed.
    for key in list(env):
        if key.endswith('API_KEY') or key.startswith(('OPENROUTER_', 'ANTHROPIC_', 'NRP_')):
            env.pop(key)
    env['UV_CACHE_DIR'] = str(root / 'cache/uv')
    env['HF_HOME'] = str(root / 'cache/huggingface')
    env['HF_TOKEN_PATH'] = str(Path.home() / '.cache/huggingface/token')
    try:
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != GUAVA:
            raise RuntimeError('Guava source differs from the reviewed pin')
        command('sam3-source-patch', ['python3', source / 'scripts/setup_sam3_compat.py'], source, env)
        studio = root / 'harnessstudio'
        if not studio.exists():
            command('clone-harnessstudio', ['git', 'clone', '--filter=blob:none',
                    'https://github.com/seemandhar/harnessstudio.git', studio], root, env)
        command('pin-harnessstudio', ['git', '-C', studio, 'checkout', '--detach', STUDIO], root, env)
        for group in (['eval', 'sam3', 'serve'] if args.group == 'all' else [args.group]):
            target = source / ('.venv-' + group)
            if state['groups'].get(group) == 'installed' and (target / 'bin/python').exists():
                continue
            group_env = {**env, 'UV_PROJECT_ENVIRONMENT': str(target)}
            command('install-' + group, [uv, 'sync', '--locked', '--no-default-groups',
                    '--group', group, '--python', '3.11'], source, group_env)
            if group == 'eval':
                # Preserve the complete locked native dependency set when adding the framework.
                constraints = root / 'eval-constraints.txt'
                code = ('import importlib.metadata as m; print("\\n".join(sorted('
                        'd.metadata["Name"]+"=="+d.version for d in m.distributions() '
                        'if d.metadata["Name"].lower() not in ("guava","robosuite"))))')
                constraints.write_text(subprocess.check_output([target/'bin/python', '-c', code], env=env, text=True))
                command('install-harnessstudio-adapter', [uv, 'pip', 'install', '--python', target/'bin/python',
                        '--constraint', constraints, '-e', studio, '-e', root/'package'], root, env)
                command('check-eval-dependencies', [uv, 'pip', 'check', '--python', target/'bin/python'], root, env)
            state['groups'][group] = 'installed'
            status(group + '-installed')
        status('runtime-installed', state='complete')
    except subprocess.CalledProcessError as error:
        status('installation-failed', state='failed', returncode=error.returncode,
               detail='Inspect the phase-specific log; no source/model substitution was made.')
        raise
    except Exception as error:
        status('setup-failed', state='failed', error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    main()
