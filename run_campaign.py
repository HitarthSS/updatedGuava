#!/usr/bin/env python3
"""Gated local-model Guava 120-case comparison through HarnessStudio. No paid APIs."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from plan import build, hashes, write
from initial_audit import PROVISIONAL_CHANGED_PIXELS


def read(path):
    return json.loads(path.read_text())


class Campaign:
    def __init__(self, root, gpu, gpu_memory_utilization=0.35, min_free_mib=26000, sam3_gpu=None):
        self.root, self.gpu = root, gpu
        self.sam3_gpu = gpu if sam3_gpu is None else sam3_gpu
        self.initial_render_roundoff_pixels = 0
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.min_free_mib = int(min_free_mib)
        minimum = 14000 if self.sam3_gpu != gpu else 20000
        if not 0.2 <= self.gpu_memory_utilization <= 0.5 or self.min_free_mib < minimum:
            raise ValueError('Unsupported shared-GPU serving allocation')
        self.package = root/'package'
        self.python = root/'source/.venv-eval/bin/python'
        self.serve_python = root/'source/.venv-serve/bin/python'
        self.sam_python = root/'source/.venv-sam3/bin/python'
        self.children = []
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.endswith('API_KEY') or key.startswith(('OPENROUTER_', 'ANTHROPIC_', 'NRP_')) or key == 'HF_TOKEN':
                self.env.pop(key)
        self.env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
            PYTHONPATH=os.pathsep.join(map(str, [self.package, root, root/'source'])),
            HF_HOME=str(root/'cache/huggingface'), HF_HUB_OFFLINE='1',
            GUAVA_SOURCE=str(root/'source'), GUAVA_MODEL_REVISION='19a8c77544cf7c161966f49bbb2654b85c6b2fa3',
            GUAVA_MODEL_URL='http://127.0.0.1:8000/v1', TOKENIZERS_PARALLELISM='false',
            GUAVA_POLICY_AUDIT=str(root/'policy-audit'), VLLM_WORKER_MULTIPROC_METHOD='spawn')
        self.env['PATH'] = str(self.serve_python.parent) + os.pathsep + self.env.get('PATH', '')
        bundled = list((root/'source/.venv-serve/lib').glob('python*/site-packages/nvidia/cu13'))
        if bundled:
            # The helper is simple; create only missing aliases inside this owned serving environment.
            cuda = bundled[0]
            for link, target in [(cuda/'lib64', 'lib'), (cuda/'lib/libcudart.so', 'libcudart.so.13')]:
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(target, target_is_directory=(link.parent/target).is_dir())
            self.env['CUDA_HOME'] = str(cuda)
            self.env['PATH'] = str(cuda/'bin') + os.pathsep + self.env['PATH']

    def status(self, phase, state='running', **fields):
        write(self.root/'STATUS.json', {'pid': os.getpid(), 'gpu': self.gpu, 'phase': phase,
              'sam3_gpu': self.sam3_gpu,
              'state': state, 'updated_unix': time.time(), 'external_model_calls': 0,
              'gpu_memory_utilization': self.gpu_memory_utilization, **fields})
        print(phase, state, flush=True)

    def command(self, name, args, env=None):
        self.status(name)
        process = self.start(name.replace('/', '-'), args, env)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, [str(v) for v in args])

    def start(self, name, argv, env=None):
        log = (self.root/'logs'/ (name+'.log')).open('a')
        process = subprocess.Popen([str(v) for v in argv], cwd=self.root, env=env or self.env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.children.append(process)
        write(self.root/'OWNED_SERVICES.json', [{'pid': p.pid} for p in self.children if p.poll() is None])
        return process

    def stop(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def ready(self, url, process, seconds=1200):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Owned service exited; inspect its log: '+url)
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(3)
        raise RuntimeError('Service readiness timed out: '+url)

    def preflight(self, model_only=False, parity_only=False):
        lock = self.root/'ADAPTER_LOCK.json'
        if lock.exists():
            for name, expected in read(lock)['files'].items():
                if hashlib.sha256((self.root/name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Prepared adapter source changed: '+name)
        setup = read(self.root/'RUNTIME_SETUP.json')
        if setup.get('state') != 'complete':
            raise RuntimeError('Locked runtime setup is incomplete')
        assets = read(self.root/'MODEL_ASSETS.json') if (self.root/'MODEL_ASSETS.json').exists() else {}
        missing = [key for key in (['guava'] if model_only else ['guava', 'sam3'])
                   if assets.get(key, {}).get('state') != 'downloaded']
        if missing:
            self.status('waiting-for-assets', state='waiting', missing=missing,
                        detail='SAM3 requires approved Hugging Face access; do not provide tokens to the assistant.')
            return None
        if assets['guava']['revision'] != self.env['GUAVA_MODEL_REVISION']:
            raise RuntimeError('Downloaded Guava checkpoint revision mismatch')
        for label in (['guava'] if model_only else ['guava', 'sam3']):
            for name, expected in assets[label]['files'].items():
                path = Path(assets[label]['snapshot'])/name
                if path.stat().st_size != expected['bytes']:
                    raise RuntimeError('Model file size changed: '+label+'/'+name)
                h = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8*1024*1024), b''):
                        h.update(block)
                if h.hexdigest() != expected['sha256']:
                    raise RuntimeError('Model file hash changed: '+label+'/'+name)
        if any(device in (2, 3, 4) or not 0 <= device <= 7 for device in (self.gpu, self.sam3_gpu)):
            raise RuntimeError('GPU 2/3/4 are excluded by the current user preference')
        free = subprocess.check_output(['nvidia-smi', '-i', str(self.gpu), '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        required = 6000 if parity_only else self.min_free_mib
        if int(free.strip()) < required:
            self.status('waiting-for-vram', state='waiting', free_mib=int(free.strip()), required_mib=required)
            return None
        if self.sam3_gpu != self.gpu and not model_only:
            sam_free = int(subprocess.check_output(['nvidia-smi', '-i', str(self.sam3_gpu),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if sam_free < 6000:
                self.status('waiting-for-sam3-vram', state='waiting', free_mib=sam_free, required_mib=6000)
                return None
        ports = [8114, 8118] if parity_only else ([8000] if model_only else [8000, 8114, 8118])
        for port in ports:
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError('Required local port already occupied; no service was taken over: '+str(port))
        actual = subprocess.check_output(['git', '-C', str(self.root/'source'), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != '8eeed908bef4fb63c8bfc7c4e2fa2f5684eb067f':
            raise RuntimeError('Guava source pin mismatch')
        subprocess.run(['git', '-C', str(self.root/'source'), 'diff', '--quiet', 'HEAD', '--', 'guava', 'configs'], check=True)
        return assets

    def model_server(self, assets):
        self.status('loading-frozen-guava')
        return self.start('vllm', [self.root/'source/.venv-serve/bin/vllm', 'serve', assets['guava']['snapshot'],
            '--served-model-name', 'AIcell/guava-v13b-qwen3.5-4b', '--host', '127.0.0.1', '--port', '8000',
            '--dtype', 'bfloat16', '--max-model-len', '32768', '--max-num-seqs', '1',
            '--gpu-memory-utilization', str(self.gpu_memory_utilization), '--limit-mm-per-prompt', '{"image":31}'])

    def hs(self, method, config, out, arms, policy='ws://127.0.0.1:8118', namespace='guava-120-v1'):
        return [self.python, '-m', 'harnessstudio.cli', 'eval', method, '--config', config,
                '--policy-url', policy, '--contract', self.package/'contracts/guava.yaml',
                '--profile', self.package/'configs/policy-profile.yaml', '--namespace', namespace,
                '--no-docker', '--arms', ','.join(arms), '--out', out]

    def native_parity(self):
        admission_root = self.root/'parity'
        root = admission_root/('reference-gpu-'+str(self.gpu))
        root.mkdir(parents=True, exist_ok=True)
        root_proof = admission_root/'PARITY.json'
        if root_proof.exists():
            prior = read(root_proof)
            if prior.get('adapter_hashes') == hashes(self.package) and prior.get('state') == 'passed':
                return
            raise RuntimeError('Previous parity receipt belongs to different adapter code')
        env = {**self.env, 'GUAVA_PARITY_ROOT': str(root),
               'GUAVA_FIXED_RESPONSES': str(admission_root/'cassette.jsonl')}
        original_result = root/'original/full/episode_000/result.json'
        if original_result.exists():
            protocol = read(root/'original/protocol.json')
            if protocol['model_revision'] != self.env['GUAVA_MODEL_REVISION']:
                raise RuntimeError('Original cassette checkpoint mismatch')
            for name, expected in protocol['source_hashes'].items():
                if hashlib.sha256((self.root/'source'/name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Original cassette source mismatch: '+name)
            cassette_rows = (root/'cassette.jsonl').read_text().splitlines()
            if len(cassette_rows) != read(original_result)['decisions']:
                raise RuntimeError('Incomplete original-runner cassette')
        else:
            self.command('original-runner', [self.python, self.root/'native_parity.py', '--backend', 'local',
            '--model', 'AIcell/guava-v13b-qwen3.5-4b', '--model-revision', self.env['GUAVA_MODEL_REVISION'],
            '--task', 'can_in_bin', '--episodes', '1', '--modes', 'full', '--max-actions', '25',
            '--max-decisions', '25', '--seed-base', '98000', '--camera', 'frontview', '--motion-speed', '1',
            '--system-prompt', self.root/'source/configs/prompts/sft_v13b_short.txt', '--table-frame',
                '--output-dir', root/'original'], env)
        attempt = root/('attempt-'+str(time.time_ns()))
        attempt.mkdir()
        replay_env = {**self.env, 'GUAVA_REPLAY_ONLY': '1', 'GUAVA_CASSETTE': str(root/'cassette.jsonl'),
                      'GUAVA_POLICY_AUDIT': str(attempt/'replay-policy-audit')}
        replay = self.start('parity-replay', [self.python, self.root/'native_parity.py', '--host', '127.0.0.1', '--port', '8118'], replay_env)
        try:
            # A WebSocket service has no health endpoint; watch its owned port and process.
            self.wait_port(8118, replay)
            config = {'server': {'timeout': 600}, 'benchmarks': [{'benchmark': 'hs_guava.benchmark:GuavaBenchmark',
                      'mode': 'sync', 'episodes_per_task': 1, 'params': {'task_names': ['can_in_bin'],
                      'seed_bases': {'can_in_bin': 98000}, 'output_dir': str(attempt/'adapter-audit')}}]}
            write(attempt/'benchmark.json', config)
            self.command('adapter-replay', self.hs('hs_guava:authors', attempt/'benchmark.json', attempt/'adapter', ['candidate']))
            originals = read(root/'original/full/episode_000/result.json')
            actions = read(root/'original/full/episode_000/actions.json') if (root/'original/full/episode_000/actions.json').exists() else []
            paths = list((attempt/'adapter-audit').glob('session-*/*/episode-0/result.json'))
            if len(paths) != 1:
                raise RuntimeError('No unique adapter parity audit')
            adapted = read(paths[0])
            initial = read(paths[0].with_name('initial.json'))
            same = (initial == read(root/'initial.json')
                    and originals['success'] == adapted['success']
                    and originals['final'] == adapted['scorer_audit'][-1]
                    and [{'name': a['name'], 'arguments': a['kwargs']} for a in actions]
                        == [{'name': a['name'], 'arguments': a['arguments']} for a in adapted['actions']])
            records = [json.loads(line) for line in (attempt/'adapter/candidate/records.jsonl').read_text().splitlines()]
            if not same or len(records) != 1 or not records[0]['valid']:
                raise RuntimeError('Native/adapter parity gate failed')
            write(root_proof, {'state': 'passed', 'scope': 'one can-in-bin seen-case fixed-response source/adapter replay on the same GPU',
                  'model_requests_exact': True, 'initial_state_and_rgb_equal': True,
                  'tools_and_final_score_equal': True, 'recorded_responses': True,
                  'adapter_hashes': hashes(self.package),
                  'gpu': self.gpu, 'source_reference': str(root),
                  'attempt': str(attempt),
                  'replay_used_for_final_evaluation': False})
        finally:
            self.stop(replay)

    def wait_port(self, port, process):
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError('Policy service exited before opening its port')
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    return
            time.sleep(1)
        raise RuntimeError('Policy port readiness timeout')

    def audit_group(self, group):
        directory = Path(group['directory'])
        episodes = group['episodes']
        audit = {'task': group['task'], 'phase': group['phase'], 'arms': {},
                 'null_check': 'native-input-and-output-provenance-v1', 'null_provenance': [],
                 'cross_rollout_action_equality': [], 'initial_observation_checks': []}
        by_arm = {}
        for arm in group['arms']:
            rows = [json.loads(line) for line in (directory/arm/'records.jsonl').read_text().splitlines()]
            if len(rows) != episodes or any(not row['valid'] for row in rows):
                raise RuntimeError('Incomplete or invalid HarnessStudio group: '+group['task']+'/'+arm)
            by_arm[arm] = {row['episode_idx']: row for row in rows}
            audit['arms'][arm] = {'episodes': len(rows), 'successes': sum(row['success'] for row in rows)}
            for row in rows:
                native = read(Path(row['metrics']['native_audit'])/'result.json')
                if native['horizon_violation'] or native['native_ignore_done'] or native['budgets']['budgeted_turns'] > native['budgets']['decision_cap']:
                    raise RuntimeError('Horizon audit failed')
        for i in range(episodes):
            base = by_arm['baseline'][i]
            from initial_audit import verify_initial_pair
            for arm in group['arms']:
                if arm == 'baseline':
                    continue
                row = by_arm[arm][i]
                if row['metrics']['initial_sha256'] != base['metrics']['initial_sha256']:
                    proof = verify_initial_pair(base['metrics']['native_audit'], row['metrics']['native_audit'],
                                                self.initial_render_roundoff_pixels)
                else:
                    proof = {'physical_state_exact': True, 'rgb_exact': True, 'changed_pixels': 0,
                             'maximum_rgb_channel_delta': 0, 'images_modified': False}
                audit['initial_observation_checks'].append({'episode_idx': i, 'arm': arm, **proof})
            null = by_arm['null'][i]
            from null_audit import verify_native_execution
            native = read(Path(null['metrics']['native_audit'])/'result.json')
            audit['null_provenance'].append(verify_native_execution(null, native, self.package))
            audit['cross_rollout_action_equality'].append(null['action_sha256'] == base['action_sha256'])
        audit['paired_cases'] = episodes
        write(directory/'AUDITED.json', audit)

    def run(self, model_only=False, parity_only=False, provisional=False):
        if provisional and (model_only or parity_only):
            raise ValueError('Provisional mode is only for the comparison')
        self.initial_render_roundoff_pixels = PROVISIONAL_CHANGED_PIXELS if provisional else 0
        mode = {'provisional': provisional, 'original_code_parity': 'unresolved' if provisional else 'required',
                'episode_initialization': 'fresh native simulator and renderer per case',
                'attempt': read(self.root/'RUN_ATTEMPT.json') if (self.root/'RUN_ATTEMPT.json').exists() else {'directory': 'runs'},
                'label': 'Provisional adapter comparison; not an exact paper reproduction' if provisional else 'Parity-gated adapter comparison',
                'gap': ('Same-GPU original-code replay matched turns 0–4, then diverged at turn 5. '
                        'Initial physical states must match exactly; provisional image matching allows at most '
                        f'{PROVISIONAL_CHANGED_PIXELS} pixels (less than 0.01% at 512x512) with a one-level RGB difference. '
                        'All raw images are retained unchanged.') if provisional else None,
                'retained_checks': ['exact physical initial states', 'bounded and audited initial rendering roundoff', 'shadow-null native input/output provenance', 'source freeze', 'native horizons'],
                'initial_rendering': {'max_changed_pixels': self.initial_render_roundoff_pixels,
                    'max_rgb_channel_delta': 1 if provisional else 0, 'raw_images_modified': False,
                    'scope': 'Provisional comparison only; strict mode still requires byte-exact initial images'},
                'null_audit_note': 'Check null actions against its own native policy outputs at the actual observation; cross-rollout byte equality is diagnostic only.',
                'gpu': self.gpu, 'sam3_gpu': self.sam3_gpu, 'created_unix': time.time()}
        write(self.root/'RUN_MODE.json', mode)
        plan, groups = build(self.root)
        assets = self.preflight(model_only, parity_only)
        if assets is None:
            return 2
        if not model_only:
            self.command('candidate-tier0', [self.python, '-m', 'harnessstudio.cli', 'verify', 'hs_guava:my_harness', '--tier', '0'])
            self.command('candidate-tier1', [self.python, '-m', 'harnessstudio.cli', 'verify', 'hs_guava:my_harness', '--tier', '1'])
        if parity_only and not (self.root/'parity/original/full/episode_000/result.json').exists():
            raise RuntimeError('Parity-only mode needs an existing original-runner cassette')
        model = None if parity_only else self.model_server(assets)
        try:
            if model is not None:
                self.ready('http://127.0.0.1:8000/v1/models', model)
            if model_only:
                request = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
                    data=json.dumps({'model': plan['model']['name'], 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
                                     'temperature': 0, 'max_tokens': 16}).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=180) as response:
                    output = json.load(response)
                write(self.root/'MODEL_PREFLIGHT.json', {'state': 'passed', 'gpu': self.gpu,
                      'model': plan['model'], 'usage': output.get('usage'), 'scope': 'text-generation preflight only'})
                self.status('model-preflight-complete', state='complete')
                return 0
            sam_env = dict(self.env, CUDA_VISIBLE_DEVICES=str(self.sam3_gpu))
            sam = self.start('sam3', [self.sam_python, '-m', 'guava.scripts.sam3_server', '--host', '127.0.0.1', '--port', '8114'], sam_env)
            self.ready('http://127.0.0.1:8114/health', sam, seconds=300)
            scene_receipt = self.root/'SIMULATOR_PREFLIGHT.json'
            scene = read(scene_receipt) if scene_receipt.exists() else {}
            if (scene.get('state') != 'passed' or scene.get('gpu') != self.gpu
                    or scene.get('source_commit') != '8eeed908bef4fb63c8bfc7c4e2fa2f5684eb067f'):
                self.command('simulator-preflight', [self.python, self.root/'simulator_preflight.py', self.root])
            if provisional:
                self.status('original-code-parity-unresolved-provisional-comparison', reproduction=mode)
            else:
                self.native_parity()
            if parity_only:
                self.status('native-parity-complete', state='complete')
                return 0
            policy = self.start('guava-policy', [self.python, '-m', 'hs_guava.model_server', '--host', '127.0.0.1', '--port', '8118'])
            self.wait_port(8118, policy)
            self.env['GUAVA_REPLAY_ONLY'] = '0'
            self.env.pop('GUAVA_CASSETTE', None)
            frozen = None
            frozen_auditor = None
            for group in groups:
                if group['phase'] == 'final' and frozen is None:
                    # Do not advance a severely regressed controller into the holdout.
                    # This is a small development safety gate, not a superiority test.
                    dev = [read(Path(g['directory'])/'AUDITED.json') for g in groups if g['phase']=='development']
                    totals = {arm: sum(g['arms'][arm]['successes'] for g in dev) for arm in ('baseline','candidate')}
                    passed = totals['candidate'] >= totals['baseline'] - 1
                    write(self.root/'DEVELOPMENT_GATE.json', {'passed': passed, 'successes': totals,
                        'episodes_per_arm': 14, 'allowed_success_deficit': 1,
                        'criterion': 'reject substantial development regression; not evidence of improvement'})
                    if not passed:
                        raise RuntimeError('Candidate failed development regression gate; final cases not started')
                    frozen = hashes(self.package)
                    frozen_auditor = {name: hashlib.sha256((self.root/name).read_bytes()).hexdigest()
                                      for name in ('run_campaign.py', 'null_audit.py', 'initial_audit.py', 'plan.py', 'report.py')}
                    write(self.root/'FROZEN.json', {'source_hashes': frozen, 'model': plan['model'],
                          'evaluation_code_hashes': frozen_auditor,
                          'reproduction': mode, 'sam3_gpu': self.sam3_gpu,
                          'gpu': self.gpu, 'gpu_memory_utilization': self.gpu_memory_utilization,
                          'development_complete': True, 'frozen_unix': time.time()})
                if frozen is not None and hashes(self.package) != frozen:
                    raise RuntimeError('Candidate source changed after design freeze')
                if frozen_auditor is not None and any(hashlib.sha256((self.root/name).read_bytes()).hexdigest() != value
                                                      for name, value in frozen_auditor.items()):
                    raise RuntimeError('Evaluation code changed after design freeze')
                directory = Path(group['directory'])
                if (directory/'AUDITED.json').exists():
                    self.audit_group(group)
                    continue
                if any((directory/arm).exists() for arm in group['arms']):
                    raise RuntimeError('Interrupted group retained; inspect it before restarting, never overwrite it')
                write(directory/'EXECUTION.json', {'gpu': self.gpu, 'sam3_gpu': self.sam3_gpu,
                      'package_hashes': hashes(self.package), 'started_unix': time.time()})
                for arm in group['arms']:
                    audit_file = self.root/'policy-audit/requests.jsonl'
                    offset = audit_file.stat().st_size if audit_file.exists() else 0
                    self.command(group['phase']+'-'+group['task']+'-'+arm, self.hs('hs_guava:my_harness',
                                 directory/'benchmark.json', directory, [arm]))
                    from report import policy_stats
                    with audit_file.open() as stream:
                        stream.seek(offset)
                        calls = [json.loads(line) for line in stream if line.strip()]
                    write(directory/arm/'POLICY_STATS.json', policy_stats(calls))
                    (directory/'summary.json').replace(directory/arm/'harnessstudio-summary.json')
                self.audit_group(group)
                from report import summarize
                summarize(self.root)
            self.status('comparison-complete', state='complete', cases_per_scored_arm=120)
            return 0
        finally:
            for child in reversed(self.children):
                self.stop(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--model-smoke', action='store_true')
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--provisional', action='store_true', help='Explicitly label unresolved original-code parity; retain all live comparison audits')
    parser.add_argument('--sam3-gpu', type=int, help='Optional separate allowed GPU for frozen SAM3')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.35)
    parser.add_argument('--min-free-mib', type=int, default=26000)
    args = parser.parse_args()
    root = args.workspace.resolve()
    (root/'logs').mkdir(exist_ok=True)
    with (root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(root, args.gpu, args.gpu_memory_utilization, args.min_free_mib, args.sam3_gpu)
        def interrupted(signum, frame):
            campaign.status('interrupted', state='interrupted', signal=signum)
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            plan, _ = build(root)
            if not args.run and not args.model_smoke and not args.parity_only:
                campaign.status('plan-ready', state='prepared', cases_per_scored_arm=120)
                print(json.dumps(plan, indent=2))
                return 0
            return campaign.run(model_only=args.model_smoke, parity_only=args.parity_only, provisional=args.provisional)
        except Exception as error:
            campaign.status('stopped-on-error', state='failed', error_type=type(error).__name__, detail=str(error))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
