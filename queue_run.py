#!/usr/bin/env python3
"""Bounded experiment queue: wait for SAM3 access and GPU memory, then run the gated campaign."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import time

from plan import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--gpus', help='Optional allowed device list; select once before launch, never mid-run')
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.35)
    parser.add_argument('--min-free-mib', type=int, default=26000)
    parser.add_argument('--require-parity-ready', action='store_true')
    parser.add_argument('--provisional', action='store_true')
    parser.add_argument('--sam3-gpu', type=int)
    args = parser.parse_args()
    root = args.workspace.resolve()
    devices = [int(item) for item in args.gpus.split(',')] if args.gpus else [args.gpu]
    if not devices or any(gpu in (2,3,4) or not 0 <= gpu <= 7 for gpu in devices):
        raise ValueError('GPU 2/3/4 excluded')
    if args.sam3_gpu is not None:
        if args.sam3_gpu in (2,3,4) or not 0 <= args.sam3_gpu <= 7:
            raise ValueError('SAM3 GPU 2/3/4 excluded')
        devices = [gpu for gpu in devices if gpu != args.sam3_gpu]
        if not devices:
            raise ValueError('Separate SAM3 placement requires another policy GPU')
    if args.provisional and args.require_parity_ready:
        raise ValueError('Provisional comparison cannot require a passed parity gate')
    deadline = time.time()+args.hours*3600
    last_token_stamp = None
    last_access_attempt = 0
    child = None
    def interrupt(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
        write(root/'QUEUE.json', {'state': 'interrupted', 'signal': signum, 'updated_unix': time.time()})
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < deadline:
            assets = json.loads((root/'MODEL_ASSETS.json').read_text()) if (root/'MODEL_ASSETS.json').exists() else {}
            missing = [label for label in ('guava','sam3') if assets.get(label,{}).get('state') != 'downloaded']
            if missing:
                token = Path.home()/'.cache/huggingface/token'
                stamp = token.stat().st_mtime_ns if token.exists() else None
                if stamp is not None and (stamp != last_token_stamp or time.time()-last_access_attempt >= 300):
                    last_token_stamp = stamp
                    last_access_attempt = time.time()
                    with (root/'logs/queued-download-assets.log').open('a') as log:
                        child = subprocess.Popen([root/'source/.venv-serve/bin/python', root/'download_assets.py',
                                                 '--workspace', root], cwd=root, stdout=log, stderr=subprocess.STDOUT)
                        child.wait()
                    continue
                write(root/'QUEUE.json', {'state': 'waiting_for_assets', 'missing': missing, 'gpu': args.gpu,
                      'pid': os.getpid(), 'expires_unix': deadline, 'updated_unix': time.time(),
                      'external_model_api_calls': 0})
                time.sleep(30)
                continue
            if args.require_parity_ready and not (root/'parity/PARITY.json').exists():
                write(root/'QUEUE.json', {'state': 'waiting_for_parity_gate', 'allowed_gpus': devices,
                      'pid': os.getpid(), 'expires_unix': deadline, 'updated_unix': time.time()})
                time.sleep(15)
                continue
            with (root/'campaign.lock').open('a') as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(probe, fcntl.LOCK_UN)
                except BlockingIOError:
                    time.sleep(15)
                    continue
            inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                                '--format=csv,noheader,nounits'],text=True)
            available = {int(line.split(',')[0]):int(line.split(',')[1]) for line in inventory.splitlines()}
            selected = max(devices, key=lambda gpu: available[gpu])
            free = available[selected]
            if free < args.min_free_mib or (args.sam3_gpu is not None and available[args.sam3_gpu] < 6000):
                write(root/'QUEUE.json', {'state': 'waiting_for_vram', 'free_mib': free, 'required_mib': args.min_free_mib,
                      'gpu': selected, 'allowed_gpus': devices, 'pid': os.getpid(), 'expires_unix': deadline, 'updated_unix': time.time()})
                write(root/'STATUS.json', {'state': 'waiting', 'phase': 'waiting-for-vram',
                      'gpu': selected, 'sam3_gpu': args.sam3_gpu, 'pid': os.getpid(),
                      'free_mib': free, 'required_mib': args.min_free_mib,
                      'external_model_calls': 0, 'updated_unix': time.time()})
                time.sleep(30)
                continue
            write(root/'QUEUE.json', {'state': 'started_provisional_campaign' if args.provisional else 'started_gated_campaign', 'gpu': selected,
                  'sam3_gpu': args.sam3_gpu, 'provisional': args.provisional,
                  'pid': os.getpid(), 'updated_unix': time.time()})
            with (root/'logs/campaign-supervisor.log').open('a') as log:
                command = [root/'source/.venv-eval/bin/python', root/'run_campaign.py',
                                        '--workspace', root, '--gpu', str(selected), '--run',
                                        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
                                        '--min-free-mib', str(args.min_free_mib)]
                if args.provisional:
                    command.append('--provisional')
                if args.sam3_gpu is not None:
                    command.extend(['--sam3-gpu', str(args.sam3_gpu)])
                child = subprocess.Popen(command,
                                        cwd=root, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
            if code == 2 and (root/'STATUS.json').exists() and json.loads((root/'STATUS.json').read_text()).get('state') == 'waiting':
                time.sleep(30)
                continue
            write(root/'QUEUE.json', {'state': 'campaign_returned', 'returncode': code,
                  'gpu': selected, 'pid': os.getpid(), 'updated_unix': time.time()})
            return code
        write(root/'QUEUE.json', {'state': 'expired_without_start', 'gpu': args.gpu, 'updated_unix': time.time()})
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
