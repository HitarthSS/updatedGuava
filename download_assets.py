#!/usr/bin/env python3
"""Download only the reported public Guava model and the gated SAM3 model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    os.environ['HF_HOME'] = str(root / 'cache/huggingface')
    os.environ['HF_TOKEN_PATH'] = str(Path.home() / '.cache/huggingface/token')
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    receipt = root / 'MODEL_ASSETS.json'
    result = json.loads(receipt.read_text()) if receipt.exists() else {}

    def save():
        result['updated_unix'] = time.time()
        p = receipt.with_suffix('.tmp')
        p.write_text(json.dumps(result, indent=2) + '\n')
        p.replace(receipt)

    for label, repo, revision, patterns in [
        ('guava', 'AIcell/guava-v13b-qwen3.5-4b', '19a8c77544cf7c161966f49bbb2654b85c6b2fa3', None),
        ('sam3', 'facebook/sam3', None, ['config.json', 'sam3.pt', '*.json']),
    ]:
        if result.get(label, {}).get('state') == 'downloaded':
            continue
        try:
            info = api.model_info(repo, revision=revision)
            sha = info.sha
            result[label] = {'repository': repo, 'revision': sha, 'state': 'downloading',
                             'paper_weight_revision_verified': label == 'guava'}
            save()
            path = Path(snapshot_download(repo, revision=sha, allow_patterns=patterns))
            # Native SAM3 asks for the default ref. Freeze it to this downloaded snapshot for offline runtime.
            if label == 'sam3':
                refs = path.parent.parent / 'refs'
                refs.mkdir(exist_ok=True)
                (refs / 'main').write_text(sha)
            files = {}
            for p in sorted(path.rglob('*')):
                if p.is_file() and (p.suffix in ('.pt', '.safetensors') or p.name == 'config.json'):
                    h = hashlib.sha256()
                    with p.open('rb') as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            h.update(block)
                    files[str(p.relative_to(path))] = {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}
            if not any(name.endswith(('.pt', '.safetensors')) for name in files):
                raise RuntimeError('No model weights downloaded')
            result[label].update(state='downloaded', snapshot=str(path), files=files)
            save()
            print(label, 'downloaded', flush=True)
        except Exception as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            result[label] = {**result.get(label, {}), 'repository': repo, 'state': 'access_required' if status in (401, 403) else 'failed',
                             'error_type': type(error).__name__, 'http_status': status}
            save()
            print(label, result[label]['state'], type(error).__name__, flush=True)
    if any(result.get(label, {}).get('state') != 'downloaded' for label in ('guava', 'sam3')):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
