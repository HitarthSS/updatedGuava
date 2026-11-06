"""Load only the source and protocol pinned by the generated manifest."""
import json
import os
from pathlib import Path
import subprocess
import sys


def manifest():
    path = Path(__file__).resolve().parents[1] / 'manifest.json'
    return json.loads(path.read_text())


def load_source():
    root = Path(os.environ['GUAVA_SOURCE']).resolve()
    expected = manifest()['source']['commit']
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise RuntimeError('Guava source revision differs from manifest')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', 'guava', 'configs'], check=True)
    sys.path.insert(0, str(root))
    from guava.inference import benchmark_engine, benchmark_compatible_tasks, sft_frame
    from guava.inference.rollout_env import RolloutEngine
    from guava.tools import tools, schema
    from guava.collection.trajectory import _fmt_tool_result
    return root, benchmark_engine, benchmark_compatible_tasks, sft_frame, RolloutEngine, tools, schema, _fmt_tool_result
