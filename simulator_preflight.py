"""Reset/render each released scene twice; no policy inference or segmentation."""
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root/'source'))
    from guava.inference.rollout_env import load_eval_config
    from guava.inference.study_protocols import BENCHMARK
    from guava.sim.tasks import make_env
    from guava.tools.tools import _capture_with_depth
    records = []
    for index, name in enumerate(BENCHMARK):
        config = load_eval_config(root/'source/configs'/ (name+'.yaml'))
        config.sim.perception = 'sam3'
        config.sim.visualize = False
        camera = 'sideview' if name == 'shell_game' else 'frontview'
        config.sim.camera.name = camera
        config.sim.camera.width = config.sim.camera.height = 512
        config.sim.camera_views = [camera]
        env = make_env(config.sim)
        try:
            if env.rob.horizon != 500000 or env.rob.ignore_done:
                raise RuntimeError('Unexpected native horizon/termination: ' + name)
            fingerprints = []
            for _ in range(2):
                env.reset(seed=99000+index)
                rgb, depth, _, _ = _capture_with_depth(env)
                fingerprints.append({'qpos': hashlib.sha256(env.rob.sim.data.qpos.tobytes()).hexdigest(),
                    'qvel': hashlib.sha256(env.rob.sim.data.qvel.tobytes()).hexdigest(),
                    'rgb': hashlib.sha256(rgb.tobytes()).hexdigest(),
                    'depth': hashlib.sha256(depth.tobytes()).hexdigest()})
            records.append({'task': name, 'horizon': env.rob.horizon, 'ignore_done': env.rob.ignore_done,
                            'camera': camera, 'deterministic_reset': fingerprints[0] == fingerprints[1],
                            'fingerprints': fingerprints})
            (root/'SIMULATOR_PREFLIGHT.json').write_text(json.dumps({'state': 'running', 'tasks': records}, indent=2))
            if fingerprints[0] != fingerprints[1]:
                raise RuntimeError('Repeated native scene reset differs: ' + name)
            print(name, 'PASS', flush=True)
        finally:
            env.rob.close()
    (root/'SIMULATOR_PREFLIGHT.json').write_text(json.dumps({'state': 'passed', 'tasks': records,
        'gpu': int(os.environ['CUDA_VISIBLE_DEVICES']),
        'source_commit': __import__('subprocess').check_output(['git','-C',str(root/'source'),'rev-parse','HEAD'],text=True).strip(),
        'scope': 'scene resets/render/horizon only; no model or SAM3 inference',
        'reported_physical_horizon_verified': False}, indent=2))


if __name__ == '__main__':
    main()
