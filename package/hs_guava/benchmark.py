"""Guava's native environment and semantic tools exposed as decision steps.

The evaluator owns engine/scorer state. Public observations contain only the
same prompt, image, gripper text and feedback the released VLM receives.
"""
import json
from pathlib import Path
import tempfile
import uuid
import hashlib
import time

import numpy as np
from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import DimSpec

from .budget import Budget, MotionMeter
from .codec import ACTION_DIM, decode
from .native import load_source, manifest


class GuavaBenchmark(StepBenchmark):
    render_backends = frozenset({'gpu'})

    def __init__(self, task_names=None, seed_base=42, seed_bases=None, output_dir='guava-audit', **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError('Unknown benchmark parameters: ' + ', '.join(kwargs))
        self.root, self.native, self.compatible, self.frame, self.Engine, self.tools, self.schema, self.format_result = load_source()
        self.plan = manifest()
        allowed = {item['name'] for item in self.plan['tasks']}
        self.names = task_names or [item['name'] for item in self.plan['tasks'] if item['unseen']]
        if not set(self.names) <= allowed:
            raise ValueError('Task was not in the inspected benchmark protocol')
        self.seed_base = int(seed_base)
        self.seed_bases = {str(k): int(v) for k, v in (seed_bases or {}).items()}
        if not set(self.seed_bases) <= allowed:
            raise ValueError('Seed map contains an unknown task')
        # Each vla-eval arm constructs its own benchmark; never overwrite another arm's audits.
        self.output = Path(output_dir) / ('session-' + uuid.uuid4().hex)
        self.output.mkdir(parents=True, exist_ok=True)
        self.engine = None
        self.active_task = None
        self.meter = None
        self.steps = 0
        self.success = False

    def get_tasks(self):
        return [{'name': name, 'suite': 'guava', 'instruction': name} for name in self.names]

    def reset(self, task):
        import yaml
        name = task['name']
        index = int(task.get('episode_idx', 0))
        self.episode_index = index
        cap = next(item['cap'] for item in self.plan['tasks'] if item['name'] == name)
        # A reused renderer can retain sub-pixel differences from the previous
        # rollout even after an exact physics reset. Give every paired case a
        # fresh native engine/context; preserve all seed and horizon settings.
        self.cleanup()
        if self.active_task != name:
            self.cleanup()
            config = yaml.safe_load((self.root / 'configs' / (name + '.yaml')).read_text())
            camera = 'sideview' if name == 'shell_game' else 'frontview'
            config.update(grasp='pca', stop_on_reward=False)
            config['sim'].update(perception='sam3', grasp='pca', motion_speed=1.0, visualize=False, camera_views=[camera])
            config['sim']['camera'] = {'name': camera, 'width': 512, 'height': 512}
            config.setdefault('robot', {}).update(sam3_url='http://127.0.0.1:8114', sam3_debug=False, bbox_debug=False)
            self.prompt = (self.root / 'configs/prompts/sft_v13b_short.txt').read_text()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as file:
                yaml.safe_dump(config, file)
                file.flush()
                self.engine = self.Engine(file.name, system_prompt=self.prompt, camera_name=camera)
            self.active_task = name
            self.engine.initial_settle_steps = 20
            self.engine.table_frame = True
            self.engine.cfg.stop_on_reward = False
            self.engine.cfg.sim.motion_speed = 1.0
            self.engine.env.cfg.motion_speed = 1.0
        self.budget = Budget(cap, cap)
        self.meter = MotionMeter(self.engine.env, self.budget)
        self.meter.attach()
        if name in ('push_pot', 'push_basket'):
            self.engine.env.rob.push_directions = ['left' if index % 2 == 0 else 'right']
        self.folder = self.output / name / ('episode-' + str(index))
        self.folder.mkdir(parents=True, exist_ok=False)
        episode = self.engine.episode(index, self.folder)
        self.seed = self.seed_bases.get(name, self.seed_base) + index
        episode.reset(seed=self.seed)
        if getattr(self.engine.env.rob, 'ignore_done', True):
            raise RuntimeError('Native simulator termination must remain enabled')
        if getattr(self.engine.env.rob, 'horizon', None) != 500000:
            raise RuntimeError('Native simulator horizon differs from the audited source setting')
        # Native reset includes 50 direct MuJoCo substeps outside env.step.
        total_setup = round(float(self.engine.env.rob.sim.data.time) / float(self.engine.env.rob.sim.model.opt.timestep))
        self.budget.physics_substeps['setup'] = total_setup
        self.meter.phase = 'execution'
        if name in ('push_pot', 'push_basket'):
            self.engine.cfg.task_name = self.engine.env.rob.task_instruction
        self.offset = self.frame.measured_offset(self.engine.env)
        from guava.tools.coordinate_frame import model_registry
        self.compatible.tool_text = self.schema.tools_to_prompt(model_registry(self.engine.ctx, self.engine.registry))
        self.steps = 0
        self.success = False
        self.feedback = ''
        self.actions = []
        self._public = self.observe(initial=True)
        self.initial = {'state': self.native.state(self.engine),
                        'qpos': self.engine.env.rob.sim.data.qpos.tolist(),
                        'qvel': self.engine.env.rob.sim.data.qvel.tolist(),
                        'rgb_sha256': hashlib.sha256(self.engine.ctx.last_rgb.tobytes()).hexdigest()}
        self.initial_sha256 = hashlib.sha256(json.dumps(self.initial, sort_keys=True).encode()).hexdigest()
        (self.folder / 'initial.json').write_text(json.dumps(self.initial, indent=2, default=str))
        self.tool_seconds = 0.0
        self.started = time.perf_counter()
        return self._public

    def observe(self, initial=False):
        self.tools._refresh(self.engine.ctx)
        self.native._save_png(self.engine.ctx.last_rgb, self.folder/('observation_'+str(self.steps).zfill(3)+'.png'))
        gripper = self.tools._gripper_state_text(self.engine.ctx)
        gripper = self.frame.gripper_text(self.engine.ctx, gripper, self.offset)
        content = self.compatible.observation('full', self.native._to_data_url(self.engine.ctx.last_rgb),
            gripper, self.feedback, initial, self.engine.cfg.task_name, tuple(self.engine.cfg.sim.object_aliases))
        return {'task_description': self.engine.cfg.task_name,
                'guava_request': {'system_prompt': self.prompt,
                                  'message': {'role': 'user' if initial else 'tool', 'content': content}}}

    def make_obs(self, raw_obs, task):
        return self._public

    def step(self, action):
        decision = decode(np.asarray(action.get('actions', action.get('action'))).reshape(-1))
        self.budget.decision()
        self.steps += 1
        terminal = decision['kind'] != 'tool'
        if decision['kind'] == 'tool':
            started = time.perf_counter()
            name, arguments = decision['name'], decision['arguments']
            entry = {'name': name, 'arguments': arguments,
                     'origin': decision.get('origin', 'model'), 'before': self.native.state(self.engine)}
            try:
                if name not in self.engine.registry:
                    raise ValueError('Unknown tool: ' + name)
                self.budget.tool()
                physical = self.frame.physical_arguments(name, arguments, self.offset)
                raw = self.engine.registry[name](**physical)
                raw = self.frame.model_result(name, raw, self.offset)
                feedback_arguments = dict(arguments)
                if name in ('get_position', 'get_position_and_size'):
                    feedback_arguments['object_name'] = feedback_arguments.get('target_name', feedback_arguments.get('object_name', 'object'))
                self.feedback = self.format_result(name, feedback_arguments,
                    json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw))
                entry['ok'] = True
            except Exception as error:
                self.feedback = 'Error calling ' + name + ': ' + str(error)
                entry.update(ok=False, error=str(error))
            finally:
                # Match original invoke(): refresh, score and snapshot before observe() refreshes again.
                # MuJoCo forward/render boundaries can affect later solver state; do not collapse them.
                self.tools._refresh(self.engine.ctx)
                entry['after'] = self.native.state(self.engine)
                self.native._save_png(self.engine.ctx.last_rgb,
                                     self.folder/('action_'+str(len(self.actions)).zfill(3)+'_'+name+'.png'))
            entry['feedback'] = self.feedback
            self.actions.append(entry)
            self.native.write(self.folder/'actions.json', self.actions)
            self.tool_seconds += time.perf_counter() - started
        finished = terminal or self.budget.ended
        if finished:
            self.finalize(decision)
        self._public = self.observe()
        return StepResult(obs=self._public, reward=float(self.success), done=finished, info={})

    def finalize(self, decision):
        final = self.native.state(self.engine)
        audits = [final]
        if self.active_task in self.native.PHYSICAL_TASKS:
            self.meter.phase = 'stability'
            for _ in range(3):
                self.engine.env._settle(5)
                audits.append(self.native.state(self.engine))
        self.success = all(item['success'] for item in audits)
        record = {'success': self.success, 'decision': decision, 'budgets': self.budget.snapshot(),
                  'task': self.active_task, 'episode_idx': self.episode_index,
                  'seed': self.seed, 'initial_sha256': self.initial_sha256,
                  'wall_seconds': time.perf_counter() - self.started, 'tool_seconds': self.tool_seconds,
                  'actions': self.actions, 'scorer_audit': audits, 'runtime_parity': 'pending',
                  'recovery_tools': sum(item['origin'] == 'harness_recovery' for item in self.actions),
                  'native_control_horizon': getattr(self.engine.env.rob, 'horizon', None),
                  'native_ignore_done': getattr(self.engine.env.rob, 'ignore_done', None),
                  'native_control_timestep': getattr(self.engine.env.rob, 'timestep', None),
                  'physics_timestep': float(self.engine.env.rob.sim.model.opt.timestep),
                  'simulated_seconds': sum(self.budget.physics_substeps.values()) * float(self.engine.env.rob.sim.model.opt.timestep),
                  'horizon_violation': self.engine.env.rob.timestep > self.engine.env.rob.horizon}
        (self.folder / 'result.json').write_text(json.dumps(record, indent=2, default=str))

    def get_step_result(self, result):
        return {'success': self.success, 'budgets': self.budget.snapshot(),
                'initial_sha256': self.initial_sha256, 'native_audit': str(self.folder)}

    def get_metadata(self):
        return {'max_steps': 30, 'step_unit': 'model_decision', 'physical_horizon_verified': False}

    def get_action_spec(self):
        return {'tool_envelope': DimSpec(name='tool_decision', dims=ACTION_DIM, format='guava_tool_decision')}

    def get_observation_spec(self):
        return {}

    def cleanup(self):
        if self.meter:
            self.meter.detach()
            self.meter = None
        if self.engine:
            self.engine.env.rob.close()
            close_ik = getattr(self.engine.arm.ik, 'close', None)
            if close_ik:
                close_ik()
            self.engine = None
            self.active_task = None
