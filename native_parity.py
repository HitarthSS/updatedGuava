"""Native original-runner recording and strict replay for adapter admission only."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
from types import SimpleNamespace


def digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def record_native():
    import openai
    from guava.inference import benchmark_engine
    root = Path(os.environ['GUAVA_PARITY_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    original = openai.OpenAI
    original_state = benchmark_engine.state
    captured_initial = False
    fixed = os.environ.get('GUAVA_FIXED_RESPONSES')
    fixed_rows = [json.loads(line) for line in Path(fixed).read_text().splitlines()] if fixed else None
    fixed_index = 0

    def state(engine):
        nonlocal captured_initial
        value = original_state(engine)
        if not captured_initial:
            captured_initial = True
            initial = {'state': value, 'qpos': engine.env.rob.sim.data.qpos.tolist(),
                       'qvel': engine.env.rob.sim.data.qvel.tolist(),
                       'rgb_sha256': hashlib.sha256(engine.ctx.last_rgb.tobytes()).hexdigest()}
            (root/'initial.json').write_text(json.dumps(initial, sort_keys=True, default=str))
        return value

    def client(*args, **kwargs):
        nonlocal fixed_index
        instance = original(*args, **kwargs) if fixed_rows is None else SimpleNamespace(
            models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id='AIcell/guava-v13b-qwen3.5-4b')])))
        create = instance.chat.completions.create if fixed_rows is None else None

        def recorded(**request):
            nonlocal fixed_index
            if fixed_rows is None:
                response = create(**request)
            else:
                from openai.types.chat import ChatCompletion
                if fixed_index >= len(fixed_rows):
                    raise RuntimeError('Original source requested more responses than preserved cassette')
                response = ChatCompletion.model_validate(fixed_rows[fixed_index]['response'])
                fixed_index += 1
            row = {'request_sha256': digest(request['messages']),
                   'model': request['model'], 'temperature': request['temperature'],
                   'max_tokens': request['max_tokens'], 'response': response.model_dump()}
            row['origin'] = 'preserved-response-source-runner-replay' if fixed_rows is not None else 'live-original-model'
            with (root/'cassette.jsonl').open('a') as file:
                file.write(json.dumps(row) + '\n')
            return response
        if fixed_rows is None:
            instance.chat.completions.create = recorded
        else:
            instance.chat = SimpleNamespace(completions=SimpleNamespace(create=recorded))
        return instance
    openai.OpenAI = client
    benchmark_engine.state = state
    # Execute the released CLI and compatible wrapper without modifying the checkout.
    sys.argv = ['guava.inference.benchmark_compatible_tasks', *sys.argv[1:]]
    runpy.run_module('guava.inference.benchmark_compatible_tasks', run_name='__main__')


def replay_server():
    from hs_guava.model_server import GuavaPolicyServer
    from hs_guava.native import load_source, manifest
    from hs_guava.request_meter import RequestMeter
    from openai.types.chat import ChatCompletion
    from vla_eval.model_servers.serve import run_server

    class ReplayServer(GuavaPolicyServer):
        def __init__(self, audit_dir: str = ''):
            # This admission-only server replays an immutable cassette and needs no live VLM.
            _, _, self.compatible, _, _, _, _, _ = load_source()
            self.model = manifest()['model']['name']
            self.states, self.indices = {}, {}
            self.lock = threading.RLock()
            self.meter = RequestMeter()
            output = Path(audit_dir or os.environ['GUAVA_POLICY_AUDIT'])
            output.mkdir(parents=True, exist_ok=True)
            self.log = (output/'requests.jsonl').open('a', buffering=1)
            self.records = [json.loads(line) for line in Path(os.environ['GUAVA_CASSETTE']).read_text().splitlines()]

        def predict(self, observation, ctx):
            key = (ctx.session_id, ctx.episode_id)
            self.indices.setdefault(key, 0)

            def create(**request):
                index = self.indices[key]
                if index >= len(self.records):
                    raise RuntimeError('Adapter requested more decisions than original runner')
                row = self.records[index]
                if digest(request['messages']) != row['request_sha256']:
                    raise RuntimeError('Original/adapter model request differs at turn ' + str(index))
                if any(request[k] != row[k] for k in ('model', 'temperature', 'max_tokens')):
                    raise RuntimeError('Original/adapter inference configuration differs')
                self.indices[key] += 1
                return ChatCompletion.model_validate(row['response'])
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            return super().predict(observation, ctx)
    run_server(ReplayServer)


if __name__ == '__main__':
    if os.environ.get('GUAVA_REPLAY_ONLY') == '1':
        replay_server()
    else:
        record_native()
