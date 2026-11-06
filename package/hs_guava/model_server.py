"""The frozen local Guava VLM, original history construction and original parser."""
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

import anyio
import numpy as np
from vla_eval.model_servers.base import ModelServer
from vla_eval.specs import DimSpec

from .codec import ACTION_DIM, encode
from .native import load_source, manifest
from .messages import prepare_messages, request_audit
from .request_meter import RequestMeter


class GuavaPolicyServer(ModelServer):
    harness_capabilities = frozenset({'act', 'chunk'})

    def __init__(self, audit_dir: str = ''):
        from openai import OpenAI
        import httpx
        _, _, self.compatible, _, _, _, _, _ = load_source()
        endpoint = os.environ.get('GUAVA_MODEL_URL', 'http://127.0.0.1:8000/v1')
        url = urlparse(endpoint)
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('This generated deployment permits a loopback model server only')
        pin = manifest()['model']
        if os.environ.get('GUAVA_MODEL_REVISION') != pin['revision']:
            raise ValueError('GUAVA_MODEL_REVISION must match the pinned serving command')
        self.model = pin['name']
        self.meter = RequestMeter()
        http = httpx.Client(event_hooks={'request': [self.meter.request], 'response': [self.meter.response]})
        self.client = OpenAI(base_url=endpoint, api_key='EMPTY', max_retries=2, timeout=180.0, http_client=http)
        if self.model not in [item.id for item in self.client.models.list().data]:
            raise RuntimeError('Exact requested model is not served')
        self.states = {}
        self.lock = threading.RLock()
        output = Path(audit_dir or os.environ.get('GUAVA_POLICY_AUDIT', 'policy-audit'))
        output.mkdir(parents=True, exist_ok=True)
        self.log = (output / 'requests.jsonl').open('a', buffering=1)

    def get_action_spec(self):
        return {'tool_envelope': DimSpec(name='tool_decision', dims=ACTION_DIM, format='guava_tool_decision')}

    def get_observation_spec(self):
        return {}

    async def on_episode_start(self, config, ctx):
        self.states.pop((ctx.session_id, ctx.episode_id), None)

    async def on_episode_end(self, result, ctx):
        self.states.pop((ctx.session_id, ctx.episode_id), None)

    def predict(self, observation, ctx):
        with self.lock:
            key = (ctx.session_id, ctx.episode_id)
            request = observation['guava_request']
            history = self.states.setdefault(key, [{'role': 'system', 'content': request['system_prompt']}])
            messages = prepare_messages(history, request)
            audit = request_audit(messages)
            started = time.perf_counter()
            self.meter.reset()
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0, max_tokens=8192)
            except Exception as error:
                self.log.write(json.dumps({'session': ctx.session_id, 'episode': ctx.episode_id,
                    'history_mode': request.get('history_mode', 'native'), 'request': audit,
                    'seconds': time.perf_counter() - started, 'error': type(error).__name__,
                    'request_attempts': len(self.meter.attempts), 'attempts': self.meter.snapshot()}) + '\n')
                raise
            choice = response.choices[0]
            text = choice.message.content or ''
            if request.get('history_mode', 'native') == 'native':
                history.append({'role': 'assistant', 'content': text})
            self.compatible.audit = []
            parser_error = None
            try:
                if choice.finish_reason == 'length':
                    decision = {'kind': 'invalid', 'reason': 'incomplete_model_output'}
                else:
                    kind, value = self.compatible.compatible(text, 'full')
                    if kind == 'terminal':
                        decision = {'kind': 'terminal', 'value': value}
                    else:
                        if (kind != 'tool' or not isinstance(value, dict)
                                or not isinstance(value.get('name'), str)
                                or not isinstance(value.get('arguments'), dict)):
                            raise ValueError('Malformed tool decision')
                        decision = {'kind': 'tool', **value}
                encoded = encode(decision)
            except (ValueError, KeyError, TypeError) as error:
                decision = {'kind': 'invalid', 'reason': 'invalid_model_output'}
                parser_error = {'type': type(error).__name__, 'detail': str(error)[:500]}
                encoded = encode(decision)
            self.log.write(json.dumps({'session': ctx.session_id, 'episode': ctx.episode_id,
                'text': text, 'decision': decision, 'parser_audit': self.compatible.audit,
                'parser_error': parser_error,
                'history_mode': request.get('history_mode', 'native'), 'request': audit,
                'seconds': time.perf_counter() - started, 'request_attempts': len(self.meter.attempts),
                'attempts': self.meter.snapshot(),
                'usage': response.usage.model_dump() if response.usage else None}) + '\n')
            return np.asarray(encoded, dtype=np.float32)

    async def on_observation(self, obs, ctx):
        action = await anyio.to_thread.run_sync(self.predict, obs, ctx)
        await ctx.send_action({'actions': action})


if __name__ == '__main__':
    from vla_eval.model_servers.serve import run_server
    run_server(GuavaPolicyServer)
