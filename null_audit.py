"""Verify shadow-null execution provenance, not cross-rollout determinism.

Independent MuJoCo renders can differ by one pixel after identical actions.
The null contract is that it executes the native policy at its actual current
observation while paying for shadow compute. Every input and output below is
checked against the pinned HarnessStudio framework's own trace events.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=4)
def load_encoder(package):
    spec = importlib.util.spec_from_file_location('_guava_locked_codec', Path(package)/'hs_guava/codec.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode


def verify_native_execution(record, native, package):
    import numpy as np
    from harnessstudio.context.trace import fingerprint

    fail = lambda detail: RuntimeError('Shadow-null native provenance failed: '+detail)
    events = [event for event in record['events'] if event.get('t') == 'native']
    sent, observations = record['action_sha256'], record['obs_sha256']
    count = len(sent)
    accounting = record['accounting']
    if (not count or len(events) != count or len(observations) != count
            or accounting['steps'] != count or accounting['native_policy_samples'] != count):
        raise fail('missing per-step native policy evidence')
    decisions = []
    for entry in native['actions']:
        if entry.get('origin', 'model') != 'model':
            raise fail('a recovery intent was executed in the null arm')
        decisions.append({'kind': 'tool', 'name': entry['name'], 'arguments': entry['arguments']})
    if native['decision']['kind'] != 'tool':
        decisions.append(native['decision'])
    if len(decisions) != count:
        raise fail('native tool audit and executed step count differ')
    encode = load_encoder(str(package))
    for step, (event, decision) in enumerate(zip(events, decisions)):
        if event['step'] != step or event['decision'] != step:
            raise fail('native sample was reused or reordered')
        if event['input_sha256'] != observations[step]:
            raise fail('native policy input differed from the executed observation')
        action = np.asarray(encode(decision), dtype=np.float32)
        # The native event fingerprints the one-action chunk; the transport
        # fingerprints the individual vector. Shape is part of the fingerprint.
        if (fingerprint(action.reshape(1, -1)) != event['action_sha256']
                or fingerprint(action) != sent[step]):
            raise fail('executed action differed from the native policy output')
    return {'state': 'passed', 'check': 'native-input-and-output-provenance-v1',
            'native_inputs_verified': count, 'native_outputs_verified': count,
            'shadow_policy_samples': accounting.get('shadow_policy_samples', 0),
            'cross_rollout_byte_equality_required': False}
