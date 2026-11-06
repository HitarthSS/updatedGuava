import copy
import unittest
from pathlib import Path

import numpy as np
from harnessstudio.context.trace import fingerprint

from null_audit import load_encoder, verify_native_execution


class NativeProvenanceChecks(unittest.TestCase):
    def fixture(self):
        package = Path(__file__).parent/'package'
        encode = load_encoder(str(package))
        decisions = [{'kind': 'tool', 'name': 'move', 'arguments': {'target_position': [.1,.2,.3]}},
                     {'kind': 'terminal', 'value': 'complete'}]
        arrays = [np.asarray(encode(d), dtype=np.float32) for d in decisions]
        row = {'events': [{'t': 'native', 'step': i, 'decision': i, 'input_sha256': 'obs'+str(i),
                           'action_sha256': fingerprint(a.reshape(1,-1))} for i,a in enumerate(arrays)],
               'obs_sha256': ['obs0','obs1'], 'action_sha256': [fingerprint(a) for a in arrays],
               'accounting': {'steps': 2, 'native_policy_samples': 2, 'shadow_policy_samples': 1}}
        native = {'actions': [{'name': 'move', 'arguments': decisions[0]['arguments'], 'origin': 'model'}],
                  'decision': decisions[-1]}
        return row, native, package

    def test_valid_native_chunk_and_sent_vector_fingerprints_are_verified(self):
        row, native, package = self.fixture()
        proof = verify_native_execution(row, native, package)
        self.assertEqual(proof['native_outputs_verified'], 2)
        self.assertEqual(proof['native_inputs_verified'], 2)

    def test_tampered_sent_action_fails_even_if_native_event_is_valid(self):
        row, native, package = self.fixture()
        row['action_sha256'][0] = 'different-action'
        with self.assertRaisesRegex(RuntimeError, 'executed action differed'):
            verify_native_execution(row, native, package)

    def test_changed_native_observation_fails(self):
        row, native, package = self.fixture()
        row['events'][0]['input_sha256'] = 'overridden-observation'
        with self.assertRaisesRegex(RuntimeError, 'input differed'):
            verify_native_execution(row, native, package)

    def test_recovery_execution_in_null_is_rejected(self):
        row, native, package = self.fixture()
        native['actions'][0]['origin'] = 'harness_recovery'
        with self.assertRaisesRegex(RuntimeError, 'recovery intent'):
            verify_native_execution(row, native, package)

    def test_missing_or_reordered_native_samples_are_rejected(self):
        row, native, package = self.fixture()
        missing = copy.deepcopy(row)
        missing['events'].pop()
        with self.assertRaisesRegex(RuntimeError, 'missing per-step'):
            verify_native_execution(missing, native, package)
        row['events'][0]['step'] = 1
        with self.assertRaisesRegex(RuntimeError, 'reused or reordered'):
            verify_native_execution(row, native, package)


if __name__ == '__main__':
    unittest.main()
