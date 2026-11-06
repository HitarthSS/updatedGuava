import json
from pathlib import Path
import tempfile
import unittest

from plan import build
from report import policy_stats, wilson
from run_campaign import Campaign
from unittest.mock import patch


class PlanChecks(unittest.TestCase):
    def test_120_cases_split_seed_spacing_and_native_turn_budgets(self):
        source = Path(__file__).resolve().parents[2]/'harness-bridge/outputs/guava-feedback-v1'
        # Optional package location in the deployed campaign.
        if not (source/'manifest.json').exists():
            source = Path(__file__).parent/'package'
        if not (source/'manifest.json').exists():
            self.skipTest('Generated package not installed beside test')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'package').mkdir()
            (root/'package/manifest.json').write_bytes((source/'manifest.json').read_bytes())
            plan, groups = build(root)
            self.assertEqual(len(plan['cases']), 120)
            self.assertEqual(len({(c['task'], c['episode_idx']) for c in plan['cases']}), 120)
            self.assertEqual(len({c['seed'] for c in plan['cases']}), 120)
            self.assertEqual(plan['physical_final_episodes'], 360)
            self.assertEqual(sum(g['episodes'] for g in groups if g['phase']=='development'), 14)
            dev = {g['task'] for g in groups if g['phase']=='development'}
            final = {g['task'] for g in groups if g['phase']=='final'}
            self.assertFalse(dev & final)
            self.assertTrue(all(c['turn_cap'] in (25,30) for c in plan['cases']))

    def test_costs_include_errors_and_missing_usage_is_not_invented(self):
        stats = policy_stats([{'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}, 'seconds': 1.0},
                              {'error': 'timeout', 'seconds': 3.0}])
        self.assertEqual(stats['vlm_calls'], 2)
        self.assertEqual(stats['errors'], 1)
        self.assertEqual(stats['usage_missing_calls'], 1)
        self.assertEqual(stats['total_tokens'], 12)
        self.assertEqual(stats['model_request_seconds'], 4.0)
        self.assertIsNone(wilson(0, 0))
        self.assertLess(wilson(60,120)[0], .5)
        self.assertGreater(wilson(60,120)[1], .5)

    def test_new_holdout_seed_base_is_used_consistently_without_changing_development(self):
        source = Path(__file__).parent/'package/manifest.json'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'package').mkdir()
            (root/'package/manifest.json').write_bytes(source.read_bytes())
            (root/'RUN_ATTEMPT.json').write_text(json.dumps({'directory':'runs-native-v3','final_seed_base':10042}))
            plan,groups=build(root)
            self.assertEqual(plan['cases'][0]['seed'],10042)
            self.assertTrue(all(c['seed']>=10042 for c in plan['cases']))
            self.assertEqual(next(g for g in groups if g['phase']=='final')['seed_base'],10042)
            self.assertEqual(next(g for g in groups if g['phase']=='development')['seed_base'],4216)
            self.assertTrue(all('runs-native-v3' in g['directory'] for g in groups))

    def test_missing_sam3_holds_before_touching_gpu_or_starting_services(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'RUNTIME_SETUP.json').write_text(json.dumps({'state': 'complete'}))
            (root/'MODEL_ASSETS.json').write_text(json.dumps({'guava': {'state': 'downloaded'}}))
            campaign = Campaign(root, 6)
            with patch('subprocess.check_output', side_effect=AssertionError('GPU must not be touched')):
                self.assertIsNone(campaign.preflight())
            self.assertEqual(json.loads((root/'STATUS.json').read_text())['missing'], ['sam3'])

    def test_pairing_gate_rejects_mismatched_initial_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            group = {'directory': str(root), 'episodes': 1, 'phase': 'final', 'task': 'fixture',
                     'arms': ['baseline','null','candidate']}
            for arm in group['arms']:
                (root/arm).mkdir()
                native = root/arm/'native'
                native.mkdir()
                (native/'result.json').write_text(json.dumps({'horizon_violation': False, 'native_ignore_done': False,
                    'budgets': {'budgeted_turns': 3, 'decision_cap': 25}}))
                (native/'initial.json').write_text(json.dumps({'qpos': [1 if arm == 'candidate' else 0],
                                                             'rgb_sha256': 'fixture'}))
                row = {'episode_idx': 0, 'valid': True, 'success': True, 'action_sha256': ['same'],
                       'metrics': {'native_audit': str(native), 'initial_sha256': 'different' if arm == 'candidate' else 'same'}}
                (root/arm/'records.jsonl').write_text(json.dumps(row)+'\n')
            campaign = Campaign(root, 6)
            with self.assertRaisesRegex(RuntimeError, 'Initial physical state mismatch'):
                campaign.audit_group(group)

    def test_adaptive_queue_refuses_excluded_devices(self):
        from queue_run import main
        with tempfile.TemporaryDirectory() as folder:
            with patch('sys.argv', ['queue', '--workspace', folder, '--gpus', '0,2,6']):
                with self.assertRaisesRegex(ValueError, 'excluded'):
                    main()

    def test_separate_perception_allocation_does_not_lower_shared_device_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                Campaign(root, 6, .27, 14500)
            separate = Campaign(root, 6, .27, 14500, sam3_gpu=5)
            self.assertEqual(separate.sam3_gpu, 5)
            self.assertEqual(separate.env['CUDA_VISIBLE_DEVICES'], '6')

    def test_provisional_queue_cannot_masquerade_as_parity_gated(self):
        from queue_run import main
        with tempfile.TemporaryDirectory() as folder:
            with patch('sys.argv', ['queue', '--workspace', folder, '--provisional', '--require-parity-ready']):
                with self.assertRaisesRegex(ValueError, 'cannot require'):
                    main()


if __name__ == '__main__':
    unittest.main()
