"""CPU checks for the generated boundary. No models, network or GPU required."""
import math
from types import SimpleNamespace
import unittest

from hs_guava.codec import ACTION_DIM, decode, encode
from hs_guava.budget import Budget, MotionMeter


class BridgeChecks(unittest.TestCase):
    def test_utf8_round_trip(self):
        value = {'kind': 'tool', 'name': 'grasp', 'arguments': {'target_name': 'café cup'}}
        action = encode(value)
        self.assertEqual(len(action), ACTION_DIM)
        self.assertEqual(decode(action), value)

    def test_terminal_and_invalid_round_trip(self):
        for value in [{'kind': 'terminal', 'value': 'done'}, {'kind': 'invalid', 'reason': 'invalid_model_output'}]:
            self.assertEqual(decode(encode(value)), value)

    def test_corrupt_transport_is_rejected(self):
        valid = encode({'kind': 'terminal'})
        for index, value in [(0, -1), (1, math.nan), (2, 0.5), (-1, 1)]:
            bad = valid.copy()
            bad[index] = value
            with self.assertRaises(ValueError):
                decode(bad)

    def test_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            encode({'kind': 'terminal', 'value': 'x' * ACTION_DIM})

    def test_decisions_and_tools_have_separate_caps(self):
        budget = Budget(2, 1)
        budget.decision()
        self.assertEqual(budget.tools, 0)
        budget.tool()
        self.assertTrue(budget.ended)
        with self.assertRaises(RuntimeError):
            budget.tool()

    def test_all_motion_phases_are_counted(self):
        data = SimpleNamespace(time=0.0)
        environment = SimpleNamespace(rob=SimpleNamespace(sim=SimpleNamespace(
            data=data, model=SimpleNamespace(opt=SimpleNamespace(timestep=0.01)))))
        def advance(action):
            data.time += 0.05
        environment.step = advance
        budget = Budget(25, 25)
        meter = MotionMeter(environment, budget)
        meter.attach()
        for phase in ['setup', 'execution', 'stability']:
            meter.phase = phase
            environment.step(None)
        meter.detach()
        self.assertIs(environment.step, advance)
        self.assertEqual(budget.control_steps, {'setup': 1, 'execution': 1, 'stability': 1})
        self.assertEqual(budget.physics_substeps, {'setup': 5, 'execution': 5, 'stability': 5})

    def test_failed_motion_still_counts_executed_physics(self):
        data = SimpleNamespace(time=0.0)
        environment = SimpleNamespace(rob=SimpleNamespace(sim=SimpleNamespace(
            data=data, model=SimpleNamespace(opt=SimpleNamespace(timestep=0.01)))))
        def advance_then_fail(action):
            data.time += 0.03
            raise RuntimeError('Controller abort')
        environment.step = advance_then_fail
        budget = Budget(25, 25)
        meter = MotionMeter(environment, budget)
        meter.phase = 'execution'
        meter.attach()
        with self.assertRaises(RuntimeError):
            environment.step(None)
        meter.detach()
        self.assertEqual(budget.control_steps['execution'], 1)
        self.assertEqual(budget.physics_substeps['execution'], 3)


if __name__ == '__main__':
    unittest.main()
