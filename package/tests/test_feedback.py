"""Semantic Guava scenarios with scripted public observations; no models or simulator."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from hs_guava.candidate import FeedbackController
from hs_guava.codec import ACTION_DIM, decode, encode
from hs_guava.feedback import EMPTY_GRASP, FeedbackState, Settings, feedback_of, text_of
from hs_guava.messages import prepare_messages, request_audit


def tool(name, **arguments):
    return {'kind': 'tool', 'name': name, 'arguments': arguments}


def observation(feedback='', initial=False, image='current-image', pose='0.100, 0.200, 0.300'):
    gripper = ('Gripper is currently at position [' + pose
               + '] with a rotation of [roll=180.0°, pitch=0.0°, yaw=0.0°] and 0% of maximum width.')
    text = ('Task: pick up carrot\nAvailable tools: grasp, align, release\n' if initial else feedback + '\n') + gripper
    return {'task_description': 'pick up carrot', 'guava_request': {
        'system_prompt': 'original system prompt', 'message': {'role': 'user' if initial else 'tool',
        'content': [{'type': 'image_url', 'image_url': {'url': image}}, {'type': 'text', 'text': text}]}}}


class StateChecks(unittest.TestCase):
    def state_with_miss(self):
        initial = observation(initial=True)
        state = FeedbackState(initial)
        aligned = observation('Aligned to right of carrot with medium clearance.')
        state.observe_execution(tool('align', target_name='carrot', position='right', clearance='medium'),
                                initial, aligned, 'model')
        missed = observation(EMPTY_GRASP)
        grasp = tool('grasp', target_name='carrot')
        state.observe_execution(grasp, aligned, missed, 'model')
        return state, grasp, missed

    def test_only_current_image_public_text_and_executed_history_reach_policy(self):
        state, grasp, missed = self.state_with_miss()
        missed['secret_goal'] = 'PRIVATE_CANARY'
        missed['scorer_audit'] = {'success': True}
        missed['guava_request']['private_state'] = 'NESTED_CANARY'
        before = deepcopy(missed)
        payload = state.request(missed, 19)
        text = text_of(payload['guava_request']['message'])
        self.assertIn('Task: pick up carrot', text)
        self.assertIn('Available tools:', text)
        self.assertIn('grasp_failed', text)
        self.assertNotIn('PRIVATE_CANARY', str(payload))
        self.assertNotIn('NESTED_CANARY', str(payload))
        self.assertNotIn('scorer_audit', str(payload))
        self.assertEqual(text.count('Gripper is currently at position'), 1)
        images = [p for p in payload['guava_request']['message']['content'] if p['type'] == 'image_url']
        self.assertEqual(images, [missed['guava_request']['message']['content'][0]])
        self.assertEqual(missed, before)

    def test_empty_grasp_replays_chosen_side_alignment_once(self):
        state, grasp, missed = self.state_with_miss()
        plan = state.recovery(grasp, missed, 4)
        self.assertEqual([d['name'] for d in plan], ['release', 'align', 'grasp'])
        self.assertEqual(plan[1]['arguments']['position'], 'right')
        self.assertEqual(state.recovery(grasp, missed, 20), [])

    def test_recovery_requires_budget_explicit_failure_and_recent_alignment(self):
        state, grasp, missed = self.state_with_miss()
        self.assertEqual(state.recovery(grasp, missed, 3), [])
        self.assertEqual(state.recovery(grasp, observation('Closed gripper — object grasped.'), 20), [])
        self.assertEqual(state.recovery(grasp, observation('Unknown result'), 20), [])
        state.executions += 3
        self.assertEqual(state.recovery(grasp, missed, 20), [])
        fresh = FeedbackState(observation(initial=True))
        self.assertEqual(fresh.recovery(grasp, missed, 20), [])

    def test_closed_width_alone_does_not_trigger_recovery(self):
        state, grasp, missed = self.state_with_miss()
        self.assertEqual(state.recovery(grasp, observation('Closed gripper — object grasped.'), 20), [])
        self.assertEqual(state.recovered_targets, set())

    def test_ledger_is_bounded_and_error_evidence_survives_recent_window(self):
        state, grasp, missed = self.state_with_miss()
        for _ in range(20):
            state.observe_execution(grasp, missed, missed, 'model')
        self.assertEqual(len(state.ledger), 6)
        self.assertEqual(sum(state.failed.values()), 21)
        self.assertLess(len(text_of(state.request(missed, 3)['guava_request']['message'])), 24000)

    def test_repeated_unchanged_pose_and_image_produces_replan_cue(self):
        obs = observation('Moved gripper to [0.1, 0.2, 0.3].')
        state = FeedbackState(observation(initial=True))
        action = tool('move', target_position=[0.1, 0.2, 0.3])
        state.observe_execution(action, obs, obs, 'model')
        state.observe_execution(action, obs, obs, 'model')
        self.assertEqual(state.last_signal, 'repeated_action_without_observed_change')
        state.observe_execution(action, obs, observation(image='changed-image'), 'model')
        self.assertIsNone(state.last_signal)

    def test_settings_and_context_bounds_are_enforced(self):
        for setting in [dict(max_recoveries=-1), dict(recent_actions=100), dict(alignment_age=True)]:
            with self.assertRaises(ValueError):
                Settings(**setting)
        obs = observation(initial=True)
        obs['guava_request']['message']['content'][1]['text'] = 'x' * 12001
        with self.assertRaises(ValueError):
            FeedbackState(obs)

    def test_native_messages_are_preserved_and_compact_mode_is_stateless(self):
        initial = observation(initial=True)
        history = [{'role': 'system', 'content': 'original system prompt'}]
        self.assertIs(prepare_messages(history, initial['guava_request']), history)
        self.assertEqual(history[1], initial['guava_request']['message'])
        history.append({'role': 'assistant', 'content': 'old reasoning'})
        state = FeedbackState(initial)
        compact = state.request(observation('new feedback', image='latest'), 10)['guava_request']
        snapshot = deepcopy(history)
        messages = prepare_messages(history, compact)
        self.assertEqual(history, snapshot)
        self.assertEqual(len(messages), 2)
        self.assertNotIn('old reasoning', str(messages))
        self.assertEqual(request_audit(messages)['image_count'], 1)
        self.assertEqual(request_audit(history)['image_count'], 1)
        compact['history_mode'] = 'unreviewed-mode'
        with self.assertRaises(ValueError):
            prepare_messages(history, compact)


class ScriptedContext:
    def __init__(self, *, horizon=9, abort_alignment=False):
        self.observation = observation(initial=True)
        self.horizon = horizon
        self.steps = 0
        self.terminal = False
        self.executed = []
        self.requests = []
        self.events = []
        self.grasps = 0
        self.abort_alignment = abort_alignment
        self.embodiment = SimpleNamespace(action_dim=ACTION_DIM, control_mode='guava_tool_decision')
        self.policy = SimpleNamespace(act=self.act)

    @property
    def remaining(self):
        return self.horizon - self.steps

    @property
    def ended(self):
        return self.terminal or self.remaining <= 0

    def observe(self):
        return deepcopy(self.observation)

    def record(self, event, **fields):
        self.events.append((event, fields))

    async def act(self, obs=None):
        self.requests.append(deepcopy(obs))
        if len(self.requests) == 1:
            decision = tool('align', target_name='carrot', position='right', clearance='medium')
        elif len(self.requests) == 2:
            decision = tool('grasp', target_name='carrot')
        else:
            decision = {'kind': 'terminal', 'value': 'Task complete.'}
        return np.array([encode(decision)], dtype=np.float32)

    async def execute(self, action):
        decision = decode(np.asarray(action).reshape(-1))
        if self.ended:
            return 0
        self.steps += 1
        self.executed.append(decision)
        if decision['kind'] == 'terminal':
            self.terminal = True
            return 1
        name = decision['name']
        if name == 'align':
            feedback = ('Error calling align: controller abort' if self.abort_alignment and self.steps > 2
                        else 'Aligned to right of carrot with medium clearance.')
        elif name == 'grasp':
            self.grasps += 1
            feedback = EMPTY_GRASP if self.grasps == 1 else 'Closed gripper — object grasped.'
        elif name == 'release':
            feedback = 'Opened gripper fully.'
        else:
            raise AssertionError(name)
        self.observation = observation(feedback)
        return 1


class ControllerChecks(unittest.IsolatedAsyncioTestCase):
    async def test_complete_recovery_uses_three_turns_and_no_intermediate_vlm_calls(self):
        ctx = ScriptedContext()
        await FeedbackController().run(ctx)
        self.assertEqual([d.get('name', 'terminal') for d in ctx.executed],
                         ['align', 'grasp', 'release', 'align', 'grasp', 'terminal'])
        self.assertEqual(ctx.steps, 6)
        self.assertEqual(len(ctx.requests), 3)
        self.assertEqual(sum(d.get('origin') == 'harness_recovery' for d in ctx.executed), 3)
        last = text_of(ctx.requests[-1]['guava_request']['message'])
        self.assertIn('harness_recovery', last)
        self.assertIn('object grasped', last)

    async def test_alignment_error_cancels_remaining_recovery_and_calls_model(self):
        ctx = ScriptedContext(abort_alignment=True)
        await FeedbackController().run(ctx)
        self.assertEqual([d.get('name', 'terminal') for d in ctx.executed],
                         ['align', 'grasp', 'release', 'align', 'terminal'])
        self.assertEqual(ctx.grasps, 1)
        self.assertIn('controller abort', text_of(ctx.requests[-1]['guava_request']['message']))

    async def test_near_horizon_skips_recovery_and_keeps_one_native_slot(self):
        ctx = ScriptedContext(horizon=5)
        await FeedbackController().run(ctx)
        self.assertEqual(ctx.steps, 3)
        self.assertFalse(any(d.get('origin') == 'harness_recovery' for d in ctx.executed))

    async def test_all_executions_stop_at_the_horizon(self):
        ctx = ScriptedContext(horizon=2)
        await FeedbackController().run(ctx)
        self.assertEqual(ctx.steps, 2)
        self.assertEqual(len(ctx.requests), 2)

    async def test_episode_state_does_not_carry_to_the_next_case(self):
        controller = FeedbackController()
        a, b = ScriptedContext(), ScriptedContext()
        await controller.run(a)
        await controller.run(b)
        self.assertEqual(a.executed, b.executed)
        self.assertEqual(a.requests, b.requests)


if __name__ == '__main__':
    unittest.main()
