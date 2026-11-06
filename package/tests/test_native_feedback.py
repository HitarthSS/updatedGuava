from copy import deepcopy
import unittest

from hs_guava.native_feedback import NativeFeedbackController, NativeFeedbackState
from hs_guava.feedback import EMPTY_GRASP, text_of
from hs_guava.messages import prepare_messages
from test_feedback import ScriptedContext, observation, tool


class NativeStateChecks(unittest.TestCase):
    def test_successful_actions_do_not_change_native_request_or_history(self):
        state = NativeFeedbackState()
        initial = observation(initial=True)
        after = observation('Closed gripper — object grasped.')
        state.observe_execution(tool('grasp', target_name='carrot'), initial, after)
        self.assertIsNone(state.request(after))
        history = [{'role': 'system', 'content': 'original system prompt'},
                   deepcopy(initial['guava_request']['message']),
                   {'role': 'assistant', 'content': 'original model reasoning and tool call'}]
        result = prepare_messages(history, after['guava_request'])
        self.assertIs(result, history)
        self.assertEqual(result[-1], after['guava_request']['message'])
        self.assertEqual(result[-2]['content'], 'original model reasoning and tool call')

    def test_explicit_empty_grasp_cue_preserves_native_image_role_and_schema(self):
        state = NativeFeedbackState()
        initial, after = observation(initial=True), observation(EMPTY_GRASP)
        after['scorer_audit'] = {'success': True, 'secret': 'PRIVATE_CANARY'}
        after['guava_request']['private_state'] = 'NESTED_CANARY'
        original = deepcopy(after)
        state.observe_execution(tool('grasp', target_name='carrot'), initial, after)
        payload = state.request(after)
        request = payload['guava_request']
        self.assertEqual(request['message']['role'], 'tool')
        self.assertEqual(request['system_prompt'], original['guava_request']['system_prompt'])
        self.assertEqual(request['message']['content'][0], original['guava_request']['message']['content'][0])
        self.assertTrue(text_of(request['message']).startswith(text_of(original['guava_request']['message'])))
        self.assertNotIn('history_mode', request)
        self.assertNotIn('CANARY', str(payload))
        self.assertEqual(after, original)
        self.assertIsNone(state.request(after))

    def test_cues_are_bounded_and_never_claim_task_success(self):
        state = NativeFeedbackState(max_cues=2)
        before, after = observation(initial=True), observation(EMPTY_GRASP)
        payloads = []
        for target in ('carrot', 'carrot', 'apple', 'potato'):
            state.observe_execution(tool('grasp', target_name=target), before, after)
            payloads.append(state.request(after))
        self.assertEqual([p is not None for p in payloads], [True, False, True, False])
        self.assertEqual(state.used, 2)

    def test_gripper_width_and_a_success_report_do_not_trigger_cues(self):
        state = NativeFeedbackState()
        obs = observation('Closed gripper — object grasped.')
        for _ in range(4):
            state.observe_execution(tool('grasp', target_name='carrot'), obs, obs)
            self.assertIsNone(state.request(obs))


class NativeControllerChecks(unittest.IsolatedAsyncioTestCase):
    async def test_empty_grasp_returns_to_model_without_automatic_replayed_actions(self):
        ctx = ScriptedContext()
        await NativeFeedbackController().run(ctx)
        self.assertEqual([d.get('name', 'terminal') for d in ctx.executed], ['align', 'grasp', 'terminal'])
        self.assertEqual(ctx.steps, len(ctx.requests))
        self.assertEqual(ctx.requests[:2], [None, None])
        self.assertIn('Recovery note:', text_of(ctx.requests[2]['guava_request']['message']))
        self.assertFalse(any(d.get('origin') == 'harness_recovery' for d in ctx.executed))

    async def test_native_controller_respects_turn_limit_and_resets_each_episode(self):
        controller = NativeFeedbackController()
        first, second = ScriptedContext(horizon=2), ScriptedContext(horizon=2)
        await controller.run(first)
        await controller.run(second)
        self.assertEqual(first.steps, 2)
        self.assertEqual(first.requests, second.requests)
        self.assertEqual(first.executed, second.executed)


if __name__ == '__main__':
    unittest.main()
