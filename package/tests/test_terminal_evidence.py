"""Public inputs, termination, and horizon contracts; no model or simulator."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

from hs_guava.codec import decode, encode
from hs_guava.terminal_evidence import TerminalEvidenceController, TerminalEvidenceState
from test_feedback import observation, tool


class EvidenceContext:
    def __init__(self, decisions, observations, remaining):
        self.decisions=list(decisions)
        self.observations=list(observations)
        self.horizon=remaining
        self.executed=[]
        self.requests=[]
        self.terminal=False
        self.embodiment=SimpleNamespace(action_dim=len(encode({'kind':'terminal','value':'complete'})))
        self.policy=SimpleNamespace(act=self.act)

    @property
    def remaining(self):return self.horizon-len(self.executed)
    @property
    def ended(self):return self.terminal or self.remaining<=0
    def observe(self):return deepcopy(self.observations[min(len(self.executed),len(self.observations)-1)])
    def record(self,*args,**kwargs):pass
    async def act(self,obs=None):
        self.requests.append(deepcopy(obs))
        return np.asarray([encode(self.decisions[len(self.requests)-1])],dtype=np.float32)
    async def execute(self,action):
        if self.ended:raise AssertionError('Horizon exceeded')
        decision=decode(np.asarray(action).reshape(-1));self.executed.append(decision)
        self.terminal=decision['kind']!='tool'
        return 1


class EvidenceStateTests(unittest.TestCase):
    def test_uses_only_a_previously_proposed_target(self):
        s=TerminalEvidenceState()
        s.observe_tool(tool('grasp',target_name='a novel object'))
        found=s.inspection({'kind':'terminal','value':'complete'},8)
        self.assertEqual(found['name'],'get_position_and_size')
        self.assertEqual(found['arguments'],{'target_name':'a novel object'})

    def test_premature_completion_uses_native_gripper_observation(self):
        s=TerminalEvidenceState()
        self.assertEqual(s.inspection({'kind':'terminal','value':'complete'},25)['name'],
                         'get_gripper_position')

    def test_failed_invalid_and_normal_tools_are_not_overridden(self):
        for d in [{'kind':'terminal','value':'failed'}, {'kind':'invalid'}, tool('release')]:
            self.assertIsNone(TerminalEvidenceState().inspection(d,25))

    def test_check_is_bounded_and_cannot_extend_last_turn(self):
        s=TerminalEvidenceState();done={'kind':'terminal','value':'complete'}
        self.assertIsNone(s.inspection(done,1))
        self.assertIsNotNone(s.inspection(done,2))
        self.assertIsNone(s.inspection(done,20))

    def test_review_preserves_native_image_role_and_input(self):
        s=TerminalEvidenceState();s.inspection({'kind':'terminal','value':'complete'},10)
        obs=observation('Position of cup is at [0.4, 0.2, 0.05].');saved=deepcopy(obs)
        actual=s.review_request(obs,9)
        self.assertEqual(obs,saved)
        self.assertEqual(actual['guava_request']['message']['role'],obs['guava_request']['message']['role'])
        self.assertEqual(actual['guava_request']['message']['content'][0],obs['guava_request']['message']['content'][0])
        self.assertEqual(actual['guava_request']['system_prompt'],obs['guava_request']['system_prompt'])
        self.assertIn('9 ordinary turns remain',actual['guava_request']['message']['content'][-1]['text'])
        self.assertIsNone(s.review_request(obs,8))

    def test_oracle_fields_cannot_change_review(self):
        out=[]
        for success in (False,True):
            s=TerminalEvidenceState();s.inspection({'kind':'terminal','value':'complete'},10)
            obs=observation();obs.update(success=success,scorer_audit=[{'success':success}])
            out.append(s.review_request(obs,9))
        self.assertEqual(*out)
        self.assertNotIn('scorer_audit',out[0])


class EvidenceControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_inspection_review_and_native_termination(self):
        decisions=[tool('grasp',target_name='cup'), {'kind':'terminal','value':'complete'},
                   {'kind':'terminal','value':'complete'}]
        ctx=EvidenceContext(decisions,[observation(),observation('Closed gripper — object grasped.'),
                                       observation('Cup is located at [0.4, 0.2, 0.05].')],remaining=5)
        await TerminalEvidenceController().run(ctx)
        self.assertEqual(len(ctx.executed),3)
        self.assertEqual(ctx.executed[1]['name'],'get_position_and_size')
        self.assertEqual(ctx.executed[-1]['kind'],'terminal')
        self.assertEqual(len(ctx.requests),3)
        self.assertEqual(sum(p is not None for p in ctx.requests),1)

    async def test_last_turn_terminates_without_unbudgeted_review(self):
        ctx=EvidenceContext([{'kind':'terminal','value':'complete'}],[observation()],remaining=1)
        await TerminalEvidenceController().run(ctx)
        self.assertEqual(ctx.executed,[{'kind':'terminal','value':'complete'}])
        self.assertEqual(len(ctx.requests),1)

    async def test_state_resets_for_every_episode(self):
        controller=TerminalEvidenceController()
        for _ in range(2):
            ctx=EvidenceContext([{'kind':'terminal','value':'complete'}]*2,
                                [observation(),observation()],remaining=3)
            await controller.run(ctx)
            self.assertEqual(ctx.executed[0]['name'],'get_gripper_position')


if __name__=='__main__':unittest.main()
