"""Semantic scenarios through the real HarnessStudio EpisodeContext, CPU only."""
from types import SimpleNamespace
import unittest

from hs_guava.candidate import FeedbackController
from hs_guava.codec import ACTION_DIM
from test_feedback import ScriptedContext

try:
    from harnessstudio.context.episode import EpisodeContext, CaseKey, Embodiment
    from harnessstudio.context.policy import PolicyProfile
    from harnessstudio.contracts import Contract
    from harnessstudio.core.arms import Arm
    from harnessstudio.core.method import Budgets
except ImportError:
    EpisodeContext = None


@unittest.skipIf(EpisodeContext is None, 'Install the pinned HarnessStudio for framework integration checks')
class HarnessStudioChecks(unittest.IsolatedAsyncioTestCase):
    async def run_arm(self, shadow=False):
        simulation = ScriptedContext(horizon=8)
        native, candidate = ScriptedContext(), ScriptedContext()
        profile = PolicyProfile(capabilities=frozenset({'act', 'chunk'}), execute_horizon=1)
        handle = SimpleNamespace(profile=profile, act=native.act if shadow else candidate.act)
        contract = Contract(name='guava-test', public_keys=('task_description', 'guava_request'),
                            instruction_key='task_description', action_dim=ACTION_DIM,
                            control_mode='guava_tool_decision', default_horizon=8)
        ctx = None

        async def emit(action):
            await simulation.execute(action)
            ctx._note_sent(action)
            ctx._set_observation(simulation.observe())
            if simulation.ended:
                ctx.stop('fixture_terminal')

        ctx = EpisodeContext(case=CaseKey('guava-test', 'synthetic-carrot', 0), contract=contract,
                             horizon=8, arm=Arm.NULL if shadow else Arm.CANDIDATE,
                             namespace='cpu-only', policy=handle, budgets=Budgets(policy_calls=8),
                             bridge=SimpleNamespace(emit=emit), shadow=shadow,
                             embodiment=Embodiment(action_dim=ACTION_DIM, control_mode='guava_tool_decision'))
        raw = simulation.observe()
        raw['success'] = 'PRIVATE_SCORER_CANARY'
        ctx._set_observation(raw)
        if shadow:
            ctx._shadow_policy = SimpleNamespace(act=candidate.act)
        await FeedbackController().run(ctx)
        return simulation, ctx

    async def test_semantic_recovery_uses_framework_budget_and_public_interface(self):
        simulation, ctx = await self.run_arm()
        self.assertEqual(ctx.accounting.steps, 6)
        self.assertEqual(ctx.accounting.method_policy_samples, 3)
        self.assertEqual(sum(d.get('origin') == 'harness_recovery' for d in simulation.executed), 3)
        self.assertNotIn('success', ctx.observe())

    async def test_shadow_null_executes_native_actions_instead_of_recovery_intents(self):
        simulation, ctx = await self.run_arm(shadow=True)
        self.assertEqual([d.get('name', 'terminal') for d in simulation.executed], ['align', 'grasp', 'terminal'])
        self.assertEqual(ctx.accounting.steps, 3)
        self.assertEqual(ctx.accounting.native_policy_samples, 3)
        self.assertEqual(ctx.accounting.shadow_policy_samples, 2)
        self.assertEqual(sum(d.get('origin') == 'harness_recovery' for d in simulation.executed), 0)
