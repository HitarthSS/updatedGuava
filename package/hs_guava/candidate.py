"""My harness: bounded public context and feedback-triggered native-tool recovery."""
from .codec import ACTION_DIM, decode, encode
from .feedback import FeedbackState, Settings, VERSION


class FeedbackController:
    def __init__(self, recent_actions=6, max_recoveries=2, alignment_age=2, feedback_chars=400):
        self.settings = Settings(recent_actions, max_recoveries, alignment_age, feedback_chars)

    async def run(self, ctx):
        observation = ctx.observe()
        if 'guava_request' not in observation:
            # Permit only HarnessStudio's separate 7-D fixture for its generic admission test.
            if (ctx.embodiment.action_dim != 7
                    or not {'images', 'state', 'task_description'} <= observation.keys()):
                raise ValueError('Guava contract did not expose guava_request')
            ctx.record('guava_feedback_fixture_passthrough', reason='non-Guava CPU fixture')
            while not ctx.ended:
                await ctx.execute(await ctx.policy.act())
            return
        state = FeedbackState(observation, self.settings)
        ctx.record('guava_feedback_start', version=VERSION, settings=vars(self.settings),
                   interventions_charge_turns=True, scorer_access=False)
        while not ctx.ended:
            before = ctx.observe()
            payload = state.request(before, ctx.remaining)
            proposed = await ctx.policy.act(obs=payload)
            # Guava's native profile executes one semantic decision per call.
            if getattr(proposed, 'shape', None) is None:
                raise ValueError('Policy did not return an action array')
            row = proposed.reshape(-1, ACTION_DIM)
            if len(row) != 1:
                raise ValueError('Guava policy returned more than one decision')
            decision = decode(row[0])
            ctx.record('guava_feedback_policy_decision', decision=decision,
                       history_mode='latest-only-v1', current_images=1)
            await ctx.execute(proposed)
            if ctx.ended or decision['kind'] != 'tool':
                break
            after = ctx.observe()
            state.observe_execution(decision, before, after, 'model')
            recovery = state.recovery(decision, after, ctx.remaining)
            if recovery:
                ctx.record('guava_feedback_recovery_start', target=decision['arguments']['target_name'],
                           planned_tools=[item['name'] for item in recovery], remaining=ctx.remaining)
            for intervention in recovery:
                if ctx.ended or ctx.remaining <= 1:
                    break
                before = ctx.observe()
                await ctx.execute(encode(intervention))
                ctx.record('guava_feedback_intervention', decision=intervention, charged_turns=1)
                if ctx.ended:
                    break
                after = ctx.observe()
                state.observe_execution(intervention, before, after, 'harness_recovery')
                if not state.recovery_can_continue(intervention, after):
                    ctx.record('guava_feedback_recovery_return_to_model', tool=intervention['name'])
                    break
