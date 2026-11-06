"""Conservative public-feedback monitoring that preserves Guava's native history."""
from copy import deepcopy

from .codec import ACTION_DIM, decode
from .feedback import EMPTY_GRASP, feedback_of, image_hash, pose_of, signature


VERSION = 'guava-native-feedback-v2'


class NativeFeedbackState:
    def __init__(self, max_cues=2):
        if type(max_cues) is not int or not 0 <= max_cues <= 3:
            raise ValueError('max_cues must be an integer between zero and three')
        self.max_cues = max_cues
        self.used = 0
        self.cued = set()
        self.pending = None
        self.previous = None

    def observe_execution(self, decision, before, after):
        self.pending = None
        key = signature(decision)
        message = after['guava_request']['message']
        feedback = feedback_of(message)
        reason = None
        if decision.get('name') == 'grasp' and feedback == EMPTY_GRASP:
            reason = 'empty_grasp'
        elif feedback.startswith('Error calling ') and self.previous == key:
            reason = 'repeated_tool_error'
        elif self.previous == key and decision.get('name') in ('align', 'move', 'rotate'):
            previous = before['guava_request']['message']
            old_pose, new_pose = pose_of(previous), pose_of(message)
            if (old_pose is not None and new_pose is not None
                    and sum((a-b)**2 for a,b in zip(old_pose,new_pose)) <= 0.000025
                    and image_hash(previous) is not None
                    and image_hash(previous) == image_hash(message)):
                reason = 'repeated_action_without_change'
        self.previous = key
        if reason and (reason, key) not in self.cued and self.used < self.max_cues:
            self.pending = (reason, key)

    def request(self, observation):
        """None means use the completely unmodified native request path."""
        if self.pending is None:
            return None
        reason, key = self.pending
        self.pending = None
        self.used += 1
        self.cued.add((reason, key))
        request = observation['guava_request']
        if request.get('history_mode', 'native') != 'native':
            raise ValueError('Native feedback requires the native conversation')
        # Explicitly whitelist the public native request, excluding audit/scorer state.
        payload = {'task_description': observation['task_description'], 'guava_request': {
            'system_prompt': request['system_prompt'], 'message': deepcopy(request['message'])}}
        notes = {
            'empty_grasp': 'The last grasp reported no object grasped. Use fresh target geometry and adjust the approach before retrying.',
            'repeated_tool_error': 'The same tool call failed twice. Check its arguments or choose a different tool instead of repeating it unchanged.',
            'repeated_action_without_change': 'Repeating that motion produced no visible or reported position change. Re-observe the target and change the approach.',
        }
        content = payload['guava_request']['message']['content']
        text_parts = [p for p in content if p.get('type') == 'text']
        if not text_parts:
            raise ValueError('Native observation has no tool feedback text')
        text_parts[-1]['text'] += '\nRecovery note: ' + notes[reason]
        return payload


class NativeFeedbackController:
    def __init__(self, max_cues=2):
        self.max_cues = max_cues

    async def run(self, ctx):
        observation = ctx.observe()
        if 'guava_request' not in observation:
            if (ctx.embodiment.action_dim != 7
                    or not {'images', 'state', 'task_description'} <= observation.keys()):
                raise ValueError('Guava contract did not expose guava_request')
            while not ctx.ended:
                await ctx.execute(await ctx.policy.act())
            return
        state = NativeFeedbackState(self.max_cues)
        ctx.record('guava_native_feedback_start', version=VERSION, max_cues=self.max_cues,
                   history_mode='native', scorer_access=False, automatic_tools=0)
        while not ctx.ended:
            before = ctx.observe()
            reason = state.pending[0] if state.pending else None
            payload = state.request(before)
            if payload is None:
                proposed = await ctx.policy.act()
            else:
                ctx.record('guava_native_feedback_cue', reason=reason, cues_used=state.used,
                           extra_policy_calls=0, extra_actions=0)
                proposed = await ctx.policy.act(obs=payload)
            row = proposed.reshape(-1, ACTION_DIM)
            if len(row) != 1:
                raise ValueError('Guava policy returned more than one decision')
            decision = decode(row[0])
            await ctx.execute(proposed)
            if ctx.ended:
                break
            if decision['kind'] == 'tool':
                state.observe_execution(decision, before, ctx.observe())
