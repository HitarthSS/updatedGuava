"""One public-perception check before accepting Guava's completion declaration.

No simulator/scorer state, task-name rules, action replay or policy updates.
The proposed terminal turn is spent on one native inspection tool instead;
the model's subsequent review consumes the next ordinary turn.
"""
from copy import deepcopy

from .codec import ACTION_DIM, decode, encode

VERSION = 'guava-terminal-evidence-v3'


class TerminalEvidenceState:
    def __init__(self, max_checks=1):
        if type(max_checks) is not int or not 0 <= max_checks <= 1:
            raise ValueError('max_checks must be zero or one')
        self.max_checks = max_checks
        self.checks = 0
        self.last_grasp_target = None
        self.last_alignment_target = None
        self.pending_review = None

    def observe_tool(self, decision):
        if decision.get('kind') != 'tool':
            return
        target = decision.get('arguments', {}).get('target_name')
        if not isinstance(target, str) or not target.strip() or len(target) > 200:
            return
        if decision.get('name') == 'grasp':
            self.last_grasp_target = target
        if decision.get('name') == 'align':
            self.last_alignment_target = target

    def inspection(self, decision, remaining):
        if (decision.get('kind') != 'terminal' or decision.get('value') != 'complete'
                or self.checks >= self.max_checks or remaining < 2):
            return None
        self.checks += 1
        target = self.last_grasp_target or self.last_alignment_target
        if target is None:
            tool = {'kind': 'tool', 'name': 'get_gripper_position', 'arguments': {}}
        else:
            tool = {'kind': 'tool', 'name': 'get_position_and_size',
                    'arguments': {'target_name': target}}
        tool['origin'] = 'harness_terminal_inspection'
        self.pending_review = {'tool': tool['name'], 'target': target}
        return tool

    def review_request(self, observation, remaining):
        if self.pending_review is None:
            return None
        request = observation['guava_request']
        if request.get('history_mode', 'native') != 'native':
            raise ValueError('Terminal evidence requires complete native history')
        # Whitelist public request fields. Evaluator labels cannot affect it.
        payload = {'task_description': observation['task_description'], 'guava_request': {
            'system_prompt': request['system_prompt'], 'message': deepcopy(request['message'])}}
        parts = payload['guava_request']['message']['content']
        texts = [part for part in parts if part.get('type') == 'text']
        if not texts:
            raise ValueError('Native observation lacks public tool feedback')
        note = ('Completion review: you proposed complete, so the harness requested one native '
                'inspection. Review its fresh reported geometry and the current image against '
                'every clause of the original task, including object identity, relative position '
                'and any requested ordering. A tool returning successfully does not by itself '
                'establish task completion. If the requested result is already visible, declare '
                'complete without disturbing it. Otherwise choose the next corrective tool. '
                f'{int(remaining)} ordinary turns remain, including this review; no extra turn allowance.')
        texts[-1]['text'] += '\n' + note
        self.pending_review = None
        return payload


class TerminalEvidenceController:
    def __init__(self, max_checks=1):
        self.max_checks = max_checks

    async def run(self, ctx):
        observation = ctx.observe()
        if 'guava_request' not in observation:
            # HarnessStudio's independent 7-D admission fixture only.
            if (ctx.embodiment.action_dim != 7
                    or not {'images', 'state', 'task_description'} <= observation.keys()):
                raise ValueError('Guava contract did not expose guava_request')
            while not ctx.ended:
                await ctx.execute(await ctx.policy.act())
            return
        state = TerminalEvidenceState(self.max_checks)
        ctx.record('guava_terminal_evidence_start', version=VERSION,
                   max_checks=self.max_checks, scorer_access=False,
                   history_mode='native', automatic_motion_tools=0)
        while not ctx.ended:
            payload = state.review_request(ctx.observe(), ctx.remaining)
            proposed = await ctx.policy.act() if payload is None else await ctx.policy.act(obs=payload)
            row = proposed.reshape(-1, ACTION_DIM)
            if len(row) != 1:
                raise ValueError('Guava policy returned more than one decision')
            decision = decode(row[0])
            inspection = state.inspection(decision, ctx.remaining)
            if inspection is not None:
                ctx.record('guava_terminal_inspection', proposed_terminal=decision,
                           inspection=inspection, charged_turns=1,
                           extra_turn_allowance=0, checks_used=state.checks)
                # Exactly one execution per model decision. Perception consumes
                # the proposed terminal's turn; review uses the next turn.
                await ctx.execute(encode(inspection))
            else:
                await ctx.execute(proposed)
                state.observe_tool(decision)
