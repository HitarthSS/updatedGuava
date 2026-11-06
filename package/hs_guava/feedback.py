"""Public-observation state for my harness; no simulator or scorer imports."""
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re


VERSION = 'guava-feedback-v1'
HISTORY_MODE = 'latest-only-v1'
EMPTY_GRASP = 'Closed gripper fully — no object grasped.'
GUIDANCE = (
    'Use the current image as visual evidence; historical observations may be stale. '
    'The ledger records executed tools and their reported outcomes, not verified task success. '
    'A grasped report or nonzero gripper width alone does not establish that the intended object is held. '
    'After a repeated failure, change the approach or request fresh geometry instead of repeating blindly. '
    'Respect the remaining turn budget; output exactly one tool or an explicit terminal declaration.'
)


def text_of(message):
    content = message.get('content', [])
    if isinstance(content, str):
        return content
    return '\n'.join(part['text'] for part in content if part.get('type') == 'text')


def feedback_of(message):
    text = text_of(message)
    # This delimiter is the released public gripper text, not privileged state.
    return text.split('Gripper is currently at position', 1)[0].strip()


def image_hash(message):
    content = message.get('content', [])
    if not isinstance(content, list):
        return None
    urls = [part.get('image_url', {}).get('url', '') for part in content if part.get('type') == 'image_url']
    return hashlib.sha256('\n'.join(urls).encode()).hexdigest() if urls else None


def pose_of(message):
    match = re.search(r'Gripper is currently at position\s*\[([^]]+)\]', text_of(message))
    if match:
        try:
            values = [float(v.strip()) for v in match[1].split(',')]
            return values if len(values) == 3 and all(abs(v) < 100 for v in values) else None
        except ValueError:
            pass
    return None


def signature(decision):
    return json.dumps({'name': decision.get('name'), 'arguments': decision.get('arguments')},
                      sort_keys=True, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class Settings:
    recent_actions: int = 6
    max_recoveries: int = 2
    alignment_age: int = 2
    feedback_chars: int = 400

    def __post_init__(self):
        for name, low, high in [('recent_actions', 1, 12), ('max_recoveries', 0, 3),
                                ('alignment_age', 1, 3), ('feedback_chars', 80, 800)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('Invalid feedback setting: ' + name)


class FeedbackState:
    def __init__(self, observation, settings=None):
        self.settings = settings or Settings()
        request = observation['guava_request']
        # Persist the original instruction/tool schema text, but no old image.
        self.initial_text = text_of(request['message']).split('Gripper is currently at position', 1)[0].strip()
        if len(self.initial_text) > 12000:
            raise ValueError('Initial public instruction/schema exceeds context bound')
        self.task = observation['task_description']
        self.ledger = deque(maxlen=self.settings.recent_actions)
        self.alignments = {}
        self.failed = Counter()
        self.recovered_targets = set()
        self.recoveries = 0
        self.executions = 0
        self.phase = 'observe'
        self.last_signal = None
        self.last_decision = None

    def request(self, observation, remaining):
        request = observation['guava_request']
        content = request['message'].get('content', [])
        images = [deepcopy(part) for part in content if part.get('type') == 'image_url']
        if len(images) != 1:
            raise ValueError('My harness requires exactly one current RGB observation')
        failures = [{'action': key, 'count': count} for key, count in sorted(self.failed.items())[-12:]]
        summary = {'phase': self.phase, 'turns_remaining': remaining,
                   'recovery_attempts_used': self.recoveries, 'last_signal': self.last_signal,
                   'recent_executions': list(self.ledger), 'failure_counts': failures}
        text = (self.initial_text + '\n\nCURRENT OBSERVATION:\n' + text_of(request['message'])
                + '\n\nEXECUTION LEDGER (reported outcomes):\n'
                + json.dumps(summary, ensure_ascii=False, sort_keys=True)
                + '\n\nHARNESS GUIDANCE:\n' + GUIDANCE)
        if len(text) > 24000:
            raise ValueError('Bounded feedback context exceeds its explicit limit')
        return {'task_description': self.task, 'guava_request': {
            'system_prompt': request['system_prompt'], 'history_mode': HISTORY_MODE,
            'message': {'role': 'user', 'content': [*images, {'type': 'text', 'text': text}]}}}

    def observe_execution(self, decision, before, after, origin):
        self.executions += 1
        message = after['guava_request']['message']
        feedback = feedback_of(message)
        key = signature(decision)
        name = decision.get('name')
        failed = (feedback == EMPTY_GRASP or feedback.startswith('Error calling ')
                  or ' failed (' in feedback)
        if failed:
            self.failed[key] += 1
        signal = 'tool_error' if failed else None
        previous = before['guava_request']['message']
        old_pose, new_pose = pose_of(previous), pose_of(message)
        # This is only a cue for the next VLM decision; it never proves failure.
        if (self.last_decision == key and name in ('move', 'align', 'rotate')
                and old_pose is not None and new_pose is not None
                and sum((a-b)**2 for a, b in zip(old_pose, new_pose)) <= 0.000025
                and image_hash(previous) is not None and image_hash(previous) == image_hash(message)):
            signal = 'repeated_action_without_observed_change'
        self.last_signal = signal
        self.last_decision = key
        self.ledger.append({'step': self.executions, 'tool': name,
                            'arguments': deepcopy(decision.get('arguments', {})),
                            'feedback': feedback[:self.settings.feedback_chars],
                            'origin': origin, 'signal': signal})
        args = decision.get('arguments', {})
        target = args.get('target_name')
        if (name == 'align' and not failed and isinstance(target, str)
                and feedback.startswith('Aligned to ' + str(args.get('position')) + ' of ' + target + ' with ')):
            self.alignments[target] = (self.executions, deepcopy(args))
        if name in ('grasp', 'close_gripper'):
            self.phase = 'grasp_unverified' if not failed else 'grasp_failed'
        elif name == 'release':
            self.phase = 'after_release'
        elif failed:
            self.phase = 'replan'

    def recovery(self, decision, after, remaining):
        # Never infer an empty grasp from width or simulator-held flags.
        if decision.get('kind') != 'tool' or decision.get('name') != 'grasp':
            return []
        target = decision.get('arguments', {}).get('target_name')
        if not isinstance(target, str) or target in self.recovered_targets:
            return []
        if feedback_of(after['guava_request']['message']) != EMPTY_GRASP:
            return []
        alignment = self.alignments.get(target)
        if (not alignment or self.executions - alignment[0] > self.settings.alignment_age
                or self.recoveries >= self.settings.max_recoveries or remaining < 4):
            return []
        args = alignment[1]
        if args.get('position') not in ('top', 'left', 'right', 'front', 'back'):
            return []
        if args.get('clearance') not in ('small', 'medium', 'large'):
            return []
        self.recovered_targets.add(target)
        self.recoveries += 1
        self.phase = 'recovering_empty_grasp'
        return [
            {'kind': 'tool', 'name': 'release', 'arguments': {}, 'origin': 'harness_recovery'},
            {'kind': 'tool', 'name': 'align', 'arguments': deepcopy(args), 'origin': 'harness_recovery'},
            {'kind': 'tool', 'name': 'grasp', 'arguments': {'target_name': target}, 'origin': 'harness_recovery'},
        ]

    def recovery_can_continue(self, decision, after):
        feedback = feedback_of(after['guava_request']['message'])
        if decision['name'] == 'release':
            return feedback == 'Opened gripper fully.'
        if decision['name'] == 'align':
            args = decision['arguments']
            return feedback.startswith('Aligned to ' + args['position'] + ' of ' + args['target_name'] + ' with ')
        return False
