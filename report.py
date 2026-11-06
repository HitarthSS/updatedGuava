"""Matched coverage and separate request, tool-wall and simulated-time accounting."""
import json
import math
from pathlib import Path
import random

from plan import write


def policy_stats(calls):
    attempts = [attempt for row in calls for attempt in (row.get('attempts') or [row])]
    return {'vlm_calls': len(attempts), 'logical_decisions': len(calls), 'errors': sum('error' in row for row in calls),
            'usage_missing_calls': sum(not row.get('usage') for row in attempts),
            'prompt_tokens': sum((row.get('usage') or {}).get('prompt_tokens', 0) or 0 for row in attempts),
            'completion_tokens': sum((row.get('usage') or {}).get('completion_tokens', 0) or 0 for row in attempts),
            'total_tokens': sum((row.get('usage') or {}).get('total_tokens', 0) or 0 for row in attempts),
            'model_request_seconds': sum(row.get('seconds', 0) for row in calls),
            'image_occurrences': sum(row.get('request', {}).get('image_count', 0) for row in calls)}


def wilson(successes, episodes):
    if not episodes:
        return None
    z = 1.959963984540054
    p = successes/episodes
    d = 1+z*z/episodes
    c = (p+z*z/(2*episodes))/d
    w = z*math.sqrt(p*(1-p)/episodes+z*z/(4*episodes*episodes))/d
    return [c-w, c+w]


def summarize(root):
    groups = json.loads((root/'GROUPS.json').read_text())
    arms = {arm: {'episodes': 0, 'successes': 0, 'vlm_calls': 0, 'prompt_tokens': 0,
                 'completion_tokens': 0, 'total_tokens': 0, 'model_request_seconds': 0,
                 'wall_seconds': 0, 'tool_seconds': 0, 'simulated_seconds': 0, 'horizon_violations': 0,
                 'usage_missing_calls': 0}
            for arm in ('baseline', 'candidate', 'null')}
    helped = harmed = matched = 0
    render_checks = []
    task_differences = []
    for group in groups:
        directory = Path(group['directory'])
        if group['phase'] != 'final' or not (directory/'AUDITED.json').exists():
            continue
        render_checks.extend(json.loads((directory/'AUDITED.json').read_text()).get('initial_observation_checks', []))
        records = {}
        for arm, total in arms.items():
            records[arm] = [json.loads(line) for line in (directory/arm/'records.jsonl').read_text().splitlines()]
            stats = json.loads((directory/arm/'POLICY_STATS.json').read_text())
            for key in ('vlm_calls', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'model_request_seconds', 'usage_missing_calls'):
                total[key] += stats[key]
            for row in records[arm]:
                native = json.loads((Path(row['metrics']['native_audit'])/'result.json').read_text())
                total['episodes'] += 1
                total['successes'] += bool(row['success'])
                for key in ('tool_seconds', 'simulated_seconds'):
                    total[key] += native[key]
                total['wall_seconds'] += row['wall_seconds']
                total['horizon_violations'] += native['horizon_violation']
        base = {row['episode_idx']: row for row in records['baseline']}
        differences = []
        for row in records['candidate']:
            before, after = bool(base[row['episode_idx']]['success']), bool(row['success'])
            matched += 1
            helped += after and not before
            harmed += before and not after
            differences.append(int(after)-int(before))
        task_differences.append(sum(differences)/len(differences))
    for total in arms.values():
        n = total['episodes']
        total['success_rate'] = total['successes']/n if n else None
        total['wilson_95'] = wilson(total['successes'], n)
    paired = {'matched_cases': matched, 'helped': helped, 'harmed': harmed,
              'difference_pp': (helped-harmed)*100/matched if matched else None}
    discordant = helped+harmed
    paired['mcnemar_exact_p'] = min(1.0, 2*sum(math.comb(discordant, k) for k in range(min(helped,harmed)+1))/2**discordant) if discordant else 1.0
    if len(task_differences) > 1:
        rng = random.Random(42)
        samples = sorted(sum(rng.choice(task_differences) for _ in task_differences)/len(task_differences)*100 for _ in range(2000))
        paired['task_cluster_bootstrap_95_pp'] = [samples[49], samples[1949]]
    result = {'coverage': '120-case unseen subset' if matched == 120 else 'partial matched subset',
              'reproduction': json.loads((root/'RUN_MODE.json').read_text()) if (root/'RUN_MODE.json').exists() else {'original_code_parity': 'unverified'},
              'initial_rendering_audit': {'pairs_checked': len(render_checks),
                  'pairs_with_roundoff': sum(not row['rgb_exact'] for row in render_checks),
                  'max_changed_pixels': max([r['changed_pixels'] for r in render_checks], default=0),
                  'max_rgb_channel_delta': max([r['maximum_rgb_channel_delta'] for r in render_checks], default=0)},
              'arms': arms, 'paired': paired, 'external_model_api_calls': 0,
              'timing_note': 'Shared GPU. Model request seconds include queue/serving overhead; tool wall includes perception and control; simulated seconds are physics time.'}
    write(root/'SUMMARY.json', result)
    return result


if __name__ == '__main__':
    import sys
    print(json.dumps(summarize(Path(sys.argv[1]).resolve()), indent=2))
