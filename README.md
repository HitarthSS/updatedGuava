# My harness for Guava — terminal evidence v3

Training-free completion verification around the frozen [Guava agent](https://arxiv.org/abs/2606.18363), evaluated through HarnessStudio. Current candidate: [`TerminalEvidenceController`](package/hs_guava/terminal_evidence.py).

## Status and results

**V3 completed the 120-case matched evaluation.** It rescued two episodes on the apple/juice reverse-order task with zero regressions across the matched cases.

| Experiment | Authors’ harness | My harness |
| --- | --- | --- |
| Previous native-feedback v2, 120 starts | 112/120 (93.33%) | 111/120 (92.50%) |
| Terminal-evidence v3, 120 fresh starts | 111/120 (92.50%) | 113/120 (94.17%) |

V3 gained **1.67 percentage points** with 14.0% more VLM calls and 21.4% more tokens. The paired result is statistically inconclusive (exact McNemar p = 0.50). See the [evaluation report](EVALUATION.md) and [audited results](results/v3-summary.json).

The paper reports 90.8% on 240 unseen-task episodes. Our 120-case evaluations used fresh starts on a reduced subset. Original source/adapter replay matched turns 0–4 and diverged at turn 5, so the adapter comparison remains provisional. The v2 code and registration are available at the [v2 commit](https://github.com/HitarthSS/updatedGuava/tree/a43dba8e27108087547e20dc39dd904781b81611).

## What v3 changes

1. Let the same frozen model act normally with its complete native conversation.
2. At its first `complete` declaration, if at least two turns remain, execute one native perception tool for the last publicly named grasp/alignment target. With no previous target, inspect the gripper position.
3. Supply the fresh image/tool result with a bounded completion-review note. The model decides whether to correct the task or stop. Check at most once per episode; add no automatic manipulation.

The proposed completion turn pays for the inspection; review consumes the next ordinary turn. Every model query corresponds to one charged decision. No oracle success, simulator object state, task-name special cases, weight updates or paid API calls are used.

Categories: `observation_grounding`, `execution_monitoring`, `agentic_orchestration`.

## Models and setup

- Frozen `AIcell/guava-v13b-qwen3.5-4b`, revision `19a8c77544cf7c161966f49bbb2654b85c6b2fa3`.
- Existing gated `facebook/sam3` checkpoint. No new models are required.
- Guava source `8eeed908bef4fb63c8bfc7c4e2fa2f5684eb067f`; HarnessStudio `785515ff63cacf70617a1bc884cf8e118ce75b85`.
- Prepare the upstream Linux/NVIDIA simulator and serving environments. Install this plugin with `python -m pip install -e package` in the compatible HarnessStudio runtime. Model weights, simulator assets and environments are external dependencies.
- `hs_guava:authors` registers the reference controller; `hs_guava:my_harness` registers v3. The policy server permits loopback model endpoints only.

## Checks

```sh
PYTHONPATH=package:package/tests python -m unittest discover -s package/tests -v
python -m unittest test_plan test_initial_audit test_null_audit -v
python tools/verify_export.py
```

41 package tests and 19 runner/audit tests passed in the existing evaluation environment. HarnessStudio admission tiers 0 and 1 passed. These checks do not establish real-simulator accuracy or original-source parity.

## Evaluation protocol

- One seen smoke case, then seven seen tasks × two development starts.
- Require no development success deficit, no horizon/provenance violations; freeze the design before final evaluation.
- Eight released unseen tasks × fifteen fresh starts: 120 cases for authors’ harness and my harness, plus 120 physical shadow-null controls.
- Final seed base 20042; development base 6200; smoke base 19000. Previously evaluated starts are excluded. Earlier final failures were used for diagnosis, so task definitions are known to the designer.
- Preserve 25/30 semantic turns per task and the released 500,000 native control-call horizon; no multiplier. Inspection and review consume existing turns.
- Audit exact physical starts and bounded initial-render roundoff. Keep the unresolved source/adapter parity gap visible.

`run_campaign.py` runs the gated comparison after the upstream runtime and assets are prepared. The evaluation ran on GPU 5 for Guava and simulation, and GPU 6 for SAM3.

[Attribution](NOTICE.md) · [Design](DESIGN_V3.md) · [Evaluation](EVALUATION.md)
