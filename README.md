# My harness for Guava — native feedback v2

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2606.18363). The evaluated behavior is in [`package/hs_guava/native_feedback.py`](package/hs_guava/native_feedback.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | Guava-4B unseen-task mean: 90.8%, 240 paper episodes |

| Our run of the authors' harness | 112/120 (93.33%) |

| **My harness** | **111/120 (92.50%)** |


Eight released unseen task definitions, fifteen fresh seeds each, plus a physical shadow-null arm. Final seed base 10042; separate seen development.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Preserve the complete native Guava conversation, images, assistant outputs and tool feedback.

2. Append at most two short public failure cues after explicit empty grasps, repeated tool errors or repeated motions with no observed change.

3. Let the same frozen model choose every action. No automatic alignment/action replay or extra model calls.


Categories: `execution_monitoring`, `agentic_orchestration`.

## Models and inputs

- AIcell/guava-v13b-qwen3.5-4b at 19a8c77544cf7c161966f49bbb2654b85c6b2fa3, plus gated facebook/sam3.

- HarnessStudio at 785515ff63cacf70617a1bc884cf8e118ce75b85; policy transport is loopback-only.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/hdacnw/guava-release) before installing its model stack.
2. Controller dependency: Install the pinned HarnessStudio runtime and package/.
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 tools/verify_export.py
PYTHONPATH=package:package/tests python -m unittest discover -s package/tests -v
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

Install the pinned HarnessStudio and pip install -e package. Register hs_guava:authors and hs_guava:my_harness; the latter selects NativeFeedbackController. The retained run_campaign.py is the audited native/custom-benchmark driver.


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

25/30 semantic turns per task and the released 500,000-control-call simulator horizon. Every action is metered; no multiplier.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `package/hs_guava/native_feedback.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- The revised candidate did not improve accuracy or compute: 1,038 versus 1,051 VLM calls.

- Original-source replay parity diverged at turn 5 and remains unresolved. Results are a provisional adapter comparison.

- Exact physical starts are required; a disclosed initial-render tolerance permits <=26 pixels at 512x512 to differ by one channel level. Raw images are unchanged.

- The earlier latest-image v1 was rejected after development regression and is not pooled with these results.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

32 CPU tests passed. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
