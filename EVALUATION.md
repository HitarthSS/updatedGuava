# Guava terminal evidence v3 evaluation

V3 completed 120 matched cases across eight released unseen tasks, with 15 fresh starts per task and a final seed base of 20042. The seen development gate passed with 11/14 successes for the authors’ harness and 12/14 for v3. The controller and evaluation sources were frozen before the final cases ran. [Design and protocol](DESIGN_V3.md)

| Final metric | Authors’ harness | V3 | Shadow null |
| --- | ---: | ---: | ---: |
| Successes | 111/120 | 113/120 | 111/120 |
| Success rate | 92.50% | 94.17% | 92.50% |
| VLM calls | 1,042 | 1,188 | 2,094 |
| Tokens | 3,775,010 | 4,584,493 | 7,722,505 |
| Summed episode wall time | 61.3 min | 65.9 min | 91.2 min |
| Horizon violations | 0 | 0 | 0 |

V3 rescued two cases and harmed zero. The paired difference was +1.67 percentage points (exact McNemar p = 0.50, task-cluster bootstrap 95% interval 0.0 to 5.0 percentage points). V3 used 14.0% more VLM calls, 21.4% more tokens and 7.5% more episode wall time on shared hardware. The run recorded zero external model API calls and complete token usage.

| Task | Authors’ harness | V3 | Shadow null |
| --- | ---: | ---: | ---: |
| `pick_up_carrot` | 15/15 | 15/15 | 15/15 |
| `tomato_near_potato` | 14/15 | 14/15 | 14/15 |
| `lemon_in_bin` | 15/15 | 15/15 | 15/15 |
| `push_pot` | 14/15 | 14/15 | 14/15 |
| `cube_stack_reverse` | 15/15 | 15/15 | 14/15 |
| `apple_juice_reverse_order` | 12/15 | 14/15 | 13/15 |
| `bin_and_tray_simple` | 13/15 | 13/15 | 13/15 |
| `set_table` | 13/15 | 13/15 | 13/15 |

Original source/adapter replay matched turns 0–4 and diverged at turn 5. This result remains a provisional adapter comparison. Initial physical states matched exactly. Four of 240 rendering comparisons showed roundoff, with at most three pixels changing by one channel level. All observations were retained as rendered. The aggregate, task counts and hashes of the evaluated sources are in the [audited v3 summary](results/v3-summary.json).

The previous v2 cohort used a separate set of starts and recorded 112/120 successes for the authors’ harness and 111/120 for v2. [Historical v2 summary](results/summary.json)
