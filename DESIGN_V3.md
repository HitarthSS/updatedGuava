# Guava terminal evidence v3

Hypothesis: a bounded native perception check before accepting completion can expose false-positive grasps and misplaced objects that the previous empty-grasp feedback missed. Earlier 120-case results are diagnostic evidence; their seeds are excluded from the next final evaluation.

The original frozen Guava-4B, SAM3, complete native history, tools and simulator remain unchanged. At the first `complete` declaration with at least two turns remaining, spend that proposed terminal turn on native `get_position_and_size` for the last publicly named grasp/alignment target, or `get_gripper_position` when none exists. Append a bounded review note to the next public feedback. The same model decides whether to correct or stop. Check at most once per episode. No automatic manipulation or oracle success input.

Every policy query maps to one metered semantic turn. The inspection replaces the terminal decision's execution; subsequent review uses the next turn. Keep 25/30 semantic turns and 500,000 native control calls, no multiplier. At the final turn, accept native termination without an extra query. No task names, seeds, evaluator outcomes or simulator object states are accepted by the harness.

Run one seen smoke case, then fourteen fresh seen development cases. Reject any aggregate accuracy regression or horizon/provenance failure before the final evaluation. Freeze all method and runner sources before 8 unseen tasks × 15 fresh starts (seed base 20042), with authors, shadow-null and candidate arms and alternating arm order. Development seed base 6200; smoke seed base 19000. Prior final task definitions were inspected, so this is fresh-start evaluation, not evidence of held-out task discovery.

Retain the disclosed source/adapter parity gap (turn-5 divergence) and initial-render roundoff audit. A passed live comparison does not prove exact original-code reproduction. No paid model APIs, new weights or model training.
