# Terminal evidence integration

Selected factory: `hs_guava:my_harness`; selected controller: `hs_guava.terminal_evidence:TerminalEvidenceController(max_checks=1)`. The authors’ factory is unchanged. Use HarnessStudio baseline, physical shadow-null and candidate arms.

The controller consumes only public observations and the previously proposed native tools. It executes one inspection tool in place of the first accepted completion proposal, then returns to the same model on the next budgeted decision. One policy call maps to one semantic execution. The original benchmark bridge and policy/history/parser code are unchanged.

The native v2 feedback controller remains in its original module for historical inspection; use the linked v2 commit from the README for its exact registration and configuration. Models and private runtime assets are not bundled. The new controller must pass development before final evaluation; no accuracy improvement is assumed.
