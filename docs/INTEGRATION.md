# Integration and arm definitions

The reusable code is the `package/` HarnessStudio plugin. After preparing the pinned upstream Guava and HarnessStudio environments, install it with `python -m pip install -e package`. Entry points are `guava-authors = hs_guava:authors` and `guava-native-feedback-v2 = hs_guava:my_harness`. The candidate maps to `NativeFeedbackController`; the authors' controller and physical shadow-null arm are retained.

`run_campaign.py` is the audited native/custom-task driver. It requires the real Guava source, checkpoints, task assets, local services, and separately installed simulation dependencies described by the preparation scripts. The package's admission metadata still marks original-runtime parity as pending; completing 120 pairs does not resolve that source-parity gap. The earlier latest-image controller is retained only because test fixtures/provenance refer to it; it is not the selected v2 method.

The root tests are dependency checks and public-observation feedback contracts. They make no model request. The actual benchmark was eight tasks × fifteen starts per arm, not 120 distinct tasks. The compact final report preserves authors, candidate and physical null counts separately.
