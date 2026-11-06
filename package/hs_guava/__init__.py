"""Generated Guava package. Full-runtime parity must pass before admission."""


def authors():
    from harnessstudio.core.method import MethodConfig, ModuleRef, Fidelity, Budgets
    return MethodConfig(
        name='guava-authors', version='0.1.0', paper='arXiv:2606.18363',
        fidelity=Fidelity.COMPONENT,
        description='Guava reference-controller bridge; runtime parity pending',
        categories=('agentic_orchestration', 'skill_composition'),
        required=frozenset({'act'}),
        controller=ModuleRef('hs_guava.controller:AuthorController'),
        null='shadow', budgets=Budgets(policy_calls=30), trainable_parameters=0,
        card={'admission': 'runtime_parity_pending',
              'step_unit': 'model decision',
              'physical_execution_counter': 'benchmark-owned'},
    )


def my_harness():
    from harnessstudio.core.method import MethodConfig, ModuleRef, Fidelity, Budgets
    return MethodConfig(
        name='guava-terminal-evidence-v3', version='0.3.0', paper='arXiv:2606.18363',
        fidelity=Fidelity.TF_VARIANT,
        substitutions=('One native perception check and model review before accepting completion; all turns charged',),
        description='My harness: bounded public terminal evidence check; development validation required',
        categories=('observation_grounding', 'agentic_orchestration', 'execution_monitoring'),
        required=frozenset({'act'}),
        controller=ModuleRef('hs_guava.terminal_evidence:TerminalEvidenceController', {'max_checks': 1}),
        null='shadow', budgets=Budgets(policy_calls=30), trainable_parameters=0,
        card={'admission': 'development_only', 'step_unit': 'budgeted semantic turn',
              'automatic_motion_tools': 0, 'max_inspection_tools': 1, 'physical_execution_counter': 'benchmark-owned',
              'original_runtime_parity': 'pending', 'accuracy_claim': None},
    )
