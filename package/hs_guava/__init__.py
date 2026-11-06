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
        name='guava-native-feedback-v2', version='0.2.0', paper='arXiv:2606.18363',
        fidelity=Fidelity.TF_VARIANT,
        substitutions=('At most two public failure cues in otherwise native conversation',),
        description='My harness: native Guava history with bounded failure feedback; development validation required',
        categories=('agentic_orchestration', 'execution_monitoring'),
        required=frozenset({'act'}),
        controller=ModuleRef('hs_guava.native_feedback:NativeFeedbackController', {'max_cues': 2}),
        null='shadow', budgets=Budgets(policy_calls=30), trainable_parameters=0,
        card={'admission': 'development_only', 'step_unit': 'budgeted semantic turn',
              'automatic_recovery_tools': 0, 'physical_execution_counter': 'benchmark-owned',
              'original_runtime_parity': 'pending', 'accuracy_claim': None},
    )
