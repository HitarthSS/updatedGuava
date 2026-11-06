"""Independent counters for decisions, tools, control steps and physics substeps."""
from dataclasses import dataclass, field


@dataclass
class Budget:
    decision_cap: int
    tool_cap: int
    decisions: int = 0
    tools: int = 0
    control_steps: dict = field(default_factory=lambda: {'setup': 0, 'execution': 0, 'stability': 0})
    physics_substeps: dict = field(default_factory=lambda: {'setup': 0, 'execution': 0, 'stability': 0})

    def decision(self):
        if self.decisions >= self.decision_cap:
            raise RuntimeError('Guava decision budget exhausted')
        self.decisions += 1

    def tool(self):
        if self.tools >= self.tool_cap:
            raise RuntimeError('Guava tool budget exhausted')
        self.tools += 1

    @property
    def ended(self):
        return self.decisions >= self.decision_cap or self.tools >= self.tool_cap

    def snapshot(self):
        return {'decisions': self.decisions, 'tool_calls': self.tools,
                'budgeted_turns': self.decisions,
                'turn_note': 'Each automatic recovery tool also consumes one turn; actual VLM calls are in policy audit logs.',
                'decision_cap': self.decision_cap, 'tool_cap': self.tool_cap,
                'control_steps': dict(self.control_steps), 'physics_substeps': dict(self.physics_substeps),
                'physical_horizon': None,
                'physical_horizon_note': 'Original simulator horizon retained. No paper-wide numeric physics cap was established during source intake.'}


class MotionMeter:
    def __init__(self, environment, budget):
        self.environment = environment
        self.budget = budget
        self.phase = 'setup'
        self.original = environment.step

    def attach(self):
        def measured(action):
            before = float(self.environment.rob.sim.data.time)
            try:
                return self.original(action)
            finally:
                after = float(self.environment.rob.sim.data.time)
                timestep = float(self.environment.rob.sim.model.opt.timestep)
                self.budget.control_steps[self.phase] += 1
                self.budget.physics_substeps[self.phase] += round((after - before) / timestep)
        self.environment.step = measured

    def detach(self):
        self.environment.step = self.original
