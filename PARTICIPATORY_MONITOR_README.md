# Participatory "It from Bit" Universe Monitor

## Overview

The **ParticipatoryMonitor** validates that the participatory dynamics in the gauge-theoretic active inference system are wired up correctly. It monitors the feedback loop where:

1. **Scale-0 agents** condense into **meta-agents** (bottom-up)
2. **Meta-agents** influence their constituents' **priors** (top-down)
3. The system operates in a **non-equilibrium dynamical regime**
4. **Emergence levels** are capped to prevent performance degradation

## Features

### 1. Condensation Monitoring
Tracks whether scale-0 agents are properly condensing into meta-agents through consensus detection:
- Counts meta-agents formed over time
- Tracks scale-0 → scale-1 information flow
- Identifies consensus opportunities vs. actual condensation

### 2. Prior Evolution Validation
Validates that priors evolve due to meta-agent activity:
- Measures KL divergence between successive prior distributions
- Tracks parent-child links (top-down influence)
- Detects whether meta-agent beliefs propagate downward

### 3. Non-Equilibrium Detection
Confirms the system is in a non-equilibrium dynamical regime:
- Tracks energy flux (rate of energy change)
- Monitors information flux (rate of information accumulation)
- Computes gradient variance across scales
- Produces equilibrium score (0=equilibrium, 1=far-from-equilibrium)

### 4. Level Cap Enforcement
Prevents runaway emergence by enforcing a hard cap on hierarchy levels:
- Configurable `max_emergence_levels` in `MultiScaleSystem`
- Prevents formation of meta-agents beyond the cap
- Tracks max scale reached and warns when cap is hit

## Usage

### Basic Setup

```python
from meta.emergence import MultiScaleSystem
from meta.consensus import ConsensusDetector
from meta.participatory_monitor import ParticipatoryMonitor

# Create multi-scale system with level cap
system = MultiScaleSystem(manifold, max_emergence_levels=4)

# Create consensus detector
consensus_detector = ConsensusDetector(
    belief_threshold=0.5,
    model_threshold=0.5,
    min_cluster_size=2
)

# Create monitor
monitor = ParticipatoryMonitor(
    system=system,
    consensus_detector=consensus_detector,
    check_interval=10,  # Take snapshots every 10 steps
    prior_change_threshold=1e-4,  # KL threshold for prior changes
    non_eq_threshold=1e-3  # Gradient threshold for non-equilibrium
)
```

### During Evolution

```python
for step in range(num_steps):
    # Your evolution logic here...

    # Take periodic snapshots
    snapshot = monitor.take_snapshot(step)

    if snapshot is not None:
        print(f"Step {step}: {snapshot.num_agents_per_scale[0]} base agents, "
              f"{snapshot.num_meta_agents_formed} meta-agents")
```

### Validation

```python
# Print comprehensive summary
monitor.print_summary(max_levels=4)

# Get detailed validation results
validation = monitor.validate_participatory_dynamics(max_levels=4)

print(f"Overall Status: {validation['overall_status']}")
print(f"Condensation: {validation['condensation']['status']}")
print(f"Prior Evolution: {validation['prior_evolution']['status']}")
print(f"Non-Equilibrium: {validation['non_equilibrium']['status']}")
print(f"Level Cap: {validation['level_cap']['status']}")
```

### Individual Analyses

```python
# Analyze condensation
condensation = monitor.analyze_condensation()
print(f"Condensation occurring: {condensation['condensation_occurring']}")
print(f"Total meta-agents: {condensation['total_meta_agents']}")

# Analyze prior evolution
prior_evolution = monitor.analyze_prior_evolution()
print(f"Top-down influence: {prior_evolution['top_down_influence']}")
print(f"Parent-child links: {prior_evolution['parent_child_links']}")

# Analyze non-equilibrium dynamics
non_eq = monitor.analyze_non_equilibrium()
print(f"Is non-equilibrium: {non_eq['is_non_equilibrium']}")
print(f"Equilibrium score: {non_eq['equilibrium_score']:.4f}")

# Check level cap
level_cap = monitor.check_level_cap(max_levels=4)
print(f"Max scale reached: {level_cap['max_scale_reached']}")
print(f"Level cap hit: {level_cap['level_cap_hit']}")
```

## Configuration

### MultiScaleSystem Parameters

- **`max_emergence_levels`**: Maximum allowed scale level
  - `None`: Unlimited emergence (not recommended - can slow down massively)
  - `3`: Allows scales 0-3 (base, groups, communities, societies)
  - `4`: Allows scales 0-4 (adds meta-societies)
  - Recommended: `3` or `4` for good performance

### ParticipatoryMonitor Parameters

- **`check_interval`**: Steps between snapshots (default: 10)
- **`prior_change_threshold`**: KL divergence threshold for prior evolution detection (default: 1e-4)
- **`non_eq_threshold`**: Gradient norm threshold for non-equilibrium detection (default: 1e-3)

## Output Example

```
======================================================================
PARTICIPATORY 'IT FROM BIT' UNIVERSE MONITOR
======================================================================

Overall Status: HEALTHY
Samples Collected: 10

----------------------------------------------------------------------
1. AGENT CONDENSATION (Scale-0 → Meta-Agents)
----------------------------------------------------------------------
   Status: ✓ OK
   Total Meta-Agents: 5
   Scale-0 → Scale-1 Flow: ✓ Yes
   Consensus Opportunities: 12

----------------------------------------------------------------------
2. PRIOR EVOLUTION (Top-Down Meta-Agent Influence)
----------------------------------------------------------------------
   Status: ✓ OK
   Parent-Child Links: 15
   Average Prior Changes by Scale:
      Scale 0: 0.000234 ✓
      Scale 1: 0.000156 ✓

----------------------------------------------------------------------
3. NON-EQUILIBRIUM DYNAMICS
----------------------------------------------------------------------
   Status: ✓ NON-EQUILIBRIUM
   Equilibrium Score: 2.4567 (0=eq, 1=far-from-eq)
   Avg Energy Flux: 0.003421
   Avg Information Flux: 0.002156
   Avg Gradient Variance: 0.001892

----------------------------------------------------------------------
4. EMERGENCE LEVEL CAP
----------------------------------------------------------------------
   Status: ✓ OK
   Max Levels Allowed: 4
   Max Scale Reached: 2
   Levels Remaining: 2

======================================================================
```

## Testing

Run the test script to verify functionality:

```bash
python test_participatory_monitor.py
```

This creates a small system, runs some evolution, and validates all monitoring features.

## Integration with Existing Code

The participatory monitor is already integrated into:

- **`simulation_suite.py`**: MultiScaleSystem now uses `max_emergence_levels=4`
- **`meta/emergence.py`**: Level cap enforcement in `form_meta_agents_at_scale()`

To use in your own scripts:

```python
# When creating MultiScaleSystem
system = MultiScaleSystem(manifold, max_emergence_levels=4)  # Add this parameter!

# Then create and use monitor as shown above
```

## Theory

The monitor validates Wheeler's "it from bit" participatory universe concept in the context of gauge-theoretic active inference:

- **"It"** = Physical agents (scale-0)
- **"Bit"** = Information/beliefs that condense into meta-agents
- **Participatory** = Meta-agents feed back to influence constituent priors
- **Non-equilibrium** = System continuously evolves (not settling into equilibrium)

The feedback loop creates a self-referential structure where:
1. Agents → form meta-agents (emergence)
2. Meta-agents → influence agent priors (top-down)
3. Agents evolve → update meta-agents (renormalization)
4. Cycle repeats at multiple scales

This is the computational realization of Wheeler's participatory universe!

## Files

- **`meta/participatory_monitor.py`**: Main monitoring utility
- **`test_participatory_monitor.py`**: Test/example script
- **`meta/emergence.py`**: MultiScaleSystem with level cap
- **`simulation_suite.py`**: Integration example
- **`PARTICIPATORY_MONITOR_README.md`**: This file

## References

- Wheeler, J. A. (1990). "Information, physics, quantum: The search for links"
- Gauge-theoretic active inference framework
- Multi-scale renormalization group theory
