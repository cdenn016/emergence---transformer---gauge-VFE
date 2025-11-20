# Meta-Agent Visualization and Analysis Tools

Comprehensive visualization and analysis toolkit for the hierarchical meta-agent system. These tools help understand, debug, and explore the emergent dynamics of gauge-theoretic hierarchical learning.

## Overview

The meta-agent system implements a sophisticated hierarchical emergence mechanism where agents at scale ζ can condense into meta-agents at scale ζ+1 through gauge-transported averaging. This toolkit provides:

- **Structure Visualization**: Hierarchy graphs, network diagrams
- **Consensus Analysis**: KL divergence matrices, coherence tracking
- **Multi-Scale Dynamics**: Population flows, scale occupancy heatmaps
- **Energy Landscapes**: Thermodynamic analysis, non-equilibrium indicators
- **Real-Time Monitoring**: Live dashboards for evolution tracking

## Installation

### Required Dependencies

```bash
pip install numpy matplotlib seaborn
```

### Optional Dependencies

For interactive visualizations and advanced features:

```bash
pip install plotly networkx
```

## Quick Start

### Basic Usage

```python
from meta.emergence import MultiScaleSystem, HierarchicalAgent
from meta.hierarchical_evolution import HierarchicalEvolutionEngine
from meta.participatory_diagnostics import ParticipatoryDiagnostics
from meta.visualization import MetaAgentAnalyzer, create_analysis_report
from meta.energy_visualization import EnergyVisualizer

# 1. Create and run system
system = MultiScaleSystem(agents={...})
engine = HierarchicalEvolutionEngine(system)
diagnostics = ParticipatoryDiagnostics(system)

analyzer = MetaAgentAnalyzer(system)

# 2. Run evolution with periodic snapshots
for i in range(100):
    engine.step()
    diagnostics.capture_snapshot()

    if i % 10 == 0:
        analyzer.capture_snapshot()

# 3. Generate all visualizations
create_analysis_report(analyzer, output_dir='./analysis')

# 4. Energy analysis
energy_viz = EnergyVisualizer(diagnostics)
energy_viz.create_energy_report(output_dir='./energy')
```

### Running the Demo

A complete demonstration is provided:

```bash
python examples/meta_agent_analysis_demo.py
```

This will:
1. Create a multi-scale system with 8 agents
2. Run 50 steps of hierarchical evolution
3. Generate all visualizations
4. Export data and detailed analysis

Output saved to `./demo_output/`

## Modules

### 1. `meta/visualization.py`

Core visualization module for hierarchical structure and dynamics.

#### Classes

##### `MetaAgentAnalyzer`

Extracts and analyzes data from `MultiScaleSystem`.

```python
analyzer = MetaAgentAnalyzer(system)

# Capture state periodically during evolution
analyzer.capture_snapshot()  # Call after each evolution step or at intervals

# Get consensus matrix (pairwise KL divergences)
kl_matrix = analyzer.get_consensus_matrix(scale=0, metric='belief')

# Get hierarchy edges for graph visualization
edges = analyzer.get_hierarchy_edges()

# Export all data to JSON
analyzer.export_to_json('snapshots.json')
```

##### `HierarchyVisualizer`

Visualize hierarchical meta-agent structure.

```python
viz = HierarchyVisualizer(analyzer)

# Static matplotlib plot
fig = viz.plot_hierarchy_tree(layout='hierarchical')
fig.savefig('hierarchy.png')

# Interactive Plotly visualization (if available)
interactive_fig = viz.plot_interactive_hierarchy()
interactive_fig.write_html('hierarchy.html')
```

**Visualization Features:**
- Nodes colored by scale (ζ)
- Node size proportional to importance
- Active vs inactive agents (solid vs hollow)
- Edge thickness shows coherence strength
- Hierarchical or spring layout

##### `ConsensusVisualizer`

Visualize consensus formation and coherence dynamics.

```python
viz = ConsensusVisualizer(analyzer)

# Consensus matrix heatmap (pairwise KL divergences)
fig = viz.plot_consensus_matrix(scale=0, metric='belief')

# Evolution over time
fig = viz.plot_consensus_evolution(scale=0)

# Coherence trajectories for all meta-agents
fig = viz.plot_coherence_trajectories()
```

**What to Look For:**
- **Dark blocks** in consensus matrix → agents in consensus
- **High coherence** (>0.95) → ready for condensation
- **Coherence trends** → stability of meta-agents

##### `DynamicsVisualizer`

Visualize multi-scale population flows and occupancy.

```python
viz = DynamicsVisualizer(analyzer)

# Heatmap: time × scale
fig = viz.plot_scale_occupancy()

# Timeline of condensation events
fig = viz.plot_condensation_timeline()

# Stacked area chart of population flows
fig = viz.plot_population_flows()
```

**Key Insights:**
- **Stars on heatmap** mark condensation events
- **Population shifts** show emergence dynamics
- **Cluster sizes** indicate condensation scale

#### Convenience Functions

```python
from meta.visualization import create_analysis_report

# Generate complete report with all visualizations
files = create_analysis_report(
    analyzer,
    output_dir='./meta_analysis'
)

# Returns dict of generated files
# files = {
#     'hierarchy_tree': './meta_analysis/hierarchy_tree.png',
#     'consensus_matrix': './meta_analysis/consensus_matrix.png',
#     ...
# }
```

---

### 2. `meta/energy_visualization.py`

Energy landscape and thermodynamic analysis.

#### `EnergyVisualizer`

Requires a `ParticipatoryDiagnostics` instance that has been tracking energy.

```python
diagnostics = ParticipatoryDiagnostics(system, track_energy=True)

# ... run evolution with diagnostics.capture_snapshot() ...

viz = EnergyVisualizer(diagnostics)
```

##### Available Plots

**Multi-Scale Energy Landscape**

```python
fig = viz.plot_energy_landscape()
```

Shows energy decomposition by scale:
- **E_self** (red): Intrinsic energy
- **E_belief_align** (cyan): Lateral coherence
- **E_prior_align** (blue): Vertical alignment
- **Total** (black line)

**Energy Flow (Flux)**

```python
fig = viz.plot_energy_flow()
```

Rate of energy change (dE/dt) across scales. Positive = energy increasing, negative = decreasing.

**Prior Evolution**

```python
fig = viz.plot_prior_evolution()
```

Shows KL divergence of priors over time. Tracks top-down information flow from meta-agents to constituents.

**Non-Equilibrium Indicators**

```python
fig = viz.plot_non_equilibrium_indicators()
```

Dashboard showing:
- Energy variance across scales
- Gradient variance
- Information flux
- Non-equilibrium score (0=equilibrium, 1=far-from-eq)

**Energy Per Agent**

```python
fig = viz.plot_energy_per_agent(scale=0, snapshot_idx=-1)
```

Bar chart showing energy decomposition for individual agents.

**Interactive 3D Energy Landscape**

```python
fig = viz.plot_interactive_energy_3d(scale=0)  # Requires Plotly
fig.write_html('energy_3d.html')
```

3D surface plot: time × agent × energy.

##### Complete Energy Report

```python
files = viz.create_energy_report(output_dir='./energy_analysis')
```

Generates all energy visualizations and saves to directory.

---

### 3. `meta/live_monitor.py`

Real-time monitoring dashboard for evolution.

#### `LiveMonitor`

Animated real-time visualization (uses matplotlib FuncAnimation).

```python
monitor = LiveMonitor(system, diagnostics, update_interval=100)

# Define update callback
def step_callback():
    engine.step()
    diagnostics.capture_snapshot()
    return True  # Continue (False to stop)

# Start monitoring (blocking)
monitor.start(update_callback=step_callback)
```

**Dashboard Panels:**
1. **Scale Occupancy**: Line plot of active agents per scale
2. **Total Energy**: Energy trajectory
3. **System Status**: Text summary (time, counts, etc.)
4. **Condensation Events**: Scatter plot of emergence
5. **Hierarchy Stats**: Coherence and cluster statistics

#### `StepwiseMonitor`

Manual control for integration into existing loops.

```python
monitor = StepwiseMonitor(system, diagnostics, update_every=10)
monitor.show()  # Display window

for i in range(n_steps):
    engine.step()
    diagnostics.capture_snapshot()
    monitor.step()  # Updates every update_every steps

monitor.save('final_state.png')
plt.show()  # Keep window open
```

#### Convenience Function

```python
from meta.live_monitor import monitor_evolution

monitor_evolution(
    system,
    evolution_step_fn=engine.step,
    n_steps=100,
    diagnostics=diagnostics,
    update_every=5
)
```

---

## Example Workflows

### Workflow 1: Post-hoc Analysis

Run evolution first, then analyze:

```python
# 1. Setup
system = create_system()
engine = HierarchicalEvolutionEngine(system)
diagnostics = ParticipatoryDiagnostics(system, track_energy=True)
analyzer = MetaAgentAnalyzer(system)

# 2. Run evolution
for i in range(100):
    engine.step()
    diagnostics.capture_snapshot()
    analyzer.capture_snapshot()

# 3. Analyze
create_analysis_report(analyzer, './analysis')

energy_viz = EnergyVisualizer(diagnostics)
energy_viz.create_energy_report('./energy')
```

### Workflow 2: Real-Time Monitoring

Monitor during evolution:

```python
from meta.live_monitor import monitor_evolution

system = create_system()
engine = HierarchicalEvolutionEngine(system)
diagnostics = ParticipatoryDiagnostics(system, track_energy=True)

monitor_evolution(
    system,
    engine.step,
    n_steps=100,
    diagnostics=diagnostics,
    update_every=5
)
```

### Workflow 3: Interactive Exploration

```python
# Run evolution
# ... (same as workflow 1)

# Generate interactive visualizations
hierarchy_viz = HierarchyVisualizer(analyzer)
interactive_fig = hierarchy_viz.plot_interactive_hierarchy()
interactive_fig.write_html('hierarchy.html')

energy_viz = EnergyVisualizer(diagnostics)
energy_3d = energy_viz.plot_interactive_energy_3d(scale=0)
energy_3d.write_html('energy_3d.html')

# Open in browser for exploration
```

### Workflow 4: Comparative Analysis

Compare different hyperparameters:

```python
results = {}

for lr in [0.05, 0.1, 0.2]:
    system = create_system()
    engine = HierarchicalEvolutionEngine(system, lr_mu_q=lr)
    analyzer = MetaAgentAnalyzer(system)

    for i in range(100):
        engine.step()
        if i % 10 == 0:
            analyzer.capture_snapshot()

    results[lr] = analyzer

# Compare final states
for lr, analyzer in results.items():
    print(f"\nLearning rate {lr}:")
    print(f"  Max scale: {max(analyzer.system.agents.keys())}")
    print(f"  Condensations: {len(analyzer.system.condensation_events)}")

    # Generate report
    create_analysis_report(analyzer, f'./analysis_lr_{lr}')
```

---

## Understanding the Visualizations

### Hierarchy Tree

**What it shows:**
- Each node = agent (or meta-agent)
- Edges = constituent relationships
- Colors = scale (ζ)
- Solid circles = active, hollow = inactive

**How to read:**
- **Bottom (ζ=0)**: Base agents observing data
- **Middle (ζ=1)**: First-order meta-agents
- **Top (ζ=max)**: Highest-level "overseer" agents
- **Edge thickness**: Coherence strength (thicker = higher)

**What to look for:**
- Balanced tree = uniform emergence
- Star patterns = dominant agents forming hubs
- Isolated nodes = agents not participating

### Consensus Matrix

**What it shows:**
Heatmap of pairwise KL divergences KL(q_i || q_j) between agent beliefs.

**How to read:**
- **Dark green**: Low KL → agents in consensus
- **Red**: High KL → divergent beliefs
- **Diagonal**: Always 0 (agent vs itself)

**What to look for:**
- **Block patterns**: Clusters of agents in consensus
- **Evolution**: Blocks forming over time → emergence
- **Threshold**: KL < 0.5 typically indicates consensus

### Scale Occupancy Heatmap

**What it shows:**
Number of active agents at each scale over time.

- **X-axis**: Time
- **Y-axis**: Scale (ζ)
- **Color**: Number of active agents
- **Stars**: Condensation events

**What to look for:**
- **Bottom-up flow**: Agents condensing from lower to higher scales
- **Discrete jumps**: Sudden emergence events
- **Stable plateaus**: Meta-agents persisting
- **Cap enforcement**: Occupancy respecting `max_emergence_levels`

### Energy Landscape

**What it shows:**
Decomposition of system energy into components:

- **E_self**: Internal prediction error (red)
- **E_belief_align**: Lateral coherence cost (cyan)
- **E_prior_align**: Vertical prior alignment (blue)

**How to read:**
- Stacked areas show contribution of each component
- Total energy = sum of all components
- Separate subplots for each scale

**What to look for:**
- **Decreasing E_self**: Agents becoming better predictors
- **Decreasing E_belief_align**: Convergence toward consensus
- **E_prior_align behavior**: Top-down influence strength
- **Total energy trends**: Optimization progress

### Coherence Trajectories

**What it shows:**
Belief and model coherence over time for each meta-agent.

**How to read:**
- Each line = one meta-agent
- Top panel = belief coherence
- Bottom panel = model coherence
- Dashed red line = high coherence threshold (0.95)

**What to look for:**
- **Rapid rise**: Meta-agent formation
- **Sustained high coherence**: Stable meta-agent
- **Decline**: Constituents diverging (instability)
- **Both high**: "Epistemic death" → potential for further condensation

### Prior Evolution

**What it shows:**
KL divergence between consecutive priors: KL(p_new || p_old).

Measures strength of top-down information flow.

**How to read:**
- **Y-axis** (log scale): Magnitude of prior change
- Each line = one agent
- Bold black line = average

**What to look for:**
- **Large spikes**: Strong top-down influence
- **Decreasing trend**: System stabilizing
- **Persistent changes**: Active hierarchical communication
- **Scale differences**: Higher scales often have larger changes

---

## Data Export

### JSON Export

```python
analyzer.export_to_json('snapshots.json')
```

**Format:**
```json
{
  "snapshots": [
    {
      "time": 0,
      "agents_by_scale": {
        "0": [
          {
            "scale": 0,
            "local_index": 0,
            "is_active": true,
            "is_meta": false,
            "mu_q": [...],
            "mu_p": [...],
            "info_accumulator": 0.0
          },
          ...
        ]
      },
      "meta_agents": [
        {
          "scale": 1,
          "local_index": 0,
          "constituents": [...],
          "emergence_time": 10,
          "belief_coherence": 0.95,
          "model_coherence": 0.92,
          "leader_index": 0,
          "leader_score": 0.87,
          "leadership_distribution": [...]
        },
        ...
      ],
      "condensation_events": [...],
      "metrics": {
        "n_agents_by_scale": {...},
        "total_agents": 10,
        "max_scale": 1
      }
    },
    ...
  ],
  "system_config": {
    "max_emergence_levels": 3,
    "max_meta_membership": 100,
    "max_total_agents": 1000
  }
}
```

This JSON can be loaded for:
- Custom analysis scripts
- Web visualizations
- Comparative studies
- Long-term storage

---

## Troubleshooting

### Common Issues

**1. "No snapshots captured"**

Make sure to call `analyzer.capture_snapshot()` during evolution:

```python
for i in range(n_steps):
    engine.step()
    analyzer.capture_snapshot()  # Add this!
```

**2. "No energy data available"**

Enable energy tracking in diagnostics:

```python
diagnostics = ParticipatoryDiagnostics(
    system,
    track_energy=True  # Important!
)
```

**3. "NetworkX not available"**

Install optional dependency:

```bash
pip install networkx
```

Or use alternative visualizations that don't require NetworkX.

**4. Live monitor not updating**

Ensure you're calling the update callback:

```python
def step_fn():
    engine.step()
    diagnostics.capture_snapshot()
    return True  # Must return True to continue!

monitor.start(update_callback=step_fn)
```

**5. Figures not displaying**

For non-interactive environments (e.g., remote servers):

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

Then save figures instead of showing:

```python
fig.savefig('output.png')
```

---

## Performance Tips

### Memory Management

For long runs with many snapshots:

```python
# Capture snapshots less frequently
if step % 50 == 0:  # Every 50 steps instead of every step
    analyzer.capture_snapshot()

# Limit snapshot history in live monitor
monitor = LiveMonitor(system, history_length=50)  # Keep only last 50
```

### Visualization Performance

```python
# For large systems, visualize only specific scales
fig = viz.plot_consensus_matrix(scale=0)  # Only scale 0

# Use static plots instead of interactive
fig = hierarchy_viz.plot_hierarchy_tree()  # Faster than interactive

# Batch visualization generation
create_analysis_report(analyzer)  # All at once, efficient
```

### Diagnostic Tracking

```python
# Disable expensive tracking if not needed
diagnostics = ParticipatoryDiagnostics(
    system,
    track_energy=True,
    track_priors=True,
    track_gradients=False  # Expensive, disable if not needed
)
```

---

## Advanced Usage

### Custom Visualizations

Extend the visualizers:

```python
from meta.visualization import MetaAgentAnalyzer

class CustomVisualizer:
    def __init__(self, analyzer: MetaAgentAnalyzer):
        self.analyzer = analyzer

    def plot_my_metric(self):
        # Access snapshot data
        for snapshot in self.analyzer.snapshots:
            # ... compute custom metrics
            pass

        # Create custom plot
        fig, ax = plt.subplots()
        # ...
        return fig
```

### Integrating with External Tools

Export data and use with other tools:

```python
# Export to pandas DataFrame
import pandas as pd

data = []
for snapshot in analyzer.snapshots:
    for scale, agents in snapshot.agents_by_scale.items():
        for agent in agents:
            data.append({
                'time': snapshot.time,
                'scale': scale,
                'agent_id': agent['local_index'],
                'is_active': agent['is_active'],
                # ... more fields
            })

df = pd.DataFrame(data)
df.to_csv('evolution_data.csv', index=False)
```

### Comparative Visualization

```python
# Compare multiple runs
analyzers = [analyzer1, analyzer2, analyzer3]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, analyzer in enumerate(analyzers):
    viz = DynamicsVisualizer(analyzer)
    # Use existing axis
    # (Modify visualizer to accept ax parameter)
    # ...
```

---

## Citation

If you use these visualization tools in your research, please cite:

```bibtex
@software{metaagent_viz,
  title={Meta-Agent Visualization Toolkit},
  author={...},
  year={2025},
  url={https://github.com/...}
}
```

---

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- See examples in `examples/meta_agent_analysis_demo.py`
- Check docstrings in source code

---

## Future Extensions

Planned features:
- [ ] Streamlit/Dash web dashboard
- [ ] Gauge frame (SO(3)) visualization
- [ ] Transport operator analysis
- [ ] Information-theoretic metrics (mutual information, etc.)
- [ ] Comparative analysis dashboard
- [ ] Video export (MP4 animations)
- [ ] Jupyter notebook widgets
