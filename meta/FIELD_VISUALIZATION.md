# Agent Field Visualization on 2D Spatial Manifolds

Visualize emergent agent fields (beliefs, priors, observables) on 2D spatial grids across hierarchical scales.

## Quick Start

### 1. Enable in Configuration

```python
from simulation_config import SimulationConfig

cfg = SimulationConfig(
    # ... other parameters ...

    # 2D spatial manifold
    spatial_shape=(9, 9),

    # Enable field visualization
    visualize_agent_fields=True,
    viz_track_interval=10,           # Record every 10 steps
    viz_scales=(0, 1),               # Image scale 0 (base) and scale 1 (meta-agents)
    viz_fields=("mu_q", "Sigma_q", "phi"),  # Which fields to visualize
    viz_latent_components=(0, 1, 2),  # Which K components (None = all)
)
```

### 2. Run Simulation

```bash
python simulation_runner.py --preset demo_2d_field_viz
```

### 3. View Results

Output structure:
```
_results/_demo_2d_field_viz/
  agent_fields/
    scale_0/                              # Base agents
      step_0000.png                       # Grid of all fields at step 0
      step_0010.png                       # Grid of all fields at step 10
      ...
    scale_1/                              # Meta-agents (if formed)
      step_0050.png
      ...
    evolution_mu_q_comp0_agent0_scale0.png    # Time evolution of mu_q[0]
    evolution_phi_comp1_agent0_scale0.png     # Time evolution of phi[1]
    comparison_mu_q_comp0_step0100_scale0.png # Compare all agents at final step
```

## Configuration Parameters

### `visualize_agent_fields` (bool, default: False)
Enable field visualization on 2D spatial manifolds.

**Only works with 2D grids** - point manifolds and 1D manifolds are skipped.

### `viz_track_interval` (int, default: 10)
Record field snapshots every N steps.

**Recommendations:**
- Fast runs: 20-50
- Standard: 10
- High temporal resolution: 5
- Every step: 1 (generates many images!)

### `viz_scales` (tuple, default: (0,))
Which hierarchical scales to image.

**Examples:**
- `(0,)` - Only base agents
- `(0, 1)` - Base agents and first level of meta-agents
- `(0, 1, 2)` - All scales up to 2

**Note:** Meta-agents (scale > 0) only visualized if they form during simulation.

### `viz_fields` (tuple, default: ("mu_q", "phi"))
Which fields to track.

**Available fields:**
- `"mu_q"` - Belief means (epistemic state)
- `"Sigma_q"` - Belief covariances (epistemic uncertainty)
- `"mu_p"` - Prior means (ontological model)
- `"Sigma_p"` - Prior covariances (ontological uncertainty)
- `"phi"` - Observable fields (gauge field)

**Typical combinations:**
- Minimal: `("mu_q", "phi")` - beliefs and observables
- Epistemic: `("mu_q", "Sigma_q")` - beliefs with uncertainty
- Full: `("mu_q", "Sigma_q", "mu_p", "Sigma_p", "phi")` - everything

### `viz_latent_components` (tuple or None, default: None)
Which latent space components to visualize.

**Examples:**
- `None` - All components (default)
- `(0,)` - Only first component
- `(0, 1, 2)` - First three components (good for K=3)

**For high-dimensional K:** Recommend selecting subset like `(0, 1, 2)` to avoid huge plots.

## Output Visualizations

### Snapshot Grids (`scale_X/step_NNNN.png`)

Multi-panel grid showing all tracked fields for all agents at a single timestep.

**Layout:**
- Rows = fields (mu_q, Sigma_q, etc.)
- Columns = (agents × components)

**Example:** With 3 agents, K=3, tracking mu_q and phi:
```
Row 1: [Agent0 mu_q[0]] [Agent0 mu_q[1]] [Agent0 mu_q[2]] [Agent1 mu_q[0]] ...
Row 2: [Agent0 phi[0]]  [Agent0 phi[1]]  [Agent0 phi[2]]  [Agent1 phi[0]]  ...
```

### Evolution Plots (`evolution_FIELD_compN_agentM_scaleS.png`)

Shows time evolution of a single field component for one agent.

**Use cases:**
- Track how beliefs converge over time
- See emergence of structure in observable fields
- Monitor stability/instability

**Layout:** Grid of 4×N heatmaps, one per recorded timestep.

### Comparison Plots (`comparison_FIELD_compN_stepT_scaleS.png`)

Side-by-side comparison of all agents for a specific field at a specific time.

**Use cases:**
- See differences between agents (before consensus)
- Visualize consensus (all agents look similar)
- Identify which agents are diverging

## Use Cases

### 1. Visualize Emergent Structure in Beliefs

**Goal:** See how spatial structure emerges in agent beliefs during learning.

```python
cfg = SimulationConfig(
    spatial_shape=(16, 16),
    visualize_agent_fields=True,
    viz_scales=(0,),
    viz_fields=("mu_q",),
    viz_track_interval=5,
    n_steps=200,
)
```

**Analysis:**
- Open `agent_fields/scale_0/step_0000.png` (initial random beliefs)
- Open `agent_fields/scale_0/step_0200.png` (final structured beliefs)
- Compare to see emergent patterns

### 2. Track Consensus Formation

**Goal:** Watch agents converge to identical beliefs before forming meta-agent.

```python
cfg = SimulationConfig(
    spatial_shape=(9, 9),
    visualize_agent_fields=True,
    viz_scales=(0,),
    viz_fields=("mu_q",),
    enable_emergence=True,
    consensus_threshold=0.05,
)
```

**Analysis:**
- View `evolution_mu_q_comp0_agent0_scale0.png`
- See beliefs become more similar over time
- Watch for convergence event

### 3. Image Meta-Agents

**Goal:** Visualize emergent meta-agents on coarser grids.

```python
cfg = SimulationConfig(
    spatial_shape=(9, 9),
    visualize_agent_fields=True,
    viz_scales=(0, 1, 2),  # All scales
    viz_fields=("mu_q", "phi"),
    enable_emergence=True,
)
```

**Analysis:**
- `scale_0/` - Base agents (9×9 grids)
- `scale_1/` - First-level meta-agents (also 9×9, representing consensus)
- Compare to see hierarchical structure

### 4. Study Observable Field Dynamics

**Goal:** Understand how gauge field φ evolves and couples to beliefs.

```python
cfg = SimulationConfig(
    spatial_shape=(12, 12),
    visualize_agent_fields=True,
    viz_fields=("phi",),
    lambda_phi=1.0,  # Enable gauge coupling
)
```

**Analysis:**
- View `evolution_phi_comp0_agent0_scale0.png`
- See how observable field structure emerges
- Compare to pullback geometry (if tracked)

### 5. Compare Epistemic vs Ontological

**Goal:** See difference between beliefs (epistemic) and priors (ontological).

```python
cfg = SimulationConfig(
    spatial_shape=(9, 9),
    visualize_agent_fields=True,
    viz_fields=("mu_q", "mu_p"),
    lambda_prior_align=2.0,
)
```

**Analysis:**
- Compare `mu_q` (beliefs) and `mu_p` (priors) in snapshot grids
- See how prior alignment pulls beliefs toward ontological model

## Performance Considerations

### Computational Cost

**Per snapshot:**
- Field extraction: O(n_agents × spatial_size × K)
- Image generation: ~1-2 seconds per scale

**Typical overhead:**
- Standard settings: ~5% slowdown
- Every step: ~10-20% slowdown

### Storage Requirements

**Per snapshot image:**
- ~100-500 KB depending on resolution and number of panels

**Example:**
- 100 steps, interval=10 → 10 images per scale
- 2 scales tracked → 20 images total
- ~5-10 MB total

### Optimization Tips

1. **Increase `viz_track_interval`** for long simulations
   - Every 10 steps: standard
   - Every 20-50 steps: memory-constrained
   - Every 5 steps: high temporal resolution

2. **Track fewer scales** - scale 0 (base agents) is most important
   - `viz_scales=(0,)` - fastest
   - `viz_scales=(0, 1)` - standard
   - `viz_scales=(0, 1, 2, ...)` - comprehensive but slow

3. **Select specific components** for high-dimensional K
   - `viz_latent_components=(0, 1, 2)` instead of None
   - Reduces plot size and generation time

4. **Track fewer fields** - mu_q and phi are most informative
   - Minimal: `("mu_q",)`
   - Standard: `("mu_q", "phi")`
   - Full: all fields (slower)

## Advanced Usage

### Custom Analysis

```python
from meta.agent_field_visualizer import AgentFieldVisualizer

# Create visualizer manually
visualizer = AgentFieldVisualizer(
    output_dir="my_custom_viz",
    scales_to_track=[0, 1],
    fields_to_track=["mu_q", "phi"],
    latent_components=[0, 1],  # First two components only
    track_interval=5
)

# Record during training
for step in range(n_steps):
    # ... your training code ...

    if visualizer.should_record(step):
        visualizer.record(step, system)

# Generate all visualizations
visualizer.generate_summary_report()

# Or generate specific plots
visualizer.plot_field_evolution(
    field_name="mu_q",
    component_idx=0,
    agent_idx=0,
    scale=0,
    save_path="my_evolution.png"
)
```

### Accessing Raw Snapshots

```python
# Load a snapshot
snapshot = visualizer.history[scale][step_idx]

# Access fields
mu_q_agent_0 = snapshot.mu_q[0]  # Shape: (H, W, K)
phi_agent_0 = snapshot.phi[0]    # Shape: (H, W, K)

# Custom plotting
import matplotlib.pyplot as plt
plt.imshow(mu_q_agent_0[:, :, 0], cmap='RdBu_r')
plt.colorbar()
plt.title(f"Agent 0 belief component 0 at step {snapshot.step}")
plt.savefig("custom_plot.png")
```

## Troubleshooting

### "No visualizations generated"

**Cause:** Simulation only has point manifolds or 1D manifolds.

**Solution:** Use 2D spatial grid: `spatial_shape=(9, 9)` or `(16, 16)`

### "scale_1/ directory empty"

**Cause:** Meta-agents didn't form during simulation.

**Solutions:**
- Run longer: increase `n_steps`
- Lower threshold: decrease `consensus_threshold` (e.g., 0.05 → 0.02)
- Check consensus diagnostics to see if KL divergences are decreasing

### Out of memory with long simulations

**Cause:** Too many snapshots stored in memory.

**Solutions:**
- Increase `viz_track_interval` (e.g., 10 → 20)
- Track fewer scales: `viz_scales=(0,)` only
- Periodically save and clear history (custom loop)

### Plots are too large / cluttered

**Cause:** Too many agents or components shown.

**Solutions:**
- Select fewer components: `viz_latent_components=(0, 1)`
- Track fewer fields: `viz_fields=("mu_q",)` only
- Use `show_all_agents=False` in custom plotting

## Integration with Other Tools

### With Pullback Geometry Tracking

Combine field visualization with geometry tracking to see both:
- **Field visualization:** What agents believe (epistemic state)
- **Geometry tracking:** Emergent spacetime structure (information geometry)

```python
cfg = SimulationConfig(
    visualize_agent_fields=True,
    track_pullback_geometry=True,
    # ... other settings ...
)
```

**Analysis:** Compare emergent structure in beliefs (field viz) with emergent spacetime dimensions (geometry tracking).

### With Meta-Agent Visualizations

Field visualization complements the hierarchical visualization tools:

```python
cfg = SimulationConfig(
    visualize_agent_fields=True,
    generate_meta_visualizations=True,
)
```

**Output:**
- `agent_fields/` - Spatial heatmaps of field values
- `meta_*.png` - Hierarchical structure diagrams

## See Also

- `meta/agent_field_visualizer.py` - Core implementation
- `presets/demo_2d_field_viz.py` - Example configuration
- `geometry/USAGE_TRACKING.md` - Pullback geometry tracking
- `meta/PARTICIPATORY_README.md` - Meta-agent diagnostics

---

**Authors:** Claude & Chris
**Date:** November 2025
**Version:** 1.0
