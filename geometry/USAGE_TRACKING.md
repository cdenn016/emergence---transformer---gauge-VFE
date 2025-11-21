# Pullback Geometry Tracking: Integration Guide

## Overview

The pullback geometry tracking system is now fully integrated into the simulation infrastructure. It monitors emergent spacetime geometry during multi-agent dynamics by computing induced metrics on the base manifold via pullback from the statistical fiber.

## Quick Start

### 1. Enable in Configuration

```python
from simulation_config import SimulationConfig

cfg = SimulationConfig(
    # ... other parameters ...

    # Enable pullback geometry tracking
    track_pullback_geometry=True,
    geometry_track_interval=10,           # Record every 10 steps
    geometry_enable_consensus=True,       # Compute consensus metrics
    geometry_enable_gauge_averaging=False, # Expensive! Keep False unless needed
    geometry_lambda_obs=0.1,              # Observable sector threshold
    geometry_lambda_dark=0.01,            # Dark sector threshold
)
```

### 2. Run Simulation

```bash
python simulation_runner.py --preset default
```

Or programmatically:

```python
from pathlib import Path
import numpy as np
from simulation_runner import build_manifold, build_supports, build_agents, build_system, run_training

# Setup
output_dir = Path("_results/_my_geometry_experiment")
output_dir.mkdir(parents=True, exist_ok=True)

# Build system
manifold = build_manifold(cfg)
supports = build_supports(manifold, cfg, rng)
agents = build_agents(manifold, supports, cfg, rng)
system = build_system(agents, cfg, rng)

# Train (automatically tracks geometry if cfg.track_pullback_geometry=True)
history = run_training(system, cfg, output_dir)
```

### 3. Analyze Results

Output files:
- `geometry_history.pkl` - Full history object for analysis
- `geometry_evolution.png` - 4-panel evolution plot
- `geometry_consensus.png` - Consensus metrics (if enabled)
- `geometry_analysis/final_eigenvalue_spectrum.png` - Final spectrum

## Configuration Parameters

### `track_pullback_geometry` (bool, default: False)
Enable pullback geometry tracking. When True, the simulation will:
- Compute induced metrics from agent beliefs and priors
- Track eigenvalue decomposition
- Monitor observable/dark/internal sectors
- Record volume elements

### `geometry_track_interval` (int, default: 10)
Record geometry snapshot every N steps. Lower values give finer temporal resolution but increase computational cost.

**Recommendations:**
- Fast exploratory runs: 20-50
- Detailed analysis: 5-10
- Publication figures: 1 (every step)

### `geometry_enable_consensus` (bool, default: False)
Compute gauge-invariant consensus metrics from multiple agents.

**Cost:** O(n_agents²) - expensive for many agents

**When to enable:**
- Few agents (< 10): Safe to enable
- Many agents (> 20): Consider disabling
- Studying collective geometry: Essential

### `geometry_enable_gauge_averaging` (bool, default: False)
Perform Monte Carlo gauge averaging for gauge-invariant individual metrics.

**Cost:** O(n_agents × n_samples × metric_computation) - VERY expensive!

**When to enable:**
- Only when studying gauge-dependent phenomena
- Small systems with few agents
- Publication-quality analysis requiring perfect gauge invariance

**Recommendation:** Keep False unless absolutely necessary. The consensus metric already provides gauge-invariant collective geometry without per-agent averaging.

### `geometry_gauge_samples` (int, default: 50)
Number of Monte Carlo samples for gauge averaging (if enabled).

**Typical values:**
- Quick test: 10-20
- Standard: 50-100
- High precision: 200-500

### `geometry_lambda_obs` (float, default: 0.1)
Threshold for observable sector (relative to max eigenvalue).

Eigenvalues λ > λ_max × geometry_lambda_obs are "observable".

**Interpretation:**
- High certainty directions → perceived spacetime dimensions
- Typically yields ~2-4 observable dimensions from high-D latent space

**Typical values:**
- Liberal (more dimensions): 0.01-0.05
- Standard: 0.1
- Conservative (fewer dimensions): 0.2-0.5

### `geometry_lambda_dark` (float, default: 0.01)
Threshold for dark sector (intermediate eigenvalues).

λ_max × geometry_lambda_dark < λ ≤ λ_max × geometry_lambda_obs

**Interpretation:**
- Intermediate information flux
- "Dark geometry" - affects dynamics but below perception threshold
- Typically ~10-100 dimensions

## Output Files

### `geometry_history.pkl`
Pickled `GeometryHistory` object containing full time series:

```python
import pickle
from pathlib import Path

# Load history
with open("_results/my_experiment/geometry_history.pkl", "rb") as f:
    history = pickle.load(f)

# Access data
steps = history.get_steps()  # Array of timesteps
n_obs = history.get_mean_observable_dims()  # Observable dimensions over time
vol_belief = history.get_mean_volume_belief()  # Belief geometry volumes
vol_prior = history.get_mean_volume_prior()  # Prior geometry volumes

# Access individual snapshots
for snapshot in history.snapshots:
    print(f"Step {snapshot.step}:")
    print(f"  Observable dims: {snapshot.mean_n_observable}")
    print(f"  Dark dims: {snapshot.mean_n_dark}")
    print(f"  Internal dims: {snapshot.mean_n_internal}")

    # Per-agent metrics
    for i, metric in enumerate(snapshot.agent_metrics_belief):
        print(f"  Agent {i} eigenvalues: {metric.eigenvalues[0]}")  # At first spatial point
```

### `geometry_evolution.png`
4-panel visualization:

1. **Observable Sector Evolution** (top-left)
   - Number of observable dimensions vs time
   - Shows emergence of effective spacetime dimensionality

2. **Volume Element Evolution** (top-right)
   - √det(G) for belief and prior geometries
   - Log scale - tracks information-geometric "volume"
   - Epistemic (belief) vs ontological (prior)

3. **Top Eigenvalues** (bottom-left)
   - Top 3 eigenvalues of belief geometry
   - Log scale - shows information flux hierarchy
   - Separation indicates sector structure

4. **Three-Sector Decomposition** (bottom-right)
   - Observable, dark, and internal dimensions
   - Stacked view of dimensional hierarchy
   - Shows stability of sector structure

### `geometry_consensus.png`
(Only generated if `geometry_enable_consensus=True`)

2-panel visualization:

1. **Consensus Volume Evolution**
   - Collective geometry volume (belief and prior)
   - Measures "size" of shared geometric structure

2. **Agent Count**
   - Number of agents contributing to consensus
   - Tracks emergence/death dynamics

### `geometry_analysis/final_eigenvalue_spectrum.png`
Detailed eigenvalue spectrum at final timestep:
- Full eigenvalue ladder (all K dimensions)
- Log scale
- Both belief and prior geometries
- Visualizes sector thresholds

## Use Cases

### 1. Study Emergent Spacetime Dimensionality

**Goal:** Understand how low-dimensional effective spacetime emerges from high-dimensional latent space.

**Configuration:**
```python
cfg = SimulationConfig(
    K_latent=767,  # High-dimensional (like transformer)
    track_pullback_geometry=True,
    geometry_track_interval=5,
    geometry_lambda_obs=0.1,
    geometry_lambda_dark=0.01,
    # ... other params ...
)
```

**Analysis:**
```python
# Load history
with open("geometry_history.pkl", "rb") as f:
    history = pickle.load(f)

# Plot observable dimensions over time
import matplotlib.pyplot as plt
steps = history.get_steps()
n_obs = history.get_mean_observable_dims()

plt.plot(steps, n_obs)
plt.xlabel('Step')
plt.ylabel('Observable Dimensions')
plt.title('Emergent Spacetime Dimensionality')
plt.show()

# Final snapshot
final = history.snapshots[-1]
print(f"Final observable dims: {final.mean_n_observable:.1f}")
print(f"Final dark dims: {final.mean_n_dark:.1f}")
print(f"Final internal dims: {final.mean_n_internal:.1f}")
```

### 2. Track Consensus Formation

**Goal:** Monitor emergence of shared geometric structure across agents.

**Configuration:**
```python
cfg = SimulationConfig(
    n_agents=5,
    track_pullback_geometry=True,
    geometry_enable_consensus=True,  # Essential!
    geometry_track_interval=10,
)
```

**Analysis:**
```python
# Load and analyze consensus
with open("geometry_history.pkl", "rb") as f:
    history = pickle.load(f)

for snapshot in history.snapshots:
    if snapshot.consensus_belief is not None:
        # Consensus volume
        G = snapshot.consensus_belief.G_consensus
        vol = np.mean(np.sqrt(np.maximum(np.linalg.det(G), 1e-10)))
        print(f"Step {snapshot.step}: consensus volume = {vol:.4e}")
```

### 3. Detect Epistemic Death via Geometry

**Goal:** Identify when agents' beliefs collapse geometrically (precursor to consensus).

**Configuration:**
```python
cfg = SimulationConfig(
    enable_emergence=True,  # Hierarchical dynamics
    track_pullback_geometry=True,
    geometry_track_interval=5,
)
```

**Analysis:**
```python
# Track volume collapse
for snapshot in history.snapshots:
    vols = [np.mean(m.volume_element()) for m in snapshot.agent_metrics_belief]
    print(f"Step {snapshot.step}: volumes = {vols}")

    # Detect collapse: volume drops below threshold
    if any(v < 1e-6 for v in vols):
        print(f"  ⚠️  Epistemic death detected!")
```

### 4. Compare Epistemic vs Ontological Geometry

**Goal:** Understand relationship between beliefs (epistemic) and world models (ontological).

**Configuration:**
```python
cfg = SimulationConfig(
    track_pullback_geometry=True,
    geometry_track_interval=5,
    lambda_prior_align=2.0,  # Strong prior alignment
)
```

**Analysis:**
```python
# Compare eigenvalue spectra
final = history.snapshots[-1]

eigvals_belief = final.mean_eigenvalues_belief
eigvals_prior = final.mean_eigenvalues_prior

plt.figure(figsize=(10, 6))
plt.semilogy(eigvals_belief, 'b-', label='Belief (epistemic)', linewidth=2)
plt.semilogy(eigvals_prior, 'r-', label='Prior (ontological)', linewidth=2)
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue')
plt.title('Epistemic vs Ontological Geometry')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Performance Considerations

### Computational Cost

**Per snapshot:**
- Metric computation: O(n_agents × n_spatial × K²)
- Eigenvalue decomposition: O(n_agents × n_spatial × n_dims³)
- Consensus (if enabled): O(n_agents² × n_spatial × K²)
- Gauge averaging (if enabled): O(n_agents × n_samples × metric_computation)

**Typical overhead:**
- Standard (no consensus): ~5-10% slowdown
- With consensus (5 agents): ~20-30% slowdown
- With gauge averaging: ~10x-100x slowdown (DON'T enable unless necessary!)

### Memory Usage

**Per snapshot:**
- Each `InducedMetric`: ~(n_spatial × n_dims²) × 8 bytes
- Full history: ~n_snapshots × n_agents × metric_size

**Example (1D manifold):**
- spatial_shape = (64,), n_dims = 1, K = 11, n_agents = 5
- Per snapshot: ~5 KB
- 100 snapshots: ~500 KB (negligible)

**Example (2D manifold):**
- spatial_shape = (64, 64), n_dims = 2, K = 11, n_agents = 5
- Per snapshot: ~160 KB
- 100 snapshots: ~16 MB (still fine)

### Optimization Tips

1. **Increase `geometry_track_interval`** if memory-limited
   - Every 10 steps: standard
   - Every 20-50 steps: memory-constrained
   - Every 1 step: publication figures

2. **Disable consensus** for exploratory runs with many agents
   - Enable only when explicitly studying collective geometry

3. **Never enable gauge averaging** unless absolutely necessary
   - Use consensus metric instead (already gauge-invariant)
   - Only enable for specialized gauge-theoretic studies

4. **Track base agents only** in hierarchical systems
   - Meta-agents don't have spatial structure
   - Focus on scale-0 agents where geometry is defined

## Advanced Usage

### Custom Analysis

```python
from geometry.geometry_tracker import GeometryTracker

# Create tracker manually
tracker = GeometryTracker(
    agents=system.agents,
    track_interval=5,
    dx=0.1,
    enable_consensus=True,
    lambda_obs=0.1,
    lambda_dark=0.01
)

# Record during custom training loop
for step in range(n_steps):
    # ... your training logic ...

    if tracker.should_record(step):
        tracker.record(step, system.agents)

# Save and analyze
tracker.save("my_geometry_history.pkl")
tracker.plot_evolution("my_geometry_plot.png")

from geometry.geometry_tracker import analyze_final_geometry
analyze_final_geometry(tracker.history, save_dir="my_analysis/")
```

### Accessing Raw Metrics

```python
# Load history
with open("geometry_history.pkl", "rb") as f:
    history = pickle.load(f)

# Get specific snapshot
snapshot = history.snapshots[50]  # Step 50×interval

# Access agent metric
agent_0_belief = snapshot.agent_metrics_belief[0]

# Raw metric tensor
G = agent_0_belief.G  # Shape: (*spatial_shape, n_dims, n_dims)

# Eigenvalues and eigenvectors
lambdas = agent_0_belief.eigenvalues  # Shape: (*spatial_shape, n_dims)
evecs = agent_0_belief.eigenvectors   # Shape: (*spatial_shape, n_dims, n_dims)

# Sector masks
obs_mask, dark_mask, int_mask = agent_0_belief.get_three_sector_decomposition(
    lambda_obs=0.1,
    lambda_dark=0.01,
    relative=True
)

# Volume element
vol = agent_0_belief.volume_element()  # Shape: (*spatial_shape,)
```

## Troubleshooting

### "Failed to compute metrics for agent X"

**Cause:** Numerical instability in covariance inversion or derivative computation.

**Solutions:**
- Check that agent covariances are well-conditioned
- Increase `eps` parameter in metric computation
- Ensure agents have non-degenerate support regions

### Geometry tracking slows simulation significantly

**Cause:** Too frequent tracking or consensus enabled with many agents.

**Solutions:**
- Increase `geometry_track_interval` (e.g., 20-50)
- Disable consensus: `geometry_enable_consensus=False`
- Track fewer timesteps: only record key snapshots

### Memory error with long simulations

**Cause:** Too many snapshots accumulated.

**Solutions:**
- Increase `geometry_track_interval`
- Periodically save and clear history:
  ```python
  if step % 1000 == 0:
      tracker.save(f"geometry_checkpoint_{step}.pkl")
      tracker.history.snapshots.clear()
  ```

### Eigenvalues are all similar (no sector structure)

**Cause:** Insufficient information flow or too short simulation.

**Solutions:**
- Run longer to allow structure to emerge
- Increase energy weights (`lambda_belief_align`, etc.)
- Check that agents are actually interacting (overlapping supports)

## Integration with Existing Code

The pullback tracking is fully integrated into `simulation_runner.py`. No changes needed to use it - just set configuration parameters!

For custom simulation loops:

```python
from geometry.geometry_tracker import GeometryTracker

# In your setup:
tracker = GeometryTracker(agents, track_interval=10, dx=0.1)

# In your training loop:
for step in range(n_steps):
    # ... your code ...

    if tracker.should_record(step):
        tracker.record(step, agents)

# After training:
tracker.save("geometry_history.pkl")
tracker.plot_evolution("geometry_evolution.png")
```

## See Also

- `geometry/README_PULLBACK.md` - Mathematical framework
- `geometry/pullback_metrics.py` - Core pullback implementation
- `geometry/gauge_consensus.py` - Gauge-invariant consensus
- `examples/example_pullback_geometry.py` - Standalone examples
- `examples/demo_pullback_tracking.py` - Integration demo

---

**Authors:** Chris & Christine
**Date:** November 2025
**Version:** 1.0
