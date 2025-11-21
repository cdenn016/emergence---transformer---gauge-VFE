# Pullback Geometry: "It From Bit" Construction

## Overview

This module implements Wheeler's "it from bit" vision through mathematical pullback: **geometric structure emerges from informational dynamics**.

Agents maintain probability distributions `q(c), p(c)` as smooth fields over a base manifold `C`. These fields induce Riemannian metrics on `C` via pullback of the Fisher-Rao metric from the statistical fiber.

## Mathematical Framework

### The Fisher-Rao Metric (Statistical Fiber)

The space of probability distributions `B` carries the Fisher-Rao metric `g_B`. For Gaussian distributions `N(μ, Σ)`:

```
g_B(δq, δq) = δμ^T Σ^{-1} δμ + (1/2)tr(Σ^{-1} δΣ Σ^{-1} δΣ)
```

This metric is intrinsic to the space of distributions, measuring statistical distinguishability via Fisher information.

### Induced Metrics via Pullback

Each smooth section `σ: C → B` induces a metric on the base manifold:

```
G(c) = σ* g_B
```

For belief section `σ^(q)`, the induced metric components are:

```
G^(q)_μν(c) = E_{q(c)}[(∂_μ log q)(∂_ν log q)]
```

Similarly for prior section `σ^(p)`:

```
G^(p)_μν(c) = E_{p(c)}[(∂_μ log p)(∂_ν log p)]
```

### For Gaussian Distributions

For `q(c) = N(μ(c), Σ(c))`, the induced metric has explicit form:

```
G_μν(c) = (∂_μ μ)^T Σ^{-1} (∂_ν μ) + (1/2)tr(Σ^{-1}(∂_μ Σ)Σ^{-1}(∂_ν Σ))
```

**Isotropic case** `Σ(c) = σ² I` (constant):

```
G_μν(c) = (1/σ²) (∂_μ μ) · (∂_ν μ)
```

This is a **conformal metric** with conformal factor `1/σ²`:
- High certainty (small σ) → magnifies distances → regions appear "larger"
- High uncertainty (large σ) → compresses distances → regions appear "smaller"

## Module Structure

### `pullback_metrics.py`

Core pullback mechanism implementing:

1. **Fisher-Rao metric on statistical fiber**
   - `fisher_rao_metric_gaussian()`: Compute Fisher metric for Gaussian
   - `fisher_rao_distance_gaussian()`: Approximate Fisher-Rao distance

2. **Pullback computation**
   - `pullback_metric_gaussian()`: General pullback for arbitrary Σ(c)
   - `pullback_metric_gaussian_isotropic()`: Optimized for isotropic case

3. **Agent metrics**
   - `agent_induced_metrics()`: Compute both G^(q) and G^(p) for an agent

4. **InducedMetric class**
   - Eigenvalue decomposition: `compute_spectral_decomposition()`
   - Sector analysis: `get_observable_sector()`, `get_three_sector_decomposition()`
   - Volume elements: `volume_element()`

### `gauge_consensus.py`

Gauge-invariant metric averaging implementing:

1. **SO(3) Haar measure sampling**
   - `sample_so3_haar()`: Uniform samples from SO(3)
   - `sample_so3_algebra_haar()`: Samples as so(3) elements

2. **Gauge averaging**
   - `gauge_average_metric_mc()`: Monte Carlo integration over gauge orbit
   - Returns `GaugeAveragedMetric` with mean and uncertainty

3. **Consensus metrics**
   - `compute_consensus_metric()`: Collective metric from multiple agents
   - `compute_consensus_metric_weighted_spatial()`: Spatially-weighted consensus
   - Returns `ConsensusMetric` with collective geometry

4. **Gaussian Fréchet mean**
   - `frechet_mean_gaussian()`: Geometric average on Fisher manifold

## Usage Examples

### Basic Pullback (1D Gaussian Field)

```python
import numpy as np
from geometry.pullback_metrics import pullback_metric_gaussian

# Create 1D base manifold
x = np.linspace(0, 2*np.pi, 64)
dx = 2*np.pi / 64

# Gaussian mean field
mu_field = np.stack([np.sin(x), np.cos(x)], axis=-1)  # (64, 2)

# Constant covariance
Sigma_field = np.repeat(np.eye(2)[None], 64, axis=0)  # (64, 2, 2)

# Compute pullback metric
metric = pullback_metric_gaussian(mu_field, Sigma_field, dx=dx)

# metric.G has shape (64, 1, 1) for 1D base manifold
print(f"Metric shape: {metric.G.shape}")
```

### Agent Induced Metrics

```python
from agent.agents import Agent
from config import AgentConfig
from geometry.pullback_metrics import agent_induced_metrics

# Create agent
config = AgentConfig(spatial_shape=(64,), K=3)
agent = Agent(0, config)

# ... initialize agent fields ...

# Compute both belief and prior metrics
G_belief, G_prior = agent_induced_metrics(agent, dx=0.1)

# G_belief: Epistemic geometry (current beliefs)
# G_prior: Ontological geometry (world model)
```

### Eigenvalue Decomposition

```python
# Compute spectral decomposition
metric.compute_spectral_decomposition()

# Get observable sector (eigenvalues > 10% of max)
obs_mask, obs_eigvals = metric.get_observable_sector(
    threshold=0.1,
    relative=True
)

# Three-sector decomposition
obs_mask, dark_mask, internal_mask = metric.get_three_sector_decomposition(
    lambda_obs=0.1,    # Observable threshold
    lambda_dark=0.01   # Dark threshold
)
```

### Consensus Metrics

```python
from geometry.gauge_consensus import compute_consensus_metric

# Create multiple agents
agents = [Agent(i, config) for i in range(5)]

# ... initialize agents ...

# Compute gauge-invariant consensus
consensus = compute_consensus_metric(
    agents,
    metric_type="prior",       # Use ontological metrics
    gauge_average=True,        # Average over gauge orbits
    n_samples_gauge=100        # MC samples
)

# consensus.G_consensus is the collective geometry
```

## Dual Geometries: Epistemic vs Ontological

### Belief-Induced Metric `G^(q)`

- **Epistemic geometry**: Reflects agent's current posterior beliefs
- **Highly dynamical**: Changes rapidly as observations arrive
- **Encodes uncertainty**: High gradient → short information distance
- **Analogous to**: Instantaneous state in phase space

### Prior-Induced Metric `G^(p)`

- **Ontological geometry**: Reflects agent's generative model
- **Quasi-static**: Evolves slowly as agent learns
- **Encodes world model**: Agent's long-term expectations
- **Analogous to**: Background geometry or potential landscape

**Conjecture**: The prior-induced metric `G^(p)` represents the agent's **perceived geometry of reality**. What the agent experiences as "spatial distance" is precisely the information-geometric distance in its generative model.

**Implication**: Different agents with different priors perceive different geometries on the same underlying base manifold `C`. There is no "true" metric independent of agents.

## Eigenvalue Sectors

For high-dimensional statistical manifolds (e.g., K=768), the induced metric naturally decomposes:

### Observable Sector
```
D_obs = {e_a : λ_a > Λ_obs}
```
Directions with high information flux → perceived spacetime dimensions (~4 for humans: 1+3)

### Dark Sector
```
D_dark = {e_a : Λ_dark < λ_a ≤ Λ_obs}
```
Intermediate information flux → additional structure below perception threshold (~10-100 dimensions)

### Internal Sector
```
D_internal = {e_a : λ_a ≤ Λ_dark}
```
Negligible information flux → pure internal degrees of freedom (~10^5 dimensions for K=768)

**Hierarchy**:
```
|D_obs| << |D_dark| << |D_internal|
λ_obs >> λ_dark >> λ_internal ≈ 0
```

## Gauge Invariance

### The Problem

Naive averaging of metrics from multiple agents:
```
Ḡ_μν(c) = (1/N) Σ w_i(c) G_i,μν(c)
```
depends on each agent's arbitrary gauge frame choice `φ_i`, violating gauge invariance.

### The Solution

**Gauge-averaged metric**:
```
⟨G_i⟩_μν(c) = ∫_G dg G_i,μν(c; φ_i → φ_i + g)
```

**Consensus metric**:
```
Ḡ^consensus_μν(c) = Σ w_i(c) ⟨G_i⟩_μν(c)
```

This is **gauge-invariant by construction** - no agent's arbitrary frame choice affects collective geometry.

### Physical Interpretation

Gauge invariance in physics may arise as a **consistency requirement for multi-agent consensus**. For agents with different internal reference frames to agree on shared geometric structure, that structure must be gauge-invariant.

**Conjecture**: Gauge invariance in fundamental physics (electromagnetism, Yang-Mills, general relativity) may be a consequence of cognitive agents requiring frame-independent descriptions of shared reality.

## Implementation Notes

### Computational Efficiency

1. **Isotropic optimization**: Use `pullback_metric_gaussian_isotropic()` when `Σ(c) = σ²I`
2. **Sparse support**: Future work - leverage agent support regions
3. **Gauge averaging**: Monte Carlo with ~50-200 samples typically sufficient
4. **Batching**: Use vectorized operations over spatial points

### Numerical Stability

- Covariances regularized via `sanitize_sigma()`
- Inverses computed via `safe_inv()` with conditioning checks
- Symmetry enforced: `G = 0.5(G + G^T)`
- Volume elements: `sqrt(max(det(G), ε))`

### Future Extensions

1. **Analytical gauge projection**: SO(3)-invariant components without MC sampling
2. **Lorentzian signature**: Mechanism for (-,+,+,+) signature emergence
3. **Temporal structure**: Belief trajectories → temporal metric components
4. **Sparse computation**: Efficient handling of localized support regions
5. **Geodesic distances**: Exact Fisher-Rao geodesics (not linearized approximation)

## References

### Wheeler's Vision
- Wheeler, J.A. (1990). "Information, Physics, Quantum: The Search for Links"

### Fisher-Rao Geometry
- Amari, S. (2016). "Information Geometry and Its Applications"
- Pennec, X. (2006). "Intrinsic Statistics on Riemannian Manifolds"

### Gauge Theory & Consensus
- Baez, J.C. (2006). "Gauge Fields, Knots and Gravity"
- Miolane, N. et al. (2020). "Geomstats: A Python Package for Riemannian Geometry"

### Cognitive Interpretations
- Clark, A. (2016). "Surfing Uncertainty"
- Hoffman, D. (2019). "The Case Against Reality"

## See Also

- `examples/example_pullback_geometry.py`: Comprehensive demonstrations
- `math_utils/fisher_metric.py`: Natural gradients on statistical manifold
- `math_utils/so3_frechet.py`: Fréchet mean on SO(3)
- `geometry/spectral_geometry.py`: Spectral methods for derivatives

---

**Authors**: Chris & Christine
**Date**: November 2025
**Version**: 1.0
