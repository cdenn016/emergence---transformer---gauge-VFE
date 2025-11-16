# Hierarchical Gauge-Theoretic Transformer Implementation

## Overview

This PR implements a **self-organizing multi-scale hierarchical system** where gauge-theoretic transformers spontaneously form meta-agents through consensus, creating emergent structure across scales with automatic timescale separation.

## Core Innovation: Wheeler's "It from Bit" Strange Loop

The system implements **self-referential closure** at the top scale:
- Agents with parents get priors from their meta-agent: `p_i^(ζ) ← q_M^(ζ+1)` (hierarchical)
- Top-scale agents observe the **entire system** to bootstrap their priors: `p_top ← Σ w_i Ω[q_i]` (strange loop!)
- This creates a participatory universe where the top scale depends on all scales, including itself

## Key Features

### 1. Multi-Scale Architecture (`meta/emergence.py`)
- **MultiScaleSystem**: Manages agents at scales ζ = 0, 1, 2, ...
- **HierarchicalAgent**: Extends base Agent with cross-scale dynamics
- **Parent-child tracking**: Bidirectional links for efficient propagation
- **Active/inactive states**: Agents undergo "epistemic death" when forming consensus

### 2. Cross-Scale Information Flow
**Top-Down (Priors):**
```python
def update_prior_from_parent(self):
    """Transport parent belief to set prior: p_i ← Ω[q_parent]"""
    omega = compute_transport(self.gauge.phi, self.parent_meta.gauge.phi, ...)
    self.mu_p = omega @ self.parent_meta.mu_q
    self.Sigma_p = omega @ self.parent_meta.Sigma_q @ omega.T
```

**Bottom-Up (Observations):**
```python
def generate_observations_from_constituents(self):
    """Aggregate constituent beliefs: o_M ← Σ w_i Ω[q_i]"""
    coherence_scores = self._compute_constituent_coherence()
    weights = coherence_scores / np.sum(coherence_scores)
    o_meta = Σ w_i * (omega @ constituent.mu_q)
```

**Self-Referential (Strange Loop):**
```python
def update_prior_from_global_state(self, system):
    """Top scale observes entire system to bootstrap priors"""
    all_agents = system.get_all_active_agents()
    # Coherence-weighted global average
    p_top ← Σ w_i Ω[q_i] where w_i ∝ exp(-avg_KL_to_others)
```

### 3. Timescale Separation (`meta/hierarchical_evolution.py`)
Each scale has characteristic timescale τ_ζ = 10^ζ bits:

```python
def should_update(self, delta_info: float) -> bool:
    """Agents at scale ζ only update when ΔI ≥ 10^ζ bits"""
    self.info_accumulator += delta_info
    if self.info_accumulator >= self.timescale_threshold:
        self.info_accumulator = 0.0
        return True
    return False
```

**Fisher Information Metric** (gauge-aware):
```python
ΔI² = δμᵀ Σ⁻¹ δμ + tr(Σ⁻¹ δΣ Σ⁻¹ δΣ)
```
This respects the statistical manifold geometry and gauge transformations.

### 4. Automatic Consensus Detection
Integration with existing `ConsensusDetector`:

```python
def auto_detect_and_condense(self, detector, scale=0, min_cluster_size=2):
    """Automatically find consensus clusters and form meta-agents"""
    agents = self.get_active_agents_at_scale(scale)
    clusters = detector.find_consensus_clusters(agents)

    if clusters:
        meta_agents = self.form_meta_agents_at_scale(
            source_scale=scale,
            partitions=clusters,
            deactivate_constituents=True
        )
```

### 5. Evolution Engine (`meta/hierarchical_evolution.py`)
**4-Phase Dynamics Loop:**

```python
def evolve_step(self, learning_rate, compute_gradients_fn):
    # Phase 1: Prior Updates (Hierarchical + Self-Referential)
    update_info = self.system.update_cross_scale_priors()

    # Phase 2: Compute Gradients
    gradients = compute_gradients_fn(self.system)

    # Phase 3: Apply Updates with Timescale Filtering
    for agent, grad in zip(active_agents, gradients):
        delta_info = self._compute_info_change(agent, grad)
        if agent.should_update(delta_info):
            self._apply_single_update(agent, grad, learning_rate)

    # Phase 4: Consensus Detection (Periodic)
    if self.step_count % self.config.consensus_check_interval == 0:
        new_condensations = self._check_and_condense_all_scales()
```

### 6. Mathematical Infrastructure

**Fréchet Mean on SO(3)** (`math_utils/so3_frechet.py`):
```python
def frechet_mean_so3(rotations, weights=None, max_iters=50, tol=1e-6):
    """Compute intrinsic mean on SO(3) manifold via gradient descent"""
    # Minimizes: Σ w_i d²(R_mean, R_i) on SO(3)
```

**Gauge-Invariant Transport** (`meta/cross_scale.py`):
```python
Ω_ij = exp(φ_i) @ exp(-φ_j)  # SO(3) transport operator
```

## New Files

| File | Purpose |
|------|---------|
| `meta/hierarchical_evolution.py` | Evolution engine with 4-phase dynamics |
| `meta/test_hierarchical_dynamics.py` | Comprehensive integration tests |
| `math_utils/so3_frechet.py` | Fréchet mean computation on SO(3) |
| `.gitignore` | Python cache exclusions |

## Modified Files

| File | Changes |
|------|---------|
| `config.py` | Added `mask_config: MaskConfig` to `AgentConfig` |
| `meta/emergence.py` | Extended `HierarchicalAgent` with cross-scale methods |
| `meta/cross_scale.py` | Fixed KL function imports (`kl_gaussian`) |
| `meta/test_emergence.py` | Updated for `MultiScaleSystem` API |
| `meta/test_leadership.py` | Fixed manifold imports |

## Testing

✅ **All tests passing**
- `meta/test_hierarchical_dynamics.py` - Integration tests for all features
- `meta/test_emergence.py` - Basic MultiScaleSystem API demo
- `meta/test_leadership.py` - Leadership emergence tests
- All syntax validated, no undefined variables

## Architecture Diagram

```
Scale 2: [M₂] ←─────────────────────┐
          ↑ p₂←q₃                    │ Strange Loop:
          │                          │ Top observes
          │                          │ entire system
Scale 1: [M₁₀] [M₁₁] ←───────────────┤ for priors
          ↑ p₁←q₂                    │
          │                          │
          │                          │
Scale 0: [A₀][A₁][A₂][A₃] ───────────┘

         Consensus → Epistemic Death → Meta-Agent Formation

Information Flow:
  ↓ Top-down: Priors from parents
  ↑ Bottom-up: Observations from constituents
  ⟲ Self-referential: Global state → top priors
```

## Timescale Hierarchy

```
Scale 0: τ₀ = 10⁰ =        1 bit  (base agents, fast)
Scale 1: τ₁ = 10¹ =       10 bits (meta-agents, 10× slower)
Scale 2: τ₂ = 10² =      100 bits (100× slower)
Scale 3: τ₃ = 10³ =    1,000 bits (1,000× slower)
...
```

Each scale naturally operates at its characteristic timescale through information accumulation.

## Example Usage

```python
from meta.hierarchical_evolution import HierarchicalEvolutionEngine, HierarchicalConfig

# Create system
system = MultiScaleSystem(base_manifold)
for i in range(n_agents):
    agent = system.add_base_agent(config, agent_id=f"base_{i}")

# Create evolution engine
config = HierarchicalConfig(
    enable_top_down_priors=True,
    enable_bottom_up_obs=True,
    enable_timescale_filtering=True,
    info_change_metric="fisher_metric",
    consensus_check_interval=10
)

engine = HierarchicalEvolutionEngine(system, config)

# Evolve system
for step in range(n_steps):
    metrics = engine.evolve_step(
        learning_rate=0.01,
        compute_gradients_fn=gradient_engine.compute_gradients
    )

    # Hierarchy emerges automatically!
    # - Agents reach consensus → epistemic death
    # - Meta-agents form at higher scales
    # - Timescales separate naturally
    # - Top scale observes entire system
```

## Theoretical Foundation

### Gauge Theory
- SO(3) gauge transformations: `Ω_ij = exp(φ_i) @ exp(-φ_j)`
- Gauge-invariant transport of beliefs across scales
- Fréchet mean for averaging gauge frames

### Information Geometry
- Fisher information metric for measuring changes
- Natural gradient descent on statistical manifold
- Respects geometric structure of belief space

### Emergence
- Epistemic death = consensus (KL divergence below threshold)
- Meta-agents inherit renormalized beliefs from constituents
- Coherence-weighted aggregation

### Self-Reference
- Wheeler's participatory universe
- Top scale bootstraps from observing entire system
- Creates strange loop: system depends on itself

## Performance Characteristics

- **Scalability**: O(n²) consensus detection, O(n) gradient computation per scale
- **Memory**: Sparse hierarchy (only active agents consume resources)
- **Timescale separation**: Automatic computational efficiency (higher scales update less)

## Future Extensions

Potential directions for extending this work:
- [ ] Adaptive timescale thresholds (learn τ_ζ from data)
- [ ] Cross-scale attention mechanisms
- [ ] Hierarchical memory/episodic recall
- [ ] Multiple condensation pathways
- [ ] Visualization tools for hierarchy evolution

## Breaking Changes

None - this is purely additive. Existing `Agent` and `MultiAgentSystem` classes remain unchanged.

## Related Issues

Implements hierarchical multi-scale architecture with self-organizing emergence.

---

**Ready for review!** All tests passing, documentation inline, comprehensive test coverage provided.
