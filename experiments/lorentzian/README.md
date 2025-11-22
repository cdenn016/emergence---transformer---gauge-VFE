# Lorentzian Signature Experiments

## Overview

This directory contains experimental testbeds for discovering mechanisms that could produce **Lorentzian signature** `(-,+,+,+)` in pullback metrics from belief dynamics.

## The Problem

Currently, the Fisher information metric is **manifestly Riemannian** (all eigenvalues positive):

```
g_ij = E[∂_i log p · ∂_j log p]  ⟹  signature (+,+,+,+)
```

However, for emergent spacetime interpretation, we want to find conditions under which the pullback metric develops Lorentzian signature, with one negative eigenvalue corresponding to a "timelike" direction.

## Potential Mechanisms

### 1. **α-Divergences** (`alpha_divergence.py`)

**Hypothesis**: For `α < 0`, the α-divergence induced metric can be indefinite, potentially exhibiting Lorentzian signature in certain parameter regimes.

**Background**:
```
D_α(p||q) = (4/(1-α²)) ∫ [1 - p^((1+α)/2) q^((1-α)/2)] dx
```

- `α = 0`: Hellinger divergence
- `α → 1`: KL divergence
- `α < 0`: Can have indefinite metric

**Test**: Scan α from -2 to +2 and check metric signature.

```bash
python -m experiments.lorentzian.alpha_divergence
```

### 2. **Hamiltonian Dynamics** (`hamiltonian_beliefs.py`)

**Hypothesis**: Hamiltonian formulation with symplectic structure might naturally induce Lorentzian geometry when the Hamiltonian (energy) is treated as a "time" coordinate.

**Background**:
- Phase space: `(q, p)` = `(μ, conjugate momentum)`
- Symplectic form: `ω = dq ∧ dp`
- Hamiltonian: `H = Free energy`

**Energy-time construction**:
```
ds² = -dH² + dq² + dp²
```

This explicitly constructs a Lorentzian metric with H as the timelike direction.

**Test**: Compare standard symplectic metric vs energy-time metric.

```bash
python -m experiments.lorentzian.hamiltonian_beliefs
```

### 3. **Lorentz Gauge Group** (`lorentz_gauge.py`)

**Hypothesis**: Using `SO(1,3)` gauge fields (Lorentz group) instead of `SO(3)` might induce Lorentzian signature through the **indefinite Killing form**.

**Background**:
- Current: `φ ∈ so(3)` (compact, positive definite Killing form)
- Proposed: `φ ∈ so(1,3)` (non-compact, indefinite Killing form)

The Lorentz algebra has:
- 3 rotations `J_i` (spatial)
- 3 boosts `K_i` (timelike)
- Commutator: `[K_i, K_j] = -ε_ijk J_k` (note minus sign!)

The Killing form for `so(1,3)` is **indefinite**, which should directly give metric signature `(-, -, +, +, +, +)` on the 6D gauge algebra.

**Test**: Compare Killing forms for SO(3) vs SO(1,3).

```bash
python -m experiments.lorentzian.lorentz_gauge
```

## Expected Outcomes

### Successful Detection Criteria

A successful mechanism should produce:

1. **One negative eigenvalue** in the metric (Lorentzian signature)
2. **Stable timelike direction** (eigenvector for negative eigenvalue)
3. **Physical interpretation**: Timelike direction should align with:
   - Gradient flow direction (`-∇F`)
   - Belief trajectory tangent (`dμ/dt`)
   - Energy gradient direction (`∇H`)

### Visualization

After running experiments, use the signature analysis tools:

```python
from geometry.signature_analysis import analyze_metric_signature
from geometry.metric_visualization import plot_signature_classification

# Analyze metric
sig = analyze_metric_signature(metric)

# Check for Lorentzian
if sig.signature == MetricSignature.LORENTZIAN:
    print("Timelike direction:", sig.timelike_direction)
    print("Light cone structure available!")
```

## Integration with Main Codebase

To integrate a successful mechanism:

1. **Modify free energy**: Use α-divergence in `free_energy_clean.py`
2. **Add Hamiltonian mode**: Create `HamiltonianTrainer` class
3. **Extend gauge fields**: Support `SO(1,3)` in `gradients/gauge_fields.py`

## Research Questions

1. **Can we find natural conditions** (not ad-hoc constructions) that produce Lorentzian signature?

2. **Is the timelike direction physically meaningful?**
   - Does it align with gradient flow?
   - Does it correspond to decreasing free energy?

3. **Does Lorentzian signature emerge dynamically?**
   - Start Riemannian, become Lorentzian after phase transition?
   - Regions of base manifold with different signatures?

4. **Connection to gauge theory?**
   - Does SO(1,3) gauge structure naturally arise from constraints?
   - Kaluza-Klein-like mechanism?

## References

- **α-divergences**: Amari (2016), "Information Geometry and Its Applications"
- **Hamiltonian mechanics**: Arnold, "Mathematical Methods of Classical Mechanics"
- **Lorentz group**: Weinberg, "The Quantum Theory of Fields, Vol 1"
- **Indefinite metrics**: O'Neill, "Semi-Riemannian Geometry"

## Next Steps

1. Run all three experiments
2. Identify most promising mechanism
3. Develop full integration with belief dynamics
4. Create comprehensive visualization suite
5. Document findings for manuscript

---

**Author**: Chris
**Date**: November 2025
**Status**: 🔬 Experimental
