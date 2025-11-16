#!/usr/bin/env python3
"""
Direct comparison of ACTUAL standard vs hierarchical training.
Forces Path A to use MultiAgentSystem and Path B to use MultiScaleSystem.
"""
import numpy as np
import sys

# Run both paths and compare
SEED = 2

print("="*70)
print("COMPARING ACTUAL TRAINING PATHS")
print("="*70)

# ============================================================================
# PATH A: Force standard (MultiAgentSystem)
# ============================================================================
print("\n" + "="*70)
print("PATH A: Standard Training (MultiAgentSystem)")
print("="*70)

# Temporarily override ENABLE_EMERGENCE
import simulation_suite
original_enable_emergence = simulation_suite.ENABLE_EMERGENCE
simulation_suite.ENABLE_EMERGENCE = False

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

manifold_a = simulation_suite.build_manifold()
supports_a = simulation_suite.build_supports(manifold_a, rng)
agents_a = simulation_suite.build_agents(manifold_a, supports_a, rng)
system_a = simulation_suite.build_system(agents_a, rng)

from free_energy_clean import compute_total_free_energy
from gradients.gradient_engine import compute_natural_gradients
from update_engine import GradientApplier
from agent.trainer import TrainingConfig

E0_a = compute_total_free_energy(system_a)
print(f"Step 0: E = {E0_a.total:.6f} [self={E0_a.self_energy:.3f}, β={E0_a.belief_align:.3f}]")

grads_a = compute_natural_gradients(system_a)
print(f"Gradients: agent[0] |∇μ_q| = {np.linalg.norm(grads_a[0].grad_mu_q):.6f}")

# Apply one update
config = TrainingConfig(lr_mu_q=0.1, lr_sigma_q=0.001, lr_mu_p=0.1, lr_sigma_p=0.001, lr_phi=0.1)
GradientApplier.apply_updates(system_a.agents, grads_a, config)

# No lock mode re-averaging (IDENTICAL_PRIORS = "init_copy")

E1_a = compute_total_free_energy(system_a)
print(f"Step 1: E = {E1_a.total:.6f} [self={E1_a.self_energy:.3f}, β={E1_a.belief_align:.3f}]")
print(f"ΔE = {E1_a.total - E0_a.total:.6f}")

# ============================================================================
# PATH B: Force hierarchical (MultiScaleSystem)
# ============================================================================
print("\n" + "="*70)
print("PATH B: Hierarchical Training (MultiScaleSystem)")
print("="*70)

# Force ENABLE_EMERGENCE = True
simulation_suite.ENABLE_EMERGENCE = True

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

manifold_b = simulation_suite.build_manifold()
supports_b = simulation_suite.build_supports(manifold_b, rng)
agents_b = simulation_suite.build_agents(manifold_b, supports_b, rng)
system_b = simulation_suite.build_system(agents_b, rng)  # Creates MultiScaleSystem

# Get active agents
active_agents_b = system_b.get_all_active_agents()
print(f"Active agents: {len(active_agents_b)}")

# Create adapter for gradient computation
from simulation_suite import _GradientSystemAdapter
adapter_b = _GradientSystemAdapter(active_agents_b, system_b.system_config)

E0_b = compute_total_free_energy(adapter_b)
print(f"Step 0: E = {E0_b.total:.6f} [self={E0_b.self_energy:.3f}, β={E0_b.belief_align:.3f}]")

grads_b = compute_natural_gradients(adapter_b)
print(f"Gradients: agent[0] |∇μ_q| = {np.linalg.norm(grads_b[0].grad_mu_q):.6f}")

# Apply one update
GradientApplier.apply_updates(active_agents_b, grads_b, config)

# No lock mode re-averaging (IDENTICAL_PRIORS = "init_copy")

# Recreate adapter after update (in case it needs fresh data)
adapter_b_after = _GradientSystemAdapter(active_agents_b, system_b.system_config)
E1_b = compute_total_free_energy(adapter_b_after)
print(f"Step 1: E = {E1_b.total:.6f} [self={E1_b.self_energy:.3f}, β={E1_b.belief_align:.3f}]")
print(f"ΔE = {E1_b.total - E0_b.total:.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print(f"\nStep 0:")
print(f"  Path A: {E0_a.total:.6f}")
print(f"  Path B: {E0_b.total:.6f}")
print(f"  Match: {np.abs(E0_a.total - E0_b.total) < 1e-5}")

print(f"\nGradients (agent 0):")
diff_mu = np.linalg.norm(grads_a[0].grad_mu_q - grads_b[0].grad_mu_q)
diff_sigma = np.linalg.norm(grads_a[0].grad_Sigma_q - grads_b[0].grad_Sigma_q)
print(f"  |∇μ_q| diff: {diff_mu:.6e}")
print(f"  |∇Σ_q| diff: {diff_sigma:.6e}")

print(f"\nStep 1:")
print(f"  Path A: {E1_a.total:.6f}")
print(f"  Path B: {E1_b.total:.6f}")
print(f"  Difference: {E1_a.total - E1_b.total:.6f}")

print(f"\nΔE:")
print(f"  Path A: {E1_a.total - E0_a.total:.6f}")
print(f"  Path B: {E1_b.total - E0_b.total:.6f}")

if np.abs(E1_a.total - E1_b.total) < 1e-5:
    print("\n✅ PASS: Dynamics match")
else:
    print("\n❌ FAIL: Dynamics diverge")
    print(f"   Investigating cause...")

    # Check if priors differ after initialization
    print(f"\n   Checking agent[0] priors:")
    print(f"   Path A μ_p: {system_a.agents[0].mu_p[:3]}")
    print(f"   Path B μ_p: {active_agents_b[0].mu_p[:3]}")
    mu_p_match = np.allclose(system_a.agents[0].mu_p, active_agents_b[0].mu_p)
    print(f"   μ_p match: {mu_p_match}")

    if not mu_p_match:
        print("   ⚠️  Priors differ - initialization bug not fully fixed!")

# Restore
simulation_suite.ENABLE_EMERGENCE = original_enable_emergence

print("\n" + "="*70)
