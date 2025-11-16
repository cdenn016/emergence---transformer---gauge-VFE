#!/usr/bin/env python3
"""
Compare step-by-step dynamics between Path A and Path B.
Identifies WHERE the dynamics diverge during training.
"""
import numpy as np
from pathlib import Path

from simulation_suite import (
    build_manifold,
    build_supports,
    build_agents,
    build_system,
    _GradientSystemAdapter,
    SEED
)

from free_energy_clean import compute_total_free_energy
from gradients.gradient_engine import compute_natural_gradients
from agent.trainer import Trainer, TrainingConfig
from update_engine import GradientApplier

def compare_single_step():
    """Compare one gradient step between both paths."""

    print("="*70)
    print("STEP-BY-STEP DYNAMICS COMPARISON")
    print("="*70)

    # Build identical initial systems
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    manifold = build_manifold()
    supports = build_supports(manifold, rng)
    agents = build_agents(manifold, supports, rng)
    system_std = build_system(agents, rng)

    print(f"\nInitial setup: {system_std.n_agents} agents")

    # ========================================================================
    # PATH A: Standard training
    # ========================================================================
    print(f"\n{'='*70}")
    print("PATH A: Standard Training")
    print(f"{'='*70}")

    # Initial energy
    E0_std = compute_total_free_energy(system_std)
    print(f"Step 0: Energy = {E0_std.total:.6f}")
    print(f"  Self: {E0_std.self_energy:.6f}, Belief: {E0_std.belief_align:.6f}")

    # Compute gradients
    grads_std = compute_natural_gradients(system_std)
    print(f"\nGradients computed for {len(grads_std)} agents")
    print(f"  Agent 0 - grad_mu_q norm: {np.linalg.norm(grads_std[0].grad_mu_q):.6f}")
    print(f"  Agent 0 - grad_Sigma_q norm: {np.linalg.norm(grads_std[0].grad_Sigma_q):.6f}")

    # Store state before update
    mu_q_before_std = [a.mu_q.copy() for a in system_std.agents]

    # Apply updates
    from agent.trainer import TrainingConfig
    config = TrainingConfig(
        n_steps=1,
        lr_mu_q=0.01,
        lr_sigma_q=0.01,
        lr_mu_p=0.001,
        lr_sigma_p=0.001,
        lr_phi=0.001
    )

    GradientApplier.apply_updates(system_std.agents, grads_std, config)

    # Apply identical priors lock if configured
    if system_std.config.identical_priors == "lock":
        GradientApplier.apply_identical_priors_lock(system_std.agents)
        print("  Applied identical_priors lock")

    # Energy after update
    E1_std = compute_total_free_energy(system_std)
    print(f"\nStep 1: Energy = {E1_std.total:.6f}")
    print(f"  Self: {E1_std.self_energy:.6f}, Belief: {E1_std.belief_align:.6f}")
    print(f"  ΔE = {E1_std.total - E0_std.total:.6f}")

    # Check parameter changes
    delta_mu = np.linalg.norm(system_std.agents[0].mu_q - mu_q_before_std[0])
    print(f"  Agent 0 Δμ_q: {delta_mu:.6f}")

    # ========================================================================
    # PATH B: Hierarchical (no emergence)
    # ========================================================================
    print(f"\n{'='*70}")
    print("PATH B: Hierarchical Training (no meta-agents)")
    print(f"{'='*70}")

    # Rebuild system with same seed to get identical initial state
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    manifold = build_manifold()
    supports = build_supports(manifold, rng)
    agents = build_agents(manifold, supports, rng)
    system_hier = build_system(agents, rng)  # This creates MultiScaleSystem if ENABLE_EMERGENCE=True

    # Get active agents (handle both MultiScaleSystem and MultiAgentSystem)
    if hasattr(system_hier, 'get_all_active_agents'):
        # MultiScaleSystem
        active_agents = system_hier.get_all_active_agents()
        system_config = system_hier.system_config
    else:
        # MultiAgentSystem
        active_agents = system_hier.agents
        system_config = system_hier.config

    print(f"Active agents: {len(active_agents)}")

    # Create adapter
    adapter = _GradientSystemAdapter(active_agents, system_config)

    # Initial energy
    E0_hier = compute_total_free_energy(adapter)
    print(f"Step 0: Energy = {E0_hier.total:.6f}")
    print(f"  Self: {E0_hier.self_energy:.6f}, Belief: {E0_hier.belief_align:.6f}")

    # Compute gradients
    grads_hier = compute_natural_gradients(adapter)
    print(f"\nGradients computed for {len(grads_hier)} agents")
    print(f"  Agent 0 - grad_mu_q norm: {np.linalg.norm(grads_hier[0].grad_mu_q):.6f}")
    print(f"  Agent 0 - grad_Sigma_q norm: {np.linalg.norm(grads_hier[0].grad_Sigma_q):.6f}")

    # Store state before update
    mu_q_before_hier = [a.mu_q.copy() for a in active_agents]

    # Apply updates
    GradientApplier.apply_updates(active_agents, grads_hier, config)

    # Apply identical priors lock if configured
    if system_config.identical_priors == "lock":
        GradientApplier.apply_identical_priors_lock(active_agents)
        print("  Applied identical_priors lock")

    # Energy after update
    E1_hier = compute_total_free_energy(adapter)
    print(f"\nStep 1: Energy = {E1_hier.total:.6f}")
    print(f"  Self: {E1_hier.self_energy:.6f}, Belief: {E1_hier.belief_align:.6f}")
    print(f"  ΔE = {E1_hier.total - E0_hier.total:.6f}")

    # Check parameter changes
    delta_mu = np.linalg.norm(active_agents[0].mu_q - mu_q_before_hier[0])
    print(f"  Agent 0 Δμ_q: {delta_mu:.6f}")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    print("\nInitial state (Step 0):")
    print(f"  Energy match: {np.abs(E0_std.total - E0_hier.total) < 1e-6}")
    print(f"    Std: {E0_std.total:.6f}, Hier: {E0_hier.total:.6f}")

    print("\nGradient comparison:")
    for i in range(min(3, len(grads_std))):
        g_std = grads_std[i]
        g_hier = grads_hier[i]

        mu_diff = np.linalg.norm(g_std.grad_mu_q - g_hier.grad_mu_q)
        sigma_diff = np.linalg.norm(g_std.grad_Sigma_q - g_hier.grad_Sigma_q)

        print(f"  Agent {i}:")
        print(f"    grad_mu_q diff: {mu_diff:.6e}")
        print(f"    grad_Sigma_q diff: {sigma_diff:.6e}")

        if mu_diff > 1e-6 or sigma_diff > 1e-6:
            print(f"    ⚠️  GRADIENTS DIFFER!")

    print("\nAfter 1 step (Step 1):")
    print(f"  Energy match: {np.abs(E1_std.total - E1_hier.total) < 1e-6}")
    print(f"    Std: {E1_std.total:.6f}, Hier: {E1_hier.total:.6f}")
    print(f"    Difference: {E1_std.total - E1_hier.total:.6e}")

    print("\nEnergy change:")
    print(f"  Path A: ΔE = {E1_std.total - E0_std.total:.6f}")
    print(f"  Path B: ΔE = {E1_hier.total - E0_hier.total:.6f}")
    print(f"  Difference in ΔE: {(E1_std.total - E0_std.total) - (E1_hier.total - E0_hier.total):.6e}")

    print("\nParameter changes:")
    for i in range(min(3, len(system_std.agents))):
        delta_std = np.linalg.norm(system_std.agents[i].mu_q - mu_q_before_std[i])
        delta_hier = np.linalg.norm(active_agents[i].mu_q - mu_q_before_hier[i])
        print(f"  Agent {i} Δμ_q: Std={delta_std:.6f}, Hier={delta_hier:.6f}, diff={delta_std-delta_hier:.6e}")

    # Final verdict
    print(f"\n{'='*70}")
    if np.abs(E1_std.total - E1_hier.total) < 1e-6:
        print("✅ PASS: Dynamics match after 1 step")
    else:
        print("❌ FAIL: Dynamics diverge after 1 step")
        if np.abs(E0_std.total - E0_hier.total) < 1e-6:
            print("   Initial states match, but gradients or updates differ")
        else:
            print("   Initial states already differ")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    compare_single_step()
