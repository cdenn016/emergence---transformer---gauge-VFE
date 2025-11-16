#!/usr/bin/env python3
"""
Debug script to investigate 2x energy discrepancy.

Run this to compare energies between enable_emergence=True and False.
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import SystemConfig, AgentConfig, TrainingConfig
from agent.system import MultiAgentSystem
from agent.trainer import Trainer
from agent.agents import Agent
from geometry.geometry_base import BaseManifold, TopologyType
from free_energy_clean import compute_total_free_energy

def create_test_system(seed=42):
    """Create a minimal test system for comparison."""
    np.random.seed(seed)

    # Configuration
    K = 3
    n_agents = 4

    # System config
    system_config = SystemConfig(
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.1,
        lambda_obs=0.0,  # Disable observations for simplicity
        kappa_beta=1.0,
        kappa_gamma=1.0,
    )

    # Agent config
    agent_config = AgentConfig(
        K=K,
        spatial_shape=(),  # 0D transformers
        init_mode_mu_q='random',
        init_mode_Sigma_q='random',
        init_mode_mu_p='random',
        init_mode_Sigma_p='random',
    )

    # Create agents
    agents = []
    for i in range(n_agents):
        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        agent = Agent(
            base_manifold=manifold,
            config=agent_config,
            agent_id=f"agent_{i}",
            seed=seed + i
        )
        agents.append(agent)

    # Create system
    system = MultiAgentSystem(agents, system_config)

    return system

def diagnose_energy_difference():
    """Compare energies between standard system."""
    print("="*70)
    print("ENERGY DIAGNOSTIC - 2x Investigation")
    print("="*70)

    # Create test system
    system = create_test_system(seed=42)

    print(f"\nSystem configuration:")
    print(f"  n_agents: {system.n_agents}")
    print(f"  len(agents): {len(system.agents)}")
    print(f"  lambda_self: {system.config.lambda_self}")
    print(f"  lambda_belief: {system.config.lambda_belief_align}")
    print(f"  lambda_prior: {system.config.lambda_prior_align}")

    # Compute energy
    print(f"\nComputing initial energy...")
    energies = compute_total_free_energy(system)

    print(f"\n{'-'*70}")
    print("ENERGY BREAKDOWN:")
    print(f"{'-'*70}")
    print(f"  Self energy:     {energies.self_energy:12.6f}")
    print(f"  Belief align:    {energies.belief_align:12.6f}")
    print(f"  Prior align:     {energies.prior_align:12.6f}")
    print(f"  Observations:    {energies.observations:12.6f}")
    print(f"{'-'*70}")
    print(f"  TOTAL:           {energies.total:12.6f}")
    print(f"{'-'*70}")

    # Diagnostic: Check overlap counting
    print(f"\nOverlap diagnostic:")
    print(f"  Stored overlaps: {len(system.overlap_masks)}")
    print(f"  Expected for {system.n_agents} agents (all-to-all): {system.n_agents * (system.n_agents - 1)}")

    # Check if overlaps are symmetric
    symmetric_count = 0
    for (i, j) in system.overlap_masks.keys():
        if (j, i) in system.overlap_masks:
            symmetric_count += 1
    print(f"  Symmetric pairs: {symmetric_count}")

    # Manually compute belief alignment energy to check for double-counting
    print(f"\nManual belief alignment check:")
    from free_energy_clean import compute_belief_alignment_energy

    E_belief_manual = 0.0
    for i in range(system.n_agents):
        E_i = compute_belief_alignment_energy(system, i)
        print(f"    Agent {i} → neighbors: {E_i:12.6f}")
        E_belief_manual += E_i

    print(f"  Manual sum: {E_belief_manual:12.6f}")
    print(f"  From breakdown: {energies.belief_align:12.6f}")
    print(f"  Match: {np.isclose(E_belief_manual, energies.belief_align)}")

    # Check if we're double-counting pairs
    print(f"\nChecking for double-counting:")
    print(f"  If double-counted: {E_belief_manual / 2:.6f}")

    return energies

if __name__ == "__main__":
    diagnose_energy_difference()
