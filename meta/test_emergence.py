# -*- coding: utf-8 -*-
"""
Test Meta-Agent Emergence
=========================

NOTE: This file demonstrates the new MultiScaleSystem API.
Most old test functions are commented out pending refactoring.

The new MultiScaleSystem provides:
- Direct hierarchical multi-scale structure
- Automatic cross-scale dynamics (top-down priors, bottom-up observations)
- Timescale separation with information accumulation
- Self-referential closure (Wheeler's "it from bit")

For working comprehensive tests, see:
- meta/test_hierarchical_dynamics.py (integration tests)
- meta/hierarchical_evolution.py (evolution engine)

Author: Chris & Christine
Updated: Nov 2025
"""

import numpy as np
from typing import Dict, List

# Import base system components
from config import SystemConfig, AgentConfig
from geometry.geometry_base import BaseManifold, TopologyType, create_full_support
from math_utils.generators import generate_so3_generators

# Import emergence machinery
from meta.emergence import (
    MultiScaleSystem,
    HierarchicalAgent,
    analyze_hierarchical_structure
)


def create_emergence_system(n_agents: int = 4, K: int = 3, seed: int = 42):
    """
    Create hierarchical multi-scale system optimized for emergence.

    Strong alignment coupling encourages consensus formation.
    """
    rng = np.random.default_rng(seed)

    # 0D base manifold for simplicity (particle-like transformers)
    base_manifold = BaseManifold(
        shape=(),
        topology=TopologyType.FLAT
    )

    # Agent config with moderate variation
    agent_cfg = AgentConfig(
        spatial_shape=(),
        K=K,
        mu_scale=0.3,      # Moderate initial spread
        sigma_scale=1.0,
        phi_scale=0.2
    )

    # System config encouraging consensus
    system_cfg = SystemConfig(
        lambda_self=0.1,           # Weak self-coupling
        lambda_belief_align=20.0,  # VERY strong belief alignment
        lambda_prior_align=10.0,   # Strong prior alignment
        lambda_obs=0.0,           # No observations (vacuum)
        lambda_phi=0.01,
        kappa_beta=0.05,          # Very low temperature (sharp attention)
        kappa_gamma=0.1,
        overlap_threshold=0.0,
        use_connection=False
    )

    # Create MultiScaleSystem
    system = MultiScaleSystem(base_manifold)
    system.system_config = system_cfg  # Attach config for energy computation

    # Add base agents at scale 0
    generators = generate_so3_generators(K)

    for i in range(n_agents):
        agent = system.add_base_agent(agent_cfg, agent_id=f"base_{i}")
        agent.support = create_full_support(base_manifold)
        agent.generators = generators

    return system, rng


def test_basic_multiscale_system():
    """Simple test demonstrating current MultiScaleSystem API."""
    print("="*70)
    print("BASIC MULTISCALE SYSTEM TEST")
    print("="*70)

    # Create system
    print("\n1. Creating MultiScaleSystem...")
    system, rng = create_emergence_system(n_agents=4, K=3)

    print(f"   Created system with {len(system.agents[0])} base agents at scale 0")

    # Show structure
    print("\n2. System structure:")
    print(system.summary())

    # Test hierarchy formation
    print("\n3. Testing meta-agent formation...")
    print("   Forming meta-agent from agents [0, 1]...")

    meta_agents = system.form_meta_agents_at_scale(
        source_scale=0,
        partitions=[[0, 1]],  # Form one meta-agent from first two base agents
        deactivate_constituents=True
    )

    print(f"   Created {len(meta_agents)} meta-agent(s) at scale 1")

    # Show updated structure
    print("\n4. Updated structure:")
    print(system.summary())

    # Test cross-scale dynamics
    print("\n5. Testing cross-scale prior updates...")
    update_info = system.update_cross_scale_priors()
    print(f"   Updated {update_info['total']} agent priors")
    print(f"     - {update_info['from_parent']} from parent meta-agents")
    print(f"     - {update_info['from_global']} from global state (strange loop)")

    # Test automatic consensus detection
    print("\n6. Testing automatic consensus detection...")
    from meta.consensus import ConsensusDetector

    detector = ConsensusDetector(
        belief_threshold=0.1,
        model_threshold=0.1,
        use_symmetric_kl=True
    )

    # Force two remaining base agents into consensus for demo
    agent_2 = system.agents[0][2]
    agent_3 = system.agents[0][3]

    # Make them identical
    agent_3.mu_q = agent_2.mu_q.copy()
    agent_3.Sigma_q = agent_2.Sigma_q.copy()

    # Check consensus
    state = detector.check_full_consensus(agent_2, agent_3)
    print(f"   Agents 2 & 3 in consensus: {state.in_consensus}")
    print(f"     Belief divergence: {state.belief_divergence:.4f}")
    print(f"     Model divergence: {state.model_divergence:.4f}")

    if state.in_consensus:
        print("   Forming meta-agent from agents [2, 3]...")
        new_meta = system.form_meta_agents_at_scale(
            source_scale=0,
            partitions=[[2, 3]],
            deactivate_constituents=True
        )
        print(f"   Created {len(new_meta)} additional meta-agent(s)")
        print("\n   Final structure:")
        print(system.summary())

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return system


# ==============================================================================
# COMMENTED OUT: Old functions requiring refactoring for MultiScaleSystem API
# ==============================================================================
#
# The following functions need updates to work with the new API:
#
# Issues to address:
# 1. MultiScaleSystem has different structure:
#    - system.agents is Dict[int, List[HierarchicalAgent]], not flat list
#    - Access agents via: system.agents[scale][local_index]
#    - Use: system.get_all_active_agents() for all active agents
#
# 2. No built-in energy computation:
#    - MultiScaleSystem doesn't have compute_free_energy()
#    - Need separate energy system or use hierarchical_evolution.py
#
# 3. Different meta-agent formation API:
#    - form_meta_agents() → form_meta_agents_at_scale()
#    - Different signature and return values
#
# 4. No emergence_events tracking:
#    - Current implementation uses condensation_events
#    - Different structure and content
#
# 5. Consensus detector API:
#    - find_consensus_clusters() expects agents list, not system
#    - Need to extract agents from system.agents[scale]
#
# For working examples of hierarchical dynamics, see:
# - meta/test_hierarchical_dynamics.py
# - meta/hierarchical_evolution.py
#
# ==============================================================================
#
# def run_emergence_experiment(...):
#     """Full emergence experiment with gradient descent."""
#     # Needs refactoring
#     pass
#
# def demonstrate_scale_separation():
#     """Demonstrate timescale separation."""
#     # Needs refactoring
#     pass
#
# def visualize_emergence(system, history):
#     """Visualize emergence process."""
#     # Needs refactoring
#     pass
#
# ==============================================================================


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HIERARCHICAL MULTI-SCALE TRANSFORMER TEST")
    print("="*70)

    # Run simple test demonstrating current API
    test_basic_multiscale_system()

    print("\n" + "="*70)
    print("ADDITIONAL RESOURCES")
    print("="*70)
    print("\nFor comprehensive hierarchical dynamics tests, see:")
    print("  - meta/test_hierarchical_dynamics.py")
    print("  - meta/hierarchical_evolution.py")
    print("\nThese files demonstrate:")
    print("  - Cross-scale prior/observation flow")
    print("  - Timescale separation (τ_ζ = 10^ζ bits)")
    print("  - Self-referential closure (Wheeler's 'it from bit')")
    print("  - Automatic consensus detection and condensation")
    print("  - Full hierarchical evolution loop")
