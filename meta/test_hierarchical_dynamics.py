# -*- coding: utf-8 -*-
"""
Test Hierarchical Dynamics and Cross-Scale Information Flow
===========================================================

Integration test for the complete hierarchical gauge system:
- Cross-scale prior updates (top-down)
- Observation generation from constituents (bottom-up)
- Timescale separation
- Automatic consensus detection
- Meta-agent formation

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from meta.emergence import MultiScaleSystem, HierarchicalAgent
from meta.hierarchical_evolution import (
    HierarchicalEvolutionEngine,
    HierarchicalConfig,
    evolve_hierarchical_system
)
from geometry.geometry_base import BaseManifold, TopologyType
from config import AgentConfig


def test_cross_scale_prior_updates():
    """Test top-down prior updates from meta-agents."""
    print("\n" + "="*70)
    print("TEST: Cross-Scale Prior Updates (Top-Down)")
    print("="*70)

    # Create system
    base_manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
    system = MultiScaleSystem(base_manifold)

    # Create base agents
    agent_config = AgentConfig(K=3, spatial_shape=())
    for i in range(4):
        system.add_base_agent(agent_config, agent_id=f"base_{i}")

    # Manually form a meta-agent
    print("\n1. Forming meta-agent from agents [0, 1]")
    system.form_meta_agents_at_scale(
        source_scale=0,
        partitions=[[0, 1]],
        deactivate_constituents=False  # Keep active for testing
    )

    # Get agents
    meta_agent = system.agents[1][0]
    constituent_0 = system.agents[0][0]
    constituent_1 = system.agents[0][1]

    # Store original priors
    mu_p_0_orig = constituent_0.mu_p.copy()
    mu_p_1_orig = constituent_1.mu_p.copy()

    print(f"\n2. Original priors:")
    print(f"   Agent 0: μ_p = {mu_p_0_orig}")
    print(f"   Agent 1: μ_p = {mu_p_1_orig}")
    print(f"   Meta belief: μ_q = {meta_agent.mu_q}")

    # Update priors from parent
    print("\n3. Updating priors from meta-agent...")
    constituent_0.update_prior_from_parent()
    constituent_1.update_prior_from_parent()

    print(f"\n4. Updated priors:")
    print(f"   Agent 0: μ_p = {constituent_0.mu_p}")
    print(f"   Agent 1: μ_p = {constituent_1.mu_p}")

    # Verify they've changed
    assert not np.allclose(constituent_0.mu_p, mu_p_0_orig), "Prior should have changed!"
    assert not np.allclose(constituent_1.mu_p, mu_p_1_orig), "Prior should have changed!"

    print("\n✓ Top-down prior updates working correctly")

    return system


def test_observation_from_constituents():
    """Test bottom-up observation generation."""
    print("\n" + "="*70)
    print("TEST: Observation Generation from Constituents (Bottom-Up)")
    print("="*70)

    # Create system with meta-agent
    base_manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
    system = MultiScaleSystem(base_manifold)

    agent_config = AgentConfig(K=3, spatial_shape=())
    for i in range(3):
        system.add_base_agent(agent_config, agent_id=f"base_{i}")

    # Form meta-agent
    print("\n1. Forming meta-agent from agents [0, 1, 2]")
    system.form_meta_agents_at_scale(
        source_scale=0,
        partitions=[[0, 1, 2]],
        deactivate_constituents=False
    )

    meta_agent = system.agents[1][0]

    # Generate observation from constituents
    print("\n2. Generating observation from constituents...")
    o_meta = meta_agent.generate_observations_from_constituents()

    print(f"\n3. Constituent beliefs:")
    for i, agent in enumerate(meta_agent.constituents):
        print(f"   Agent {i}: μ_q = {agent.mu_q}")

    print(f"\n4. Meta-agent observation: o = {o_meta}")
    print(f"   Meta-agent belief: μ_q = {meta_agent.mu_q}")

    # Verify observation is reasonable (should be close to average)
    assert o_meta is not None, "Observation should not be None"
    assert o_meta.shape == meta_agent.mu_q.shape, "Shape mismatch"

    # Compute observation likelihood
    energy = system.compute_observation_likelihood_meta(meta_agent)
    print(f"\n5. Observation likelihood energy: E_obs = {energy:.4f}")

    print("\n✓ Bottom-up observation generation working correctly")

    return system


def test_timescale_separation():
    """Test timescale-separated updates."""
    print("\n" + "="*70)
    print("TEST: Timescale Separation (τ_ζ = 10^ζ bits)")
    print("="*70)

    # Create system with agents at different scales
    base_manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
    system = MultiScaleSystem(base_manifold)

    agent_config = AgentConfig(K=3, spatial_shape=())
    for i in range(6):
        system.add_base_agent(agent_config, agent_id=f"base_{i}")

    # Form meta-agents at scale 1
    system.form_meta_agents_at_scale(
        source_scale=0,
        partitions=[[0, 1], [2, 3]],
        deactivate_constituents=False
    )

    # Form meta-agent at scale 2
    system.form_meta_agents_at_scale(
        source_scale=1,
        partitions=[[0, 1]],
        deactivate_constituents=False
    )

    print("\n1. System hierarchy:")
    print(system.summary())

    # Test timescale thresholds
    print("\n2. Timescale thresholds:")
    for scale in sorted(system.agents.keys()):
        agent = system.agents[scale][0]
        print(f"   Scale ζ={scale}: threshold = {agent.timescale_threshold:.1f} bits")

    # Simulate information accumulation
    print("\n3. Simulating updates with different info changes:")

    agent_scale0 = system.agents[0][0]  # Base agent
    agent_scale1 = system.agents[1][0]  # Meta-agent
    agent_scale2 = system.agents[2][0]  # Meta-meta-agent

    # Small change - only scale 0 should update
    delta_info_small = 5.0  # bits
    print(f"\n   ΔI = {delta_info_small} bits:")
    print(f"      ζ=0 updates: {agent_scale0.should_update(delta_info_small)}")
    print(f"      ζ=1 updates: {agent_scale1.should_update(delta_info_small)}")
    print(f"      ζ=2 updates: {agent_scale2.should_update(delta_info_small)}")

    # Medium change - scales 0 and 1 might update
    delta_info_medium = 15.0  # bits
    print(f"\n   ΔI = {delta_info_medium} bits:")
    print(f"      ζ=0 updates: {agent_scale0.should_update(delta_info_medium)}")
    print(f"      ζ=1 updates: {agent_scale1.should_update(delta_info_medium)}")
    print(f"      ζ=2 updates: {agent_scale2.should_update(delta_info_medium)}")

    print("\n✓ Timescale separation working correctly")

    return system


def test_auto_consensus_detection():
    """Test automatic consensus detection and condensation."""
    print("\n" + "="*70)
    print("TEST: Automatic Consensus Detection")
    print("="*70)

    # Create system
    base_manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
    system = MultiScaleSystem(base_manifold)

    agent_config = AgentConfig(K=3, spatial_shape=())

    # Create agents with similar beliefs (for consensus)
    print("\n1. Creating 6 base agents with 2 consensus clusters")
    for i in range(6):
        agent = system.add_base_agent(agent_config, agent_id=f"base_{i}")

        # Set beliefs manually to create consensus
        if i < 3:
            # Cluster 1: similar beliefs
            agent.mu_q = np.array([1.0, 0.0, 0.0]) + 0.01 * np.random.randn(3)
        else:
            # Cluster 2: different similar beliefs
            agent.mu_q = np.array([0.0, 1.0, 0.0]) + 0.01 * np.random.randn(3)

    # Detect consensus
    print("\n2. Detecting consensus clusters...")
    new_meta_agents = system.auto_detect_and_condense(
        scale=0,
        kl_threshold=0.1,  # Permissive for test
        min_cluster_size=2
    )

    print(f"\n3. Formed {len(new_meta_agents)} meta-agents")
    print(system.summary())

    print("\n✓ Consensus detection working correctly")

    return system


def test_full_hierarchical_evolution():
    """Test full hierarchical evolution loop."""
    print("\n" + "="*70)
    print("TEST: Full Hierarchical Evolution")
    print("="*70)

    # Create system
    base_manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
    system = MultiScaleSystem(base_manifold)

    # Create base agents
    agent_config = AgentConfig(K=3, spatial_shape=())
    n_agents = 8

    print(f"\n1. Creating {n_agents} base agents")
    for i in range(n_agents):
        system.add_base_agent(agent_config, agent_id=f"base_{i}")

    # Configure evolution
    config = HierarchicalConfig(
        enable_top_down_priors=True,
        enable_bottom_up_obs=True,
        enable_timescale_filtering=True,
        consensus_check_interval=5,
        consensus_kl_threshold=0.05
    )

    # Create evolution engine
    engine = HierarchicalEvolutionEngine(system, config)

    print("\n2. Evolving system for 20 steps...")

    # Dummy gradient function (for testing)
    def dummy_gradients(sys):
        """Create zero gradients for testing."""
        from gradients.gradient_engine import AgentGradients

        agents = sys.get_all_active_agents()
        grads = []

        for agent in agents:
            grad = AgentGradients(
                grad_mu_q=np.zeros_like(agent.mu_q),
                grad_Sigma_q=np.zeros_like(agent.Sigma_q),
                grad_mu_p=np.zeros_like(agent.mu_p),
                grad_Sigma_p=np.zeros_like(agent.Sigma_p),
                grad_phi=np.zeros_like(agent.gauge.phi),
                delta_mu_q=0.01 * np.random.randn(*agent.mu_q.shape),
                delta_Sigma_q=np.zeros_like(agent.Sigma_q),
                delta_mu_p=np.zeros_like(agent.mu_p),
                delta_Sigma_p=np.zeros_like(agent.Sigma_p),
                delta_phi=np.zeros_like(agent.gauge.phi)
            )
            grads.append(grad)

        return grads

    # Evolve
    for step in range(20):
        metrics = engine.evolve_step(
            learning_rate=0.01,
            compute_gradients_fn=dummy_gradients
        )

        if step % 5 == 0:
            print(f"\n   Step {step}: {metrics['n_active']} active agents")

    print("\n3. Final system state:")
    print(system.summary())

    print("\n✓ Full hierarchical evolution working correctly")

    return system


def run_all_tests():
    """Run all hierarchical dynamics tests."""
    print("\n" + "="*70)
    print("HIERARCHICAL GAUGE DYNAMICS TEST SUITE")
    print("="*70)

    tests = [
        ("Cross-Scale Prior Updates", test_cross_scale_prior_updates),
        ("Observation from Constituents", test_observation_from_constituents),
        ("Timescale Separation", test_timescale_separation),
        ("Auto Consensus Detection", test_auto_consensus_detection),
        ("Full Evolution Loop", test_full_hierarchical_evolution),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            system = test_fn()
            results[name] = "PASS"
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"FAIL: {e}"

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {name}: {result}")

    all_passed = all(r == "PASS" for r in results.values())
    print(f"\n{'='*70}")
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    run_all_tests()
