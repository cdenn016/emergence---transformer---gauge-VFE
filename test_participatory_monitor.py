#!/usr/bin/env python3
"""
Test script for Participatory Monitor

Demonstrates usage of the participatory dynamics monitoring utility.
"""

import numpy as np
from pathlib import Path

from config import SystemConfig, AgentConfig, TrainingConfig
from geometry.geometry_base import BaseManifold, create_full_support, TopologyType
from agent.masking import MaskConfig
from meta.emergence import MultiScaleSystem, HierarchicalAgent
from meta.consensus import ConsensusDetector
from meta.hierarchical_evolution import evolve_hierarchical
from math_utils.generators import generate_so3_generators
from participatory_monitor import ParticipatoryMonitor


def test_participatory_monitor():
    """Test the participatory dynamics monitor"""

    print("="*70)
    print("TESTING PARTICIPATORY 'IT FROM BIT' MONITOR")
    print("="*70)

    # Create a simple system for testing
    K = 3  # Latent dimension

    # Configuration
    system_cfg = SystemConfig(
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=1.0,
        lambda_obs=1.0,
        lambda_phi=1.0,
        identical_priors="off"
    )

    agent_cfg = AgentConfig(
        K=K,
        observation_noise=0.1,
        mask_config=MaskConfig(mask_type="gaussian")
    )

    training_cfg = TrainingConfig(
        n_steps=100,
        lr_mu_q=0.01,
        lr_sigma_q=0.005,
        lr_phi=0.001,
    )

    # Create point manifold for particle agents (no spatial dimensions)
    # This ensures agents have simple K-dimensional fields, not spatial fields
    manifold = BaseManifold(shape=(), topology=TopologyType.PERIODIC)
    support = create_full_support(manifold)

    # Create multi-scale system with level cap
    print("\n1. Creating MultiScaleSystem with max_emergence_levels=3...")
    system = MultiScaleSystem(manifold, max_emergence_levels=3)

    # Create some base agents
    n_agents = 12
    generators = generate_so3_generators(K)

    print(f"2. Adding {n_agents} base agents at scale 0...")
    for i in range(n_agents):
        agent = system.add_base_agent(agent_cfg, agent_id=f"agent_{i}")
        agent.support = support
        agent.generators = generators

    print(f"   Created {len(system.agents[0])} agents at scale 0")

    # Create consensus detector
    print("\n3. Creating ConsensusDetector...")
    consensus_detector = ConsensusDetector(
        belief_threshold=0.5,
        model_threshold=0.5
    )

    # Create participatory monitor
    print("\n4. Creating ParticipatoryMonitor...")
    monitor = ParticipatoryMonitor(
        system=system,
        consensus_detector=consensus_detector,
        check_interval=5,
        prior_change_threshold=1e-4,
        non_eq_threshold=1e-3
    )

    # Simulate some evolution
    print("\n5. Running hierarchical evolution...")
    print("   (This will take a moment...)")

    # Generate some observations for agents
    rng = np.random.default_rng(42)
    observations = rng.normal(0, 0.1, size=(n_agents, K))

    # Run evolution with monitoring
    num_steps = 50
    for step in range(num_steps):
        # Take snapshot periodically
        snapshot = monitor.take_snapshot(step)

        if snapshot is not None:
            print(f"   Step {step}: {snapshot.num_agents_per_scale.get(0, 0)} base agents, "
                  f"{snapshot.num_meta_agents_formed} meta-agents")

        # Simple update step (just for testing - not full evolution)
        # In real usage, this would be the full hierarchical evolution loop
        for scale, agents in system.agents.items():
            for agent in agents:
                if agent.is_active:
                    # Simple belief update (just add noise for testing)
                    agent.mu_q += rng.normal(0, 0.001, size=K)

                    # Simple prior update (for testing prior evolution)
                    if agent.parent_meta is not None:
                        agent.mu_p += rng.normal(0, 0.002, size=K)

        # Try to form meta-agents at intervals (manually create clusters for testing)
        if step % 20 == 10 and step > 0:
            print(f"\n   Attempting condensation at step {step}...")
            # Manually create some clusters for testing (not using consensus detection)
            if 0 in system.agents and len(system.agents[0]) >= 4:
                # Group agents into pairs for testing
                clusters = [[0, 1], [2, 3]]
                print(f"   Creating {len(clusters)} test clusters")
                system.form_meta_agents_at_scale(0, clusters)
                print(f"   Formed {len(system.agents.get(1, []))} meta-agents at scale 1")

    # Final snapshot
    print("\n6. Taking final snapshot...")
    monitor.take_snapshot(num_steps, force=True)

    # Print comprehensive summary
    print("\n7. Validation Results:")
    monitor.print_summary(max_levels=3)

    # Get detailed validation
    validation = monitor.validate_participatory_dynamics(max_levels=3)

    print("\n8. Detailed Results:")
    print(f"   Overall Status: {validation['overall_status']}")
    print(f"   Samples Collected: {validation['samples_collected']}")

    # Test individual analyses
    print("\n9. Component Analysis:")

    condensation = monitor.analyze_condensation()
    print(f"\n   Condensation:")
    print(f"     Status: {condensation['status']}")
    print(f"     Condensation occurring: {condensation.get('condensation_occurring', False)}")
    print(f"     Total meta-agents: {condensation.get('total_meta_agents', 0)}")

    prior_evolution = monitor.analyze_prior_evolution()
    print(f"\n   Prior Evolution:")
    print(f"     Status: {prior_evolution['status']}")
    print(f"     Top-down influence: {prior_evolution.get('top_down_influence', False)}")
    print(f"     Parent-child links: {prior_evolution.get('parent_child_links', 0)}")

    non_eq = monitor.analyze_non_equilibrium()
    print(f"\n   Non-Equilibrium:")
    print(f"     Status: {non_eq['status']}")
    if non_eq['status'] != 'insufficient_data':
        print(f"     Is non-equilibrium: {non_eq.get('is_non_equilibrium', False)}")
        print(f"     Equilibrium score: {non_eq.get('equilibrium_score', 0):.4f}")

    level_cap = monitor.check_level_cap(max_levels=3)
    print(f"\n   Level Cap:")
    print(f"     Status: {level_cap['status']}")
    print(f"     Max levels allowed: {level_cap.get('max_levels_allowed', 'N/A')}")
    print(f"     Max scale reached: {level_cap.get('max_scale_reached', 0)}")
    print(f"     Level cap hit: {level_cap.get('level_cap_hit', False)}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return monitor, system


if __name__ == "__main__":
    monitor, system = test_participatory_monitor()

    print("\nTo use this monitor in your own code:")
    print("  1. Create a MultiScaleSystem with max_emergence_levels set")
    print("  2. Create a ParticipatoryMonitor instance")
    print("  3. Call monitor.take_snapshot(step) in your evolution loop")
    print("  4. Call monitor.print_summary() or monitor.validate_participatory_dynamics()")
    print("     to check the status of participatory dynamics")
