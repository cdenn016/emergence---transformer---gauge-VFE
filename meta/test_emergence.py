# -*- coding: utf-8 -*-
"""
Test Meta-Agent Emergence
=========================

NOTE: This file has been updated to use the new MultiScaleSystem API.
The old test functions are commented out pending full refactoring.

The new MultiScaleSystem provides:
- Direct hierarchical multi-scale structure
- Automatic cross-scale dynamics (top-down priors, bottom-up observations)
- Timescale separation with information accumulation
- Self-referential closure (Wheeler's "it from bit")

For working examples, see:
- meta/test_hierarchical_dynamics.py (comprehensive integration tests)
- meta/hierarchical_evolution.py (evolution engine)

Author: Chris & Christine
Updated: Nov 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Import base system components
from config import SystemConfig, AgentConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem
from geometry.geometry_base import BaseManifold, TopologyType, create_full_support
from math_utils.generators import generate_so3_generators

# Import consensus detection
from meta.consensus import ConsensusDetector

# Import emergence machinery
from meta.emergence import (
    MultiScaleSystem,
    HierarchicalAgent,
    analyze_hierarchical_structure
)

# Import gradient computation
from gradients.gradient_engine import compute_all_gradients as compute_gradients_dict
from math_utils.fisher_metric import natural_gradient_gaussian
from retraction import retract_spd


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


def run_emergence_experiment(
    n_steps: int = 100,
    consensus_check_interval: int = 10,
    consensus_threshold: float = 0.05,
    min_cluster_size: int = 2,
    lr_mu: float = 0.05,
    lr_sigma: float = 0.001,
    lr_phi: float = 0.05
):
    """
    Run full emergence experiment.
    
    Watches for consensus formation and creates meta-agents dynamically.
    """
    print("="*70)
    print("META-AGENT EMERGENCE EXPERIMENT")
    print("="*70)
    
    # Create hierarchical multi-scale system
    print("\nInitializing hierarchical system with 4 agents...")
    system, rng = create_emergence_system(n_agents=4, K=3)
    
    # Create consensus detector
    detector = ConsensusDetector(
        belief_threshold=consensus_threshold,
        model_threshold=consensus_threshold,
        use_symmetric_kl=True
    )
    
    # Storage
    history = {
        'energy': [],
        'emergence_events': [],
        'structure_metrics': [],
        'consensus_matrix': []
    }
    
    print(f"\nRunning {n_steps} steps with emergence detection...")
    print("-"*70)
    
    for step in range(n_steps):
        # 1. Compute energy
        try:
            energy_dict = base_system.compute_free_energy()
            energy = energy_dict.get('total', 0.0)
        except:
            energy = 0.0
        history['energy'].append(energy)
        
        # 2. Check for consensus and form meta-agents
        if step % consensus_check_interval == 0 and step > 0:
            # Find consensus clusters
            clusters = detector.find_consensus_clusters(base_system)
            
            # Filter for minimum size
            valid_clusters = [c for c in clusters if len(c) >= min_cluster_size]
            
            if valid_clusters:
                print(f"\nStep {step}: Found {len(valid_clusters)} consensus clusters!")
                
                # Compute coherence scores
                coherence_scores = []
                for cluster in valid_clusters:
                    # Sample pairwise divergences
                    belief_divs = []
                    model_divs = []
                    for i in range(len(cluster)-1):
                        state = detector.check_full_consensus(
                            base_system.agents[cluster[i]],
                            base_system.agents[cluster[i+1]]
                        )
                        belief_divs.append(state.belief_divergence)
                        model_divs.append(state.model_divergence)
                    
                    coherence_scores.append({
                        'belief_coherence': 1.0 / (1.0 + np.mean(belief_divs)),
                        'model_coherence': 1.0 / (1.0 + np.mean(model_divs))
                    })
                
                # Form meta-agents
                new_meta = h_system.form_meta_agents(valid_clusters, coherence_scores)
                
                for meta_agent in new_meta:
                    print(f"  Created Meta-{meta_agent.meta.agent_id}: "
                          f"scale={meta_agent.scale}, "
                          f"constituents={meta_agent.meta.constituent_ids}, "
                          f"timescale={meta_agent.meta.characteristic_timescale:.0f}x")
                
                # Add new meta-agents to base system for gradient computation
                base_system.agents.extend(new_meta)
                base_system.n_agents = len(base_system.agents)
        
        # 3. Record structure metrics
        if step % 50 == 0:
            metrics = analyze_hierarchical_structure(h_system)
            history['structure_metrics'].append((step, metrics))
            
            if metrics['n_scales'] > 1:
                print(f"\nStep {step}: Hierarchical structure:")
                for scale in range(metrics['n_scales']):
                    if scale in metrics['active_per_scale']:
                        print(f"  Scale {scale}: {metrics['active_per_scale'][scale]} active agents")
        
        # 4. Compute and apply gradients
        active_agents = h_system.get_active_agents()
        
        if len(active_agents) > 0:
            # Compute gradients
            try:
                all_gradients = compute_gradients_dict(base_system)
                
                # Apply updates with scale-dependent learning rates
                for agent in active_agents:
                    if agent.agent_id not in all_gradients:
                        continue
                    
                    grads = all_gradients[agent.agent_id]
                    
                    # Scale learning rate for meta-agents
                    lr_scale = agent.effective_learning_rate_scale
                    
                    # Update parameters
                    if hasattr(grads, 'grad_mu_q') and grads.grad_mu_q is not None:
                        agent.mu_q -= lr_mu * lr_scale * grads.grad_mu_q
                    
                    if hasattr(grads, 'grad_Sigma_q') and grads.grad_Sigma_q is not None:
                        _, nat_grad = natural_gradient_gaussian(
                            agent.mu_q, agent.Sigma_q,
                            np.zeros_like(agent.mu_q), grads.grad_Sigma_q
                        )
                        try:
                            agent.Sigma_q = retract_spd(
                                agent.Sigma_q, -lr_sigma * lr_scale * nat_grad
                            )
                        except:
                            pass
                    
                    if hasattr(grads, 'grad_phi') and grads.grad_phi is not None:
                        agent.gauge.phi -= lr_phi * lr_scale * grads.grad_phi
                        agent.gauge.phi = np.clip(agent.gauge.phi, -np.pi, np.pi)
            except Exception as e:
                # Fallback: simple consensus dynamics
                if step % 10 == 0:
                    avg_mu = np.mean([a.mu_q for a in active_agents], axis=0)
                    for agent in active_agents:
                        lr = 0.01 * agent.effective_learning_rate_scale
                        agent.mu_q += lr * (avg_mu - agent.mu_q)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    # Final summary
    print("\n" + h_system.summary())
    
    # Analyze emergence patterns
    if h_system.emergence_events:
        print(f"\n{len(h_system.emergence_events)} emergence events occurred:")
        for event in h_system.emergence_events[:5]:  # Show first 5
            print(f"  Step {event['time']}: "
                  f"Scale {event['scale']} meta-agent from {event['constituents']}")
    
    final_metrics = analyze_hierarchical_structure(h_system)
    print(f"\nFinal structure:")
    print(f"  Scales: {final_metrics['n_scales']}")
    print(f"  Mean cluster size: {final_metrics['mean_cluster_size']:.1f}")
    print(f"  Max hierarchy depth: {final_metrics['max_hierarchy_depth']}")
    
    return h_system, history


def demonstrate_scale_separation():
    """
    Demonstrate time-scale separation in hierarchical system.
    """
    print("\n" + "="*70)
    print("TIME-SCALE SEPARATION DEMONSTRATION")
    print("="*70)
    
    # Create simple 3-agent system
    base_system, _ = create_emergence_system(n_agents=3, K=3)
    h_system = HierarchicalMultiAgentSystem(base_system)
    
    # Get K from the agents
    K = base_system.agents[0].K
    
    # Force agents 0,1 into consensus to form meta-agent
    base_system.agents[0].mu_q = np.zeros(K)
    base_system.agents[0].mu_q[0] = 1.0
    base_system.agents[1].mu_q = base_system.agents[0].mu_q.copy()
    base_system.agents[0].Sigma_q = np.eye(K)
    base_system.agents[1].Sigma_q = np.eye(K)
    
    # Create meta-agent
    meta_agents = h_system.form_meta_agents([[0, 1]])
    meta = meta_agents[0]
    
    print(f"\nCreated meta-agent from agents [0,1]")
    print(f"  Meta-agent scale: {meta.scale}")
    print(f"  Characteristic timescale: {meta.meta.characteristic_timescale:.0f}x base")
    print(f"  Effective learning rate scale: {meta.effective_learning_rate_scale:.2e}")
    
    # Simulate dynamics
    print("\nSimulating 100 steps:")
    print("  Base agent 2: full learning rate")
    print(f"  Meta-agent: {meta.effective_learning_rate_scale:.2e}x learning rate")
    
    mu_base_history = []
    mu_meta_history = []
    
    # ⚡ FIX: Target with correct dimension K
    target = np.zeros(K)
    target[1] = 1.0  # Set second component to 1.0
    
    for step in range(100):
        # Update base agent quickly
        base_agent = base_system.agents[2]
        base_agent.mu_q += 0.05 * (target - base_agent.mu_q)
        
        # Update meta-agent slowly
        meta.mu_q += 0.05 * meta.effective_learning_rate_scale * (target - meta.mu_q)
        
        if step % 20 == 0:
            base_dist = np.linalg.norm(base_agent.mu_q - target)
            meta_dist = np.linalg.norm(meta.mu_q - target)
            print(f"  Step {step:3d}: Base dist={base_dist:.3f}, Meta dist={meta_dist:.3f}")
        
        mu_base_history.append(np.linalg.norm(base_agent.mu_q - target))
        mu_meta_history.append(np.linalg.norm(meta.mu_q - target))
    
    print("\nTime-scale separation demonstrated:")
    print(f"  Base agent converged to {np.linalg.norm(base_agent.mu_q - target):.4f}")
    print(f"  Meta-agent at {np.linalg.norm(meta.mu_q - target):.4f} (10^4x slower)")
    
    return mu_base_history, mu_meta_history

def visualize_emergence(h_system, history):
    """Visualize the emergence process."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Energy evolution
        ax = axes[0, 0]
        ax.plot(history['energy'], 'b-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Free Energy')
        ax.set_title('Energy Minimization')
        
        # Mark emergence events
        for event in h_system.emergence_events:
            ax.axvline(x=event['time'], color='r', alpha=0.3, linestyle='--')
        
        # 2. Number of active agents over time
        ax = axes[0, 1]
        steps = [m[0] for m in history['structure_metrics']]
        n_active = [sum(m[1]['active_per_scale'].values()) 
                   for m in history['structure_metrics']]
        ax.plot(steps, n_active, 'go-', linewidth=2, markersize=6)
        ax.set_xlabel('Step')
        ax.set_ylabel('Active Agents')
        ax.set_title('Agent Count (decreases as meta-agents form)')
        
        # 3. Hierarchical structure
        ax = axes[1, 0]
        if history['structure_metrics']:
            final_metrics = history['structure_metrics'][-1][1]
            scales = list(final_metrics['agents_per_scale'].keys())
            counts = list(final_metrics['agents_per_scale'].values())
            ax.bar(scales, counts, color='blue', alpha=0.5, label='Total')
            
            active_counts = [final_metrics['active_per_scale'].get(s, 0) for s in scales]
            ax.bar(scales, active_counts, color='green', alpha=0.8, label='Active')
            
            ax.set_xlabel('Scale ζ')
            ax.set_ylabel('Number of Agents')
            ax.set_title('Final Hierarchical Structure')
            ax.legend()
        
        # 4. Emergence timeline
        ax = axes[1, 1]
        if h_system.emergence_events:
            times = [e['time'] for e in h_system.emergence_events]
            scales = [e['scale'] for e in h_system.emergence_events]
            ax.scatter(times, scales, s=100, c='red', marker='*')
            ax.set_xlabel('Time (step)')
            ax.set_ylabel('Emergent Scale')
            ax.set_title('Meta-Agent Formation Events')
            ax.set_ylim([0.5, max(scales) + 0.5])
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


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

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return system


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HIERARCHICAL MULTI-SCALE TRANSFORMER TEST")
    print("="*70)

    # Run simple test demonstrating current API
    test_basic_multiscale_system()

    print("\nNOTE: Full emergence experiment (run_emergence_experiment) is")
    print("commented out pending refactoring to new MultiScaleSystem API.")
    print("\nFor comprehensive hierarchical dynamics tests, see:")
    print("  - meta/test_hierarchical_dynamics.py")
    print("  - meta/hierarchical_evolution.py")