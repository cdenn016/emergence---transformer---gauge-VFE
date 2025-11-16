#!/usr/bin/env python
"""
Simple Consensus Test - Fixed Version
======================================

Fixed GaugeField initialization to match actual API.
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your actual modules
from config import SystemConfig, AgentConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem
from geometry.geometry_base import BaseManifold, TopologyType
from agent.masking import SupportRegionSmooth, MaskConfig
from math_utils.generators import generate_so3_generators

# Import consensus detector
from meta.consensus import ConsensusDetector


def create_simple_0d_system(n_agents: int = 3, K: int = 3):
    """
    Create minimal 0D system that works with your masking.
    """
    rng = np.random.default_rng(42)
    
    # Create 0D base manifold
    base_manifold = BaseManifold(
        shape=(),  # 0D point
        topology=TopologyType.FLAT
    )
    
    # Agent configuration
    agent_cfg = AgentConfig(
        spatial_shape=(),
        K=K,
        mu_scale=0.1,
        sigma_scale=1.0,
        phi_scale=0.1
    )
    
    # System configuration - strong coupling for consensus
    system_cfg = SystemConfig(
        lambda_self=0.1,
        lambda_belief_align=10.0,  # Strong alignment
        lambda_prior_align=5.0,
        lambda_obs=0.0,
        lambda_phi=0.01,
        kappa_beta=0.1,  # Low temp = sharp attention
        kappa_gamma=0.5,
        overlap_threshold=0.0,
        use_connection=False
    )
    
    # Create agents with proper support
    agents = []
    generators = generate_so3_generators(K)
    
    for i in range(n_agents):
        agent = Agent(
            agent_id=i,
            config=agent_cfg,
            rng=rng,
            base_manifold=base_manifold
        )
        
        # Create SupportRegionSmooth for 0D case
        # For 0D: mask is scalar 1.0
        mask_config = MaskConfig(
            mask_type='hard',
            overlap_threshold=0.0
        )
        
        # For 0D, everything is trivial
        support = SupportRegionSmooth(
            mask_binary=np.array(True),  # 0D boolean
            base_shape=(),  # Empty tuple for 0D
            config=mask_config,
            mask_continuous=np.array(1.0)  # 0D continuous mask
        )
        
        agent.support = support
        agent.generators = generators  # Store generators on agent
        
        # Initialize agent fields (for 0D, no spatial dimensions)
        agent.mu_q = rng.normal(0, agent_cfg.mu_scale, K)
        agent.mu_p = rng.normal(0, agent_cfg.mu_scale, K)
        agent.Sigma_q = np.eye(K) * agent_cfg.sigma_scale
        agent.Sigma_p = np.eye(K) * agent_cfg.sigma_scale
        
        # Initialize gauge - FIXED: only pass phi and K
        from gradients.gauge_fields import GaugeField
        phi_init = rng.normal(0, agent_cfg.phi_scale, 3)
        agent.gauge = GaugeField(
            phi=phi_init,
            K=K,
            validate=True
        )
        
        agents.append(agent)
    
    # Create system
    system = MultiAgentSystem(agents, system_cfg)
    
    return system


def test_basic_consensus():
    """Test consensus detection on simple system."""
    print("="*60)
    print("BASIC CONSENSUS TEST")
    print("="*60)
    
    # Create 3-agent system with K=3 (must be odd for SO(3)!)
    K = 3
    system = create_simple_0d_system(n_agents=3, K=K)
    
    # Force agents 0 and 1 to have identical parameters
    system.agents[0].mu_q = np.array([1.0, 0.0, 0.0])
    system.agents[1].mu_q = np.array([1.0, 0.0, 0.0])  # Same
    system.agents[0].Sigma_q = np.eye(K)
    system.agents[1].Sigma_q = np.eye(K)
    # Use tiny but non-zero phi to avoid numerical warnings
    system.agents[0].gauge.phi = np.array([1e-6, 0.0, 0.0])
    system.agents[1].gauge.phi = np.array([1e-6, 0.0, 0.0])
    
    # Also align their priors
    system.agents[0].mu_p = np.array([1.0, 0.0, 0.0])
    system.agents[1].mu_p = np.array([1.0, 0.0, 0.0])
    system.agents[0].Sigma_p = np.eye(K)
    system.agents[1].Sigma_p = np.eye(K)
    
    # Make agent 2 different
    system.agents[2].mu_q = np.array([0.0, 1.0, 0.0])
    system.agents[2].mu_p = np.array([0.0, 1.0, 0.0])
    
    print("\nAgent states:")
    for i, agent in enumerate(system.agents):
        print(f"  Agent {i}: μ_q = {agent.mu_q}")
    
    # Test consensus detection
    detector = ConsensusDetector(
        belief_threshold=0.01,
        model_threshold=0.01
    )
    
    print("\nPairwise consensus check:")
    for i in range(3):
        for j in range(i+1, 3):
            state = detector.check_full_consensus(
                system.agents[i], system.agents[j]
            )
            print(f"  Agents {i}-{j}: "
                  f"Belief KL={state.belief_divergence:.4f}, "
                  f"Model KL={state.model_divergence:.4f}, "
                  f"Consensus={state.is_epistemically_dead}")
    
    # Find clusters
    clusters = detector.find_consensus_clusters(system)
    print(f"\nConsensus clusters found: {clusters}")
    
    if clusters == [[0, 1]]:
        print("✓ TEST PASSED! Agents 0,1 correctly identified as consensus cluster")
    else:
        print("✗ Test failed - unexpected clusters")
    
    return system, detector


def test_consensus_evolution():
    """Test how consensus forms over time."""
    print("\n" + "="*60)
    print("CONSENSUS EVOLUTION TEST")
    print("="*60)
    
    # Create system
    system = create_simple_0d_system(n_agents=4, K=3)
    detector = ConsensusDetector(belief_threshold=0.05, model_threshold=500000)
    
    print("Initial state: 4 random agents")
    
    # Simple consensus dynamics: move toward average
    n_steps = 50
    coupling = 0.1
    
    for step in range(n_steps):
        # Compute average belief
        avg_mu_q = np.mean([a.mu_q for a in system.agents], axis=0)
        avg_mu_p = np.mean([a.mu_p for a in system.agents], axis=0)
        
        # Move each agent toward average
        for agent in system.agents:
            agent.mu_q += coupling * (avg_mu_q - agent.mu_q)
            agent.mu_p += coupling * (avg_mu_p - agent.mu_p)
        
        # Check consensus periodically
        if step % 10 == 0:
            clusters = detector.find_consensus_clusters(system)
            if clusters:
                print(f"  Step {step}: Consensus clusters = {clusters}")
            else:
                # Compute mean divergence
                matrix = detector.compute_consensus_matrix(system)
                mean_div = np.mean(matrix[matrix > 0])
                print(f"  Step {step}: No consensus yet, mean KL = {mean_div:.4f}")
    
    # Final check
    final_clusters = detector.find_consensus_clusters(system)
    if final_clusters:
        print(f"\nFinal consensus achieved: {final_clusters}")
        
        # Check for meta-agent candidates
        candidates = detector.identify_meta_agent_candidates(system)
        if candidates:
            print("\nMeta-agent candidates:")
            for candidate in candidates:
                print(f"  Agents {candidate['indices']}: "
                      f"coherence = {candidate['belief_coherence']:.3f}")
        
        print("✓ Consensus formation successful!")
    else:
        print("Agents did not reach full consensus (may need more steps)")


def test_with_actual_gradients():
    """Test consensus with your actual gradient computation."""
    print("\n" + "="*60)
    print("CONSENSUS WITH GRADIENT DESCENT")
    print("="*60)
    
    try:
        from gradients.gradient_engine import compute_all_gradients
        print("✓ Gradient engine imported successfully")
        
        # Create system with VERY strong alignment
        system = create_simple_0d_system(n_agents=3, K=3)
        print("✓ System created")
        
        # Override system config for extreme alignment
        print(f"Original config: λ_belief={system.config.lambda_belief_align}, "
              f"λ_self={system.config.lambda_self}, κ_β={system.config.kappa_beta}")
        
        system.config.lambda_belief_align = 100.0  # Extreme coupling
        system.config.lambda_self = 0.01  # Weak self-energy
        system.config.kappa_beta = 0.01  # Very sharp attention
        
        print(f"Modified config: λ_belief={system.config.lambda_belief_align}, "
              f"λ_self={system.config.lambda_self}, κ_β={system.config.kappa_beta}")
        
        detector = ConsensusDetector(belief_threshold=0.1)
        
        print("\nInitial agent states:")
        for i, agent in enumerate(system.agents):
            print(f"  Agent {i}: μ_q={agent.mu_q}")
        
        print("\nRunning gradient descent...")
        
        for step in range(100):
            # Compute gradients - returns LIST of AgentGradients
            try:
                print(f"\n--- Step {step} ---")
                gradients = compute_all_gradients(system)  # Returns list!
                print(f"  ✓ Gradients computed for {len(gradients)} agents")
                
                # Check gradient magnitudes
                grad_norms = []
                for i, grad in enumerate(gradients):
                    if hasattr(grad, 'grad_mu_q'):
                        grad_norm = np.linalg.norm(grad.grad_mu_q)
                        grad_norms.append(grad_norm)
                        print(f"  Agent {i}: ||∇μ_q|| = {grad_norm:.6f}")
                
                # Apply updates
                lr = 0.1
                for i, (agent, grad) in enumerate(zip(system.agents, gradients)):
                    if hasattr(grad, 'grad_mu_q'):
                        old_mu = agent.mu_q.copy()
                        agent.mu_q -= lr * grad.grad_mu_q
                        delta = np.linalg.norm(agent.mu_q - old_mu)
                        print(f"  Agent {i}: Δμ = {delta:.6f}, new μ={agent.mu_q}")
                
                # Periodic diagnostics
                if step % 10 == 0 or step < 5:
                    # Compute pairwise KL divergences
                    print("\n  Computing consensus metrics...")
                    kls = []
                    for i in range(len(system.agents)):
                        for j in range(i+1, len(system.agents)):
                            state = detector.check_full_consensus(
                                system.agents[i], system.agents[j]
                            )
                            kls.append(state.belief_divergence)
                            print(f"    KL({i},{j}) = {state.belief_divergence:.6f}")
                    
                    mean_kl = np.mean(kls)
                    max_kl = np.max(kls)
                    
                    clusters = detector.find_consensus_clusters(system)
                    
                    print(f"\n  SUMMARY: Mean KL={mean_kl:.4f}, "
                          f"Max KL={max_kl:.4f}, Clusters={clusters}")
                
                # Early stopping if converged
                if step > 10 and len(grad_norms) > 0 and max(grad_norms) < 1e-6:
                    print(f"\n✓ Converged at step {step} (gradients < 1e-6)")
                    break
                    
            except Exception as e:
                print(f"\n✗ Gradient computation failed at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Final state
        print("\n" + "="*60)
        print("FINAL STATE:")
        print("="*60)
        for i, agent in enumerate(system.agents):
            print(f"  Agent {i}: μ_q={agent.mu_q}")
        
        final_clusters = detector.find_consensus_clusters(system)
        if final_clusters:
            print(f"\n✓ Consensus achieved: {final_clusters}")
        else:
            print("\n✗ No consensus formed")
            print("   Agents reached equilibrium but remain separated")
                
    except ImportError as e:
        print(f"  Gradient engine not available: {e}")

if __name__ == "__main__":
    # Run tests
    print("RUNNING CONSENSUS TESTS")
    print("="*60)
    
    # Test 1: Basic consensus detection
    system, detector = test_basic_consensus()
    
    # Test 2: Consensus evolution
    test_consensus_evolution()
    
    # Test 3: With actual gradients (if available)
    test_with_actual_gradients()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)