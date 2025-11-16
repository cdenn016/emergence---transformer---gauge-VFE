# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 11:39:18 2025

@author: chris and christine
"""

#!/usr/bin/env python
"""
Simplified Consensus Test - Minimal Dependencies
================================================

A simpler test that creates mock agents to test consensus detection
without needing the full simulation infrastructure.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys
import os

# Add meta_agent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import just the consensus detector
from meta.consensus import ConsensusDetector, ConsensusState


@dataclass 
class MockGauge:
    """Mock gauge field"""
    phi: np.ndarray


@dataclass
class MockAgent:
    """Minimal agent for testing consensus"""
    agent_id: int
    mu_q: np.ndarray      # Belief mean
    Sigma_q: np.ndarray   # Belief covariance
    mu_p: np.ndarray      # Model mean  
    Sigma_p: np.ndarray   # Model covariance
    gauge: MockGauge      # Gauge field
    generators: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Set up SO(3) generators if not provided"""
        if self.generators is None:
            K = len(self.mu_q)
            # Simple generators for testing (not full SO(3), but sufficient)
            self.generators = np.zeros((3, K, K))
            if K >= 3:
                # Basic rotation generators
                self.generators[0, 0, 1] = 1
                self.generators[0, 1, 0] = -1
                self.generators[1, 0, 2] = 1 if K > 2 else 0
                self.generators[1, 2, 0] = -1 if K > 2 else 0
                self.generators[2, 1, 2] = 1 if K > 2 else 0
                self.generators[2, 2, 1] = -1 if K > 2 else 0


class MockSystem:
    """Minimal system for testing"""
    def __init__(self, agents):
        self.agents = agents
        self.n_agents = len(agents)


def create_consensus_groups():
    """
    Create agents with predetermined consensus groups.
    
    Returns:
        system: MockSystem with 8 agents in various consensus states
        expected_clusters: What clusters we expect to find
    """
    K = 3  # Latent dimension
    agents = []
    
    # Group 1: Agents 0, 1, 2 - perfect consensus
    shared_mu_1 = np.array([1.0, 0.5, -0.5])
    shared_sigma_1 = np.eye(K) * 0.5
    shared_phi_1 = np.array([0.1, 0.0, 0.0])
    
    for i in [0, 1, 2]:
        agent = MockAgent(
            agent_id=i,
            mu_q=shared_mu_1 + np.random.normal(0, 0.0001, K),  # Tiny variation
            Sigma_q=shared_sigma_1,
            mu_p=shared_mu_1 + np.random.normal(0, 0.0001, K),
            Sigma_p=shared_sigma_1,
            gauge=MockGauge(shared_phi_1 + np.random.normal(0, 0.0001, 3))
        )
        agents.append(agent)
    
    # Group 2: Agents 3, 4 - consensus  
    shared_mu_2 = np.array([-0.5, 1.0, 0.0])
    shared_sigma_2 = np.eye(K) * 0.8
    shared_phi_2 = np.array([0.0, 0.2, 0.0])
    
    for i in [3, 4]:
        agent = MockAgent(
            agent_id=i,
            mu_q=shared_mu_2 + np.random.normal(0, 0.0001, K),
            Sigma_q=shared_sigma_2,
            mu_p=shared_mu_2 + np.random.normal(0, 0.0001, K),
            Sigma_p=shared_sigma_2,
            gauge=MockGauge(shared_phi_2 + np.random.normal(0, 0.0001, 3))
        )
        agents.append(agent)
    
    # Agents 5, 6, 7 - independent (no consensus)
    for i in [5, 6, 7]:
        agent = MockAgent(
            agent_id=i,
            mu_q=np.random.normal(0, 1, K),
            Sigma_q=np.eye(K) * np.random.uniform(0.5, 1.5),
            mu_p=np.random.normal(0, 1, K),
            Sigma_p=np.eye(K) * np.random.uniform(0.5, 1.5),
            gauge=MockGauge(np.random.normal(0, 0.5, 3))
        )
        agents.append(agent)
    
    system = MockSystem(agents)
    expected_clusters = [[0, 1, 2], [3, 4]]
    
    return system, expected_clusters


def test_pairwise_consensus():
    """Test consensus detection between agent pairs."""
    print("="*60)
    print("TESTING PAIRWISE CONSENSUS DETECTION")
    print("="*60)
    
    K = 3
    
    # Create two agents with identical beliefs
    agent1 = MockAgent(
        agent_id=0,
        mu_q=np.array([1.0, 0.0, 0.0]),
        Sigma_q=np.eye(K),
        mu_p=np.array([1.0, 0.0, 0.0]),
        Sigma_p=np.eye(K),
        gauge=MockGauge(np.zeros(3))
    )
    
    agent2 = MockAgent(
        agent_id=1,
        mu_q=np.array([1.0, 0.0, 0.0]),  # Same as agent1
        Sigma_q=np.eye(K),
        mu_p=np.array([1.0, 0.0, 0.0]),
        Sigma_p=np.eye(K),
        gauge=MockGauge(np.zeros(3))  # Same gauge frame
    )
    
    detector = ConsensusDetector(belief_threshold=0.01, model_threshold=0.01)
    
    # Test consensus
    state = detector.check_full_consensus(agent1, agent2)
    print(f"\nIdentical agents:")
    print(f"  Belief KL: {state.belief_divergence:.6f}")
    print(f"  Model KL: {state.model_divergence:.6f}")
    print(f"  Epistemic death: {state.is_epistemically_dead}")
    
    # Now test with different beliefs
    agent3 = MockAgent(
        agent_id=2,
        mu_q=np.array([0.0, 1.0, 0.0]),  # Different
        Sigma_q=np.eye(K),
        mu_p=np.array([0.0, 1.0, 0.0]),
        Sigma_p=np.eye(K),
        gauge=MockGauge(np.zeros(3))
    )
    
    state = detector.check_full_consensus(agent1, agent3)
    print(f"\nDifferent agents:")
    print(f"  Belief KL: {state.belief_divergence:.6f}")
    print(f"  Model KL: {state.model_divergence:.6f}")
    print(f"  Epistemic death: {state.is_epistemically_dead}")
    
    # Test with different gauge frames but same belief
    agent4 = MockAgent(
        agent_id=3,
        mu_q=np.array([1.0, 0.0, 0.0]),
        Sigma_q=np.eye(K),
        mu_p=np.array([1.0, 0.0, 0.0]),
        Sigma_p=np.eye(K),
        gauge=MockGauge(np.array([0.5, 0.0, 0.0]))  # Different gauge
    )
    
    state = detector.check_full_consensus(agent1, agent4)
    print(f"\nSame beliefs, different gauge frames:")
    print(f"  Belief KL: {state.belief_divergence:.6f}")
    print(f"  Model KL: {state.model_divergence:.6f}")
    print(f"  Epistemic death: {state.is_epistemically_dead}")
    print(f"  (Should show consensus after gauge transport)")


def test_cluster_detection():
    """Test consensus cluster detection."""
    print("\n" + "="*60)
    print("TESTING CLUSTER DETECTION")
    print("="*60)
    
    system, expected_clusters = create_consensus_groups()
    detector = ConsensusDetector(belief_threshold=0.01, model_threshold=0.01)
    
    print(f"\nCreated system with {system.n_agents} agents")
    print(f"Expected clusters: {expected_clusters}")
    
    # Detect clusters
    found_clusters = detector.find_consensus_clusters(system)
    print(f"Found clusters: {found_clusters}")
    
    # Verify
    def normalize_clusters(clusters):
        """Sort clusters for comparison"""
        return sorted([sorted(c) for c in clusters])
    
    expected_norm = normalize_clusters(expected_clusters)
    found_norm = normalize_clusters(found_clusters)
    
    if expected_norm == found_norm:
        print("✓ Cluster detection PASSED")
    else:
        print("✗ Cluster detection FAILED")
        print(f"  Expected: {expected_norm}")
        print(f"  Found: {found_norm}")
    
    # Show consensus matrix
    print("\nConsensus matrix (total KL divergence):")
    matrix = detector.compute_consensus_matrix(system)
    
    # Pretty print matrix with threshold coloring
    print("    ", end="")
    for j in range(system.n_agents):
        print(f"   A{j:1d}", end="")
    print()
    
    for i in range(system.n_agents):
        print(f"A{i:1d}: ", end="")
        for j in range(system.n_agents):
            if i == j:
                print("  ---", end="")
            else:
                val = matrix[i, j]
                # Color code: consensus (<0.02) vs no consensus
                if val < 0.02:
                    print(f" {val:4.3f}", end="")  # Consensus
                else:
                    print(f" {val:4.1f}", end="")  # No consensus
        print()
    
    # Test meta-agent candidates
    print("\nMeta-agent candidates:")
    candidates = detector.identify_meta_agent_candidates(system)
    
    for i, candidate in enumerate(candidates):
        print(f"  Candidate {i}: agents {candidate['indices']}")
        print(f"    Belief coherence: {candidate['belief_coherence']:.3f}")
        print(f"    Model coherence: {candidate['model_coherence']:.3f}")


def test_consensus_evolution():
    """Test how consensus evolves under simulated dynamics."""
    print("\n" + "="*60)
    print("TESTING CONSENSUS EVOLUTION")
    print("="*60)
    
    K = 3
    n_agents = 4
    n_steps = 50
    
    # Create agents with random initial conditions
    agents = []
    for i in range(n_agents):
        agent = MockAgent(
            agent_id=i,
            mu_q=np.random.normal(0, 0.5, K),
            Sigma_q=np.eye(K),
            mu_p=np.random.normal(0, 0.5, K),
            Sigma_p=np.eye(K),
            gauge=MockGauge(np.random.normal(0, 0.1, 3))
        )
        agents.append(agent)
    
    system = MockSystem(agents)
    detector = ConsensusDetector(belief_threshold=0.01, model_threshold=0.01)
    
    print(f"Initial divergences:")
    matrix = detector.compute_consensus_matrix(system)
    print(f"  Mean: {np.mean(matrix[matrix > 0]):.3f}")
    print(f"  Min: {np.min(matrix[matrix > 0]):.3f}")
    print(f"  Max: {np.max(matrix):.3f}")
    
    # Simulate consensus formation by gradually averaging beliefs
    coupling_strength = 0.1
    
    for step in range(n_steps):
        # Simple consensus dynamics: move toward average
        avg_mu_q = np.mean([a.mu_q for a in agents], axis=0)
        avg_mu_p = np.mean([a.mu_p for a in agents], axis=0)
        
        for agent in agents:
            # Move toward consensus
            agent.mu_q += coupling_strength * (avg_mu_q - agent.mu_q)
            agent.mu_p += coupling_strength * (avg_mu_p - agent.mu_p)
        
        if step % 10 == 0:
            matrix = detector.compute_consensus_matrix(system)
            clusters = detector.find_consensus_clusters(system)
            mean_div = np.mean(matrix[matrix > 0]) if np.any(matrix > 0) else 0
            print(f"Step {step:3d}: Mean div={mean_div:.4f}, Clusters={clusters}")
    
    # Final state
    print(f"\nFinal divergences:")
    matrix = detector.compute_consensus_matrix(system)
    print(f"  Mean: {np.mean(matrix[matrix > 0]):.3f}")
    print(f"  Min: {np.min(matrix[matrix > 0]):.3f}") 
    print(f"  Max: {np.max(matrix):.3f}")
    
    final_clusters = detector.find_consensus_clusters(system)
    if final_clusters:
        print(f"Final consensus clusters: {final_clusters}")
        print("✓ Consensus achieved!")
    else:
        print("No complete consensus (may need more steps or stronger coupling)")


if __name__ == "__main__":
    # Run all tests
    test_pairwise_consensus()
    test_cluster_detection()
    test_consensus_evolution()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)