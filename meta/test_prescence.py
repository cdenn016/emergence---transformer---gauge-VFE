# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 09:45:16 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Example: Presence × Coherence Weighting in Meta-Agent Formation
================================================================

Demonstrates how coherence weighting works for the specific case:
- Agent 1: χ₁(c) = 1.0 (strong presence)
- Agent 2: χ₂(c) = 0.5 (medium presence)
- Agent 3: χ₃(c) = 0.1 (weak presence, near boundary)

Coherence structure:
- KL₁₂ = 0.001 (agents 1&2 very coherent)
- KL₁₃ = 0.009 (agents 1&3 somewhat coherent)
- KL₂₃ = 0.0001 (agents 2&3 extremely coherent!)

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import List, Tuple


# =============================================================================
# Example Data Structure
# =============================================================================

class MockAgent:
    """Simple agent for demonstration."""
    def __init__(self, chi_value: float, mu: np.ndarray, Sigma: np.ndarray):
        self.chi = chi_value  # Presence at location c
        self.mu = mu          # Mean
        self.Sigma = Sigma    # Covariance


def kl_divergence(mu_p, Sigma_p, mu_q, Sigma_q) -> float:
    """Simplified KL divergence for demonstration."""
    k = len(mu_p)
    Sigma_q_inv = np.linalg.inv(Sigma_q)
    
    term1 = np.trace(Sigma_q_inv @ Sigma_p)
    term2 = (mu_q - mu_p) @ Sigma_q_inv @ (mu_q - mu_p)
    term3 = np.log(np.linalg.det(Sigma_q) / np.linalg.det(Sigma_p))
    
    return 0.5 * (term1 + term2 - k + term3)


# =============================================================================
# Coherence Score Computation
# =============================================================================

def compute_coherence_scores(agents: List[MockAgent], 
                            kl_matrix: np.ndarray) -> np.ndarray:
    """
    Compute coherence score for each agent.
    
    C̄ᵢ = exp(-average_KL_with_others)
    """
    n = len(agents)
    coherence_scores = np.zeros(n)
    
    for i in range(n):
        # Average KL with all others
        kl_sum = sum(kl_matrix[i, j] for j in range(n) if i != j)
        avg_kl = kl_sum / (n - 1)
        
        # Coherence via exponential decay
        coherence_scores[i] = np.exp(-avg_kl)
    
    return coherence_scores


# =============================================================================
# Meta-Agent Support Computation
# =============================================================================

def compute_meta_support_presence_coherence(
    agents: List[MockAgent],
    coherence_scores: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute meta-agent support at location c.
    
    χ_M(c) = Σᵢ wᵢ · χᵢ
    where wᵢ ∝ χᵢ · C̄ᵢ
    
    Returns:
        chi_M: Meta-agent support value
        weights: Normalized weights used
    """
    n = len(agents)
    
    # Unnormalized weights: presence × coherence
    weights_unnorm = np.array([
        agents[i].chi * coherence_scores[i]
        for i in range(n)
    ])
    
    # Normalize
    total_weight = np.sum(weights_unnorm)
    weights = weights_unnorm / total_weight
    
    # Weighted combination
    chi_M = sum(weights[i] * agents[i].chi for i in range(n))
    
    return chi_M, weights


def compute_meta_support_union(agents: List[MockAgent]) -> float:
    """Old method: union = max{χᵢ}."""
    return max(agent.chi for agent in agents)


def compute_meta_support_coherence_only(
    agents: List[MockAgent],
    coherence_scores: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Alternative: weight by coherence only (no presence)."""
    weights = coherence_scores / np.sum(coherence_scores)
    chi_M = sum(weights[i] * agents[i].chi for i in range(len(agents)))
    return chi_M, weights


# =============================================================================
# Meta-Agent Field Computation
# =============================================================================

def compute_meta_fields(
    agents: List[MockAgent],
    coherence_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute renormalized fields for meta-agent.
    
    Returns:
        mu_M: Renormalized mean
        Sigma_M: Renormalized covariance
        weights: Normalized weights used
    """
    n = len(agents)
    k = len(agents[0].mu)
    
    # Weights: presence × coherence
    weights_unnorm = np.array([
        agents[i].chi * coherence_scores[i]
        for i in range(n)
    ])
    weights = weights_unnorm / np.sum(weights_unnorm)
    
    # Weighted combination
    mu_M = sum(weights[i] * agents[i].mu for i in range(n))
    Sigma_M = sum(weights[i] * agents[i].Sigma for i in range(n))
    
    return mu_M, Sigma_M, weights


# =============================================================================
# Run Example
# =============================================================================

def main():
    print("=" * 70)
    print("Presence × Coherence Weighting Example")
    print("=" * 70)
    
    # Define agents with specific presence values
    print("\n1. SETUP")
    print("-" * 70)
    
    # Agent 1: Strong presence
    agent1 = MockAgent(
        chi_value=1.0,
        mu=np.array([1.0, 0.0, 0.0]),
        Sigma=np.eye(3)
    )
    
    # Agent 2: Medium presence
    agent2 = MockAgent(
        chi_value=0.5,
        mu=np.array([1.1, 0.1, 0.0]),
        Sigma=np.eye(3) * 1.1
    )
    
    # Agent 3: Weak presence (near boundary)
    agent3 = MockAgent(
        chi_value=0.1,
        mu=np.array([1.05, 0.05, 0.0]),
        Sigma=np.eye(3) * 0.95
    )
    
    agents = [agent1, agent2, agent3]
    
    print(f"Agent 1: χ = {agent1.chi}, μ = {agent1.mu}")
    print(f"Agent 2: χ = {agent2.chi}, μ = {agent2.mu}")
    print(f"Agent 3: χ = {agent3.chi}, μ = {agent3.mu}")
    
    # Define KL matrix (symmetric)
    print("\n2. COHERENCE STRUCTURE")
    print("-" * 70)
    
    kl_matrix = np.array([
        [0.0,    0.001,  0.009],
        [0.001,  0.0,    0.0001],
        [0.009,  0.0001, 0.0]
    ])
    
    print("KL Divergence Matrix:")
    print(f"  KL₁₂ = {kl_matrix[0,1]:.4f} (agents 1&2 very coherent)")
    print(f"  KL₁₃ = {kl_matrix[0,2]:.4f} (agents 1&3 somewhat coherent)")
    print(f"  KL₂₃ = {kl_matrix[1,2]:.4f} (agents 2&3 extremely coherent!)")
    
    # Compute coherence scores
    coherence_scores = compute_coherence_scores(agents, kl_matrix)
    
    print("\nCoherence Scores C̄ᵢ = exp(-avg_KL):")
    for i, score in enumerate(coherence_scores):
        avg_kl = -np.log(score)
        print(f"  C̄₁ = {score:.6f} (avg KL = {avg_kl:.6f})")
    
    # Meta-agent support
    print("\n3. META-AGENT SUPPORT χ_M(c)")
    print("-" * 70)
    
    # Method 1: Union (old)
    chi_union = compute_meta_support_union(agents)
    print(f"Union (max): χ_M = {chi_union:.3f}")
    print("  → Too conservative, ignores coherence")
    
    # Method 2: Coherence only
    chi_coherence, weights_coh = compute_meta_support_coherence_only(agents, coherence_scores)
    print(f"\nCoherence only: χ_M = {chi_coherence:.3f}")
    print(f"  Weights: {weights_coh}")
    print("  → Doesn't emphasize strong presence enough")
    
    # Method 3: Presence × Coherence (recommended)
    chi_meta, weights_pc = compute_meta_support_presence_coherence(agents, coherence_scores)
    print(f"\nPresence × Coherence: χ_M = {chi_meta:.3f} ⭐")
    print(f"  Weights: {weights_pc}")
    print("  → Strong agents dominate, weak agents contribute little")
    
    print("\nWeight Breakdown (Presence × Coherence):")
    for i, agent in enumerate(agents):
        w_unnorm = agent.chi * coherence_scores[i]
        print(f"  Agent {i+1}: χᵢ={agent.chi:.1f} × C̄ᵢ={coherence_scores[i]:.3f} "
              f"= {w_unnorm:.4f} → w={weights_pc[i]:.3f}")
    
    # Meta-agent fields
    print("\n4. META-AGENT FIELDS μ_M, Σ_M")
    print("-" * 70)
    
    mu_M, Sigma_M, field_weights = compute_meta_fields(agents, coherence_scores)
    
    print(f"Field weights (same as support): {field_weights}")
    print(f"\nRenormalized mean μ_M:")
    print(f"  {mu_M}")
    print(f"\nRenormalized covariance Σ_M (diagonal):")
    print(f"  {np.diag(Sigma_M)}")
    
    # Analysis
    print("\n5. PHYSICAL INTERPRETATION")
    print("-" * 70)
    print("""
Key insights:
    
1. Agent 1 (χ=1.0, strong) gets weight 0.624
   → Dominates the meta-agent despite being only 1/3 of constituents
   
2. Agent 2 (χ=0.5, medium) gets weight 0.313
   → Moderate contribution, proportional to presence
   
3. Agent 3 (χ=0.1, weak) gets weight 0.063
   → Minimal contribution despite high coherence with agent 2!
   → Weak presence near boundary → shouldn't define meta-agent strongly
   
4. Quadratic weighting χᵢ² · C̄ᵢ emerges naturally:
   - wᵢ ∝ χᵢ · C̄ᵢ (presence × coherence)
   - χ_M = Σᵢ wᵢ · χᵢ (weighted average of presence)
   - Result: χᵢ appears twice → χᵢ² effect
   
5. Meta-agent support (0.787) is:
   - Much less than union (1.0)
   - Much more than coherence-only (0.534)
   - Reflects dominance of strong coherent constituents
    """)
    
    print("=" * 70)
    print("Conclusion: Presence × Coherence gives physically correct weighting!")
    print("=" * 70)


if __name__ == "__main__":
    main()