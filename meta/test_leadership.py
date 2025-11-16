# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 09:48:26 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Leadership Emergence in Meta-Agent Formation
=============================================

Demonstrates how natural leaders emerge based on L_i = χ_i² · C̄_i

Example scenario:
- 6 base agents form 2 meta-agents
- Each cluster has a clear leader who "templates" the meta-agent

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from meta.emergence import MultiScaleSystem, print_leadership_summary
from geometry.geometry_base import BaseManifold, TopologyType
from config import AgentConfig


def create_test_system():
    """Create a simple test system with 6 agents."""
    # 0D manifold for transformers (particles)
    manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
    system = MultiScaleSystem(manifold)
    
    # Add 6 base agents
    print("Creating 6 base agents...")
    for i in range(6):
        config = AgentConfig(spatial_shape=(), K=3)
        agent = system.add_base_agent(config, agent_id=f"base_{i}")
        
        # Initialize with slight variations
        agent.mu_q = np.random.randn(3) * 0.1 + np.array([i % 3, 0, 0])
        agent.Sigma_q = np.eye(3) * (1.0 + 0.1 * i)
        agent.gauge.phi = np.random.randn(3) * 0.05
    
    return system


def demonstrate_leadership():
    """Show leadership emergence in meta-agent formation."""
    
    print("=" * 70)
    print("LEADERSHIP EMERGENCE DEMONSTRATION")
    print("=" * 70)
    
    # Create system
    system = create_test_system()
    
    print(f"\nInitial system: {len(system.agents[0])} base agents at ζ=0")
    
    # Form two meta-agents from specific clusters
    print("\n" + "-" * 70)
    print("CONDENSATION EVENT: ζ=0 → ζ=1")
    print("-" * 70)
    print("\nForming 2 meta-agents from 2 clusters:")
    print("  Cluster A: [0, 1, 2] → meta_1[0]")
    print("  Cluster B: [3, 4, 5] → meta_1[1]")
    
    meta_agents = system.form_meta_agents_at_scale(
        source_scale=0,
        partitions=[
            [0, 1, 2],  # Cluster A
            [3, 4, 5]   # Cluster B
        ]
    )
    
    print("\n" + system.summary())
    
    # Detailed leadership analysis
    print_leadership_summary(system)
    
    # Explain the physics
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print("""
Leadership Score: L_i = χ_i² · C̄_i

For transformers (0D):
- All χ_i = 1.0 (all agents equally present)
- Leader is simply the most coherent agent: max{C̄_i}

For spatial cases:
- χ_i varies with location
- Strong presence + high coherence = leadership
- Weak agents at boundaries don't lead even if coherent

The leader agent "templates" the meta-agent:
- Meta-agent fields weighted most heavily by leader
- Leader's gauge frame dominates φ_M
- Leader's statistics dominate μ_M, Σ_M

This naturally emerges from Presence × Coherence weighting!
No explicit leadership mechanism needed - it's built into the physics.
    """)
    
    # Show what happens with second-order condensation
    if len(meta_agents) >= 2:
        print("\n" + "-" * 70)
        print("SECOND CONDENSATION: ζ=1 → ζ=2")
        print("-" * 70)
        print("\nCombining both meta-agents into a single ζ=2 meta-agent:")
        
        super_meta = system.form_meta_agents_at_scale(
            source_scale=1,
            partitions=[[0, 1]]  # Combine both ζ=1 agents
        )
        
        print("\n" + system.summary())
        print_leadership_summary(system)
        
        print("""
Notice: The ζ=2 meta-agent's leader is one of the ζ=1 meta-agents!
This creates a leadership chain from base agents → meta-agents → super-meta-agents.

Leadership hierarchies emerge naturally across scales!
        """)


def demonstrate_spatial_leadership():
    """
    Show how leadership varies spatially.
    
    In spatial cases, different agents can lead at different locations.
    """
    print("\n" + "=" * 70)
    print("SPATIAL LEADERSHIP (TODO)")
    print("=" * 70)
    print("""
For spatial fields (2D lattices), leadership can vary by location:

At center (c₁):
  Agent 1: χ=1.0, C̄=0.95 → L=0.950 ← Leader at center!
  Agent 2: χ=0.5, C̄=0.99 → L=0.248
  
At boundary (c₂):
  Agent 1: χ=0.2, C̄=0.95 → L=0.038
  Agent 2: χ=0.9, C̄=0.99 → L=0.802 ← Different leader!
  
The meta-agent is a spatially-varying blend where leadership shifts
across the base manifold. This creates smooth transitions in the
emergent coarse-grained structure.
    """)


if __name__ == "__main__":
    np.random.seed(42)
    demonstrate_leadership()
    demonstrate_spatial_leadership()
    
    print("\n" + "=" * 70)
    print("Leadership emerges naturally from χ² · C̄ weighting! 🏆")
    print("=" * 70)