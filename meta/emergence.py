# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 15:18:40 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Meta-Agent Emergence and Hierarchical Structure
================================================

Creates meta-agents from consensus clusters through renormalization.

Theory:
-------
When agents reach epistemic death (consensus), they can be integrated out
and replaced by a single meta-agent with renormalized parameters:

    q_M = average(Ω_ij[q_j]) for j in cluster
    p_M = average(Ω_ij[p_j]) for j in cluster  
    φ_M = mean_SO3(φ_j)       (Fréchet mean on SO(3))

The meta-agent operates at scale ζ+1 with emergent dynamics τ_{ζ+1} >> τ_ζ.

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Import from existing codebase
from agent.agents import Agent
from geometry.geometry_base import BaseManifold, SupportRegion, create_full_support
from math_utils.transport import compute_transport
from math_utils.generators import generate_so3_generators
from config import AgentConfig


class ScaleLevel(Enum):
    """Hierarchical scale levels"""
    BASE = 0      # Individual agents (ζ=0)
    GROUP = 1     # First-order meta-agents (ζ=1)
    COMMUNITY = 2 # Second-order meta-agents (ζ=2)
    SOCIETY = 3   # Third-order meta-agents (ζ=3)
    
    def __int__(self):
        return self.value


@dataclass
class MetaAgentDescriptor:
    """
    Metadata for a meta-agent.
    
    Tracks hierarchical structure and emergence history.
    """
    agent_id: int                      # Unique ID in system
    scale: int                        # Hierarchical scale ζ
    constituent_ids: List[int]        # IDs of constituent agents
    emergence_time: int               # Simulation step when formed
    parent_meta_id: Optional[int]     # Parent meta-agent if part of higher structure
    
    # Renormalized parameters
    belief_coherence: float           # How aligned constituents were
    model_coherence: float            # How aligned models were
    
    # Dynamics
    characteristic_timescale: float   # τ_ζ relative to base agents
    
    def __repr__(self):
        return (f"MetaAgent(id={self.agent_id}, scale={self.scale}, "
                f"constituents={self.constituent_ids})")


class HierarchicalAgent(Agent):
    """
    Extended agent with hierarchical awareness.
    
    Can be either a base agent or meta-agent.
    """
    
    def __init__(self, *args, meta_descriptor: Optional[MetaAgentDescriptor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = meta_descriptor
        self.is_meta = meta_descriptor is not None
        self.scale = meta_descriptor.scale if meta_descriptor else 0
        self.is_active = True  # Can be deactivated when absorbed into meta-agent
        
        # Constituent management (for meta-agents)
        self.constituents: List['HierarchicalAgent'] = []
        self.parent_meta: Optional['HierarchicalAgent'] = None
        
    
    @property
    def effective_learning_rate_scale(self) -> float:
        """No artificial scaling - let dynamics be natural."""
        return 1.0  # Same timescale for all agents!


class MetaAgentFactory:
    """
    Factory for creating and managing meta-agents.
    
    Handles:
    - Renormalization of parameters
    - Gauge frame averaging on SO(3)
    - Coupling constant renormalization
    - Constituent deactivation
    """
    
    def __init__(self, 
                 timescale_ratio: float = 1e4,
                 preserve_constituents: bool = True):
        """
        Initialize factory.
        
        Args:
            timescale_ratio: τ_{ζ+1}/τ_ζ ratio between scales
            preserve_constituents: Keep constituents inactive vs delete
        """
        
        """TODO REMOVE HARDCODED TIMESCALE!"""
        
        self.timescale_ratio = timescale_ratio
        self.preserve_constituents = preserve_constituents
        self.next_meta_id = 10000  # Start meta IDs at 10000 to distinguish
        
    def create_meta_agent(self,
                         system,
                         constituent_indices: List[int],
                         emergence_time: int = 0,
                         coherence_scores: Optional[Dict] = None) -> HierarchicalAgent:
        """
        Create a meta-agent from constituent agents.
        """
        if len(constituent_indices) < 2:
            raise ValueError("Meta-agent requires at least 2 constituents")
        
        # Get constituent agents
        constituents = [system.agents[i] for i in constituent_indices]
        
        # Determine scale (one level above highest constituent)
        max_constituent_scale = max(
            getattr(agent, 'scale', 0) for agent in constituents
        )
        meta_scale = max_constituent_scale + 1
        
        # Compute renormalized parameters
        mu_q, Sigma_q = self._renormalize_beliefs(constituents, system)
        mu_p, Sigma_p = self._renormalize_models(constituents, system)
        phi = self._average_gauge_frames(constituents)
        
        # Create meta-agent configuration
        K = constituents[0].K
        base_manifold = constituents[0].base_manifold
        
        meta_config = AgentConfig(
            spatial_shape=base_manifold.shape,
            K=K,
            mu_scale=1.0,
            sigma_scale=1.0,
            phi_scale=1.0
        )
        
        # ⚡ Add mask_config
        from agent.masking import MaskConfig
        meta_config.mask_config = MaskConfig()
        
        # Create metadata
        meta_descriptor = MetaAgentDescriptor(
            agent_id=self.next_meta_id,
            scale=meta_scale,
            constituent_ids=constituent_indices,
            emergence_time=emergence_time,
            parent_meta_id=None,
            belief_coherence=coherence_scores.get('belief_coherence', 1.0) if coherence_scores else 1.0,
            model_coherence=coherence_scores.get('model_coherence', 1.0) if coherence_scores else 1.0,
            characteristic_timescale=self.timescale_ratio ** meta_scale
        )
        
        # Create the meta-agent
        meta_agent = HierarchicalAgent(
            agent_id=self.next_meta_id,
            config=meta_config,
            rng=np.random.default_rng(self.next_meta_id),
            base_manifold=base_manifold,
            meta_descriptor=meta_descriptor
        )
        
        # ⚡⚡⚡ CRITICAL: Initialize fields before setting values! ⚡⚡⚡
        meta_agent.support = self._compute_meta_support(constituents)
        meta_agent.generators = constituents[0].generators
        
        # Initialize field structures
        meta_agent._initialize_belief_cholesky()
        meta_agent._initialize_prior_cholesky()
        meta_agent._initialize_gauge()
        
        # Now set renormalized parameters (overwrite initialized values)
        meta_agent.mu_q = mu_q
        meta_agent.Sigma_q = Sigma_q  # This will compute L_q via setter
        meta_agent.mu_p = mu_p
        meta_agent.Sigma_p = Sigma_p  # This will compute L_p via setter
        meta_agent.gauge.phi = phi
        
        # Link constituents
        meta_agent.constituents = constituents
        for agent in constituents:
            if hasattr(agent, 'parent_meta'):
                agent.parent_meta = meta_agent
        
        self.next_meta_id += 1
        
        # Deactivate constituents if requested
        if not self.preserve_constituents:
            self._deactivate_constituents(constituents)
        
        return meta_agent
    
    def _renormalize_beliefs(self,
                            constituents: List[Agent],
                            system) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute renormalized belief parameters.
        
        Uses gauge transport to align all beliefs before averaging.
        """
        # Use first agent as reference frame
        ref_agent = constituents[0]
        K = ref_agent.K
        
        # DEBUG: Check dimensions
        print(f"\n[DEBUG] Renormalizing beliefs:")
        print(f"  Reference agent K: {K}")
        print(f"  Reference mu_q shape: {ref_agent.mu_q.shape}")
        print(f"  Reference generators shape: {ref_agent.generators.shape}")
        print(f"  Reference phi shape: {ref_agent.gauge.phi.shape}")
        
        # For 0D case
        if ref_agent.base_manifold.is_point:
            mu_sum = ref_agent.mu_q.copy()
            Sigma_sum = ref_agent.Sigma_q.copy()
            
            for i, agent in enumerate(constituents[1:], 1):
                print(f"\n  Processing constituent {i}:")
                print(f"    mu_q shape: {agent.mu_q.shape}")
                print(f"    phi shape: {agent.gauge.phi.shape}")
                
                # Transport to reference frame
                omega = compute_transport(
                    ref_agent.gauge.phi,
                    agent.gauge.phi,
                    ref_agent.generators,
                    validate=False
                )
                
                print(f"    omega shape: {omega.shape}")
                print(f"    About to compute: omega @ mu_q")
                print(f"      omega.shape: {omega.shape}")
                print(f"      mu_q.shape: {agent.mu_q.shape}")
                # Transport and accumulate
                mu_transported = omega @ agent.mu_q
                Sigma_transported = omega @ agent.Sigma_q @ omega.T
                
                mu_sum += mu_transported
                Sigma_sum += Sigma_transported
            
            # Average
            n = len(constituents)
            mu_M = mu_sum / n
            Sigma_M = Sigma_sum / n
            
        else:
            # For spatial case, need to handle field averaging
            spatial_shape = ref_agent.support.base_manifold.shape
            mu_M = np.zeros((*spatial_shape, K))
            Sigma_M = np.zeros((*spatial_shape, K, K))
            
            # Average at each spatial point
            # (Implementation depends on overlap structure)
            # For now, simple average
            for agent in constituents:
                mu_M += agent.mu_q
                Sigma_M += agent.Sigma_q
            
            mu_M /= len(constituents)
            Sigma_M /= len(constituents)
        
        return mu_M, Sigma_M
    
    def _renormalize_models(self,
                          constituents: List[Agent],
                          system) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute renormalized model parameters.
        
        Similar to beliefs but for the model/prior distributions.
        """
        # Use same approach as beliefs
        return self._renormalize_beliefs(constituents, system)
    
    def _average_gauge_frames(self, constituents: List[Agent]) -> np.ndarray:
        """
        Compute average gauge frame on SO(3).
        
        Uses Euclidean average in so(3) then projects back.
        More sophisticated: Use Fréchet mean on SO(3).
        
        Returns:
            phi_M: Averaged gauge frame in so(3)
        """
        # Simple approach: Euclidean average in Lie algebra
        phi_sum = np.zeros(3)
        
        for agent in constituents:
            phi_sum += agent.gauge.phi
        
        phi_avg = phi_sum / len(constituents)
        
        # Ensure within SO(3) bounds
        phi_avg = np.clip(phi_avg, -np.pi, np.pi)
        
        return phi_avg
    
    def _compute_meta_support(self, constituents: List[Agent]) -> SupportRegion:
        """
        Compute support region for meta-agent.
        
        Options:
        1. Union of constituent supports
        2. Intersection (common support)
        3. Weighted average of χ functions
        
        Returns:
            support: SupportRegion for meta-agent
        """
        # For now: union approach (maximum of χ weights)
        ref_agent = constituents[0]
        base_manifold = ref_agent.base_manifold
        
        if base_manifold.is_point:
            # 0D: support is trivial
            return create_full_support(base_manifold)
        
        # Compute union of supports
        chi_union = np.zeros_like(ref_agent.support.chi_weight)
        
        for agent in constituents:
            chi_union = np.maximum(chi_union, agent.support.chi_weight)
        
        return SupportRegion(
            base_manifold=base_manifold,
            chi_weight=chi_union
        )
    
    def _deactivate_constituents(self, constituents: List[Agent]):
        """Mark constituent agents as inactive."""
        for agent in constituents:
            if hasattr(agent, 'is_active'):
                agent.is_active = False
    
    def dissolve_meta_agent(self,
                           meta_agent: HierarchicalAgent,
                           perturbation_scale: float = 0.01) -> List[Agent]:
        """
        Dissolve meta-agent back into constituents.
        
        Breaks consensus with small perturbations.
        
        Args:
            meta_agent: Meta-agent to dissolve
            perturbation_scale: Size of perturbation to break consensus
            
        Returns:
            reactivated_agents: List of reactivated constituents
        """
        if not meta_agent.is_meta:
            raise ValueError("Can only dissolve meta-agents")
        
        constituents = meta_agent.constituents
        rng = np.random.default_rng(42)
        
        for agent in constituents:
            # Add small perturbation to break consensus
            perturbation = rng.normal(0, perturbation_scale, agent.mu_q.shape)
            agent.mu_q += perturbation
            
            # Reactivate
            if hasattr(agent, 'is_active'):
                agent.is_active = True
            
            # Clear parent link
            if hasattr(agent, 'parent_meta'):
                agent.parent_meta = None
        
        return constituents


class RenormalizationGroup:
    """
    Renormalization group flow for coupling constants.
    
    Computes effective couplings at higher scales:
        β^{(ζ+1)}_{ij} = f_β({β^{(ζ)}_{kl}})
        γ^{(ζ+1)}_{ij} = f_γ({γ^{(ζ)}_{kl}})
    """
    
    def __init__(self, averaging_method: str = 'mean'):
        """
        Initialize RG flow.
        
        Args:
            averaging_method: How to combine lower-scale couplings
                             ('mean', 'max', 'rms')
        """
        self.averaging_method = averaging_method
    
    def renormalize_attention_weights(self,
                                     beta_matrix: np.ndarray,
                                     cluster_mapping: Dict[int, List[int]]) -> np.ndarray:
        """
        Compute renormalized attention weights β^{(ζ+1)}.
        
        Args:
            beta_matrix: Lower scale attention weights, shape (N, N)
            cluster_mapping: Maps meta-agent index -> constituent indices
            
        Returns:
            beta_renorm: Renormalized weights, shape (M, M) where M < N
        """
        n_meta = len(cluster_mapping)
        beta_renorm = np.zeros((n_meta, n_meta))
        
        for i, cluster_i in enumerate(cluster_mapping.values()):
            for j, cluster_j in enumerate(cluster_mapping.values()):
                if i == j:
                    continue
                
                # Extract block of couplings between clusters
                block_couplings = []
                for idx_i in cluster_i:
                    for idx_j in cluster_j:
                        block_couplings.append(beta_matrix[idx_i, idx_j])
                
                # Renormalize
                if self.averaging_method == 'mean':
                    beta_renorm[i, j] = np.mean(block_couplings)
                elif self.averaging_method == 'max':
                    beta_renorm[i, j] = np.max(block_couplings)
                elif self.averaging_method == 'rms':
                    beta_renorm[i, j] = np.sqrt(np.mean(np.square(block_couplings)))
        
        return beta_renorm


# =============================================================================
# Hierarchical System Extension
# =============================================================================

class HierarchicalMultiAgentSystem:
    """
    Multi-agent system with hierarchical emergence.
    
    Manages agents at multiple scales with cross-scale interactions.
    """
    
    def __init__(self, base_system):
        """
        Initialize from a base MultiAgentSystem.
        
        Args:
            base_system: Standard MultiAgentSystem with scale-0 agents
        """
        self.base_system = base_system
        
        # Convert base agents to HierarchicalAgents
        self.agents_by_scale = {0: []}
        for agent in base_system.agents:
            hierarchical_agent = self._upgrade_to_hierarchical(agent)
            self.agents_by_scale[0].append(hierarchical_agent)
        
        # Replace base system agents
        base_system.agents = self.agents_by_scale[0]
        
        # Emergence machinery
        self.factory = MetaAgentFactory()
        self.renormalization = RenormalizationGroup()
        
        # Track emergence history
        self.emergence_events = []
        self.current_time = 0
        
    def _upgrade_to_hierarchical_v2(self, agent: Agent) -> HierarchicalAgent:
        """
        Alternative: Copy entire __dict__ then fix up references.
        
        This is the most robust approach.
        """
        # Create HierarchicalAgent shell
        h_agent = HierarchicalAgent(
            agent_id=agent.agent_id,
            config=agent.config,
            rng=np.random.default_rng(agent.agent_id),
            base_manifold=agent.base_manifold
        )
        
        # Copy ALL fields from base agent (except private and methods)
        for key, value in agent.__dict__.items():
            if not key.startswith('_') and not callable(value):
                try:
                    if isinstance(value, np.ndarray):
                        h_agent.__dict__[key] = value.copy()
                    else:
                        h_agent.__dict__[key] = value
                except Exception as e:
                    print(f"Warning: Failed to copy {key}: {e}")
        
        # Verify critical fields exist
        for field in ['mu_q', 'mu_p', 'L_q', 'L_p']:
            if not hasattr(h_agent, field):
                raise RuntimeError(f"HierarchicalAgent missing critical field: {field}")
        
        return h_agent

    
    # ======================================
    
    def form_meta_agents(self,
                        consensus_clusters: List[List[int]],
                        coherence_scores: Optional[List[Dict]] = None):
        """
        Form meta-agents from consensus clusters.
        
        Args:
            consensus_clusters: List of agent index clusters
            coherence_scores: Optional coherence metrics for each cluster
        """
        new_meta_agents = []
        
        for i, cluster in enumerate(consensus_clusters):
            if len(cluster) < 2:
                continue
            
            # Get coherence scores if provided
            scores = coherence_scores[i] if coherence_scores else None
            
            # Create meta-agent
            meta_agent = self.factory.create_meta_agent(
                self.base_system,
                cluster,
                emergence_time=self.current_time,
                coherence_scores=scores
            )
            
            # Add to appropriate scale
            scale = meta_agent.scale
            if scale not in self.agents_by_scale:
                self.agents_by_scale[scale] = []
            
            self.agents_by_scale[scale].append(meta_agent)
            new_meta_agents.append(meta_agent)
            
            # Record emergence event
            self.emergence_events.append({
                'time': self.current_time,
                'scale': scale,
                'meta_id': meta_agent.meta.agent_id,
                'constituents': cluster,
                'coherence': scores
            })
            
            # Deactivate constituents in base system
            for idx in cluster:
                self.base_system.agents[idx].is_active = False
        
        print(f"Formed {len(new_meta_agents)} meta-agents at time {self.current_time}")
        return new_meta_agents
    
    def get_active_agents(self) -> List[HierarchicalAgent]:
        """Get all active agents across all scales."""
        active = []
        for scale in sorted(self.agents_by_scale.keys()):
            for agent in self.agents_by_scale[scale]:
                if agent.is_active:
                    active.append(agent)
        return active
    
    def step(self):
        """Perform one update step with scale-dependent learning rates."""
        self.current_time += 1
        
        # Update each scale with appropriate timescale
        for scale in sorted(self.agents_by_scale.keys()):
            agents = self.agents_by_scale[scale]
            
            for agent in agents:
                if not agent.is_active:
                    continue
                
                # Scale learning rate by hierarchical timescale
                lr_scale = agent.effective_learning_rate_scale
                
                # Get gradients (would need to modify gradient computation)
                # For now, simple placeholder
                if lr_scale > 0:
                    # Update with scaled learning rate
                    pass
    
    def summary(self) -> str:
        """Get hierarchical structure summary."""
        lines = ["Hierarchical System Structure:"]
        lines.append("=" * 50)
        
        for scale in sorted(self.agents_by_scale.keys()):
            agents = self.agents_by_scale[scale]
            active = sum(1 for a in agents if a.is_active)
            lines.append(f"Scale {scale}: {active}/{len(agents)} active agents")
            
            if scale > 0:  # Meta-agents
                for agent in agents:
                    if agent.is_active and agent.is_meta:
                        lines.append(f"  Meta-{agent.meta.agent_id}: "
                                   f"constituents={agent.meta.constituent_ids}")
        
        lines.append(f"\nTotal emergence events: {len(self.emergence_events)}")
        return "\n".join(lines)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_hierarchical_structure(hierarchical_system: HierarchicalMultiAgentSystem) -> Dict:
    """
    Analyze the hierarchical structure and emergence patterns.
    
    Returns:
        Dictionary with structure metrics
    """
    metrics = {
        'n_scales': len(hierarchical_system.agents_by_scale),
        'agents_per_scale': {},
        'active_per_scale': {},
        'emergence_events': len(hierarchical_system.emergence_events),
        'mean_cluster_size': 0,
        'max_hierarchy_depth': 0
    }
    
    # Count agents per scale
    cluster_sizes = []
    for scale, agents in hierarchical_system.agents_by_scale.items():
        metrics['agents_per_scale'][scale] = len(agents)
        metrics['active_per_scale'][scale] = sum(1 for a in agents if a.is_active)
        
        # Track cluster sizes
        for agent in agents:
            if agent.is_meta:
                cluster_sizes.append(len(agent.meta.constituent_ids))
    
    if cluster_sizes:
        metrics['mean_cluster_size'] = np.mean(cluster_sizes)
        metrics['max_hierarchy_depth'] = max(hierarchical_system.agents_by_scale.keys())
    
    return metrics