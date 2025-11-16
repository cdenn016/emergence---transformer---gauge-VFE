# -*- coding: utf-8 -*-
"""
Meta-Agent Emergence and Hierarchical Structure
================================================

Creates meta-agents from consensus clusters through renormalization.

Theory:
-------
At each base point c ∈ C, agents at scale ζ can condense into meta-agents
at scale ζ+1 through renormalization:

    μ_M(c) = (1/n) Σᵢ Ωᵢⱼ[μᵢ(c)]     (gauge-transported average)
    Σ_M(c) = (1/n) Σᵢ Ωᵢⱼ[Σᵢ(c)]Ωᵢⱼᵀ
    φ_M(c) = average_SO3({φᵢ(c)})    (Fréchet mean on SO(3))

Multiple condensations can occur iteratively:
    ζ=0: 3000 agents → [partition] → ζ=1: 50 meta-agents
    ζ=1: 50 meta-agents → [partition] → ζ=2: 6 meta-agents
    etc.

No artificial timescale separation - dynamics emerge naturally!

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# Import from existing codebase
from agent.agents import Agent
from geometry.geometry_base import BaseManifold, SupportRegion, create_full_support
from math_utils.transport import compute_transport
from math_utils.generators import generate_so3_generators
from math_utils.so3_frechet import average_gauge_frames_so3, so3_exp, so3_log
from config import AgentConfig
from agent.masking import MaskConfig


# =============================================================================
# Scale Indexing
# =============================================================================

@dataclass
class ScaleIndex:
    """
    Index for agents at a specific (scale, position) in the hierarchy.
    
    At each base point c ∈ C, we can have multiple agents at each scale ζ.
    """
    scale: int          # ζ ∈ {0, 1, 2, ...}
    local_index: int    # Index within this scale
    
    def __repr__(self):
        return f"ζ{self.scale}[{self.local_index}]"


class ScaleLevel(Enum):
    """Hierarchical scale levels (for convenience)."""
    BASE = 0       # Individual agents (ζ=0)
    GROUP = 1      # First-order meta-agents (ζ=1)
    COMMUNITY = 2  # Second-order meta-agents (ζ=2)
    SOCIETY = 3    # Third-order meta-agents (ζ=3)
    
    def __int__(self):
        return self.value


@dataclass
class MetaAgentDescriptor:
    """
    Metadata for a meta-agent.
    
    Tracks hierarchical structure and emergence history.
    """
    scale_index: ScaleIndex           # Unique scale + local index
    constituent_indices: List[ScaleIndex]  # Lower-scale constituents
    emergence_time: int                    # Simulation step when formed
    
    # Coherence metrics
    belief_coherence: float           # Mean coherence: avg(C̄ᵢ)
    model_coherence: float            # Model coherence
    
    # Leadership structure
    leader_index: Optional[int] = None      # Index of dominant constituent
    leader_score: Optional[float] = None    # Leadership score L = χ² · C̄
    leadership_distribution: Optional[np.ndarray] = None  # L_i for all constituents
    
    def __repr__(self):
        leader_str = f", leader={self.leader_index}" if self.leader_index is not None else ""
        return (f"MetaAgent({self.scale_index}, "
                f"n_constituents={len(self.constituent_indices)}{leader_str})")


# =============================================================================
# Hierarchical Agent
# =============================================================================

class HierarchicalAgent(Agent):
    """
    Extended agent with hierarchical awareness.
    
    Can be either:
    - Base agent (scale ζ=0)
    - Meta-agent (scale ζ≥1) with renormalized fields
    
    Fields (μ_q, Σ_q, μ_p, Σ_p, φ) are smooth sections over support region.
    """
    
    def __init__(self,
                 scale: int,
                 local_index: int,
                 constituent_indices: Optional[List[ScaleIndex]] = None,
                 meta_descriptor: Optional[MetaAgentDescriptor] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Scale identification
        self.scale = scale
        self.local_index = local_index
        self.scale_index = ScaleIndex(scale, local_index)
        
        # Hierarchical structure
        self.constituent_indices = constituent_indices or []
        self.is_meta = len(self.constituent_indices) > 0
        self.meta = meta_descriptor
        
        # Activity status (can be deactivated when absorbed into meta-agent)
        self.is_active = True
    
    def __repr__(self):
        status = "active" if self.is_active else "inactive"
        meta_str = f", meta" if self.is_meta else ""
        return f"HAgent({self.scale_index}, {status}{meta_str})"


# =============================================================================
# Multi-Scale System
# =============================================================================

class MultiScaleSystem:
    """
    System with agents at multiple scales at each base point.
    
    Structure:
    - agents[ζ] = list of agents at scale ζ
    - At each c ∈ C, multiple agents can coexist at each scale
    
    Supports iterative condensation:
        Scale 0 → partition → Scale 1
        Scale 1 → partition → Scale 2
        etc.
    """
    
    def __init__(self, base_manifold: BaseManifold):
        self.base_manifold = base_manifold
        
        # agents[scale] = list of agents at that scale
        self.agents: Dict[int, List[HierarchicalAgent]] = defaultdict(list)
        
        # Track condensation history
        self.condensation_events = []
        self.current_time = 0
    
    def add_base_agent(self, agent_config: AgentConfig, agent_id: str = None) -> HierarchicalAgent:
        """
        Add a new scale-0 base agent.
        
        Args:
            agent_config: Configuration for the agent
            agent_id: Optional identifier
        
        Returns:
            Created HierarchicalAgent at scale 0
        """
        local_index = len(self.agents[0])
        
        if agent_id is None:
            agent_id = f"base_{local_index}"
        
        agent = HierarchicalAgent(
            scale=0,
            local_index=local_index,
            agent_id=agent_id,
            config=agent_config,
            rng=np.random.default_rng(local_index),
            base_manifold=self.base_manifold
        )
        
        self.agents[0].append(agent)
        return agent
    
    def form_meta_agents_at_scale(self,
                                  source_scale: int,
                                  partitions: List[List[int]],
                                  deactivate_constituents: bool = True) -> List[HierarchicalAgent]:
        """
        Form meta-agents at scale ζ+1 from partitions at scale ζ.
        
        Args:
            source_scale: Scale ζ of constituents
            partitions: List of constituent clusters (local indices at source_scale)
                       Each cluster becomes one meta-agent at scale ζ+1
            deactivate_constituents: Mark constituents as inactive after condensation
        
        Returns:
            List of newly formed meta-agents
        
        Example:
            # Condense scale-0 agents [0,1,2] and [3,4] into two scale-1 meta-agents
            system.form_meta_agents_at_scale(
                source_scale=0,
                partitions=[[0, 1, 2], [3, 4]]
            )
        """
        target_scale = source_scale + 1
        source_agents = self.agents[source_scale]
        
        if not source_agents:
            raise ValueError(f"No agents at source scale {source_scale}")
        
        new_meta_agents = []
        
        for partition in partitions:
            if len(partition) < 2:
                print(f"[Warning] Skipping singleton cluster: {partition}")
                continue
            
            # Get constituent agents
            constituents = [source_agents[i] for i in partition]
            
            # Verify all constituents are active
            inactive = [c for c in constituents if not c.is_active]
            if inactive:
                print(f"[Warning] Partition contains inactive agents: {inactive}")
            
            # Compute coherence scores ONCE (used for all renormalization)
            coherence_scores = self._compute_coherence_scores(constituents, field_type='belief')
            
            # Identify leader (agent with max χ² · C̄)
            leader_idx, leader_score, leadership_dist = self._identify_leader(constituents, coherence_scores)
            
            # Compute renormalized fields via coherence-weighted gauge transport
            mu_q, Sigma_q = self._renormalize_beliefs(constituents, coherence_scores)
            mu_p, Sigma_p = self._renormalize_models(constituents, coherence_scores)
            phi = self._average_gauge_frames(constituents, coherence_scores)
            
            # Create meta-agent configuration
            meta_index = len(self.agents[target_scale])
            meta_id = f"meta_{target_scale}_{meta_index}"
            
            constituent_scale_indices = [
                ScaleIndex(source_scale, i) for i in partition
            ]
            
            # Compute coherence metrics from scores
            belief_coherence = np.mean(coherence_scores)
            
            # Also compute model coherence separately
            model_coherence_scores = self._compute_coherence_scores(constituents, field_type='model')
            model_coherence = np.mean(model_coherence_scores)
            
            meta_descriptor = MetaAgentDescriptor(
                scale_index=ScaleIndex(target_scale, meta_index),
                constituent_indices=constituent_scale_indices,
                emergence_time=self.current_time,
                belief_coherence=belief_coherence,
                model_coherence=model_coherence,
                leader_index=leader_idx,
                leader_score=leader_score,
                leadership_distribution=leadership_dist
            )
            
            # Create meta-agent
            meta_agent = HierarchicalAgent(
                scale=target_scale,
                local_index=meta_index,
                constituent_indices=constituent_scale_indices,
                meta_descriptor=meta_descriptor,
                agent_id=meta_id,
                config=constituents[0].config,
                rng=np.random.default_rng(hash(meta_id) % 2**32),
                base_manifold=self.base_manifold
            )
            
            # Initialize field structures
            meta_agent.support = self._compute_meta_support(constituents, coherence_scores)
            meta_agent.generators = constituents[0].generators
            
            meta_agent._initialize_belief_cholesky()
            meta_agent._initialize_prior_cholesky()
            meta_agent._initialize_gauge()
            
            # Set renormalized values (overwrites initialized values)
            meta_agent.mu_q = mu_q
            meta_agent.Sigma_q = Sigma_q  # Triggers L_q recomputation via setter
            meta_agent.mu_p = mu_p
            meta_agent.Sigma_p = Sigma_p  # Triggers L_p recomputation via setter
            meta_agent.gauge.phi = phi
            
            # Add to system
            self.agents[target_scale].append(meta_agent)
            new_meta_agents.append(meta_agent)
            
            # Deactivate constituents if requested
            if deactivate_constituents:
                for agent in constituents:
                    agent.is_active = False
            
            # Record condensation event
            self.condensation_events.append({
                'time': self.current_time,
                'source_scale': source_scale,
                'target_scale': target_scale,
                'n_constituents': len(partition),
                'constituent_indices': constituent_scale_indices,
                'leader_index': leader_idx,
                'leader_score': leader_score,
                'coherence': {
                    'belief': belief_coherence,
                    'model': model_coherence
                }
            })
        
        print(f"[Condensation ζ={source_scale}→{target_scale}] "
              f"{len(partitions)} clusters → {len(new_meta_agents)} meta-agents")
        
        # Print leader info for each meta-agent
        for meta in new_meta_agents:
            leader_constituent_idx = meta.meta.constituent_indices[meta.meta.leader_index]
            print(f"  {meta.scale_index}: leader={leader_constituent_idx} "
                  f"(L={meta.meta.leader_score:.3f})")
        
        return new_meta_agents
    
    # =========================================================================
    # Renormalization Methods
    # =========================================================================
    
    def _renormalize_beliefs(self,
                           constituents: List[HierarchicalAgent],
                           coherence_scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute renormalized belief fields via coherence-weighted gauge transport.
        
        Uses Presence × Coherence weighting:
            wᵢ(c) ∝ χᵢ(c) · C̄ᵢ
            μ_M(c) = Σᵢ wᵢ(c) · Ωᵢⱼ[μᵢ(c)]
        
        For 0D: χᵢ = 1, so weights are purely coherence-based
        For spatial: weights vary with location based on presence
        
        Returns:
            (mu_M, Sigma_M): Renormalized belief parameters
        """
        ref = constituents[0]
        K = ref.K
        
        # Compute coherence scores if not provided
        if coherence_scores is None:
            coherence_scores = self._compute_coherence_scores(constituents)
        
        if ref.base_manifold.is_point:
            # === 0D CASE (Transformers) ===
            # All χᵢ = 1, weight by coherence only
            
            mu_weighted = np.zeros(K)
            Sigma_weighted = np.zeros((K, K))
            total_weight = 0.0
            
            for i, agent in enumerate(constituents):
                C_i = coherence_scores[i]
                
                # Transport to reference frame
                omega = compute_transport(
                    ref.gauge.phi,
                    agent.gauge.phi,
                    ref.generators,
                    validate=False
                )
                
                # Accumulate weighted transported statistics
                mu_weighted += C_i * (omega @ agent.mu_q)
                Sigma_weighted += C_i * (omega @ agent.Sigma_q @ omega.T)
                total_weight += C_i
            
            # Normalize
            mu_M = mu_weighted / total_weight
            Sigma_M = Sigma_weighted / total_weight
            
            return mu_M, Sigma_M
        
        else:
            # === SPATIAL CASE ===
            # Weight by presence × coherence at each location
            
            spatial_shape = ref.base_manifold.shape
            mu_M = np.zeros((*spatial_shape, K))
            Sigma_M = np.zeros((*spatial_shape, K, K))
            
            # Precompute transports to reference frame
            transports = []
            for agent in constituents:
                omega = compute_transport(
                    ref.gauge.phi,
                    agent.gauge.phi,
                    ref.generators,
                    validate=False
                )
                transports.append(omega)
            
            # At each spatial location
            for c_idx in np.ndindex(spatial_shape):
                mu_weighted = np.zeros(K)
                Sigma_weighted = np.zeros((K, K))
                total_weight = 0.0
                
                for i, agent in enumerate(constituents):
                    # Presence at this location
                    chi_i = agent.support.chi_weight[c_idx]
                    
                    # Skip if negligible presence
                    if chi_i < 1e-6:
                        continue
                    
                    # Combined weight: presence × coherence
                    w_i = chi_i * coherence_scores[i]
                    
                    # Transport and accumulate
                    omega = transports[i]
                    mu_weighted += w_i * (omega @ agent.mu_q[c_idx])
                    Sigma_weighted += w_i * (omega @ agent.Sigma_q[c_idx] @ omega.T)
                    total_weight += w_i
                
                # Normalize
                if total_weight > 1e-6:
                    mu_M[c_idx] = mu_weighted / total_weight
                    Sigma_M[c_idx] = Sigma_weighted / total_weight
                else:
                    # Fallback: no substantial presence
                    mu_M[c_idx] = np.zeros(K)
                    Sigma_M[c_idx] = np.eye(K)
            
            return mu_M, Sigma_M
    
    def _renormalize_models(self,
                          constituents: List[HierarchicalAgent],
                          coherence_scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute renormalized model/prior fields via coherence-weighted transport.
        
        Uses same Presence × Coherence weighting as beliefs:
            wᵢ(c) ∝ χᵢ(c) · C̄ᵢ
            p_M(c) = Σᵢ wᵢ(c) · Ωᵢⱼ[pᵢ(c)]
        
        Returns:
            (mu_M, Sigma_M): Renormalized model parameters
        """
        # Same mathematical procedure as beliefs, just different fields
        ref = constituents[0]
        K = ref.K
        
        if coherence_scores is None:
            coherence_scores = self._compute_coherence_scores(constituents, field_type='model')
        
        if ref.base_manifold.is_point:
            # 0D case
            mu_weighted = np.zeros(K)
            Sigma_weighted = np.zeros((K, K))
            total_weight = 0.0
            
            for i, agent in enumerate(constituents):
                C_i = coherence_scores[i]
                
                omega = compute_transport(
                    ref.gauge.phi,
                    agent.gauge.phi,
                    ref.generators,
                    validate=False
                )
                
                mu_weighted += C_i * (omega @ agent.mu_p)
                Sigma_weighted += C_i * (omega @ agent.Sigma_p @ omega.T)
                total_weight += C_i
            
            return mu_weighted / total_weight, Sigma_weighted / total_weight
        
        else:
            # Spatial case
            spatial_shape = ref.base_manifold.shape
            mu_M = np.zeros((*spatial_shape, K))
            Sigma_M = np.zeros((*spatial_shape, K, K))
            
            transports = []
            for agent in constituents:
                omega = compute_transport(
                    ref.gauge.phi,
                    agent.gauge.phi,
                    ref.generators,
                    validate=False
                )
                transports.append(omega)
            
            for c_idx in np.ndindex(spatial_shape):
                mu_weighted = np.zeros(K)
                Sigma_weighted = np.zeros((K, K))
                total_weight = 0.0
                
                for i, agent in enumerate(constituents):
                    chi_i = agent.support.chi_weight[c_idx]
                    if chi_i < 1e-6:
                        continue
                    
                    w_i = chi_i * coherence_scores[i]
                    omega = transports[i]
                    
                    mu_weighted += w_i * (omega @ agent.mu_p[c_idx])
                    Sigma_weighted += w_i * (omega @ agent.Sigma_p[c_idx] @ omega.T)
                    total_weight += w_i
                
                if total_weight > 1e-6:
                    mu_M[c_idx] = mu_weighted / total_weight
                    Sigma_M[c_idx] = Sigma_weighted / total_weight
                else:
                    mu_M[c_idx] = np.zeros(K)
                    Sigma_M[c_idx] = np.eye(K)
            
            return mu_M, Sigma_M
    
    def _average_gauge_frames(self,
                            constituents: List[HierarchicalAgent],
                            coherence_scores: Optional[np.ndarray] = None,
                            method: str = 'frechet') -> np.ndarray:
        """
        Compute average gauge frame on SO(3) with coherence weighting.
        
        Uses proper Fréchet mean (geometric average) on the SO(3) manifold:
        1. Map gauge frames to rotation matrices: Rᵢ = exp(φᵢ)
        2. Compute weighted Fréchet mean R̄ 
        3. Map back to Lie algebra: φ̄ = log(R̄)
        
        Weighting:
            - 0D: wᵢ ∝ C̄ᵢ (coherence only)
            - Spatial: wᵢ ∝ χᵢ(center) · C̄ᵢ (presence × coherence at center)
        
        Args:
            constituents: List of agents to average
            coherence_scores: Optional precomputed coherence scores
            method: 'frechet' (geometric, default) or 'euclidean' (fast approximation)
        
        Returns:
            phi_M: Averaged gauge frame in so(3), shape (3,)
        
        Notes:
            - Euclidean average only valid for small deviations (< 0.5 rad)
            - Fréchet mean is geometrically correct but ~10x slower
            - For meta-agent formation, geometric correctness is essential!
        """
        if coherence_scores is None:
            coherence_scores = self._compute_coherence_scores(constituents)
        
        # Extract gauge frames
        phis = [agent.gauge.phi for agent in constituents]
        
        # Compute weights
        if constituents[0].base_manifold.is_point:
            # 0D: Weight by coherence only (all χᵢ = 1)
            weights = coherence_scores
        else:
            # Spatial: Weight by presence × coherence at center
            center_idx = tuple(s//2 for s in constituents[0].base_manifold.shape)
            weights = np.array([
                agent.support.chi_weight[center_idx] * coherence_scores[i]
                for i, agent in enumerate(constituents)
            ])
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Check if all frames are nearly identical (fast path)
        phi_std = np.std(phis, axis=0)
        if np.max(phi_std) < 0.1 and method == 'frechet':
            # All frames within 0.1 rad - Euclidean average is accurate enough
            method = 'euclidean'
        
        # Compute average
        phi_avg = average_gauge_frames_so3(phis, weights=weights, method=method)
        
        return phi_avg
    
    def _compute_meta_support(self,
                             constituents: List[HierarchicalAgent],
                             coherence_scores: Optional[np.ndarray] = None) -> SupportRegion:
        """
        Compute meta-agent support via Presence × Coherence weighting.
        
        χ_M(c) = Σᵢ wᵢ(c) · χᵢ(c)
        where wᵢ(c) ∝ χᵢ(c) · C̄ᵢ
        
        This gives quadratic weighting: χ_M(c) ∝ Σᵢ χᵢ(c)² · C̄ᵢ
        
        Strong coherent agents dominate; weak agents contribute little.
        
        Args:
            constituents: List of constituent agents
            coherence_scores: Optional precomputed coherence scores
        
        Returns:
            support: SupportRegion for meta-agent
        """
        ref = constituents[0]
        
        if ref.base_manifold.is_point:
            return create_full_support(ref.base_manifold)
        
        # Compute coherence scores if not provided
        if coherence_scores is None:
            coherence_scores = self._compute_coherence_scores(constituents)
        
        spatial_shape = ref.base_manifold.shape
        chi_meta = np.zeros(spatial_shape, dtype=np.float32)
        
        # At each spatial location
        for c_idx in np.ndindex(spatial_shape):
            chi_values = []
            weights_unnorm = []
            
            for i, agent in enumerate(constituents):
                chi_i = agent.support.chi_weight[c_idx]
                C_i = coherence_scores[i]
                
                chi_values.append(chi_i)
                weights_unnorm.append(chi_i * C_i)  # Presence × Coherence
            
            # Normalize weights
            total_weight = sum(weights_unnorm)
            
            if total_weight > 1e-6:
                weights = [w / total_weight for w in weights_unnorm]
                chi_meta[c_idx] = sum(w * chi for w, chi in zip(weights, chi_values))
            else:
                # No substantive presence - leave as zero
                chi_meta[c_idx] = 0.0
        
        return SupportRegion(
            base_manifold=ref.base_manifold,
            chi_weight=chi_meta
        )
    
    def _compute_coherence(self,
                          constituents: List[HierarchicalAgent],
                          field_type: str = 'belief') -> float:
        """
        Compute coherence metric for constituent cluster.
        
        Measures how aligned the constituents are (lower KL = higher coherence).
        
        Args:
            constituents: List of agents in cluster
            field_type: 'belief' or 'model'
        
        Returns:
            coherence: Value in [0, 1] (1 = perfect consensus)
        """
        from math_utils.kl_divergence import kl_divergence_gaussians
        
        # Compute average pairwise KL divergence
        kl_sum = 0.0
        n_pairs = 0
        
        for i, agent_i in enumerate(constituents):
            for agent_j in constituents[i+1:]:
                if field_type == 'belief':
                    mu_i, Sigma_i = agent_i.mu_q, agent_i.Sigma_q
                    mu_j, Sigma_j = agent_j.mu_q, agent_j.Sigma_q
                else:
                    mu_i, Sigma_i = agent_i.mu_p, agent_i.Sigma_p
                    mu_j, Sigma_j = agent_j.mu_p, agent_j.Sigma_p
                
                # Symmetrized KL
                kl_ij = kl_divergence_gaussians(mu_i, Sigma_i, mu_j, Sigma_j)
                kl_ji = kl_divergence_gaussians(mu_j, Sigma_j, mu_i, Sigma_i)
                kl_sum += (kl_ij + kl_ji) / 2
                n_pairs += 1
        
        if n_pairs == 0:
            return 1.0
        
        avg_kl = kl_sum / n_pairs
        
        # Convert to coherence metric (exponential decay)
        coherence = np.exp(-avg_kl)
        
        return coherence
    
    def _compute_coherence_scores(self,
                                 constituents: List[HierarchicalAgent],
                                 field_type: str = 'belief') -> np.ndarray:
        """
        Compute coherence score for each agent with the cluster.
        
        C̄ᵢ = exp(-average_KL_with_others)
        
        Args:
            constituents: List of agents in cluster
            field_type: 'belief' or 'model' - which field to measure coherence
        
        Returns:
            scores: Array of coherence values in (0, 1], shape (n,)
        
        Notes:
            - Perfect consensus: C̄ᵢ = 1.0 (KL = 0)
            - Weak coherence: C̄ᵢ ≈ 0.0 (large KL)
            - Used to weight agents in meta-agent formation
        """
        from math_utils.kl_divergence import kl_divergence_gaussians
        
        n = len(constituents)
        coherence_scores = np.zeros(n)
        
        for i, agent_i in enumerate(constituents):
            kl_sum = 0.0
            count = 0
            
            for j, agent_j in enumerate(constituents):
                if i == j:
                    continue
                
                # Compute KL at representative location
                if agent_i.base_manifold.is_point:
                    # 0D case: use fields directly
                    if field_type == 'belief':
                        mu_i, Sigma_i = agent_i.mu_q, agent_i.Sigma_q
                        mu_j, Sigma_j = agent_j.mu_q, agent_j.Sigma_q
                    else:
                        mu_i, Sigma_i = agent_i.mu_p, agent_i.Sigma_p
                        mu_j, Sigma_j = agent_j.mu_p, agent_j.Sigma_p
                else:
                    # Spatial case: use center of support
                    center_idx = tuple(s//2 for s in agent_i.base_manifold.shape)
                    
                    if field_type == 'belief':
                        mu_i = agent_i.mu_q[center_idx]
                        Sigma_i = agent_i.Sigma_q[center_idx]
                        mu_j = agent_j.mu_q[center_idx]
                        Sigma_j = agent_j.Sigma_q[center_idx]
                    else:
                        mu_i = agent_i.mu_p[center_idx]
                        Sigma_i = agent_i.Sigma_p[center_idx]
                        mu_j = agent_j.mu_p[center_idx]
                        Sigma_j = agent_j.Sigma_p[center_idx]
                
                # Symmetric KL divergence
                kl_ij = kl_divergence_gaussians(mu_i, Sigma_i, mu_j, Sigma_j)
                kl_ji = kl_divergence_gaussians(mu_j, Sigma_j, mu_i, Sigma_i)
                kl_sym = (kl_ij + kl_ji) / 2.0
                
                kl_sum += kl_sym
                count += 1
            
            # Average KL with all others
            avg_kl = kl_sum / count if count > 0 else 0.0
            
            # Coherence score via exponential decay
            # KL = 0 → C̄ = 1.0 (perfect coherence)
            # KL = 1 → C̄ ≈ 0.37
            # KL = 5 → C̄ ≈ 0.007 (weak coherence)
            coherence_scores[i] = np.exp(-avg_kl)
        
        return coherence_scores
    
    def _identify_leader(self,
                        constituents: List[HierarchicalAgent],
                        coherence_scores: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Identify the leader agent in the cluster.
        
        Leader = agent with max leadership score L_i = χ_i² · C̄_i
        
        For spatial cases, evaluates at center of support.
        For 0D (transformers), all χ_i = 1, so leader is most coherent agent.
        
        Args:
            constituents: List of agents in cluster
            coherence_scores: Precomputed coherence scores C̄_i
        
        Returns:
            leader_idx: Index of leader in constituents list
            leader_score: Leadership score L_leader
            leadership_distribution: All leadership scores [L_1, ..., L_n]
        
        Physical Meaning:
            The leader is the agent that most strongly "templates" the meta-agent.
            It has strong presence (high χ) and high coherence with the cluster.
        """
        n = len(constituents)
        leadership_scores = np.zeros(n)
        
        for i, agent in enumerate(constituents):
            if agent.base_manifold.is_point:
                # 0D: all χ = 1, leader is most coherent
                chi_i = 1.0
            else:
                # Spatial: evaluate at center
                center_idx = tuple(s//2 for s in agent.base_manifold.shape)
                chi_i = agent.support.chi_weight[center_idx]
            
            # Leadership score: L_i = χ_i² · C̄_i
            leadership_scores[i] = (chi_i ** 2) * coherence_scores[i]
        
        # Identify leader
        leader_idx = int(np.argmax(leadership_scores))
        leader_score = leadership_scores[leader_idx]
        
        return leader_idx, leader_score, leadership_scores
    
    # =========================================================================
    # System Queries
    # =========================================================================
    
    def get_active_agents_at_scale(self, scale: int) -> List[HierarchicalAgent]:
        """Get all active agents at a specific scale."""
        return [a for a in self.agents[scale] if a.is_active]
    
    def get_all_active_agents(self) -> List[HierarchicalAgent]:
        """Get all active agents across all scales."""
        active = []
        for scale in sorted(self.agents.keys()):
            active.extend(self.get_active_agents_at_scale(scale))
        return active
    
    def max_scale(self) -> int:
        """Get maximum scale present in system."""
        return max(self.agents.keys()) if self.agents else 0
    
    def summary(self) -> str:
        """Hierarchical structure summary."""
        lines = ["Multi-Scale System Structure"]
        lines.append("=" * 60)
        lines.append(f"Base manifold: {self.base_manifold}")
        lines.append("")
        
        for scale in sorted(self.agents.keys()):
            agents_at_scale = self.agents[scale]
            active_count = sum(1 for a in agents_at_scale if a.is_active)
            lines.append(f"Scale ζ={scale}: {active_count}/{len(agents_at_scale)} active")
            
            if scale > 0:  # Show meta-agent structure
                for agent in agents_at_scale[:5]:  # First 5
                    if agent.is_active and agent.is_meta:
                        lines.append(f"  {agent.scale_index}: "
                                   f"from {len(agent.constituent_indices)} constituents "
                                   f"(coherence: {agent.meta.belief_coherence:.3f})")
        
        lines.append("")
        lines.append(f"Total condensation events: {len(self.condensation_events)}")
        
        return "\n".join(lines)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_hierarchical_structure(system: MultiScaleSystem) -> Dict:
    """
    Analyze the hierarchical structure and emergence patterns.
    
    Returns:
        Dictionary with structure metrics including leadership
    """
    metrics = {
        'n_scales': len(system.agents),
        'max_scale': system.max_scale(),
        'agents_per_scale': {},
        'active_per_scale': {},
        'total_agents': 0,
        'total_active': 0,
        'condensation_events': len(system.condensation_events),
        'mean_cluster_size': 0,
        'mean_coherence': 0,
        'leadership_stats': {}
    }
    
    # Count agents per scale
    cluster_sizes = []
    coherences = []
    leader_scores = []
    
    for scale, agents in system.agents.items():
        metrics['agents_per_scale'][scale] = len(agents)
        metrics['active_per_scale'][scale] = sum(1 for a in agents if a.is_active)
        metrics['total_agents'] += len(agents)
        metrics['total_active'] += sum(1 for a in agents if a.is_active)
        
        # Track meta-agent properties
        for agent in agents:
            if agent.is_meta:
                cluster_sizes.append(len(agent.meta.constituent_indices))
                coherences.append(agent.meta.belief_coherence)
                if agent.meta.leader_score is not None:
                    leader_scores.append(agent.meta.leader_score)
    
    if cluster_sizes:
        metrics['mean_cluster_size'] = np.mean(cluster_sizes)
        metrics['mean_coherence'] = np.mean(coherences)
    
    if leader_scores:
        metrics['leadership_stats'] = {
            'mean_score': np.mean(leader_scores),
            'min_score': np.min(leader_scores),
            'max_score': np.max(leader_scores),
            'std_score': np.std(leader_scores)
        }
    
    return metrics


def analyze_leadership_chains(system: MultiScaleSystem) -> Dict:
    """
    Trace leadership chains from base agents through meta-agents.
    
    Shows which base agents influence higher-scale structures.
    
    Returns:
        Dictionary mapping base agent indices to their influence chain
    """
    if 0 not in system.agents:
        return {}
    
    chains = {}
    base_agents = system.agents[0]
    
    # Initialize chains for each base agent
    for i, agent in enumerate(base_agents):
        chains[i] = {
            'scales_present': [0],
            'leadership_roles': [],  # (scale, meta_index, is_leader)
            'influence_score': 1.0 if agent.is_active else 0.0
        }
    
    # Trace through meta-agent hierarchy
    for scale in sorted(system.agents.keys()):
        if scale == 0:
            continue
        
        for meta_agent in system.agents[scale]:
            if not meta_agent.is_meta:
                continue
            
            # Get constituent indices at scale-1
            for constituent_idx in meta_agent.meta.constituent_indices:
                # This is tricky - need to map back to base agents
                # For now, record direct parent relationships
                pass
    
    return chains


def print_leadership_summary(system: MultiScaleSystem):
    """
    Print detailed leadership analysis.
    """
    print("\n" + "=" * 70)
    print("LEADERSHIP STRUCTURE ANALYSIS")
    print("=" * 70)
    
    for scale in sorted(system.agents.keys()):
        if scale == 0:
            continue  # Base agents have no leaders
        
        agents_at_scale = [a for a in system.agents[scale] if a.is_active and a.is_meta]
        
        if not agents_at_scale:
            continue
        
        print(f"\nScale ζ={scale}:")
        print("-" * 70)
        
        for agent in agents_at_scale:
            meta = agent.meta
            leader_idx = meta.leader_index
            leader_constituent = meta.constituent_indices[leader_idx]
            
            print(f"\n  Meta-agent {agent.scale_index}:")
            print(f"    Constituents: {meta.constituent_indices}")
            print(f"    Leader: {leader_constituent} (L={meta.leader_score:.4f})")
            print(f"    Leadership distribution:")
            
            # Show all leadership scores
            for i, (const_idx, L_i) in enumerate(zip(meta.constituent_indices, 
                                                     meta.leadership_distribution)):
                is_leader = "← LEADER" if i == leader_idx else ""
                pct = 100 * L_i / np.sum(meta.leadership_distribution)
                print(f"      {const_idx}: L={L_i:.4f} ({pct:.1f}%) {is_leader}")
            
            print(f"    Coherence: {meta.belief_coherence:.4f}")
    
    print("\n" + "=" * 70)