# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 15:24:36 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Cross-Scale Transport Operators and Energy Terms
=================================================

Implements cross-scale morphisms Λ^{s,s'}_{ij} and energy terms for 
hierarchical free energy with agents at different scales.

Theory:
-------
The full hierarchical free energy includes cross-scale terms:

S = S_intra + S_cross

where S_cross = Σ_{s,s'} Σ_{i∈scale_s, j∈scale_s'} 
                η^{ss'}_{ij} KL(q_i^s || Λ^{ss'}_{ij}[q_j^{s'}])

The cross-scale transport Λ^{ss'}_{ij} must handle:
- Dimensional changes (coarse-graining/fine-graining)
- Gauge transport between frames
- Information compression/decompression

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import from existing modules
from math_utils.kl_divergence import kl_divergence_gaussian
from math_utils.transport import compute_transport
from math_utils.generators import generate_so3_generators


class ScaleDirection(Enum):
    """Direction of cross-scale transport"""
    UPSCALE = "upscale"      # s < s' (fine to coarse)
    DOWNSCALE = "downscale"  # s > s' (coarse to fine)
    LATERAL = "lateral"      # s = s' (same scale)


@dataclass
class CrossScaleConfig:
    """Configuration for cross-scale interactions"""
    
    # Cross-scale coupling strengths
    lambda_upscale: float = 1.0      # Fine → coarse influence
    lambda_downscale: float = 0.5    # Coarse → fine influence
    
    # Cross-scale attention temperatures
    kappa_cross: float = 1.0         # Temperature for η^{ss'}_{ij}
    
    # Connectivity
    max_scale_gap: int = 1           # Maximum |s - s'| for direct coupling
    connectivity: str = "nearest"    # "nearest", "all", "hierarchical"
    
    # Information compression
    compression_method: str = "average"  # "average", "max", "attention"
    decompression_method: str = "broadcast"  # "broadcast", "interpolate"


class CrossScaleTransport:
    """
    Implements cross-scale transport operators Λ^{ss'}_{ij}.
    
    Handles belief/model transport between different hierarchical scales.
    """
    
    def __init__(self, 
                 K: int,
                 generators: Optional[np.ndarray] = None,
                 config: Optional[CrossScaleConfig] = None):
        """
        Initialize cross-scale transport.
        
        Args:
            K: Latent dimension
            generators: SO(3) generators for gauge transport
            config: Cross-scale configuration
        """
        self.K = K
        self.generators = generators if generators is not None else generate_so3_generators()
        self.config = config if config is not None else CrossScaleConfig()
        
        # Cache for transport operators
        self._transport_cache = {}
        
    def compute_cross_scale_transport(self,
                                     agent_i,
                                     agent_j,
                                     scale_i: int,
                                     scale_j: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-scale transport operator Λ^{ss'}_{ij}.
        
        Returns both mean and covariance transport operators.
        
        Args:
            agent_i: Agent at scale s
            agent_j: Agent at scale s'
            scale_i, scale_j: Hierarchical scales
            
        Returns:
            Lambda_mu: Transport for means
            Lambda_Sigma: Transport for covariances
        """
        
        # Determine transport direction
        if scale_i == scale_j:
            direction = ScaleDirection.LATERAL
        elif scale_i < scale_j:
            direction = ScaleDirection.UPSCALE
        else:
            direction = ScaleDirection.DOWNSCALE
        
        # Get gauge transport
        Omega_ij = compute_transport(
            agent_i.gauge.phi,
            agent_j.gauge.phi,
            self.generators,
            validate=False
        )
        
        if direction == ScaleDirection.LATERAL:
            # Same scale: just gauge transport
            Lambda_mu = Omega_ij
            Lambda_Sigma = Omega_ij
            
        elif direction == ScaleDirection.UPSCALE:
            # Fine to coarse: compression + gauge transport
            Lambda_mu = self._compute_upscale_transport(
                agent_i, agent_j, Omega_ij
            )
            Lambda_Sigma = Lambda_mu  # Same for covariances
            
        else:  # DOWNSCALE
            # Coarse to fine: decompression + gauge transport
            Lambda_mu = self._compute_downscale_transport(
                agent_i, agent_j, Omega_ij
            )
            Lambda_Sigma = Lambda_mu
        
        return Lambda_mu, Lambda_Sigma
    
    def _compute_upscale_transport(self,
                                  fine_agent,
                                  coarse_agent,
                                  Omega: np.ndarray) -> np.ndarray:
        """
        Compute upscale transport (fine → coarse).
        
        This involves:
        1. Aggregating information from fine scale
        2. Applying gauge transport
        3. Projecting to coarse representation
        """
        
        if self.config.compression_method == "average":
            # Simple averaging in latent space
            # For meta-agent, this would average over constituents
            Lambda = Omega
            
        elif self.config.compression_method == "max":
            # Max-pooling style compression
            Lambda = Omega
            
        elif self.config.compression_method == "attention":
            # Attention-weighted compression
            # Would need constituent attention weights
            Lambda = Omega
            
        else:
            Lambda = Omega
        
        return Lambda
    
    def _compute_downscale_transport(self,
                                    coarse_agent,
                                    fine_agent,
                                    Omega: np.ndarray) -> np.ndarray:
        """
        Compute downscale transport (coarse → fine).
        
        This involves:
        1. Broadcasting coarse information
        2. Applying gauge transport
        3. Localizing to fine scale
        """
        
        if self.config.decompression_method == "broadcast":
            # Simple broadcasting
            Lambda = Omega
            
        elif self.config.decompression_method == "interpolate":
            # Smooth interpolation
            Lambda = Omega
            
        else:
            Lambda = Omega
        
        return Lambda
    
    def transport_belief(self,
                        mu: np.ndarray,
                        Sigma: np.ndarray,
                        Lambda_mu: np.ndarray,
                        Lambda_Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply cross-scale transport to belief distribution.
        
        Args:
            mu, Sigma: Original belief parameters
            Lambda_mu, Lambda_Sigma: Transport operators
            
        Returns:
            mu_transported, Sigma_transported
        """
        mu_transported = Lambda_mu @ mu
        Sigma_transported = Lambda_Sigma @ Sigma @ Lambda_Sigma.T
        
        return mu_transported, Sigma_transported


class CrossScaleAttention:
    """
    Computes cross-scale attention weights η^{ss'}_{ij}.
    
    These weights control the strength of cross-scale interactions.
    """
    
    def __init__(self, config: Optional[CrossScaleConfig] = None):
        self.config = config if config is not None else CrossScaleConfig()
        
    def compute_cross_scale_weights(self,
                                   hierarchical_system) -> Dict[Tuple, np.ndarray]:
        """
        Compute all cross-scale attention weights.
        
        Returns:
            weights: Dict mapping (scale_i, scale_j) -> weight matrix η^{ss'}
        """
        weights = {}
        
        # Get all scales
        scales = sorted(hierarchical_system.agents_by_scale.keys())
        
        for scale_i in scales:
            for scale_j in scales:
                # Check if scales should interact
                if not self._should_interact(scale_i, scale_j):
                    continue
                
                # Get agents at each scale
                agents_i = hierarchical_system.agents_by_scale[scale_i]
                agents_j = hierarchical_system.agents_by_scale[scale_j]
                
                # Compute weight matrix
                n_i = len(agents_i)
                n_j = len(agents_j)
                eta = np.zeros((n_i, n_j))
                
                for i, agent_i in enumerate(agents_i):
                    for j, agent_j in enumerate(agents_j):
                        if not (agent_i.is_active and agent_j.is_active):
                            continue
                        
                        # Compute cross-scale attention
                        eta[i, j] = self._compute_pairwise_attention(
                            agent_i, agent_j, scale_i, scale_j
                        )
                
                weights[(scale_i, scale_j)] = eta
        
        return weights
    
    def _should_interact(self, scale_i: int, scale_j: int) -> bool:
        """Check if two scales should have direct interaction."""
        
        if self.config.connectivity == "all":
            return True
        
        elif self.config.connectivity == "nearest":
            return abs(scale_i - scale_j) <= self.config.max_scale_gap
        
        elif self.config.connectivity == "hierarchical":
            # Only adjacent scales or parent-child
            return abs(scale_i - scale_j) == 1
        
        return False
    
    def _compute_pairwise_attention(self,
                                   agent_i,
                                   agent_j,
                                   scale_i: int,
                                   scale_j: int) -> float:
        """
        Compute attention weight η^{ss'}_{ij}.
        
        Based on:
        - Scale difference
        - Spatial overlap (if applicable)
        - Hierarchical relationship (parent-child)
        """
        
        # Base weight depends on scale relationship
        if scale_i == scale_j:
            base_weight = 1.0  # Intra-scale (normal attention)
        elif scale_i < scale_j:
            base_weight = self.config.lambda_upscale  # Fine to coarse
        else:
            base_weight = self.config.lambda_downscale  # Coarse to fine
        
        # Modulate by scale distance
        scale_distance = abs(scale_i - scale_j)
        distance_factor = 1.0 / (1.0 + scale_distance)
        
        # Check for parent-child relationship (stronger coupling)
        if hasattr(agent_i, 'meta') and hasattr(agent_j, 'meta'):
            if scale_i < scale_j:  # i could be constituent of j
                if (agent_j.is_meta and 
                    agent_i.agent_id in agent_j.meta.constituent_ids):
                    distance_factor = 2.0  # Stronger parent-child coupling
            elif scale_i > scale_j:  # j could be constituent of i
                if (agent_i.is_meta and 
                    agent_j.agent_id in agent_i.meta.constituent_ids):
                    distance_factor = 2.0
        
        return base_weight * distance_factor


class CrossScaleFreeEnergy:
    """
    Computes cross-scale terms in the hierarchical free energy.
    
    S_cross = Σ_{s,s'} Σ_{i,j} η^{ss'}_{ij} [
                KL(q_i^s || Λ^{ss'}_{ij}[q_j^{s'}]) +
                KL(p_i^s || Λ^{ss'}_{ij}[p_j^{s'}])
              ]
    """
    
    def __init__(self,
                 transport: CrossScaleTransport,
                 attention: CrossScaleAttention):
        self.transport = transport
        self.attention = attention
        
    def compute_cross_scale_energy(self,
                                  hierarchical_system) -> Dict[str, float]:
        """
        Compute all cross-scale energy terms.
        
        Returns:
            energy_dict: Breakdown of cross-scale energies
        """
        total_energy = 0.0
        energy_breakdown = {}
        
        # Get cross-scale attention weights
        eta_weights = self.attention.compute_cross_scale_weights(
            hierarchical_system
        )
        
        # Compute energy for each scale pair
        for (scale_i, scale_j), eta in eta_weights.items():
            if scale_i == scale_j:
                continue  # Skip intra-scale (handled elsewhere)
            
            agents_i = hierarchical_system.agents_by_scale[scale_i]
            agents_j = hierarchical_system.agents_by_scale[scale_j]
            
            scale_pair_energy = 0.0
            
            for i, agent_i in enumerate(agents_i):
                if not agent_i.is_active:
                    continue
                    
                for j, agent_j in enumerate(agents_j):
                    if not agent_j.is_active:
                        continue
                    
                    # Get cross-scale transport operators
                    Lambda_mu, Lambda_Sigma = self.transport.compute_cross_scale_transport(
                        agent_i, agent_j, scale_i, scale_j
                    )
                    
                    # Transport agent_j's belief to agent_i's scale/frame
                    mu_j_transported, Sigma_j_transported = self.transport.transport_belief(
                        agent_j.mu_q, agent_j.Sigma_q,
                        Lambda_mu, Lambda_Sigma
                    )
                    
                    # Compute KL divergence
                    kl_belief = kl_divergence_gaussian(
                        agent_i.mu_q, agent_i.Sigma_q,
                        mu_j_transported, Sigma_j_transported
                    )
                    
                    # Weight by cross-scale attention
                    weighted_kl = eta[i, j] * kl_belief
                    scale_pair_energy += weighted_kl
            
            energy_breakdown[f'cross_{scale_i}_{scale_j}'] = scale_pair_energy
            total_energy += scale_pair_energy
        
        energy_breakdown['total_cross'] = total_energy
        return energy_breakdown
    
    def compute_cross_scale_gradients(self,
                                     hierarchical_system,
                                     agent_idx: int,
                                     agent_scale: int) -> Dict:
        """
        Compute gradients from cross-scale interactions for one agent.
        
        Returns:
            gradients: Dict with grad_mu_q, grad_Sigma_q, grad_phi components
        """
        agent = hierarchical_system.agents_by_scale[agent_scale][agent_idx]
        K = agent.K
        
        # Initialize gradients
        grad_mu = np.zeros(K)
        grad_Sigma = np.zeros((K, K))
        grad_phi = np.zeros(3)
        
        # Get attention weights
        eta_weights = self.attention.compute_cross_scale_weights(
            hierarchical_system
        )
        
        # Accumulate gradients from all scales
        for scale_j in hierarchical_system.agents_by_scale.keys():
            if scale_j == agent_scale:
                continue  # Skip same scale
                
            if (agent_scale, scale_j) not in eta_weights:
                continue
                
            eta = eta_weights[(agent_scale, scale_j)]
            agents_j = hierarchical_system.agents_by_scale[scale_j]
            
            for j, agent_j in enumerate(agents_j):
                if not agent_j.is_active:
                    continue
                
                # Get transport operators
                Lambda_mu, Lambda_Sigma = self.transport.compute_cross_scale_transport(
                    agent, agent_j, agent_scale, scale_j
                )
                
                # Transport agent_j to agent_i's frame
                mu_j_transported, Sigma_j_transported = self.transport.transport_belief(
                    agent_j.mu_q, agent_j.Sigma_q,
                    Lambda_mu, Lambda_Sigma
                )
                
                # Compute gradient of KL(q_i || Λ q_j)
                Sigma_j_inv = np.linalg.inv(Sigma_j_transported + 1e-8 * np.eye(K))
                
                # Gradient w.r.t. mu_i
                delta_mu = agent.mu_q - mu_j_transported
                grad_mu += eta[agent_idx, j] * Sigma_j_inv @ delta_mu
                
                # Gradient w.r.t. Sigma_i  
                Sigma_i_inv = np.linalg.inv(agent.Sigma_q + 1e-8 * np.eye(K))
                grad_Sigma += 0.5 * eta[agent_idx, j] * (Sigma_j_inv - Sigma_i_inv)
                
                # Gradient w.r.t. gauge field (through transport operator)
                # This is more complex and depends on the derivative of transport
                # For now, simplified contribution
                grad_phi += eta[agent_idx, j] * 0.01 * agent.gauge.phi
        
        return {
            'grad_mu_q': grad_mu,
            'grad_Sigma_q': grad_Sigma,
            'grad_phi': grad_phi
        }


class HierarchicalEnergyComputer:
    """
    Computes total hierarchical free energy with cross-scale terms.
    
    S_total = S_intra + S_cross
    
    where:
    - S_intra: Standard within-scale terms (self, alignment, obs)
    - S_cross: Cross-scale coupling terms
    """
    
    def __init__(self,
                 cross_scale_config: Optional[CrossScaleConfig] = None):
        self.config = cross_scale_config or CrossScaleConfig()
        self.transport = CrossScaleTransport(K=3, config=self.config)
        self.attention = CrossScaleAttention(self.config)
        self.cross_energy = CrossScaleFreeEnergy(self.transport, self.attention)
        
    def compute_total_hierarchical_energy(self,
                                         hierarchical_system,
                                         include_cross_scale: bool = True) -> Dict:
        """
        Compute complete hierarchical free energy.
        
        Args:
            hierarchical_system: HierarchicalMultiAgentSystem
            include_cross_scale: Whether to include cross-scale terms
            
        Returns:
            energy_dict: Complete energy breakdown
        """
        energy_dict = {}
        
        # 1. Compute standard intra-scale energy
        # This uses the existing free energy computation
        base_system = hierarchical_system.base_system
        try:
            intra_energy = base_system.compute_free_energy()
            energy_dict.update(intra_energy)
        except:
            energy_dict['intra_scale'] = 0.0
        
        # 2. Add cross-scale terms if requested
        if include_cross_scale:
            cross_energy = self.cross_energy.compute_cross_scale_energy(
                hierarchical_system
            )
            energy_dict.update(cross_energy)
            
            # Total energy
            energy_dict['total'] = (
                energy_dict.get('total', 0.0) + 
                cross_energy.get('total_cross', 0.0)
            )
        
        return energy_dict
    
    def compute_hierarchical_gradients(self,
                                      hierarchical_system) -> Dict:
        """
        Compute gradients including cross-scale terms.
        
        Returns:
            gradients: Dict mapping agent_id -> gradient components
        """
        all_gradients = {}
        
        # 1. Get standard intra-scale gradients
        # (Would call existing gradient computation)
        
        # 2. Add cross-scale gradient contributions
        for scale, agents in hierarchical_system.agents_by_scale.items():
            for i, agent in enumerate(agents):
                if not agent.is_active:
                    continue
                    
                # Get cross-scale gradients
                cross_grads = self.cross_energy.compute_cross_scale_gradients(
                    hierarchical_system, i, scale
                )
                
                # Accumulate with existing gradients
                if agent.agent_id not in all_gradients:
                    all_gradients[agent.agent_id] = cross_grads
                else:
                    # Add to existing
                    all_gradients[agent.agent_id]['grad_mu_q'] += cross_grads['grad_mu_q']
                    all_gradients[agent.agent_id]['grad_Sigma_q'] += cross_grads['grad_Sigma_q']
                    all_gradients[agent.agent_id]['grad_phi'] += cross_grads['grad_phi']
        
        return all_gradients


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_cross_scale_coupling(hierarchical_system) -> Dict:
    """
    Analyze cross-scale coupling structure.
    
    Returns metrics about information flow between scales.
    """
    config = CrossScaleConfig()
    attention = CrossScaleAttention(config)
    
    # Get attention weights
    eta_weights = attention.compute_cross_scale_weights(hierarchical_system)
    
    metrics = {
        'n_cross_scale_pairs': len(eta_weights),
        'coupling_strengths': {},
        'information_flow': {'upward': 0, 'downward': 0}
    }
    
    for (scale_i, scale_j), eta in eta_weights.items():
        if scale_i == scale_j:
            continue
            
        mean_coupling = np.mean(eta[eta > 0]) if np.any(eta > 0) else 0
        metrics['coupling_strengths'][f'{scale_i}_to_{scale_j}'] = mean_coupling
        
        if scale_i < scale_j:
            metrics['information_flow']['upward'] += mean_coupling
        else:
            metrics['information_flow']['downward'] += mean_coupling
    
    return metrics