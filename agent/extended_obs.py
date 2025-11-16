# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 17:13:01 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Observation Model Extensions
============================

Methods to extend MultiAgentSystem with different observation modes:
1. Ground truth observations
2. Dynamic observations
3. Shared observations
4. Multiple observations per agent

Add these to your MultiAgentSystem class or use as standalone utilities.

Author: Chris 
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Extension Methods for MultiAgentSystem
# =============================================================================

class ObservationExtensions:
    """Mix-in class for observation model extensions."""
    
    def set_observations_from_ground_truth(
        self,
        x_true: np.ndarray,
        noise_scale: float = 0.2,
        per_agent_bias: bool = True,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Set observations from ground truth latent state.
        
        Use case: Reconstruction problem where true state exists.
        Each agent observes at their location with noise.
        
        Args:
            x_true: Ground truth latent state, shape (*S, K)
            noise_scale: Observation noise std dev
            per_agent_bias: If True, add agent-specific bias
            rng: Random generator (creates new if None)
        
        Example:
            >>> # Create ground truth (e.g., smooth function)
            >>> x_true = initialize_ground_truth_field(manifold.shape, K)
            >>> 
            >>> # Set observations from it
            >>> system.set_observations_from_ground_truth(x_true, noise_scale=0.1)
            >>> 
            >>> # Train to reconstruct
            >>> history = trainer.train(system, n_epochs=1000)
            >>> 
            >>> # Measure reconstruction error
            >>> error = np.mean([np.linalg.norm(agent.mu_q - x_true) 
            >>>                  for agent in system.agents])
        """
        if rng is None:
            rng = np.random.default_rng()
        
        D_x = self.W_obs.shape[0]
        
        for agent in self.agents:
            # Find observation location (strongest point)
            chi = agent.support.chi_weight
            
            if agent.base_manifold.is_point:
                coord = ()
                x_true_at_coord = x_true
            else:
                flat_idx = np.argmax(chi)
                coord = np.unravel_index(flat_idx, chi.shape)
                x_true_at_coord = x_true[coord]
            
            # Generate observation: o = W @ x_true + noise + bias
            y_true = self.W_obs @ x_true_at_coord
            noise = rng.normal(scale=noise_scale, size=(D_x,))
            
            if per_agent_bias:
                if not hasattr(agent, 'obs_bias'):
                    agent.obs_bias = rng.normal(scale=0.1, size=(D_x,))
                bias = agent.obs_bias
            else:
                bias = 0.0
            
            observation = (y_true + noise + bias).astype(np.float32)
            agent.observations = {coord: observation}
        
        print(f"✓ Set observations from ground truth (noise_scale={noise_scale})")
    
    
    def update_observations_dynamic(
        self,
        noise_scale: float = 0.2,
        resample_from: str = 'beliefs',
        rng: Optional[np.random.Generator] = None
    ):
        """
        Update observations with new measurements (dynamic sensing).
        
        Use case: Online learning with streaming data.
        Call this each epoch to simulate new observations arriving.
        
        Args:
            noise_scale: Observation noise std dev
            resample_from: 'beliefs' (current μ_q) or 'priors' (p)
            rng: Random generator
        
        Example:
            >>> for epoch in range(n_epochs):
            >>>     # New observations arrive
            >>>     system.update_observations_dynamic(noise_scale=0.15)
            >>>     
            >>>     # Update beliefs to match new data
            >>>     system.step()
        """
        if rng is None:
            rng = np.random.default_rng()
        
        D_x = self.W_obs.shape[0]
        
        for agent in self.agents:
            chi = agent.support.chi_weight
            
            # Get coordinate
            if agent.base_manifold.is_point:
                coord = ()
                if resample_from == 'beliefs':
                    mu_source = agent.mu_q
                else:
                    mu_source = agent.mu_p
            else:
                flat_idx = np.argmax(chi)
                coord = np.unravel_index(flat_idx, chi.shape)
                if resample_from == 'beliefs':
                    mu_source = agent.mu_q[coord]
                else:
                    mu_source = agent.mu_p[coord]
            
            # Generate new observation
            y_pred = self.W_obs @ mu_source
            noise = rng.normal(scale=noise_scale, size=(D_x,))
            
            # Add persistent bias if exists
            bias = agent.obs_bias if hasattr(agent, 'obs_bias') else 0.0
            
            observation = (y_pred + noise + bias).astype(np.float32)
            agent.observations = {coord: observation}
    
    
    def set_shared_observations(
        self,
        observation_field: np.ndarray,
        noise_per_agent: bool = False,
        noise_scale: float = 0.1,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Set same observations for all agents (shared data).
        
        Use case: Ensemble learning - multiple models fitting same dataset.
        All agents see the same observations but may interpret differently.
        
        Args:
            observation_field: Observation at each location, shape (*S, D_x)
            noise_per_agent: If True, add different noise per agent
            noise_scale: Noise std dev if noise_per_agent=True
            rng: Random generator
        
        Example:
            >>> # Generate observations from true state
            >>> obs_field = W_obs @ x_true  # (*S, D_x)
            >>> 
            >>> # All agents see same data (with optional per-agent noise)
            >>> system.set_shared_observations(obs_field, noise_per_agent=True)
            >>> 
            >>> # Agents will converge differently based on priors/initialization
        """
        if rng is None:
            rng = np.random.default_rng()
        
        for agent in self.agents:
            chi = agent.support.chi_weight
            
            # Get observation at agent's location
            if agent.base_manifold.is_point:
                coord = ()
                obs_base = observation_field
            else:
                flat_idx = np.argmax(chi)
                coord = np.unravel_index(flat_idx, chi.shape)
                obs_base = observation_field[coord]
            
            # Add per-agent noise if requested
            if noise_per_agent:
                noise = rng.normal(scale=noise_scale, size=obs_base.shape)
                observation = (obs_base + noise).astype(np.float32)
            else:
                observation = obs_base.astype(np.float32)
            
            agent.observations = {coord: observation}
        
        mode = "with per-agent noise" if noise_per_agent else "exact"
        print(f"✓ Set shared observations {mode}")
    
    
    def add_multiple_observations(
        self,
        agent_idx: int,
        coords: List[Tuple],
        observations: List[np.ndarray]
    ):
        """
        Add multiple observations for a single agent.
        
        Use case: Rich sensing with multiple measurements per agent.
        
        Args:
            agent_idx: Agent index
            coords: List of spatial coordinates
            observations: List of observation vectors (D_x,)
        
        Example:
            >>> # Agent 0 makes observations at 3 locations
            >>> coords = [(5, 5), (6, 6), (7, 7)]
            >>> obs = [o1, o2, o3]  # Each shape (D_x,)
            >>> 
            >>> system.add_multiple_observations(0, coords, obs)
            >>> 
            >>> # Energy will integrate over all 3 observations
        """
        if len(coords) != len(observations):
            raise ValueError("coords and observations must have same length")
        
        agent = self.agents[agent_idx]
        
        # Replace or extend observations
        agent.observations = {
            coord: obs.astype(np.float32)
            for coord, obs in zip(coords, observations)
        }
        
        print(f"✓ Agent {agent_idx}: {len(coords)} observations set")


# =============================================================================
# Ground Truth Generation Utilities
# =============================================================================

def initialize_ground_truth_smooth(
    manifold_shape: Tuple[int, ...],
    K: int,
    n_modes: int = 3,
    amplitude: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate smooth ground truth using sum of sinusoids.
    
    Args:
        manifold_shape: Spatial shape (*S,)
        K: Latent dimension
        n_modes: Number of Fourier modes per dimension
        amplitude: Signal amplitude
        rng: Random generator
    
    Returns:
        x_true: Ground truth latent state (*S, K)
    
    Example:
        >>> x_true = initialize_ground_truth_smooth((16, 16), K=3)
        >>> # Result: smooth 16x16 field with 3 latent dimensions
    """
    if rng is None:
        rng = np.random.default_rng()
    
    ndim = len(manifold_shape)
    
    if ndim == 0:
        # 0D: just return random vector
        return rng.normal(size=(K,)).astype(np.float32)
    
    # Create coordinate grids
    coords = np.meshgrid(*[np.linspace(0, 2*np.pi, n) for n in manifold_shape], 
                         indexing='ij')
    
    # Initialize field
    x_true = np.zeros((*manifold_shape, K), dtype=np.float32)
    
    # Generate each latent component
    for k in range(K):
        field_k = np.zeros(manifold_shape, dtype=np.float32)
        
        # Sum of sinusoids
        for _ in range(n_modes):
            # Random frequencies and phases
            freqs = rng.integers(1, 4, size=ndim)
            phases = rng.uniform(0, 2*np.pi, size=ndim)
            
            # Compute wave
            wave = np.ones(manifold_shape)
            for d in range(ndim):
                wave *= np.sin(freqs[d] * coords[d] + phases[d])
            
            field_k += wave
        
        # Normalize
        field_k = amplitude * field_k / np.std(field_k)
        x_true[..., k] = field_k
    
    return x_true


def initialize_ground_truth_localized(
    manifold_shape: Tuple[int, ...],
    K: int,
    n_centers: int = 3,
    amplitude: float = 1.0,
    width: float = 0.2,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate localized ground truth (sum of Gaussians).
    
    Args:
        manifold_shape: Spatial shape (*S,)
        K: Latent dimension
        n_centers: Number of Gaussian bumps
        amplitude: Signal amplitude
        width: Gaussian width (relative to domain size)
        rng: Random generator
    
    Returns:
        x_true: Ground truth latent state (*S, K)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    ndim = len(manifold_shape)
    
    if ndim == 0:
        return rng.normal(size=(K,)).astype(np.float32)
    
    # Create coordinate grids
    grids = np.meshgrid(*[np.arange(n) for n in manifold_shape], indexing='ij')
    
    x_true = np.zeros((*manifold_shape, K), dtype=np.float32)
    
    for k in range(K):
        field_k = np.zeros(manifold_shape, dtype=np.float32)
        
        # Add Gaussian bumps
        for _ in range(n_centers):
            # Random center
            center = [rng.uniform(0, n) for n in manifold_shape]
            
            # Compute distance
            dist_sq = sum((grid - c)**2 for grid, c in zip(grids, center))
            
            # Gaussian bump
            sigma = width * np.mean(manifold_shape)
            bump = np.exp(-dist_sq / (2 * sigma**2))
            
            field_k += rng.normal() * bump
        
        # Normalize
        field_k = amplitude * field_k / (np.std(field_k) + 1e-8)
        x_true[..., k] = field_k
    
    return x_true


# =============================================================================
# Observation Quality Metrics
# =============================================================================

def compute_reconstruction_error(
    system,
    x_true: np.ndarray,
    metric: str = 'mse'
) -> Dict[str, float]:
    """
    Compute reconstruction error between beliefs and ground truth.
    
    Args:
        system: MultiAgentSystem
        x_true: Ground truth latent state (*S, K)
        metric: 'mse', 'mae', or 'correlation'
    
    Returns:
        errors: Dict with per-agent and mean errors
    
    Example:
        >>> x_true = initialize_ground_truth_smooth(manifold.shape, K=3)
        >>> system.set_observations_from_ground_truth(x_true)
        >>> train(system)
        >>> errors = compute_reconstruction_error(system, x_true)
        >>> print(f"Reconstruction MSE: {errors['mean']:.4f}")
    """
    errors = {}
    
    for i, agent in enumerate(system.agents):
        mu_q = agent.mu_q
        
        if metric == 'mse':
            error = float(np.mean((mu_q - x_true)**2))
        elif metric == 'mae':
            error = float(np.mean(np.abs(mu_q - x_true)))
        elif metric == 'correlation':
            # Correlation coefficient
            mu_flat = mu_q.reshape(-1)
            true_flat = x_true.reshape(-1)
            corr = np.corrcoef(mu_flat, true_flat)[0, 1]
            error = 1.0 - corr  # Convert to error (0 = perfect)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        errors[f'agent_{i}'] = error
    
    errors['mean'] = np.mean([v for k, v in errors.items() if k.startswith('agent_')])
    errors['std'] = np.std([v for k, v in errors.items() if k.startswith('agent_')])
    
    return errors


def compute_observation_likelihood(system) -> Dict[str, float]:
    """
    Compute log-likelihood of observations under current beliefs.
    
    Higher is better (beliefs explain observations well).
    
    Returns:
        likelihoods: Dict with per-agent and total log-likelihoods
    """
    from free_energy.free_energy_clean import expected_log_likelihood_gaussian
    
    likelihoods = {}
    
    for i, agent in enumerate(system.agents):
        if not hasattr(agent, 'observations') or not agent.observations:
            likelihoods[f'agent_{i}'] = 0.0
            continue
        
        total_ll = 0.0
        
        for coord, o_obs in agent.observations.items():
            if agent.base_manifold.is_point:
                mu_q = agent.mu_q
                Sigma_q = agent.Sigma_q
            else:
                mu_q = agent.mu_q[coord]
                Sigma_q = agent.Sigma_q[coord]
            
            ll = expected_log_likelihood_gaussian(
                o_obs, mu_q, Sigma_q,
                agent.C_obs, agent.R_obs
            )
            total_ll += ll
        
        likelihoods[f'agent_{i}'] = total_ll
    
    likelihoods['total'] = sum(v for k, v in likelihoods.items() 
                               if k.startswith('agent_'))
    
    return likelihoods


# =============================================================================
# Example Usage
# =============================================================================

def example_reconstruction_experiment():
    """
    Example: Reconstruction from noisy observations.
    """
    
    from agent.trainer import Trainer
    
    # Create system (assume already initialized)
    system = ...  # Your MultiAgentSystem
    
    # 1. Generate smooth ground truth
    manifold_shape = system.agents[0].support.base_shape
    K = system.agents[0].K
    
    x_true = initialize_ground_truth_smooth(
        manifold_shape, K,
        n_modes=3,
        amplitude=1.0
    )
    
    # 2. Set observations from ground truth
    system.set_observations_from_ground_truth(
        x_true,
        noise_scale=0.1,
        per_agent_bias=True
    )
    
    # 3. Train to reconstruct
    trainer = Trainer(config=...)
    history = trainer.train(system, n_epochs=1000)
    
    # 4. Measure reconstruction quality
    errors = compute_reconstruction_error(system, x_true, metric='mse')
    print(f"Final reconstruction MSE: {errors['mean']:.4f} ± {errors['std']:.4f}")
    
    return system, x_true, history


def example_dynamic_sensing():
    """
    Example: Online learning with streaming observations.
    """
    system = ...  # Your MultiAgentSystem
    
    history = {'energies': [], 'likelihoods': []}
    
    for epoch in range(100):
        # New observations arrive
        system.update_observations_dynamic(
            noise_scale=0.15,
            resample_from='beliefs'  # or 'priors'
        )
        
        # Update beliefs
        energies = system.step()
        
        # Track metrics
        lls = compute_observation_likelihood(system)
        history['energies'].append(energies['total'])
        history['likelihoods'].append(lls['total'])
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: E={energies['total']:.2f}, "
                  f"LL={lls['total']:.2f}")
    
    return system, history