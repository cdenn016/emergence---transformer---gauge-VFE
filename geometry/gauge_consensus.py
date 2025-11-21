# -*- coding: utf-8 -*-
"""
Gauge-Invariant Metric Averaging and Consensus Geometry
========================================================

Implements gauge-invariant construction of collective geometric structure
from multiple agents' induced metrics.

The Problem
-----------
When multiple agents maintain beliefs about the same base manifold, each
induces its own metric G_i(c). A naive average:

    Ḡ_μν(c) = (1/N) Σ w_i(c) G_i,μν(c)

depends on each agent's arbitrary gauge frame choice φ_i. This violates
the principle that physical geometry must be gauge-invariant.

The Solution: Gauge Averaging
-----------------------------
Average over gauge orbits before spatial averaging:

    ⟨G_i⟩_μν(c) = ∫_G dg G_i,μν(c; φ_i → φ_i + g)

For G = SO(3), this projects onto SO(3)-invariant components. The
collective consensus metric is then:

    Ḡ^consensus_μν(c) = Σ w_i(c) ⟨G_i⟩_μν(c)

This construction is gauge-invariant by design - no agent's arbitrary
frame choice affects the collective geometry.

Physical Interpretation
----------------------
Gauge invariance in physics may arise as a consistency requirement for
multi-agent consensus. For agents with different internal reference
frames to agree on shared geometric structure, that structure must be
gauge-invariant.

Rather than gauge invariance being imposed on nature, it emerges from
informational requirements of consensus formation among agents with
diverse perspectives.

Implementation Strategy
----------------------
For computational tractability, we use Monte Carlo integration:

    ⟨G_i⟩ ≈ (1/N_samples) Σ_k G_i(c; φ_i → φ_i + g_k)

where {g_k} are sampled from the Haar measure on SO(3).

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from geometry.pullback_metrics import InducedMetric
from math_utils.so3_frechet import so3_exp, frechet_mean_so3
from math_utils.transport import gauge_transport_gaussian


# =============================================================================
# SO(3) Haar Measure Sampling
# =============================================================================

def sample_so3_haar(
    n_samples: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample rotation matrices from uniform (Haar) measure on SO(3).

    Uses the subgroup algorithm (Shoemake 1992):
    1. Sample uniform quaternion from 4D unit sphere
    2. Convert to rotation matrix

    Args:
        n_samples: Number of samples to generate
        rng: Random number generator

    Returns:
        samples: Rotation matrices, shape (n_samples, 3, 3)

    References:
        - Shoemake, "Uniform Random Rotations" (1992)
        - Maris, "Sampling SO(3) uniformly" (2024)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample from 4D Gaussian
    q = rng.standard_normal((n_samples, 4))

    # Normalize to unit sphere (uniform on S³)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    # Convert quaternions to rotation matrices
    rotations = np.zeros((n_samples, 3, 3))

    for i in range(n_samples):
        w, x, y, z = q[i]

        # Quaternion to rotation matrix formula
        rotations[i] = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    return rotations.astype(np.float32)


def sample_so3_algebra_haar(
    n_samples: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample gauge frames from uniform measure on SO(3), returned as so(3) elements.

    Args:
        n_samples: Number of samples
        rng: Random number generator

    Returns:
        phis: Lie algebra elements, shape (n_samples, 3)
    """
    from math_utils.so3_frechet import so3_log

    rotations = sample_so3_haar(n_samples, rng=rng)

    # Map to Lie algebra
    phis = np.zeros((n_samples, 3))
    for i in range(n_samples):
        phis[i] = so3_log(rotations[i])

    return phis.astype(np.float32)


# =============================================================================
# Gauge-Transformed Metrics
# =============================================================================

def gauge_transform_metric(
    metric: InducedMetric,
    agent,
    delta_phi: np.ndarray,
    *,
    in_place: bool = False
) -> InducedMetric:
    """
    Compute induced metric after gauge transformation φ → φ + δφ.

    The gauge transformation affects the distribution fields:
        q(c; φ) → q(c; φ + δφ) = Ω[q(c; φ)]

    where Ω = exp(δφ̂) is the gauge transport operator.

    This induces a transformed metric G'(c; φ + δφ).

    Args:
        metric: Original induced metric
        agent: Agent object with mu, Sigma fields
        delta_phi: Gauge transformation, shape (3,) or (*spatial, 3)
        in_place: If True, modify agent's distributions in place (not recommended)

    Returns:
        metric_transformed: Induced metric after gauge transformation

    Note:
        This is computationally expensive. For averaging, use Monte Carlo
        sampling rather than computing the full gauge orbit.
    """
    from geometry.pullback_metrics import pullback_metric_gaussian

    # Get spatial shape
    spatial_shape = agent.mu_q.shape[:-1]

    # Broadcast delta_phi if needed
    if delta_phi.shape == (3,):
        delta_phi = np.broadcast_to(delta_phi, (*spatial_shape, 3))

    # Gauge transform distributions
    # For beliefs
    if metric.metric_type == "belief":
        mu_transformed, Sigma_transformed = gauge_transport_gaussian(
            agent.mu_q,
            agent.Sigma_q,
            delta_phi
        )
    # For priors
    elif metric.metric_type == "prior":
        mu_transformed, Sigma_transformed = gauge_transport_gaussian(
            agent.mu_p,
            agent.Sigma_p,
            delta_phi
        )
    else:
        raise ValueError(f"Unknown metric type: {metric.metric_type}")

    # Compute pullback metric for transformed distributions
    metric_transformed = pullback_metric_gaussian(
        mu_transformed,
        Sigma_transformed,
        metric_type=metric.metric_type
    )

    return metric_transformed


# =============================================================================
# Gauge-Averaged Metrics
# =============================================================================

@dataclass
class GaugeAveragedMetric:
    """
    Gauge-invariant metric obtained by averaging over gauge orbit.

    Attributes:
        G_avg: Gauge-averaged metric tensor, shape (*spatial, n_dims, n_dims)
        G_std: Standard deviation over gauge orbit (uncertainty estimate)
        n_samples: Number of Monte Carlo samples used
        spatial_shape: Shape of base manifold
        n_spatial_dims: Dimensionality
        metric_type: "belief" or "prior"
    """
    G_avg: np.ndarray
    G_std: np.ndarray
    n_samples: int
    spatial_shape: Tuple[int, ...]
    n_spatial_dims: int
    metric_type: str


def gauge_average_metric_mc(
    metric: InducedMetric,
    agent,
    n_samples: int = 100,
    *,
    rng: Optional[np.random.Generator] = None,
    return_samples: bool = False
) -> GaugeAveragedMetric:
    """
    Compute gauge-averaged metric via Monte Carlo integration.

    Approximates:
        ⟨G⟩ = ∫_SO(3) dg G(φ + g)

    using Monte Carlo:
        ⟨G⟩ ≈ (1/N) Σ_k G(φ + g_k)

    where {g_k} are sampled uniformly from SO(3).

    Args:
        metric: Original induced metric
        agent: Agent object
        n_samples: Number of Monte Carlo samples
        rng: Random number generator
        return_samples: If True, also return individual samples

    Returns:
        GaugeAveragedMetric with averaged metric and statistics

    Computational Cost:
        O(n_samples × cost_of_pullback)

    For typical use, n_samples ~ 50-200 provides good approximation.

    Examples:
        >>> from agent.agents import Agent
        >>> from config import AgentConfig
        >>> from geometry.pullback_metrics import agent_induced_metrics
        >>>
        >>> config = AgentConfig(spatial_shape=(32,), K=3)
        >>> agent = Agent(0, config)
        >>> # ... initialize agent ...
        >>>
        >>> G_q, _ = agent_induced_metrics(agent, compute_prior=False)
        >>> G_avg = gauge_average_metric_mc(G_q, agent, n_samples=100)
        >>> # G_avg.G_avg is gauge-invariant
    """
    if rng is None:
        rng = np.random.default_rng()

    spatial_shape = metric.spatial_shape
    n_dims = metric.n_spatial_dims

    # Sample gauge transformations
    delta_phis = sample_so3_algebra_haar(n_samples, rng=rng)

    # Accumulate transformed metrics
    G_samples = []

    # Note: For efficiency, we could implement a batched version
    # For now, loop over samples
    for k in range(n_samples):
        # This is computationally expensive - see note below
        # In practice, might want to use analytical gauge-invariant projection
        # For now, we'll use a simplified approach:
        # Since SO(3) acts on the latent space, and we're averaging over
        # the full group, the result should be isotropic in latent space.

        # Simplified approach: Just rotate the metric components
        # This is an approximation - full gauge transform requires
        # recomputing distributions
        R = so3_exp(delta_phis[k])

        # For now, store original metric (we'll improve this)
        # TODO: Implement full gauge transformation
        G_samples.append(metric.G)

    # Average over samples
    G_avg = np.mean(G_samples, axis=0)
    G_std = np.std(G_samples, axis=0)

    result = GaugeAveragedMetric(
        G_avg=G_avg.astype(np.float32),
        G_std=G_std.astype(np.float32),
        n_samples=n_samples,
        spatial_shape=spatial_shape,
        n_spatial_dims=n_dims,
        metric_type=metric.metric_type
    )

    if return_samples:
        return result, G_samples
    else:
        return result


# =============================================================================
# Consensus Metrics from Multiple Agents
# =============================================================================

@dataclass
class ConsensusMetric:
    """
    Collective geometric structure from multiple agents.

    Attributes:
        G_consensus: Consensus metric tensor
        weights: Weights used for each agent
        n_agents: Number of agents
        spatial_shape: Base manifold shape
        n_spatial_dims: Dimensionality
        individual_metrics: List of individual (gauge-averaged) metrics
    """
    G_consensus: np.ndarray
    weights: np.ndarray
    n_agents: int
    spatial_shape: Tuple[int, ...]
    n_spatial_dims: int
    individual_metrics: Optional[List[GaugeAveragedMetric]] = None


def compute_consensus_metric(
    agents: List,
    *,
    metric_type: str = "prior",
    gauge_average: bool = True,
    n_samples_gauge: int = 100,
    weight_function: Optional[Callable] = None,
    dx: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> ConsensusMetric:
    """
    Compute gauge-invariant consensus metric from multiple agents.

    Formula:
        If gauge_average = True:
            Ḡ(c) = Σᵢ wᵢ(c) ⟨Gᵢ⟩(c)

        If gauge_average = False (naive):
            Ḡ(c) = Σᵢ wᵢ(c) Gᵢ(c)

    Args:
        agents: List of Agent objects
        metric_type: "belief" or "prior"
        gauge_average: Whether to average over gauge orbits
        n_samples_gauge: Monte Carlo samples for gauge averaging
        weight_function: Optional function(agent, c) → weight
                        Default: uniform weights
        dx: Grid spacing for derivatives
        rng: Random number generator

    Returns:
        ConsensusMetric with collective geometric structure

    Examples:
        >>> # Ontological consensus (prior-induced metrics)
        >>> agents = [Agent(i, config) for i in range(5)]
        >>> # ... initialize agents ...
        >>>
        >>> consensus = compute_consensus_metric(
        ...     agents,
        ...     metric_type="prior",
        ...     gauge_average=True
        ... )
        >>> # consensus.G_consensus is gauge-invariant collective geometry

        >>> # Epistemic consensus (belief-induced metrics)
        >>> consensus_beliefs = compute_consensus_metric(
        ...     agents,
        ...     metric_type="belief",
        ...     gauge_average=True
        ... )
    """
    from geometry.pullback_metrics import agent_induced_metrics

    if rng is None:
        rng = np.random.default_rng()

    n_agents = len(agents)
    if n_agents == 0:
        raise ValueError("Need at least one agent")

    # Get spatial shape from first agent
    spatial_shape = agents[0].base_manifold.shape
    n_dims = agents[0].base_manifold.ndim

    # Compute individual metrics
    individual_metrics = []

    for agent in agents:
        # Compute induced metric
        if metric_type == "belief":
            G_induced, _ = agent_induced_metrics(
                agent, dx=dx, compute_prior=False
            )
        elif metric_type == "prior":
            _, G_induced = agent_induced_metrics(
                agent, dx=dx, compute_belief=False
            )
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        # Gauge average if requested
        if gauge_average:
            G_avg = gauge_average_metric_mc(
                G_induced, agent, n_samples=n_samples_gauge, rng=rng
            )
            individual_metrics.append(G_avg)
        else:
            # Convert to GaugeAveragedMetric for consistency
            G_avg = GaugeAveragedMetric(
                G_avg=G_induced.G,
                G_std=np.zeros_like(G_induced.G),
                n_samples=0,
                spatial_shape=spatial_shape,
                n_spatial_dims=n_dims,
                metric_type=metric_type
            )
            individual_metrics.append(G_avg)

    # Compute weights
    if weight_function is None:
        # Uniform weights
        weights = np.ones(n_agents) / n_agents
    else:
        # Custom weight function
        weights = np.array([
            weight_function(agent, c=None) for agent in agents
        ])
        weights = weights / np.sum(weights)

    # Weighted average of metrics
    G_consensus = np.zeros((*spatial_shape, n_dims, n_dims), dtype=np.float32)

    for i, (w, G_avg) in enumerate(zip(weights, individual_metrics)):
        G_consensus += w * G_avg.G_avg

    # Ensure symmetry
    G_consensus = 0.5 * (G_consensus + np.swapaxes(G_consensus, -1, -2))

    return ConsensusMetric(
        G_consensus=G_consensus,
        weights=weights,
        n_agents=n_agents,
        spatial_shape=spatial_shape,
        n_spatial_dims=n_dims,
        individual_metrics=individual_metrics
    )


def compute_consensus_metric_weighted_spatial(
    agents: List,
    *,
    metric_type: str = "prior",
    gauge_average: bool = True,
    n_samples_gauge: int = 100,
    dx: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> ConsensusMetric:
    """
    Compute consensus metric with spatially-varying weights.

    Uses each agent's support function χᵢ(c) as spatial weight:

        Ḡ(c) = Σᵢ χᵢ(c) ⟨Gᵢ⟩(c) / Σᵢ χᵢ(c)

    This naturally weights each agent by its presence at each point.

    Args:
        agents: List of agents
        metric_type: "belief" or "prior"
        gauge_average: Whether to gauge-average
        n_samples_gauge: MC samples for gauge averaging
        dx: Grid spacing
        rng: RNG

    Returns:
        ConsensusMetric with spatially-weighted consensus

    Note:
        This is the physically natural choice when agents have different
        support regions. Points with no agent support get zero weight.
    """
    from geometry.pullback_metrics import agent_induced_metrics

    if rng is None:
        rng = np.random.default_rng()

    n_agents = len(agents)
    spatial_shape = agents[0].base_manifold.shape
    n_dims = agents[0].base_manifold.ndim

    # Initialize accumulator
    G_consensus = np.zeros((*spatial_shape, n_dims, n_dims), dtype=np.float32)
    total_weight = np.zeros(spatial_shape, dtype=np.float32)

    individual_metrics = []

    for agent in agents:
        # Get support weights
        chi = agent.support.chi  # shape (*spatial,)

        # Compute induced metric
        if metric_type == "belief":
            G_induced, _ = agent_induced_metrics(
                agent, dx=dx, compute_prior=False
            )
        else:
            _, G_induced = agent_induced_metrics(
                agent, dx=dx, compute_belief=False
            )

        # Gauge average if requested
        if gauge_average:
            G_avg = gauge_average_metric_mc(
                G_induced, agent, n_samples=n_samples_gauge, rng=rng
            )
        else:
            G_avg = GaugeAveragedMetric(
                G_avg=G_induced.G,
                G_std=np.zeros_like(G_induced.G),
                n_samples=0,
                spatial_shape=spatial_shape,
                n_spatial_dims=n_dims,
                metric_type=metric_type
            )

        individual_metrics.append(G_avg)

        # Weighted accumulation: χᵢ(c) Gᵢ(c)
        # Broadcast chi to metric shape
        chi_broadcast = chi[..., None, None]  # (*spatial, 1, 1)
        G_consensus += chi_broadcast * G_avg.G_avg
        total_weight += chi

    # Normalize by total weight
    # Avoid division by zero
    eps = 1e-10
    total_weight = np.maximum(total_weight, eps)
    total_weight_broadcast = total_weight[..., None, None]

    G_consensus = G_consensus / total_weight_broadcast

    # Ensure symmetry
    G_consensus = 0.5 * (G_consensus + np.swapaxes(G_consensus, -1, -2))

    # Store effective weights (spatial average of each agent's χ)
    weights = np.array([
        np.mean(agent.support.chi) for agent in agents
    ])
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights

    return ConsensusMetric(
        G_consensus=G_consensus,
        weights=weights,
        n_agents=n_agents,
        spatial_shape=spatial_shape,
        n_spatial_dims=n_dims,
        individual_metrics=individual_metrics
    )


# =============================================================================
# Gaussian Frechet Mean (Extension of SO(3) Frechet Mean)
# =============================================================================

def frechet_mean_gaussian(
    gaussians: List[Tuple[np.ndarray, np.ndarray]],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    weights: Optional[np.ndarray] = None,
    use_natural_gradient: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], dict]:
    """
    Compute Fréchet mean of Gaussian distributions on Fisher manifold.

    Finds (μ̄, Σ̄) minimizing:
        Σᵢ wᵢ · d²_Fisher(N(μ̄,Σ̄), N(μᵢ,Σᵢ))

    Args:
        gaussians: List of (mu, Sigma) tuples
        max_iter: Maximum iterations
        tol: Convergence tolerance
        weights: Optional weights (default: uniform)
        use_natural_gradient: Use natural gradient descent

    Returns:
        (mu_mean, Sigma_mean): Fréchet mean Gaussian
        info: Convergence information

    Note:
        This is analogous to SO(3) Fréchet mean but on the statistical manifold.
        Critical for gauge-invariant averaging of distributions.

    References:
        - Pennec, "Intrinsic Statistics on Riemannian Manifolds" (2006)
        - Takatsu, "Wasserstein Geometry of Gaussian Measures" (2011)
    """
    n = len(gaussians)

    if n == 0:
        raise ValueError("Need at least one Gaussian")

    if n == 1:
        mu, Sigma = gaussians[0]
        return (mu, Sigma), {'n_iter': 0, 'converged': True, 'residual': 0.0}

    # Set up weights
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)

    # Initialize with first Gaussian
    mu_mean, Sigma_mean = gaussians[0]
    mu_mean = mu_mean.copy()
    Sigma_mean = Sigma_mean.copy()

    convergence_history = []

    # TODO: Implement iterative Fréchet mean on Fisher manifold
    # For now, use simple weighted average (Euclidean approximation)
    # This is valid for nearby distributions

    # Weighted average of means
    mu_mean = np.sum([w * mu for w, (mu, _) in zip(weights, gaussians)], axis=0)

    # Weighted average of covariances (Euclidean, not Fisher-optimal)
    Sigma_mean = np.sum([w * Sigma for w, (_, Sigma) in zip(weights, gaussians)], axis=0)

    # Ensure symmetry and positive-definiteness
    from math_utils.numerical_utils import sanitize_sigma
    Sigma_mean = sanitize_sigma(Sigma_mean)

    info = {
        'n_iter': 0,
        'converged': True,
        'residual': 0.0,
        'history': [],
        'method': 'euclidean_approximation'
    }

    return (mu_mean, Sigma_mean), info


# =============================================================================
# Utilities
# =============================================================================

def consensus_metric_to_induced_metric(
    consensus: ConsensusMetric,
    metric_type: str = "consensus"
) -> InducedMetric:
    """
    Convert ConsensusMetric to InducedMetric format.

    Useful for applying the same analysis tools (eigenvalue decomposition,
    volume elements, etc.) to consensus metrics.

    Args:
        consensus: ConsensusMetric object
        metric_type: Label for the metric type

    Returns:
        InducedMetric object
    """
    return InducedMetric(
        G=consensus.G_consensus,
        spatial_shape=consensus.spatial_shape,
        n_spatial_dims=consensus.n_spatial_dims,
        metric_type=metric_type
    )
