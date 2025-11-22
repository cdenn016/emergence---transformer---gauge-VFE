"""
Metric Signature Analysis Tools
================================

Tools for analyzing the signature of metric tensors to detect emergent
Lorentzian structure in statistical manifolds.

The goal is to find regions where the pullback metric g_pullback has
signature (-,+,+,+) instead of the usual Riemannian (+,+,+,+).

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class MetricSignature(Enum):
    """Metric signature types."""
    RIEMANNIAN = "riemannian"          # All positive eigenvalues
    LORENTZIAN = "lorentzian"          # One negative, rest positive
    EUCLIDEAN = "euclidean"            # Flat (+,+,+,+)
    MINKOWSKI = "minkowski"            # Flat (-,+,+,+)
    DEGENERATE = "degenerate"          # Has zero eigenvalues
    INDEFINITE = "indefinite"          # Multiple negative eigenvalues
    UNKNOWN = "unknown"


@dataclass
class SignatureAnalysis:
    """Results of metric signature analysis."""
    eigenvalues: np.ndarray           # Sorted eigenvalues
    eigenvectors: np.ndarray          # Corresponding eigenvectors
    signature: MetricSignature        # Classified signature type
    signature_tuple: Tuple[int, ...]  # (n_negative, n_zero, n_positive)
    timelike_direction: Optional[np.ndarray] = None  # If Lorentzian
    spacelike_directions: Optional[np.ndarray] = None

    def __str__(self):
        """Human-readable description."""
        sig_str = f"({'-' * self.signature_tuple[0]}"
        sig_str += f"{'0' * self.signature_tuple[1]}"
        sig_str += f"{'+' * self.signature_tuple[2]})"

        return f"Signature: {sig_str} ({self.signature.value})"


def analyze_metric_signature(
    metric: np.ndarray,
    tol: float = 1e-10
) -> SignatureAnalysis:
    """
    Analyze the signature of a metric tensor.

    Args:
        metric: (d, d) metric tensor g_ij
        tol: Tolerance for zero eigenvalues

    Returns:
        SignatureAnalysis object with full signature information

    Example:
        >>> g = np.diag([-1, 1, 1, 1])  # Minkowski metric
        >>> sig = analyze_metric_signature(g)
        >>> print(sig)
        Signature: (-+++) (lorentzian)
    """
    # Ensure symmetric
    metric_sym = 0.5 * (metric + metric.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(metric_sym)

    # Sort by eigenvalue magnitude (largest absolute value first)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Classify eigenvalues
    n_negative = np.sum(eigenvalues < -tol)
    n_zero = np.sum(np.abs(eigenvalues) <= tol)
    n_positive = np.sum(eigenvalues > tol)

    signature_tuple = (n_negative, n_zero, n_positive)

    # Determine signature type
    if n_zero > 0:
        signature = MetricSignature.DEGENERATE
    elif n_negative == 0:
        signature = MetricSignature.RIEMANNIAN
    elif n_negative == 1:
        signature = MetricSignature.LORENTZIAN
        # Check if approximately flat
        if np.allclose(np.abs(eigenvalues), 1.0, rtol=0.1):
            signature = MetricSignature.MINKOWSKI
    elif n_negative > 1:
        signature = MetricSignature.INDEFINITE
    else:
        signature = MetricSignature.UNKNOWN

    # Extract special directions for Lorentzian metrics
    timelike_direction = None
    spacelike_directions = None

    if signature == MetricSignature.LORENTZIAN or signature == MetricSignature.MINKOWSKI:
        # First eigenvector (largest absolute eigenvalue) is timelike
        timelike_direction = eigenvectors[:, 0] if eigenvalues[0] < 0 else None

        # Find the negative eigenvalue index
        neg_idx = np.where(eigenvalues < -tol)[0][0]
        timelike_direction = eigenvectors[:, neg_idx]

        # Remaining are spacelike
        spacelike_mask = np.arange(len(eigenvalues)) != neg_idx
        spacelike_directions = eigenvectors[:, spacelike_mask]

    return SignatureAnalysis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        signature=signature,
        signature_tuple=signature_tuple,
        timelike_direction=timelike_direction,
        spacelike_directions=spacelike_directions
    )


def compute_pullback_metric(
    agent,
    point_idx: Optional[int] = None,
    include_internal: bool = True,
    include_dark: bool = True
) -> np.ndarray:
    """
    Compute the pullback metric at a point in the base manifold.

    The pullback metric is induced by the Fisher information metric
    on the statistical manifold:

        g_pullback = g_spatial + g_dark + g_internal

    where:
        - g_spatial: Induced by varying position in base manifold
        - g_dark: Contribution from gauge field φ
        - g_internal: Contribution from varying μ, Σ

    Args:
        agent: Agent with μ_q, Sigma_q, phi fields
        point_idx: Index of point in flattened base manifold (None = center)
        include_internal: Include internal metric contribution
        include_dark: Include dark (gauge) metric contribution

    Returns:
        metric: Pullback metric tensor g_αβ
    """
    # Get agent's belief parameters
    mu_q = agent.mu_q
    Sigma_q = agent.Sigma_q
    phi = agent.phi

    # Determine point
    if point_idx is None:
        # Use center of support
        if hasattr(agent, 'support') and agent.support is not None:
            point_idx = agent.support.center_idx
        else:
            point_idx = 0

    # Get dimensionality
    K = mu_q.shape[-1]  # Latent dimension

    # For now, return a placeholder metric
    # TODO: Implement full pullback computation from Fisher metric

    # Spatial metric component (simplified)
    if mu_q.ndim > 1:
        # Compute gradient of μ with respect to spatial coordinates
        # This is approximate - full implementation needs proper covariant derivatives
        spatial_dim = 2  # Assuming 2D base manifold
        g_spatial = np.eye(spatial_dim)
    else:
        spatial_dim = 0
        g_spatial = np.zeros((0, 0))

    # Dark (gauge) metric component
    # Contribution from gauge field φ curvature
    if include_dark and phi is not None:
        # φ has shape (*S, 3) for SO(3) gauge group
        # Dark metric measures how much φ varies spatially
        g_dark = 0.1 * np.eye(max(spatial_dim, 1))  # Placeholder
    else:
        g_dark = np.zeros_like(g_spatial)

    # Internal metric component (Fisher metric on (μ,Σ) space)
    if include_internal:
        # Fisher metric for Gaussian: g_μμ = Σ^{-1}, g_ΣΣ = Tr[Σ^{-1}...Σ^{-1}]
        # Simplified: just take magnitude
        g_internal = np.eye(max(spatial_dim, K))
    else:
        g_internal = np.zeros((max(spatial_dim, K), max(spatial_dim, K)))

    # Combine (this is simplified - full version needs careful tensor algebra)
    total_dim = max(spatial_dim, K)
    g_pullback = np.zeros((total_dim, total_dim))

    # Add components (proper implementation requires projection operators)
    if spatial_dim > 0:
        g_pullback[:spatial_dim, :spatial_dim] += g_spatial + g_dark

    # Add small internal contribution
    g_pullback += 0.01 * g_internal

    return g_pullback


def signature_field_2d(
    system,
    agent_idx: int = 0,
    grid_size: Tuple[int, int] = (20, 20)
) -> Dict[str, np.ndarray]:
    """
    Compute metric signature across a 2D grid in the base manifold.

    This maps out regions where the metric signature might change,
    potentially revealing Lorentzian regions.

    Args:
        system: MultiAgentSystem
        agent_idx: Which agent to analyze
        grid_size: (H, W) resolution of grid

    Returns:
        Dictionary with:
            - 'signatures': (H, W) array of signature classifications
            - 'eigenvalues': (H, W, d) array of eigenvalues
            - 'n_negative': (H, W) count of negative eigenvalues
            - 'n_positive': (H, W) count of positive eigenvalues
            - 'lorentzian_mask': (H, W) boolean mask of Lorentzian regions
    """
    agent = system.agents[agent_idx]
    H, W = grid_size

    # Get base manifold shape
    if hasattr(agent, 'base_manifold'):
        base_shape = agent.base_manifold.shape
    else:
        base_shape = (H, W)

    # Initialize output arrays
    signatures = np.zeros((H, W), dtype=int)
    eigenvalues_field = np.zeros((H, W, 4))  # Assume max 4D
    n_negative = np.zeros((H, W), dtype=int)
    n_positive = np.zeros((H, W), dtype=int)

    # Compute signature at each grid point
    for i in range(H):
        for j in range(W):
            # Map grid point to base manifold index
            point_idx = i * W + j

            # Compute pullback metric
            g = compute_pullback_metric(agent, point_idx=point_idx)

            # Analyze signature
            sig_analysis = analyze_metric_signature(g)

            # Store results
            d = len(sig_analysis.eigenvalues)
            eigenvalues_field[i, j, :d] = sig_analysis.eigenvalues
            n_negative[i, j] = sig_analysis.signature_tuple[0]
            n_positive[i, j] = sig_analysis.signature_tuple[2]

            # Encode signature as integer
            if sig_analysis.signature == MetricSignature.LORENTZIAN:
                signatures[i, j] = -1
            elif sig_analysis.signature == MetricSignature.RIEMANNIAN:
                signatures[i, j] = 1
            else:
                signatures[i, j] = 0

    # Create Lorentzian mask
    lorentzian_mask = (signatures == -1)

    return {
        'signatures': signatures,
        'eigenvalues': eigenvalues_field,
        'n_negative': n_negative,
        'n_positive': n_positive,
        'lorentzian_mask': lorentzian_mask,
        'grid_size': grid_size
    }


def compute_light_cone_structure(
    metric: np.ndarray,
    point: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute light cone structure at a point if metric is Lorentzian.

    Args:
        metric: (d, d) metric tensor (must be Lorentzian)
        point: (d,) point in manifold

    Returns:
        Dictionary with:
            - 'timelike_vectors': Vectors inside light cone
            - 'null_vectors': Vectors on light cone boundary
            - 'spacelike_vectors': Vectors outside light cone
            - 'light_cone_angle': Opening angle of light cone
    """
    sig = analyze_metric_signature(metric)

    if sig.signature != MetricSignature.LORENTZIAN:
        raise ValueError(f"Metric must be Lorentzian, got {sig.signature}")

    # For a Lorentzian metric g with signature (-,+,+,+),
    # a vector v is:
    #   - Timelike if g(v,v) < 0
    #   - Null if g(v,v) = 0
    #   - Spacelike if g(v,v) > 0

    # Light cone is defined by g(v,v) = 0

    # Generate sample vectors
    d = len(metric)
    n_samples = 100
    vectors = np.random.randn(n_samples, d)

    # Compute g(v,v) for each vector
    inner_products = np.array([
        v @ metric @ v for v in vectors
    ])

    # Classify vectors
    timelike_mask = inner_products < -1e-6
    null_mask = np.abs(inner_products) < 1e-6
    spacelike_mask = inner_products > 1e-6

    # Compute light cone opening angle
    # For Minkowski metric, this is 45 degrees
    # For general Lorentzian metric, depends on curvature
    timelike_dir = sig.timelike_direction
    if timelike_dir is not None:
        # Angle between timelike direction and null directions
        # (This is approximate - proper calculation requires full geometry)
        light_cone_angle = np.pi / 4  # Placeholder
    else:
        light_cone_angle = None

    return {
        'timelike_vectors': vectors[timelike_mask],
        'null_vectors': vectors[null_mask],
        'spacelike_vectors': vectors[spacelike_mask],
        'light_cone_angle': light_cone_angle,
        'timelike_direction': timelike_dir
    }
