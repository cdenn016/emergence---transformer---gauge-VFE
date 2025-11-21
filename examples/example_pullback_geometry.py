# -*- coding: utf-8 -*-
"""
Example: Pullback Geometry and "It From Bit" Construction
==========================================================

Demonstrates Wheeler's "it from bit" vision: geometric structure emerging
from informational dynamics via pullback of Fisher-Rao metrics.

This example shows:
1. Computing induced metrics from agent belief/prior fields
2. Eigenvalue decomposition into observable/dark/internal sectors
3. Isotropic vs anisotropic Gaussian examples
4. Gauge-invariant metric averaging
5. Consensus geometry from multiple agents

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from agent.agents import Agent
from config import AgentConfig
from geometry.pullback_metrics import (
    pullback_metric_gaussian,
    pullback_metric_gaussian_isotropic,
    agent_induced_metrics,
    InducedMetric
)
from geometry.gauge_consensus import (
    compute_consensus_metric,
    compute_consensus_metric_weighted_spatial,
    consensus_metric_to_induced_metric
)


# =============================================================================
# Example 1: Basic Pullback Metric from Gaussian Field
# =============================================================================

def example_1_basic_pullback():
    """
    Compute induced metric for a simple 1D Gaussian field.

    Setup:
        - Base manifold: [0, 2π] with 64 points
        - Gaussian field: μ(x) = [sin(x), cos(x)], Σ = I

    Result:
        - Metric measures how rapidly beliefs change with position
    """
    print("=" * 70)
    print("Example 1: Basic Pullback Metric (1D)")
    print("=" * 70)

    # Create 1D base manifold
    n_points = 64
    x = np.linspace(0, 2*np.pi, n_points)
    dx = 2*np.pi / n_points

    # Gaussian mean field: circular trajectory in 2D latent space
    mu_field = np.stack([np.sin(x), np.cos(x)], axis=-1)  # (64, 2)

    # Constant identity covariance
    K = 2
    Sigma_field = np.repeat(np.eye(K)[None], n_points, axis=0)  # (64, 2, 2)

    # Compute pullback metric
    metric = pullback_metric_gaussian(
        mu_field,
        Sigma_field,
        dx=dx,
        metric_type="belief",
        periodic=True
    )

    print(f"\nMetric tensor shape: {metric.G.shape}")
    print(f"Spatial shape: {metric.spatial_shape}")
    print(f"Base manifold dimensionality: {metric.n_spatial_dims}")

    # Since this is 1D, G is (64, 1, 1)
    G_11 = metric.G[:, 0, 0]

    print(f"\nMetric component G_11:")
    print(f"  Min: {G_11.min():.4f}")
    print(f"  Max: {G_11.max():.4f}")
    print(f"  Mean: {G_11.mean():.4f}")

    # For circular trajectory with constant speed, metric should be constant
    print(f"\nExpected: Constant metric ≈ 1.0 (since ||dμ/dx|| = 1)")
    print(f"Actual variation: {G_11.std():.4e}")

    # Compute eigenvalue decomposition
    metric.compute_spectral_decomposition()

    print(f"\nEigenvalues (all points): {metric.eigenvalues[:5, 0]}")

    print("\n✓ Example 1 complete!\n")

    return metric


# =============================================================================
# Example 2: Isotropic vs Anisotropic Gaussian
# =============================================================================

def example_2_isotropic_vs_anisotropic():
    """
    Compare isotropic (Σ = σ²I) and anisotropic covariance effects.

    Shows how uncertainty structure affects induced geometry.
    """
    print("=" * 70)
    print("Example 2: Isotropic vs Anisotropic Gaussian")
    print("=" * 70)

    n_points = 64
    x = np.linspace(0, 2*np.pi, n_points)
    dx = 2*np.pi / n_points

    # Same mean field
    mu_field = np.stack([np.sin(x), np.cos(x), np.zeros(n_points)], axis=-1)  # (64, 3)
    K = 3

    # === Isotropic Case ===
    sigma = 0.5
    metric_iso = pullback_metric_gaussian_isotropic(
        mu_field,
        sigma,
        dx=dx,
        metric_type="belief"
    )

    # === Anisotropic Case ===
    # Varying covariance: more uncertain in z-direction
    Sigma_field = np.zeros((n_points, K, K))
    for i in range(n_points):
        Sigma_field[i] = np.diag([0.25, 0.25, 1.0])  # Different uncertainties

    metric_aniso = pullback_metric_gaussian(
        mu_field,
        Sigma_field,
        dx=dx,
        metric_type="belief"
    )

    print("\nIsotropic metric (σ² = 0.25):")
    G_iso = metric_iso.G[:, 0, 0]
    print(f"  Mean G_11: {G_iso.mean():.4f}")
    print(f"  Conformal factor: 1/σ² = {1/sigma**2:.4f}")
    print(f"  Expected G_11 ≈ {1/sigma**2:.4f} (for unit speed)")

    print("\nAnisotropic metric:")
    G_aniso = metric_aniso.G[:, 0, 0]
    print(f"  Mean G_11: {G_aniso.mean():.4f}")

    print("\nInterpretation:")
    print("  - Isotropic: metric ∝ 1/σ² (high certainty → large distances)")
    print("  - Anisotropic: metric depends on full Σ structure")

    print("\n✓ Example 2 complete!\n")

    return metric_iso, metric_aniso


# =============================================================================
# Example 3: Agent Induced Metrics (Belief vs Prior)
# =============================================================================

def example_3_agent_metrics():
    """
    Compute belief-induced and prior-induced metrics for an agent.

    Demonstrates the dual geometries:
    - G^(q): Epistemic geometry (dynamical, reflects current beliefs)
    - G^(p): Ontological geometry (quasi-static, reflects world model)
    """
    print("=" * 70)
    print("Example 3: Agent Induced Metrics (Belief vs Prior)")
    print("=" * 70)

    # Create agent
    config = AgentConfig(
        spatial_shape=(64,),
        K=3,
        alpha=0.1,
        covariance_kwargs={'base_scale': 0.5}
    )

    rng = np.random.default_rng(42)
    agent = Agent(0, config, rng=rng)

    # Initialize with some structure
    x = np.linspace(0, 2*np.pi, 64)

    # Belief: following an observation
    agent.mu_q = np.stack([
        np.sin(2*x),
        np.cos(2*x),
        0.5 * np.sin(x)
    ], axis=-1)

    # Prior: smoother, reflecting long-term structure
    agent.mu_p = np.stack([
        np.sin(x),
        np.cos(x),
        np.zeros(64)
    ], axis=-1)

    # Compute both induced metrics
    G_belief, G_prior = agent_induced_metrics(
        agent,
        dx=2*np.pi/64,
        compute_belief=True,
        compute_prior=True
    )

    print(f"\nBelief-induced metric G^(q):")
    print(f"  Shape: {G_belief.G.shape}")
    G_q_11 = G_belief.G[:, 0, 0]
    print(f"  Mean: {G_q_11.mean():.4f}")
    print(f"  Std: {G_q_11.std():.4f}")
    print(f"  Interpretation: Epistemic geometry, rapidly changing")

    print(f"\nPrior-induced metric G^(p):")
    print(f"  Shape: {G_prior.G.shape}")
    G_p_11 = G_prior.G[:, 0, 0]
    print(f"  Mean: {G_p_11.mean():.4f}")
    print(f"  Std: {G_p_11.std():.4f}")
    print(f"  Interpretation: Ontological geometry, agent's perceived reality")

    print("\nKey Insight:")
    print("  - G^(q): What the agent currently believes (changes with observations)")
    print("  - G^(p): How the agent models reality (stable world model)")
    print("  - Conjecture: G^(p) is the agent's perceived spacetime geometry")

    print("\n✓ Example 3 complete!\n")

    return agent, G_belief, G_prior


# =============================================================================
# Example 4: Eigenvalue Decomposition and Sectors
# =============================================================================

def example_4_eigenvalue_sectors():
    """
    Demonstrate eigenvalue decomposition into observable/dark/internal sectors.

    For high-dimensional latent spaces, most dimensions are "internal" with
    negligible information flux. Only a few are "observable".
    """
    print("=" * 70)
    print("Example 4: Eigenvalue Decomposition and Sectors")
    print("=" * 70)

    # Create 2D base manifold with high-dimensional latent space
    config = AgentConfig(
        spatial_shape=(16, 16),  # 2D base manifold
        K=10,  # 10-dimensional latent space
        alpha=0.1
    )

    rng = np.random.default_rng(42)
    agent = Agent(0, config, rng=rng)

    # Create structured mean field: most dimensions inactive
    H, W = 16, 16
    x = np.linspace(0, 2*np.pi, W)
    y = np.linspace(0, 2*np.pi, H)
    X, Y = np.meshgrid(x, y)

    # Initialize mu_p with structure in first 3 dimensions only
    mu_p = np.zeros((H, W, 10))
    mu_p[..., 0] = np.sin(X) * np.cos(Y)  # Active
    mu_p[..., 1] = np.cos(X) * np.sin(Y)  # Active
    mu_p[..., 2] = 0.5 * np.sin(2*X)       # Active
    # Dimensions 3-9: inactive (zero gradient)

    agent.mu_p = mu_p

    # Compute prior-induced metric
    _, G_prior = agent_induced_metrics(
        agent,
        dx=2*np.pi/16,
        compute_belief=False,
        compute_prior=True
    )

    # Compute eigenvalue decomposition
    G_prior.compute_spectral_decomposition()

    print(f"\nMetric shape: {G_prior.G.shape}")  # (16, 16, 2, 2) for 2D base
    print(f"Eigenvalues shape: {G_prior.eigenvalues.shape}")  # (16, 16, 2)

    # Average eigenvalues over space
    eigvals_mean = np.mean(G_prior.eigenvalues, axis=(0, 1))
    print(f"\nMean eigenvalues across space: {eigvals_mean}")

    # Define sectors
    lambda_obs = 0.1  # Observable threshold (relative to max)
    lambda_dark = 0.01  # Dark threshold

    obs_mask, dark_mask, internal_mask = G_prior.get_three_sector_decomposition(
        lambda_obs, lambda_dark, relative=True
    )

    n_obs = np.sum(obs_mask, axis=-1)  # Number of observable dimensions at each point
    n_dark = np.sum(dark_mask, axis=-1)
    n_internal = np.sum(internal_mask, axis=-1)

    print(f"\nSector decomposition (averaged over space):")
    print(f"  Observable dimensions: {np.mean(n_obs):.2f}")
    print(f"  Dark dimensions: {np.mean(n_dark):.2f}")
    print(f"  Internal dimensions: {np.mean(n_internal):.2f}")

    print("\nInterpretation:")
    print("  - Observable: Directions with high information flux → perceived space")
    print("  - Dark: Intermediate information → additional structure")
    print("  - Internal: Negligible flux → pure internal degrees of freedom")

    # Volume element
    vol = G_prior.volume_element()
    print(f"\nVolume element sqrt(det G):")
    print(f"  Mean: {np.mean(vol):.4f}")
    print(f"  Std: {np.std(vol):.4f}")

    print("\n✓ Example 4 complete!\n")

    return G_prior


# =============================================================================
# Example 5: Consensus Metrics from Multiple Agents
# =============================================================================

def example_5_consensus_metrics():
    """
    Compute gauge-invariant consensus metric from multiple agents.

    Shows how collective geometry emerges from individual agent perspectives.
    """
    print("=" * 70)
    print("Example 5: Consensus Metrics from Multiple Agents")
    print("=" * 70)

    # Create multiple agents
    n_agents = 5
    config = AgentConfig(
        spatial_shape=(32,),
        K=3,
        alpha=0.1
    )

    rng = np.random.default_rng(42)
    agents = [Agent(i, config, rng=rng) for i in range(n_agents)]

    # Initialize with slightly different priors
    x = np.linspace(0, 2*np.pi, 32)

    for i, agent in enumerate(agents):
        # Each agent has slightly shifted/scaled prior
        phase = 2*np.pi * i / n_agents
        agent.mu_p = np.stack([
            np.sin(x + phase),
            np.cos(x + phase),
            0.5 * np.sin(2*x + phase)
        ], axis=-1)

    print(f"\nComputing consensus from {n_agents} agents...")

    # Compute consensus metric (naive, no gauge averaging)
    consensus_naive = compute_consensus_metric(
        agents,
        metric_type="prior",
        gauge_average=False,
        dx=2*np.pi/32
    )

    print(f"\nNaive consensus (no gauge averaging):")
    G_naive = consensus_naive.G_consensus[:, 0, 0]
    print(f"  Mean G_11: {G_naive.mean():.4f}")
    print(f"  Std G_11: {G_naive.std():.4f}")

    # Note: Gauge averaging is computationally expensive
    # For this example, we demonstrate the structure
    print("\nGauge-invariant consensus would:")
    print("  1. Average each agent's metric over SO(3) gauge orbit")
    print("  2. Combine gauge-averaged metrics")
    print("  3. Result is independent of arbitrary frame choices")

    # Convert to InducedMetric for analysis
    consensus_metric = consensus_metric_to_induced_metric(
        consensus_naive, metric_type="consensus"
    )

    consensus_metric.compute_spectral_decomposition()

    eigvals_mean = np.mean(consensus_metric.eigenvalues)
    print(f"\nConsensus metric eigenvalues (mean): {eigvals_mean:.4f}")

    print("\n✓ Example 5 complete!\n")

    return consensus_naive


# =============================================================================
# Example 6: Visualizations
# =============================================================================

def example_6_visualizations():
    """
    Create visualizations of induced metrics and their properties.
    """
    print("=" * 70)
    print("Example 6: Visualizations")
    print("=" * 70)

    # Create 1D example for visualization
    n_points = 128
    x = np.linspace(0, 4*np.pi, n_points)
    dx = 4*np.pi / n_points

    # Gaussian field with varying mean
    mu_field = np.stack([
        np.sin(x),
        np.cos(x),
        0.3 * np.sin(2*x)
    ], axis=-1)

    # Varying uncertainty (conformal factor)
    sigma_field = 0.3 + 0.2 * np.sin(x/2)  # σ ∈ [0.1, 0.5]

    # Compute isotropic metric
    metric = pullback_metric_gaussian_isotropic(
        mu_field,
        sigma_field,
        dx=dx
    )

    # Compute eigenvalues
    metric.compute_spectral_decomposition()

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Mean field components
    ax = axes[0]
    ax.plot(x, mu_field[:, 0], label='μ₁(x)', linewidth=2)
    ax.plot(x, mu_field[:, 1], label='μ₂(x)', linewidth=2)
    ax.plot(x, mu_field[:, 2], label='μ₃(x)', linewidth=2)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Mean μ(x)')
    ax.set_title('Gaussian Mean Field μ(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Uncertainty and metric
    ax = axes[1]
    ax2 = ax.twinx()

    line1 = ax.plot(x, sigma_field, 'b-', label='σ(x)', linewidth=2)
    line2 = ax2.plot(x, metric.G[:, 0, 0], 'r-', label='G₁₁(x)', linewidth=2)

    ax.set_xlabel('Position x')
    ax.set_ylabel('Uncertainty σ(x)', color='b')
    ax2.set_ylabel('Metric G₁₁(x)', color='r')
    ax.set_title('Uncertainty σ(x) and Induced Metric G₁₁(x)')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    # Plot 3: Volume element
    ax = axes[2]
    vol = metric.volume_element()
    ax.plot(x, vol, 'g-', linewidth=2)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Volume element √det(G)')
    ax.set_title('Information-Geometric Volume Element')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "pullback_geometry_example.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.close()

    print("\n✓ Example 6 complete!\n")


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Run all pullback geometry examples."""

    print("\n" + "=" * 70)
    print("PULLBACK GEOMETRY: IT FROM BIT CONSTRUCTION")
    print("Demonstrating Wheeler's Vision via Fisher-Rao Metrics")
    print("=" * 70 + "\n")

    # Example 1: Basic pullback
    metric_1d = example_1_basic_pullback()

    # Example 2: Isotropic vs anisotropic
    metric_iso, metric_aniso = example_2_isotropic_vs_anisotropic()

    # Example 3: Agent metrics
    agent, G_belief, G_prior = example_3_agent_metrics()

    # Example 4: Eigenvalue sectors
    G_2d = example_4_eigenvalue_sectors()

    # Example 5: Consensus
    consensus = example_5_consensus_metrics()

    # Example 6: Visualizations
    example_6_visualizations()

    print("=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Geometric structure (metrics) emerges from information (distributions)")
    print("  2. Belief metrics G^(q) encode epistemic geometry (current knowledge)")
    print("  3. Prior metrics G^(p) encode ontological geometry (world model)")
    print("  4. High-dimensional spaces decompose into observable/dark/internal sectors")
    print("  5. Gauge-invariant consensus produces collective geometry")
    print("\nThis is Wheeler's 'It From Bit' realized mathematically:")
    print("  - 'It' = Riemannian metric tensor on base manifold")
    print("  - 'Bit' = Probability distributions as information")
    print("  - Mechanism = Pullback of Fisher-Rao metric")
    print("\nSee outputs/pullback_geometry_example.png for visualizations.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
