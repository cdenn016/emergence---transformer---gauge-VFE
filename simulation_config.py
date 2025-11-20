"""
Simulation Configuration Dataclass

Consolidates all simulation parameters into a single, well-organized configuration.
Replaces 50+ global variables with a structured, type-safe configuration.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class SimulationConfig:
    """Complete configuration for multi-agent simulation with emergence."""

    # =============================================================================
    # Experiment Metadata
    # =============================================================================
    experiment_name: str = "_playground"
    experiment_description: str = "Multi-agent with smooth support boundaries"
    output_dir: str = "_results"
    seed: int = 2

    # =============================================================================
    # Spatial Geometry
    # =============================================================================
    spatial_shape: Tuple = ()
    manifold_topology: str = "periodic"  # periodic, flat, sphere

    # =============================================================================
    # Training Loop
    # =============================================================================
    n_steps: int = 50
    log_every: int = 1
    skip_initial_steps: int = 0  # For analysis plots (ignore transients)

    # Early stopping conditions (any condition triggers stop)
    stop_if_n_scales_reached: Optional[int] = None  # Stop when this many scales exist
    stop_if_n_condensations: Optional[int] = None  # Stop after this many meta-agents formed
    stop_if_min_active_agents: Optional[int] = None  # Stop if active agents drops below this

    # =============================================================================
    # Agents & Latent Space
    # =============================================================================
    n_agents: int = 5
    K_latent: int = 11
    D_x: int = 5  # Observation dimension

    # Field initialization scales
    mu_scale: float = 0.5
    sigma_scale: float = 1.0
    phi_scale: float = 0.1
    mean_smoothness: float = 1.0

    # Connection
    connection_type: str = 'flat'  # flat, random, constant
    use_connection: bool = False

    # =============================================================================
    # Hierarchical Emergence
    # =============================================================================
    enable_emergence: bool = True
    consensus_threshold: float = 0.05  # KL threshold for epistemic death
    consensus_check_interval: int = 5  # Check every N steps
    min_cluster_size: int = 2  # Min agents to form meta-agent
    max_scale: int = 3  # Highest scale (prevents runaway emergence)
    max_meta_membership: int = 10  # Max constituents per meta-agent
    max_total_agents: int = 1000  # Hard cap across ALL scales

    enable_cross_scale_priors: bool = True  # Top-down prior propagation
    enable_timescale_sep: bool = False  # Timescale separation
    info_metric: str = "fisher_metric"  # Information change metric

    # Ouroboros Tower: Multi-scale hyperprior propagation
    enable_hyperprior_tower: bool = False  # Wheeler's "it from bit" extended
    max_hyperprior_depth: int = 3  # How many levels up to receive priors
    hyperprior_decay: float = 0.3  # Exponential decay for ancestral priors

    # =============================================================================
    # Energy Weights (Cultural/Hierarchical Tension)
    # =============================================================================
    lambda_self: float = 3.0  # Individual identity (resist conformity)
    lambda_belief_align: float = 2.0  # Peer pressure (social)
    lambda_prior_align: float = 2.5  # Cultural authority (top-down)
    lambda_obs: float = 0.0  # External observations
    lambda_phi: float = 0.0  # Gauge coupling

    kappa_beta: float = 1.0  # Softmax temperature (belief align)
    kappa_gamma: float = 1.0  # Softmax temperature (prior align)

    identical_priors: str = "off"  # off, lock, init_copy
    identical_priors_source: str = "first"  # first or mean

    # =============================================================================
    # Learning Rates
    # =============================================================================
    lr_mu_q: float = 0.08
    lr_sigma_q: float = 0.001
    lr_mu_p: float = 0.2
    lr_sigma_p: float = 0.01
    lr_phi: float = 0.1

    # =============================================================================
    # Support Geometry
    # =============================================================================
    support_pattern: str = "point"  # point, circles_2d, full, intervals_1d
    agent_placement_2d: str = "center"  # center, random, grid
    agent_radius: float = 5.0  # For 2D circular supports
    random_radius_range: Optional[Tuple[float, float]] = None  # (min, max) or None
    interval_overlap_fraction: float = 0.25  # For 1D intervals

    # =============================================================================
    # Masking (Smooth Support Boundaries)
    # =============================================================================
    mask_type: str = "gaussian"  # hard, smooth, gaussian
    overlap_threshold: float = 1e-1  # Ignore overlaps below this
    min_mask_for_normal_cov: float = 1e-1  # Below this, use large Σ

    # Gaussian mask parameters
    gaussian_sigma: float = field(init=False)  # Computed from overlap_threshold
    gaussian_cutoff_sigma: float = 3.0  # Hard cutoff at N*σ

    # Smooth mask parameters
    smooth_width: float = 0.1  # Transition width (relative to radius)

    # Covariance outside support
    covariance_strategy: str = "smooth"  # Gaussian-filtered Cholesky
    outside_cov_scale: float = 1e3  # Scale for diagonal Σ outside support
    use_smooth_cov_transition: bool = False  # Interpolate Σ at boundaries

    # =============================================================================
    # Observation Model
    # =============================================================================
    obs_bias_scale: float = 0.5
    obs_noise_scale: float = 1.0
    obs_w_scale: float = 0.5
    obs_r_scale: float = 1.0
    obs_ground_truth_amplitude: float = 0.5
    obs_ground_truth_modes: int = 3

    # =============================================================================
    # Diagnostics
    # =============================================================================
    run_initial_diagnostics: bool = False
    run_final_diagnostics: bool = False
    save_diagnostic_plots: bool = True
    save_diagnostic_report: bool = False

    # Comprehensive meta-agent visualizations (hierarchy, consensus, energy)
    generate_meta_visualizations: bool = True
    snapshot_interval: int = 5  # Capture analyzer snapshots every N steps

    def __post_init__(self):
        """Compute derived parameters."""
        # Compute gaussian_sigma from overlap_threshold
        if self.overlap_threshold > 0:
            self.gaussian_sigma = 1.0 / np.sqrt(-2 * np.log(self.overlap_threshold))
        else:
            self.gaussian_sigma = 1.0

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    def save(self, filepath: str):
        """Save configuration to text file for reproducibility."""
        with open(filepath, 'w') as f:
            f.write("# Simulation Configuration\n")
            f.write(f"# {'='*60}\n\n")

            sections = {
                "Experiment": ["experiment_name", "experiment_description", "seed"],
                "Manifold": ["spatial_shape", "manifold_topology"],
                "Training": ["n_steps", "log_every", "skip_initial_steps",
                            "stop_if_n_scales_reached", "stop_if_n_condensations",
                            "stop_if_min_active_agents"],
                "Agents": ["n_agents", "K_latent", "D_x", "mu_scale", "sigma_scale",
                          "phi_scale", "mean_smoothness"],
                "Emergence": ["enable_emergence", "consensus_threshold", "consensus_check_interval",
                             "min_cluster_size", "max_scale", "max_meta_membership",
                             "max_total_agents", "enable_cross_scale_priors", "enable_timescale_sep"],
                "Energy": ["lambda_self", "lambda_belief_align", "lambda_prior_align",
                          "lambda_obs", "lambda_phi", "kappa_beta", "kappa_gamma"],
                "Learning Rates": ["lr_mu_q", "lr_sigma_q", "lr_mu_p", "lr_sigma_p", "lr_phi"],
                "Support": ["support_pattern", "agent_placement_2d", "agent_radius",
                           "random_radius_range", "interval_overlap_fraction"],
                "Masking": ["mask_type", "gaussian_sigma", "gaussian_cutoff_sigma",
                           "smooth_width", "overlap_threshold", "outside_cov_scale"],
                "Observations": ["obs_bias_scale", "obs_noise_scale", "obs_w_scale",
                               "obs_r_scale", "obs_ground_truth_amplitude", "obs_ground_truth_modes"],
                "Diagnostics": ["run_initial_diagnostics", "run_final_diagnostics",
                              "save_diagnostic_plots", "save_diagnostic_report"],
            }

            for section_name, keys in sections.items():
                f.write(f"[{section_name}]\n")
                for key in keys:
                    if hasattr(self, key):
                        value = getattr(self, key)
                        f.write(f"{key:<30} = {value}\n")
                f.write("\n")


# =============================================================================
# Preset Configurations
# =============================================================================

def default_config() -> SimulationConfig:
    """Default configuration for standard runs."""
    return SimulationConfig()


def emergence_demo_config() -> SimulationConfig:
    """Configuration optimized for demonstrating hierarchical emergence."""
    return SimulationConfig(
        experiment_name="_emergence_demo",
        experiment_description="Optimized for demonstrating meta-agent formation",
        n_agents=8,
        n_steps=100,
        enable_emergence=True,
        consensus_threshold=0.05,
        consensus_check_interval=5,
        lambda_self=3.0,
        lambda_belief_align=2.0,
        lambda_prior_align=2.5,
        enable_cross_scale_priors=True,
        # Early stopping: stop once we reach 5 scales or form 15 meta-agents
        stop_if_n_scales_reached=5,
        stop_if_n_condensations=15,
    )


def ouroboros_config() -> SimulationConfig:
    """Configuration with Ouroboros Tower (multi-scale hyperpriors)."""
    return SimulationConfig(
        experiment_name="_ouroboros_tower",
        experiment_description="Wheeler's 'it from bit' with ancestral priors",
        enable_emergence=True,
        enable_hyperprior_tower=True,
        max_hyperprior_depth=3,
        hyperprior_decay=0.3,
    )


def flat_agents_config() -> SimulationConfig:
    """Configuration for flat multi-agent system (no emergence)."""
    return SimulationConfig(
        experiment_name="_flat_agents",
        experiment_description="Standard multi-agent without emergence",
        enable_emergence=False,
        n_agents=5,
        n_steps=50,
    )
