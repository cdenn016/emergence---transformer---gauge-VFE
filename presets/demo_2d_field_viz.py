#!/usr/bin/env python3
"""
Demo: 2D Agent Field Visualization

Shows emergent agents on a 2D spatial grid with field imaging.

Configuration:
- 9×9 periodic 2D manifold
- K=3 latent dimensions (SO(3) gauge group)
- 5 base agents with hierarchical emergence
- Visualizes mu_q, Sigma_q, and phi fields at base scale (0)
- Images emergent meta-agents at scale 1

Usage:
    python simulation_runner.py --preset demo_2d_field_viz

Output:
    _results/_demo_2d_field_viz/
        agent_fields/
            scale_0/
                step_0000.png, step_0010.png, ...
            scale_1/ (if meta-agents form)
            evolution_mu_q_comp0_agent0_scale0.png
            comparison_mu_q_comp0_step0100_scale0.png

Author: Claude & Chris
Date: November 2025
"""

from simulation_config import SimulationConfig


def get_config() -> SimulationConfig:
    """2D spatial manifold with agent field visualization."""
    return SimulationConfig(
        # Experiment
        experiment_name="_demo_2d_field_viz",
        experiment_description="2D agent field visualization with hierarchical emergence",
        seed=42,

        # 2D spatial manifold
        spatial_shape=(9, 9),
        manifold_topology="periodic",

        # Agents
        n_agents=5,
        K_latent=3,  # SO(3) - odd for gauge group
        D_x=5,

        # Support pattern
        support_pattern="full",  # All agents cover full manifold

        # Training
        n_steps=100,
        log_every=5,

        # Hierarchical emergence
        enable_emergence=True,
        consensus_threshold=0.05,
        consensus_check_interval=2,
        min_cluster_size=2,
        max_scale=2,

        # Energy landscape
        lambda_self=1.0,
        lambda_belief_align=2.0,
        lambda_prior_align=1.5,
        lambda_obs=0.0,
        lambda_phi=0.0,

        # Learning rates
        lr_mu_q=0.05,
        lr_sigma_q=0.005,
        lr_mu_p=0.01,
        lr_sigma_p=0.001,
        lr_phi=0.1,

        # ===================================
        # AGENT FIELD VISUALIZATION
        # ===================================
        visualize_agent_fields=True,
        viz_track_interval=10,  # Record every 10 steps
        viz_scales=(0, 1),  # Image base agents and meta-agents
        viz_fields=("mu_q", "Sigma_q", "phi"),  # Show beliefs, uncertainties, observables
        viz_latent_components=(0, 1, 2),  # All 3 components of SO(3)

        # Optional: Also track pullback geometry
        track_pullback_geometry=False,  # Set True to see emergent spacetime geometry

        # Meta visualizations
        generate_meta_visualizations=True,
        snapshot_interval=5,
    )
