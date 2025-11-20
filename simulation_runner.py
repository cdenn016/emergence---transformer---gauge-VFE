#!/usr/bin/env python3
"""
Streamlined Simulation Runner

Clean, modular simulation orchestrator that replaces the bloated simulation_suite.py

Key improvements:
- Uses SimulationConfig dataclass (no more 50+ globals!)
- Extracted GradientSystemAdapter to meta/gradient_adapter.py
- Unified training interface (less duplication)
- Clear separation of concerns
- ~500 lines vs 1345 lines (62% reduction!)

Usage:
    python simulation_runner.py                    # Default config
    python simulation_runner.py --preset emergence # Emergence demo
    python simulation_runner.py --preset ouroboros # Ouroboros Tower
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from simulation_config import (
    SimulationConfig,
    default_config,
    emergence_demo_config,
    ouroboros_config,
    flat_agents_config
)
from config import AgentConfig, SystemConfig, TrainingConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem
from agent.trainer import Trainer, TrainingHistory
from geometry.geometry_base import BaseManifold, TopologyType
from agent.masking import SupportRegionSmooth, SupportPatternsSmooth, MaskConfig


# =============================================================================
# System Building (Clean Helper Functions)
# =============================================================================

def build_manifold(cfg: SimulationConfig) -> BaseManifold:
    """Create BaseManifold from configuration."""
    topology_map = {
        "periodic": TopologyType.PERIODIC,
        "flat": TopologyType.FLAT,
        "sphere": TopologyType.SPHERE,
    }
    topology_key = cfg.manifold_topology.lower()
    if topology_key not in topology_map:
        raise ValueError(f"Unknown topology '{topology_key}'. Valid options: {list(topology_map.keys())}")

    return BaseManifold(
        shape=cfg.spatial_shape,
        topology=topology_map[topology_key]
    )


def build_supports(manifold: BaseManifold,
                  cfg: SimulationConfig,
                  rng: np.random.Generator):
    """Build support regions for all agents."""
    from geometry.geometry_base import SupportPatterns

    mask_config = MaskConfig(
        mask_type=cfg.mask_type,
        smooth_width=cfg.smooth_width,
        gaussian_sigma=cfg.gaussian_sigma,
        gaussian_cutoff_sigma=cfg.gaussian_cutoff_sigma,
        overlap_threshold=cfg.overlap_threshold,
        min_mask_for_normal_cov=cfg.min_mask_for_normal_cov,
        outside_cov_scale=cfg.outside_cov_scale,
        use_smooth_cov_transition=cfg.use_smooth_cov_transition
    )

    ndim = manifold.ndim

    # 0D: Point manifold (all agents at same point)
    if ndim == 0:
        return [
            SupportRegionSmooth(
                mask_binary=np.array(True),
                base_shape=(),
                config=mask_config
            ) for _ in range(cfg.n_agents)
        ]

    # 1D: Intervals or full support
    if ndim == 1:
        if cfg.support_pattern == "full":
            basic = [SupportPatterns.full(manifold) for _ in range(cfg.n_agents)]
        elif cfg.support_pattern == "intervals_1d":
            basic = _build_intervals_1d(manifold, cfg)
        else:
            raise ValueError(f"Unsupported 1D pattern: {cfg.support_pattern}")

        return [
            SupportRegionSmooth(s.mask, s.base_shape, mask_config) for s in basic
        ]

    # 2D: Circles or full support
    if ndim == 2:
        if cfg.support_pattern == "full":
            basic = [SupportPatterns.full(manifold) for _ in range(cfg.n_agents)]
            return [SupportRegionSmooth(s.mask, s.base_shape, mask_config) for s in basic]
        elif cfg.support_pattern == "circles_2d":
            return _build_circles_2d(manifold, cfg, mask_config, rng)
        else:
            raise ValueError(f"Unsupported 2D pattern: {cfg.support_pattern}")

    # Higher dimensions: Full support
    basic = [SupportPatterns.full(manifold) for _ in range(cfg.n_agents)]
    return [SupportRegionSmooth(s.mask, s.base_shape, mask_config) for s in basic]


def _build_intervals_1d(manifold, cfg):
    """Build overlapping intervals for 1D manifold."""
    from geometry.geometry_base import SupportPatterns

    n_points = manifold.shape[0]
    base_width = n_points / cfg.n_agents
    overlap = int(base_width * cfg.interval_overlap_fraction)

    supports = []
    for i in range(cfg.n_agents):
        start = int(max(0, round(i * base_width) - overlap // 2))
        end = int(min(n_points, round((i + 1) * base_width) + overlap // 2))
        supports.append(SupportPatterns.interval(manifold, start=start, end=end))
    return supports


def _build_circles_2d(manifold, cfg, mask_config, rng):
    """Build circular supports for 2D manifold."""
    H, W = manifold.shape
    supports = []

    if cfg.agent_placement_2d == "center":
        center = (H // 2, W // 2)
        for _ in range(cfg.n_agents):
            supports.append(SupportPatternsSmooth.circle(
                manifold_shape=manifold.shape,
                center=center,
                radius=cfg.agent_radius,
                config=mask_config
            ))
    elif cfg.agent_placement_2d == "random":
        for _ in range(cfg.n_agents):
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            radius = (rng.uniform(*cfg.random_radius_range)
                     if cfg.random_radius_range else cfg.agent_radius)
            supports.append(SupportPatternsSmooth.circle(
                manifold_shape=manifold.shape,
                center=(cy, cx),
                radius=radius,
                config=mask_config
            ))
    else:
        raise ValueError(f"Unknown placement: {cfg.agent_placement_2d}")

    return supports


def build_agents(manifold, supports, cfg: SimulationConfig, rng):
    """Create Agent objects with support enforcement."""
    agent_cfg = AgentConfig(
        spatial_shape=cfg.spatial_shape,
        K=cfg.K_latent,
        mu_scale=cfg.mu_scale,
        sigma_scale=cfg.sigma_scale,
        phi_scale=cfg.phi_scale,
        mean_smoothness_scale=cfg.mean_smoothness
    )

    mask_config = MaskConfig(
        mask_type=cfg.mask_type,
        overlap_threshold=cfg.overlap_threshold,
        outside_cov_scale=cfg.outside_cov_scale
    )
    agent_cfg.mask_config = mask_config

    agents = []
    for i in range(cfg.n_agents):
        agent = Agent(agent_id=i, config=agent_cfg, rng=rng, base_manifold=manifold)
        agent.support = supports[i]

        # Re-initialize with support enforcement
        agent._initialize_belief_cholesky()
        agent._initialize_prior_cholesky()
        agent._initialize_gauge()
        agent.geometry.support = supports[i]
        agent.geometry.n_active = supports[i].n_active

        agents.append(agent)

    return agents


def build_system(agents, cfg: SimulationConfig, rng):
    """Create MultiAgentSystem or MultiScaleSystem."""
    system_cfg = SystemConfig(
        lambda_self=cfg.lambda_self,
        lambda_belief_align=cfg.lambda_belief_align,
        lambda_prior_align=cfg.lambda_prior_align,
        lambda_obs=cfg.lambda_obs,
        lambda_phi=cfg.lambda_phi,
        identical_priors=cfg.identical_priors,
        identical_priors_source=cfg.identical_priors_source,
        kappa_beta=cfg.kappa_beta,
        kappa_gamma=cfg.kappa_gamma,
        overlap_threshold=cfg.overlap_threshold,
        use_connection=cfg.use_connection,
        connection_init_mode=cfg.connection_type,
        D_x=cfg.D_x,
        obs_W_scale=cfg.obs_w_scale,
        obs_R_scale=cfg.obs_r_scale,
        obs_noise_scale=cfg.obs_noise_scale,
        obs_bias_scale=cfg.obs_bias_scale,
        obs_ground_truth_modes=cfg.obs_ground_truth_modes,
        obs_ground_truth_amplitude=cfg.obs_ground_truth_amplitude,
        seed=int(rng.integers(0, 2**31)),
    )

    if cfg.enable_emergence:
        return _build_hierarchical_system(agents, system_cfg, cfg)
    else:
        return _build_standard_system(agents, system_cfg)


def _build_standard_system(agents, system_cfg):
    """Build standard MultiAgentSystem."""
    print("  Mode: STANDARD (no emergence)")
    system = MultiAgentSystem(agents, system_cfg)
    if system.config.has_observations:
        system.ensure_observation_model()
    return system


def _build_hierarchical_system(agents, system_cfg, cfg):
    """Build hierarchical MultiScaleSystem."""
    from meta.emergence import MultiScaleSystem
    from math_utils.generators import generate_so3_generators

    print("  Mode: HIERARCHICAL (emergence enabled)")
    print(f"  Consensus threshold: {cfg.consensus_threshold}")
    print(f"  Max scales: {cfg.max_scale}")

    manifold = agents[0].base_manifold
    system = MultiScaleSystem(
        manifold,
        max_emergence_levels=cfg.max_scale,
        max_meta_membership=cfg.max_meta_membership,
        max_total_agents=cfg.max_total_agents
    )
    system.system_config = system_cfg

    # Add base agents
    generators = generate_so3_generators(cfg.K_latent)
    for agent in agents:
        h_agent = system.add_base_agent(agent.config, agent_id=agent.agent_id)
        h_agent.support = agent.support
        h_agent.generators = generators
        # Copy state
        h_agent.mu_q = agent.mu_q.copy()
        h_agent.Sigma_q = agent.Sigma_q.copy()
        h_agent.mu_p = agent.mu_p.copy()
        h_agent.Sigma_p = agent.Sigma_p.copy()
        if hasattr(agent, 'gauge'):
            h_agent.gauge.phi = agent.gauge.phi.copy()

    # Apply identical priors if configured
    if system_cfg.identical_priors in ("init_copy", "lock"):
        _apply_identical_priors(system, system_cfg)

    return system


def _apply_identical_priors(system, system_cfg):
    """Apply identical priors to base agents."""
    base_agents = system.agents[0]
    if not base_agents:
        return

    if system_cfg.identical_priors_source == "mean":
        mu_p_shared = sum(a.mu_p for a in base_agents) / len(base_agents)
        L_p_shared = sum(a.L_p for a in base_agents) / len(base_agents)
    else:
        mu_p_shared = base_agents[0].mu_p.copy()
        L_p_shared = base_agents[0].L_p.copy()

    for a in base_agents:
        a.mu_p = mu_p_shared.copy()
        a.L_p = L_p_shared.copy()
        if hasattr(a, 'invalidate_caches'):
            a.invalidate_caches()


# =============================================================================
# Training (Unified Interface)
# =============================================================================

def run_training(system, cfg: SimulationConfig, output_dir: Path):
    """
    Unified training interface for both standard and hierarchical systems.

    Automatically detects system type and runs appropriate training.
    """
    if cfg.enable_emergence:
        return _run_hierarchical_training(system, cfg, output_dir)
    else:
        return _run_standard_training(system, cfg, output_dir)


def _run_standard_training(system, cfg, output_dir):
    """Run standard training with Trainer."""
    print(f"\n{'='*70}")
    print("STANDARD TRAINING")
    print(f"{'='*70}")

    training_cfg = TrainingConfig(
        n_steps=cfg.n_steps,
        log_every=cfg.log_every,
        lr_mu_q=cfg.lr_mu_q,
        lr_sigma_q=cfg.lr_sigma_q,
        lr_mu_p=cfg.lr_mu_p,
        lr_sigma_p=cfg.lr_sigma_p,
        lr_phi=cfg.lr_phi,
        checkpoint_every=1,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )

    trainer = Trainer(system, training_cfg)
    history = trainer.train()

    # Save history
    _save_history(history, output_dir)
    _plot_energy(history, output_dir)

    return history


def _run_hierarchical_training(system, cfg, output_dir):
    """Run hierarchical training with emergence."""
    from meta.hierarchical_evolution import HierarchicalEvolutionEngine, HierarchicalConfig
    from meta.gradient_adapter import GradientSystemAdapter
    from gradients.gradient_engine import compute_natural_gradients
    from free_energy_clean import compute_total_free_energy

    print(f"\n{'='*70}")
    print("HIERARCHICAL TRAINING WITH EMERGENCE")
    print(f"{'='*70}")

    hier_config = HierarchicalConfig(
        enable_top_down_priors=cfg.enable_cross_scale_priors,
        enable_hyperprior_tower=cfg.enable_hyperprior_tower,
        max_hyperprior_depth=cfg.max_hyperprior_depth,
        hyperprior_decay=cfg.hyperprior_decay,
        enable_timescale_filtering=cfg.enable_timescale_sep,
        consensus_check_interval=cfg.consensus_check_interval,
        consensus_kl_threshold=cfg.consensus_threshold,
        min_cluster_size=cfg.min_cluster_size,
        lr_mu_q=cfg.lr_mu_q,
        lr_sigma_q=cfg.lr_sigma_q,
        lr_mu_p=cfg.lr_mu_p,
        lr_sigma_p=cfg.lr_sigma_p,
        lr_phi=cfg.lr_phi
    )

    engine = HierarchicalEvolutionEngine(system, hier_config)

    # Initialize comprehensive visualization tools
    analyzer = None
    diagnostics = None
    if cfg.generate_meta_visualizations:
        from meta.visualization import MetaAgentAnalyzer
        from meta.participatory_diagnostics import ParticipatoryDiagnostics

        print("  Initializing comprehensive visualization tools...")
        analyzer = MetaAgentAnalyzer(system)

        # Note: compute_full_energies=True enables belief/prior alignment energy tracking
        # This is VERY EXPENSIVE (10-100x slower) but gives detailed energy decomposition
        # Set to False for fast performance with only self-energy tracking
        diagnostics = ParticipatoryDiagnostics(
            system=system,
            track_agent_ids=None,  # Auto-selects first 3 scale-0 agents
            compute_full_energies=False  # Set True for detailed energy (SLOW!)
        )
        print(f"  Snapshot interval: every {cfg.snapshot_interval} steps")
        print(f"  Full energy computation: {'ENABLED (slow)' if diagnostics.compute_full_energies else 'DISABLED (fast, self-energy only)'}")

    # History tracking
    history = {
        'step': [],
        'total': [],
        'n_scales': [],
        'n_active_agents': [],
        'n_condensations': [],
        'emergence_events': []
    }

    # Training loop
    for step in range(cfg.n_steps):
        active_agents = system.get_all_active_agents()
        if not active_agents:
            break

        # Create adapter for gradient computation
        adapter = GradientSystemAdapter(active_agents, system.system_config)

        # Compute energy before updates
        energies = compute_total_free_energy(adapter)

        # Gradient computation wrapper
        def compute_grads(sys):
            return compute_natural_gradients(adapter)

        # Evolve one step
        metrics = engine.evolve_step(learning_rate=cfg.lr_mu_q, compute_gradients_fn=compute_grads)

        # Capture visualization snapshots
        if cfg.generate_meta_visualizations:
            if step % cfg.snapshot_interval == 0 or step == cfg.n_steps - 1:
                analyzer.capture_snapshot()
            diagnostics.record_snapshot(step)

        # Record metrics - use ACTUAL system state, not potentially stale metrics
        actual_n_scales = len(system.agents)  # Number of scales = number of keys in agents dict
        actual_n_active = sum(sum(1 for a in agents if a.is_active)
                             for agents in system.agents.values())

        history['step'].append(step)
        history['total'].append(energies.total)
        history['n_scales'].append(actual_n_scales)
        history['n_active_agents'].append(actual_n_active)
        history['n_condensations'].append(metrics.get('n_condensations', 0))

        # Log emergence events
        if metrics.get('n_condensations', 0) > 0:
            event = {
                'step': step,
                'n_condensations': metrics['n_condensations'],
                'n_scales': len(metrics.get('n_active', {}))
            }
            history['emergence_events'].append(event)
            print(f"\n🌟 EMERGENCE at step {step}! {metrics['n_condensations']} new meta-agents")

        if step % cfg.log_every == 0:
            print(f"Step {step:4d} | Energy: {energies.total:.4f} | "
                  f"Scales: {history['n_scales'][-1]} | Active: {history['n_active_agents'][-1]}")

    # Save history
    _save_history(history, output_dir)

    # Generate visualizations
    if cfg.generate_meta_visualizations and analyzer and diagnostics:
        _generate_comprehensive_visualizations(system, analyzer, diagnostics, output_dir)
    else:
        # Minimal visualization
        _plot_emergence(history, output_dir)

    return history


# =============================================================================
# Saving and Visualization
# =============================================================================

def _save_history(history, output_dir):
    """Save training history."""
    hist_path = output_dir / "history.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    print(f"✓ Saved {hist_path}")


def _plot_energy(history, output_dir):
    """Plot energy evolution for standard training."""
    if isinstance(history, TrainingHistory):
        steps, energy = history.steps, history.total_energy
    else:
        steps, energy = history['step'], history['total']

    plt.figure(figsize=(10, 6))
    plt.plot(steps, energy, linewidth=2, color='black')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Energy Evolution')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / "energy_evolution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved {fig_path}")


def _plot_emergence(history, output_dir):
    """Plot emergence evolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy with emergence markers
    ax1.plot(history['step'], history['total'], 'b-', linewidth=2)
    for event in history['emergence_events']:
        ax1.axvline(event['step'], color='red', alpha=0.3, linestyle='--')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy (red = emergence)')
    ax1.grid(alpha=0.3)

    # Hierarchy evolution
    ax2.plot(history['step'], history['n_scales'], 'g-', marker='o', label='# Scales')
    ax2.plot(history['step'], history['n_active_agents'], 'b-', marker='s', label='# Active')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Count')
    ax2.set_title('Hierarchical Structure')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "emergence_evolution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved {fig_path}")


def _generate_comprehensive_visualizations(system, analyzer, diagnostics, output_dir):
    """
    Generate comprehensive meta-agent visualizations.

    Uses the new visualization toolkit to create:
    - Hierarchy graphs (static and interactive)
    - Consensus matrices
    - Scale occupancy heatmaps
    - Energy landscapes
    - Coherence trajectories
    - And more!
    """
    from meta.visualization import (
        HierarchyVisualizer,
        ConsensusVisualizer,
        DynamicsVisualizer,
        create_analysis_report
    )
    from meta.energy_visualization import EnergyVisualizer

    print(f"\n{'='*70}")
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print(f"{'='*70}")

    # Create output directories
    meta_dir = output_dir / "meta_analysis"
    energy_dir = output_dir / "energy_analysis"
    meta_dir.mkdir(exist_ok=True, parents=True)
    energy_dir.mkdir(exist_ok=True, parents=True)

    # Generate structure and dynamics visualizations
    print("\n1. Meta-Agent Structure and Dynamics...")
    try:
        create_analysis_report(analyzer, str(meta_dir))
    except Exception as e:
        print(f"  ⚠️  Error generating meta-agent analysis: {e}")

    # Generate energy visualizations
    print("\n2. Energy Landscapes and Thermodynamics...")
    try:
        energy_viz = EnergyVisualizer(diagnostics)
        energy_viz.create_energy_report(str(energy_dir))
    except Exception as e:
        print(f"  ⚠️  Error generating energy analysis: {e}")

    # Generate interactive hierarchy (if possible)
    print("\n3. Interactive Visualizations...")
    try:
        hierarchy_viz = HierarchyVisualizer(analyzer)
        interactive_fig = hierarchy_viz.plot_interactive_hierarchy()
        if interactive_fig:
            interactive_path = output_dir / "interactive_hierarchy.html"
            interactive_fig.write_html(str(interactive_path))
            print(f"  ✓ Saved interactive hierarchy to {interactive_path}")
    except Exception as e:
        print(f"  ⚠️  Plotly not available or error: {e}")

    # Export data for external analysis
    print("\n4. Exporting Raw Data...")
    try:
        data_path = output_dir / "snapshots.json"
        analyzer.export_to_json(str(data_path))
        print(f"  ✓ Saved raw data to {data_path}")
    except Exception as e:
        print(f"  ⚠️  Error exporting data: {e}")

    # Print summary
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Meta-agent analysis: {meta_dir}/")
    print(f"Energy analysis:     {energy_dir}/")
    print(f"Interactive plots:   {output_dir}/interactive_*.html")
    print(f"Raw data:            {output_dir}/snapshots.json")
    print(f"{'='*70}\n")

    # Print analysis summary
    if analyzer.snapshots:
        final_snapshot = analyzer.snapshots[-1]
        print("Final System State:")
        print(f"  Total agents:    {final_snapshot.metrics['total_agents']}")
        print(f"  Active agents:   {final_snapshot.metrics['total_active']}")
        print(f"  Max scale:       {final_snapshot.metrics['max_scale']}")
        print(f"  Meta-agents:     {len(final_snapshot.meta_agents)}")
        print(f"  Condensations:   {len(system.condensation_events)}")
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Streamlined Simulation Runner")
    parser.add_argument('--preset', type=str, default='default',
                       choices=['default', 'emergence', 'ouroboros', 'flat'],
                       help='Configuration preset')
    args = parser.parse_args()

    # Load configuration
    preset_map = {
        'default': default_config,
        'emergence': emergence_demo_config,
        'ouroboros': ouroboros_config,
        'flat': flat_agents_config
    }
    cfg = preset_map[args.preset]()

    # Setup
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("MULTI-AGENT SIMULATION")
    print(f"{'='*70}")
    print(f"Preset: {args.preset}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Build and run
    manifold = build_manifold(cfg)
    supports = build_supports(manifold, cfg, rng)
    agents = build_agents(manifold, supports, cfg, rng)
    system = build_system(agents, cfg, rng)

    # Save config
    cfg.save(str(output_dir / "config.txt"))

    # Train
    history = run_training(system, cfg, output_dir)

    # Summary
    print(f"\n{'='*70}")
    print("✓ SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
