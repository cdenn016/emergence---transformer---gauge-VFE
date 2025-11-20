"""
Meta-Agent Analysis and Visualization Demo

This script demonstrates how to use the comprehensive visualization and analysis
tools for the hierarchical meta-agent system. It runs a complete simulation,
captures data, and generates all visualizations.

Usage:
    python examples/meta_agent_analysis_demo.py

Outputs:
    - ./meta_analysis/: Structure and dynamics visualizations
    - ./energy_analysis/: Energy landscape and thermodynamics
    - ./snapshots.json: Raw data export
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta.emergence import MultiScaleSystem, HierarchicalAgent
from meta.hierarchical_evolution import HierarchicalEvolutionEngine
from meta.participatory_diagnostics import ParticipatoryDiagnostics
from meta.visualization import (
    MetaAgentAnalyzer,
    HierarchyVisualizer,
    ConsensusVisualizer,
    DynamicsVisualizer,
    create_analysis_report
)
from meta.energy_visualization import EnergyVisualizer


def create_demo_system(n_agents: int = 8,
                      dim: int = 3,
                      max_emergence_levels: int = 3) -> MultiScaleSystem:
    """
    Create a multi-scale system for demonstration.

    Args:
        n_agents: Number of initial agents at scale 0
        dim: Dimensionality of beliefs
        max_emergence_levels: Maximum hierarchy depth

    Returns:
        Initialized MultiScaleSystem
    """
    print(f"\n{'='*70}")
    print("Creating Multi-Scale System")
    print(f"{'='*70}")
    print(f"  Initial agents: {n_agents}")
    print(f"  Belief dimension: {dim}")
    print(f"  Max emergence levels: {max_emergence_levels}\n")

    # Create initial agents at scale 0
    agents = []
    for i in range(n_agents):
        # Random initial beliefs clustered in groups
        group = i // (n_agents // 2)  # Two groups
        mu_q = np.random.randn(dim) + group * 3.0
        Sigma_q = np.eye(dim) * 0.5

        # Priors start diffuse
        mu_p = np.zeros(dim)
        Sigma_p = np.eye(dim) * 5.0

        # Gauge frame (identity for simplicity)
        phi = np.eye(dim)

        agent = HierarchicalAgent(
            mu_q=mu_q,
            Sigma_q=Sigma_q,
            mu_p=mu_p,
            Sigma_p=Sigma_p,
            phi=phi,
            scale=0,
            local_index=i
        )
        agents.append(agent)

    # Create system
    system = MultiScaleSystem(
        agents={0: agents},
        max_emergence_levels=max_emergence_levels,
        max_meta_membership=n_agents,  # Allow all agents to condense
        max_total_agents=n_agents * 10  # Generous cap
    )

    print(f"✓ System initialized with {len(agents)} agents at scale 0")
    return system


def run_simulation(system: MultiScaleSystem,
                  n_steps: int = 50,
                  snapshot_interval: int = 5,
                  lr_mu_q: float = 0.1,
                  lr_Sigma_q: float = 0.05) -> tuple:
    """
    Run hierarchical evolution with diagnostics and snapshot capture.

    Args:
        system: MultiScaleSystem to evolve
        n_steps: Number of evolution steps
        snapshot_interval: Capture snapshot every N steps
        lr_mu_q: Learning rate for belief means
        lr_Sigma_q: Learning rate for belief covariances

    Returns:
        (analyzer, diagnostics, evolution_results)
    """
    print(f"\n{'='*70}")
    print("Running Hierarchical Evolution")
    print(f"{'='*70}")
    print(f"  Steps: {n_steps}")
    print(f"  Snapshot interval: {snapshot_interval}")
    print(f"  Learning rates: μ={lr_mu_q}, Σ={lr_Sigma_q}\n")

    # Initialize diagnostics
    diagnostics = ParticipatoryDiagnostics(
        system=system,
        track_energy=True,
        track_priors=True,
        track_gradients=False  # Can be expensive
    )

    # Initialize analyzer
    analyzer = MetaAgentAnalyzer(system)

    # Create evolution engine
    engine = HierarchicalEvolutionEngine(
        system=system,
        lr_mu_q=lr_mu_q,
        lr_Sigma_q=lr_Sigma_q,
        lr_mu_p=0.05,
        lr_Sigma_p=0.02,
        belief_kl_threshold=0.5,  # Relatively permissive
        model_kl_threshold=0.5,
        timescale_filtering=True,
        deactivate_constituents=False,  # Keep constituents active
        ouroboros_tower=False  # Standard Markov hierarchy
    )

    # Run evolution with periodic snapshots
    print("Evolution progress:")

    for step in range(n_steps):
        # Evolve one step
        engine.step()

        # Capture diagnostics
        diagnostics.capture_snapshot()

        # Capture analyzer snapshot at intervals
        if step % snapshot_interval == 0 or step == n_steps - 1:
            analyzer.capture_snapshot()
            print(f"  Step {step:3d}: "
                  f"Scales={list(system.agents.keys())}, "
                  f"Active={sum(len([a for a in agents if a.is_active]) for agents in system.agents.values())}, "
                  f"Condensations={len(system.condensation_events)}")

    print(f"\n✓ Evolution complete!")
    print(f"  Total condensations: {len(system.condensation_events)}")
    print(f"  Final scales: {sorted(system.agents.keys())}")
    print(f"  Snapshots captured: {len(analyzer.snapshots)}")

    # Get final metrics
    final_metrics = {
        'total_condensations': len(system.condensation_events),
        'max_scale_reached': max(system.agents.keys()),
        'final_active_agents': sum(
            len([a for a in agents if a.is_active])
            for agents in system.agents.values()
        )
    }

    return analyzer, diagnostics, final_metrics


def generate_visualizations(analyzer: MetaAgentAnalyzer,
                           diagnostics: ParticipatoryDiagnostics,
                           output_dir: str = './demo_output'):
    """
    Generate all visualizations and analysis reports.

    Args:
        analyzer: MetaAgentAnalyzer with captured snapshots
        diagnostics: ParticipatoryDiagnostics with energy data
        output_dir: Base output directory
    """
    print(f"\n{'='*70}")
    print("Generating Visualizations")
    print(f"{'='*70}\n")

    # Create output directories
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    meta_dir = Path(output_dir) / 'meta_analysis'
    energy_dir = Path(output_dir) / 'energy_analysis'

    # Generate meta-agent structure visualizations
    print("1. Meta-Agent Structure and Dynamics...")
    meta_files = create_analysis_report(analyzer, str(meta_dir))

    # Generate energy visualizations
    print("\n2. Energy Landscapes and Thermodynamics...")
    energy_viz = EnergyVisualizer(diagnostics)
    energy_files = energy_viz.create_energy_report(str(energy_dir))

    # Generate interactive visualizations (if Plotly available)
    print("\n3. Interactive Visualizations...")
    try:
        hierarchy_viz = HierarchyVisualizer(analyzer)
        interactive_fig = hierarchy_viz.plot_interactive_hierarchy()

        if interactive_fig:
            interactive_path = Path(output_dir) / 'interactive_hierarchy.html'
            interactive_fig.write_html(str(interactive_path))
            print(f"  ✓ Saved interactive hierarchy to {interactive_path}")
    except Exception as e:
        print(f"  ✗ Failed to generate interactive visualizations: {e}")

    # Export raw data
    print("\n4. Exporting Raw Data...")
    data_path = Path(output_dir) / 'snapshots.json'
    analyzer.export_to_json(str(data_path))

    print(f"\n{'='*70}")
    print("Visualization Complete!")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - Meta-agent analysis: {meta_dir}/")
    print(f"  - Energy analysis: {energy_dir}/")
    print(f"  - Raw data: {data_path}")
    print(f"\nTotal files generated: {len(meta_files) + len(energy_files) + 1}")


def analyze_specific_features(analyzer: MetaAgentAnalyzer,
                              diagnostics: ParticipatoryDiagnostics):
    """
    Perform specific analyses and print insights.

    Args:
        analyzer: MetaAgentAnalyzer with captured snapshots
        diagnostics: ParticipatoryDiagnostics with energy data
    """
    print(f"\n{'='*70}")
    print("Detailed Analysis")
    print(f"{'='*70}\n")

    # 1. Condensation statistics
    if analyzer.system.condensation_events:
        print("Condensation Events:")
        for i, event in enumerate(analyzer.system.condensation_events):
            print(f"  Event {i+1}:")
            print(f"    Time: {event['time']}")
            print(f"    {event['source_scale']} → {event['target_scale']}")
            print(f"    Constituents: {event['n_constituents']}")
            print(f"    Belief coherence: {event['coherence']['belief']:.3f}")
            print(f"    Model coherence: {event['coherence']['model']:.3f}")
            print(f"    Leader score: {event['leader_score']:.3f}")
            print()

    # 2. Energy evolution
    if diagnostics.energy_snapshots:
        print("Energy Evolution:")
        first = diagnostics.energy_snapshots[0]
        last = diagnostics.energy_snapshots[-1]

        print(f"  Initial total energy: {sum(s['E_total'] for s in first['by_scale'].values()):.2f}")
        print(f"  Final total energy: {sum(s['E_total'] for s in last['by_scale'].values()):.2f}")

        for scale in sorted(last['by_scale'].keys()):
            if scale in first['by_scale']:
                print(f"  Scale {scale}:")
                print(f"    ΔE_self: {last['by_scale'][scale]['E_self'] - first['by_scale'][scale]['E_self']:.2f}")
                print(f"    ΔE_belief: {last['by_scale'][scale]['E_belief_align'] - first['by_scale'][scale]['E_belief_align']:.2f}")
                print(f"    ΔE_prior: {last['by_scale'][scale]['E_prior_align'] - first['by_scale'][scale]['E_prior_align']:.2f}")
        print()

    # 3. Hierarchy depth progression
    print("Hierarchy Depth Progression:")
    for snapshot in analyzer.snapshots[::max(1, len(analyzer.snapshots) // 5)]:
        scales_repr = ', '.join(
            f"ζ{s}:{snapshot.metrics['n_active_by_scale'].get(s, 0)}"
            for s in sorted(snapshot.metrics['n_active_by_scale'].keys())
        )
        print(f"  t={snapshot.time:3d}: [{scales_repr}]")
    print()

    # 4. Consensus analysis
    print("Final Consensus Status:")
    final_snapshot = analyzer.snapshots[-1]

    for scale in sorted(final_snapshot.agents_by_scale.keys()):
        kl_matrix = analyzer.get_consensus_matrix(scale, 'belief')

        if kl_matrix.size > 0:
            # Check for consensus clusters
            n_agents = kl_matrix.shape[0]
            avg_kl = np.mean(kl_matrix[np.triu_indices(n_agents, k=1)])
            max_kl = np.max(kl_matrix)
            min_kl = np.min(kl_matrix[kl_matrix > 0]) if np.any(kl_matrix > 0) else 0

            print(f"  Scale {scale}:")
            print(f"    Agents: {n_agents}")
            print(f"    Avg pairwise KL: {avg_kl:.3f}")
            print(f"    Min KL: {min_kl:.3f}")
            print(f"    Max KL: {max_kl:.3f}")

            if avg_kl < 0.5:
                print(f"    Status: ✓ Strong consensus")
            elif avg_kl < 2.0:
                print(f"    Status: ○ Partial consensus")
            else:
                print(f"    Status: ✗ Divergent beliefs")
    print()


def main():
    """Main demonstration workflow."""
    print("\n" + "="*70)
    print(" "*15 + "META-AGENT ANALYSIS DEMO")
    print("="*70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Create system
    system = create_demo_system(
        n_agents=8,
        dim=3,
        max_emergence_levels=3
    )

    # 2. Run simulation
    analyzer, diagnostics, metrics = run_simulation(
        system,
        n_steps=50,
        snapshot_interval=5,
        lr_mu_q=0.1,
        lr_Sigma_q=0.05
    )

    # 3. Analyze specific features
    analyze_specific_features(analyzer, diagnostics)

    # 4. Generate all visualizations
    generate_visualizations(
        analyzer,
        diagnostics,
        output_dir='./demo_output'
    )

    print("\n" + "="*70)
    print(" "*20 + "DEMO COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Explore the generated visualizations in ./demo_output/")
    print("  2. Try different system parameters (n_agents, learning rates, etc.)")
    print("  3. Enable Ouroboros Tower mode for non-Markov hierarchies")
    print("  4. Experiment with timescale filtering and constituent deactivation")
    print("\n")


if __name__ == '__main__':
    main()
