#!/usr/bin/env python3
"""
Demonstration: Pullback Geometry Tracking in Simulation

Shows how emergent spacetime geometry evolves during multi-agent dynamics.

This example:
1. Creates a 1D periodic manifold with 3 agents
2. Enables pullback geometry tracking
3. Runs standard (non-hierarchical) training
4. Visualizes emergence of observable/dark/internal sectors
5. Tracks evolution of information-geometric volume elements

Usage:
    python examples/demo_pullback_tracking.py

Output:
    _results/_demo_pullback_tracking/
        geometry_evolution.png          - 4-panel geometry evolution
        geometry_history.pkl            - Full history for analysis
        geometry_analysis/              - Detailed final geometry analysis
        final_eigenvalue_spectrum.png   - Final eigenvalue decomposition

Author: Chris & Christine
Date: November 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_config import SimulationConfig
from simulation_runner import main as run_simulation

def demo_pullback_tracking_config() -> SimulationConfig:
    """
    Configuration demonstrating pullback geometry tracking.

    Features:
    - 1D periodic manifold (64 points)
    - 3 agents with full support
    - Standard (non-hierarchical) evolution
    - Pullback geometry tracked every 5 steps
    - Observable/dark/internal sector decomposition
    """
    return SimulationConfig(
        # Experiment metadata
        experiment_name="_demo_pullback_tracking",
        experiment_description="Demonstrate emergent geometry evolution",
        seed=42,

        # Simple 1D manifold
        spatial_shape=(64,),
        manifold_topology="periodic",

        # Few agents for clarity
        n_agents=3,
        K_latent=11,  # Odd for SO(3)
        support_pattern="full",  # All agents cover full manifold

        # Sufficient evolution time
        n_steps=100,
        log_every=10,

        # Standard evolution (no emergence)
        enable_emergence=False,

        # Energy landscape
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.5,  # Weaker prior alignment
        lambda_obs=0.0,
        lambda_phi=0.0,

        # Moderate learning rates
        lr_mu_q=0.05,
        lr_sigma_q=0.005,
        lr_mu_p=0.01,
        lr_sigma_p=0.001,
        lr_phi=0.1,

        # ===================================
        # PULLBACK GEOMETRY TRACKING
        # ===================================
        track_pullback_geometry=True,
        geometry_track_interval=5,  # Record every 5 steps
        geometry_enable_consensus=True,  # Compute consensus (only 3 agents, cheap)
        geometry_enable_gauge_averaging=False,  # Expensive, disabled
        geometry_gauge_samples=50,
        geometry_lambda_obs=0.1,  # Observable sector threshold
        geometry_lambda_dark=0.01,  # Dark sector threshold
    )


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("PULLBACK GEOMETRY TRACKING DEMONSTRATION")
    print(f"{'='*70}\n")

    print("This demo shows how to track emergent spacetime geometry during simulation.")
    print("\nKey outputs:")
    print("  - geometry_evolution.png: 4-panel plot showing:")
    print("      * Observable sector dimensions over time")
    print("      * Volume elements (epistemic vs ontological)")
    print("      * Top eigenvalues (information flux directions)")
    print("      * Three-sector decomposition")
    print("")
    print("  - geometry_history.pkl: Full history for custom analysis")
    print("  - geometry_analysis/: Detailed eigenvalue spectrum")
    print("")

    # Create config
    cfg = demo_pullback_tracking_config()

    # Inject config into simulation_runner
    # (Normally you'd use command-line args, but we're injecting for demo)
    import simulation_runner
    original_main = simulation_runner.main

    def demo_main():
        import numpy as np
        from pathlib import Path

        # Setup
        np.random.seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        output_dir = Path(cfg.output_dir) / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("MULTI-AGENT SIMULATION")
        print(f"{'='*70}")
        print(f"Preset: demo_pullback_tracking")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")

        # Build and run
        manifold = simulation_runner.build_manifold(cfg)
        supports = simulation_runner.build_supports(manifold, cfg, rng)
        agents = simulation_runner.build_agents(manifold, supports, cfg, rng)
        system = simulation_runner.build_system(agents, cfg, rng)

        # Save config
        cfg.save(str(output_dir / "config.txt"))

        # Train
        history = simulation_runner.run_training(system, cfg, output_dir)

        # Summary
        print(f"\n{'='*70}")
        print("✓ SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Results: {output_dir}")
        print(f"{'='*70}\n")

        print("\nGeometry tracking results:")
        print(f"  {output_dir}/geometry_evolution.png")
        print(f"  {output_dir}/geometry_history.pkl")
        print(f"  {output_dir}/geometry_analysis/")
        print("")

    # Run demo
    demo_main()

    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"{'='*70}\n")

    print("Next steps:")
    print("  1. View geometry_evolution.png to see emergent structure")
    print("  2. Load geometry_history.pkl to analyze specific snapshots")
    print("  3. Try enabling consensus with gauge averaging (expensive!)")
    print("  4. Integrate into your own simulations via SimulationConfig")
    print("")
