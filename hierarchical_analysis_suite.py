#!/usr/bin/env python3
"""
Hierarchical Multi-Scale Analysis Suite
=======================================

Visualization and analysis tools for hierarchical meta-agent emergence.

Usage:
    python hierarchical_analysis_suite.py --run-dir _results/_playground
    python hierarchical_analysis_suite.py --run-dir _results/_playground --all

Author: Claude
Date: November 2025
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# Data Loading
# =============================================================================

def load_hierarchical_history(run_dir: Path) -> Optional[Dict]:
    """
    Load hierarchical evolution history.

    Looks for hierarchical_history.pkl (preferred) or hierarchical_history.npz.

    Returns:
        Dictionary with metrics or None if not found
    """
    pkl_path = run_dir / "hierarchical_history.pkl"
    npz_path = run_dir / "hierarchical_history.npz"

    # Prefer PKL
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            history = pickle.load(f)
        print(f"✓ Loaded hierarchical history from {pkl_path}")
        return history

    # Fallback to NPZ
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        history = {k: data[k] for k in data.files}
        print(f"✓ Loaded hierarchical history from {npz_path}")
        return history

    print("⚠️  No hierarchical_history.(pkl|npz) found")
    return None


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary(history: Dict, run_dir: Path):
    """Print summary statistics of hierarchical evolution."""
    print("\n" + "=" * 70)
    print(f"HIERARCHICAL EVOLUTION SUMMARY – {run_dir.name}")
    print("=" * 70 + "\n")

    steps = np.array(history['step'])
    n_steps = len(steps)

    print(f"Total steps: {n_steps}")
    print(f"Step range: {steps[0]} → {steps[-1]}")
    print()

    # Count condensations
    if 'n_condensations' in history:
        condensations = np.array(history['n_condensations'])
        total_condensations = np.sum(condensations)
        emergence_steps = steps[condensations > 0]

        print(f"Total condensations: {total_condensations}")
        print(f"Emergence events at steps: {list(emergence_steps)}")
        print()

    # Scale evolution
    if 'n_scales' in history:
        n_scales = np.array(history['n_scales'])
        max_scale = np.max(n_scales) - 1  # Scale 0, 1, 2, ... so -1 for max index

        print(f"Maximum scale reached: {max_scale}")
        print(f"Final number of scales: {n_scales[-1]}")
        print()

    # Agent counts
    if 'n_active_agents' in history:
        n_active = np.array(history['n_active_agents'])

        print(f"Initial active agents: {n_active[0]}")
        print(f"Final active agents: {n_active[-1]}")
        print()

    # Energy
    if 'total_energy' in history:
        energy = np.array(history['total_energy'])

        print(f"Initial energy: {energy[0]:.4f}")
        print(f"Final energy: {energy[-1]:.4f}")
        print(f"Energy change: {energy[-1] - energy[0]:.4f} ({((energy[-1] - energy[0]) / energy[0] * 100):.1f}%)")
        print()

    print("=" * 70 + "\n")


# =============================================================================
# Visualizations
# =============================================================================

def plot_overview(history: Dict, output_path: Optional[Path] = None):
    """
    Plot overview of hierarchical evolution.

    Shows:
    - Energy evolution
    - Agent counts per scale
    - Condensation events
    """
    steps = np.array(history['step'])

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # (1) Total energy
    ax1 = fig.add_subplot(gs[0, :])
    if 'total_energy' in history:
        energy = np.array(history['total_energy'])
        ax1.plot(steps, energy, 'b-', linewidth=2, label='Total Energy')
        ax1.set_ylabel('Free Energy', fontsize=12)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_title('Energy Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No energy data available',
                ha='center', va='center', transform=ax1.transAxes)

    # (2) Number of scales
    ax2 = fig.add_subplot(gs[1, 0])
    if 'n_scales' in history:
        n_scales = np.array(history['n_scales'])
        ax2.plot(steps, n_scales, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_ylabel('Number of Scales', fontsize=12)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_title('Scale Hierarchy Growth', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0.5)  # Start slightly below 1
    else:
        ax2.text(0.5, 0.5, 'No scale data available',
                ha='center', va='center', transform=ax2.transAxes)

    # (3) Total active agents
    ax3 = fig.add_subplot(gs[1, 1])
    if 'n_active_agents' in history:
        n_active = np.array(history['n_active_agents'])
        ax3.plot(steps, n_active, 'r-', linewidth=2)
        ax3.set_ylabel('Active Agents', fontsize=12)
        ax3.set_xlabel('Step', fontsize=12)
        ax3.set_title('Total Active Agents (All Scales)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No agent count data available',
                ha='center', va='center', transform=ax3.transAxes)

    # (4) Agents per scale (stacked area)
    ax4 = fig.add_subplot(gs[2, :])
    if 'n_active_per_scale' in history:
        # Convert list of dicts to arrays per scale
        n_active_per_scale = history['n_active_per_scale']

        # Find all scales that ever existed
        all_scales = set()
        for scale_dict in n_active_per_scale:
            all_scales.update(scale_dict.keys())
        all_scales = sorted(all_scales)

        # Build arrays
        scale_arrays = {}
        for scale in all_scales:
            scale_arrays[scale] = np.array([
                scale_dict.get(scale, 0) for scale_dict in n_active_per_scale
            ])

        # Stacked area plot
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_scales)))

        bottom = np.zeros(len(steps))
        for i, scale in enumerate(all_scales):
            ax4.fill_between(steps, bottom, bottom + scale_arrays[scale],
                           label=f'Scale {scale}', color=colors[i], alpha=0.7)
            bottom += scale_arrays[scale]

        ax4.set_ylabel('Active Agents', fontsize=12)
        ax4.set_xlabel('Step', fontsize=12)
        ax4.set_title('Active Agents Per Scale (Stacked)', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No per-scale data available',
                ha='center', va='center', transform=ax4.transAxes)

    # Mark condensation events
    if 'n_condensations' in history:
        condensations = np.array(history['n_condensations'])
        emergence_steps = steps[condensations > 0]

        for ax in [ax1, ax2, ax3, ax4]:
            for step in emergence_steps:
                ax.axvline(step, color='orange', alpha=0.5, linestyle='--',
                          linewidth=1, label='Condensation' if step == emergence_steps[0] else '')

    plt.suptitle('Hierarchical Multi-Scale Evolution', fontsize=16, fontweight='bold', y=0.995)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved overview plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_condensation_timeline(history: Dict, output_path: Optional[Path] = None):
    """
    Plot detailed condensation timeline.

    Shows when and where condensations occurred.
    """
    if 'n_condensations' not in history:
        print("⚠️  No condensation data available")
        return

    steps = np.array(history['step'])
    condensations = np.array(history['n_condensations'])

    emergence_indices = np.where(condensations > 0)[0]

    if len(emergence_indices) == 0:
        print("ℹ️  No condensations occurred during simulation")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot condensations
    ax.stem(steps[emergence_indices], condensations[emergence_indices],
           linefmt='orange', markerfmt='o', basefmt=' ', label='Condensations')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Number of Meta-Agents Formed', fontsize=12)
    ax.set_title('Condensation Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Annotate each event
    for idx in emergence_indices:
        step = steps[idx]
        count = condensations[idx]
        ax.annotate(f'{int(count)}', xy=(step, count), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved condensation timeline to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_scale_breakdown(history: Dict, output_path: Optional[Path] = None):
    """
    Plot active vs total agents per scale.

    Shows how agents are distributed across scales and what fraction is active.
    """
    if 'n_active_per_scale' not in history or 'n_agents_per_scale' not in history:
        print("⚠️  No per-scale agent data available")
        return

    steps = np.array(history['step'])
    n_active_per_scale = history['n_active_per_scale']
    n_agents_per_scale = history['n_agents_per_scale']

    # Find all scales
    all_scales = set()
    for scale_dict in n_agents_per_scale:
        all_scales.update(scale_dict.keys())
    all_scales = sorted(all_scales)

    n_scales = len(all_scales)

    fig, axes = plt.subplots(n_scales, 1, figsize=(14, 4 * n_scales),
                            sharex=True, squeeze=False)
    axes = axes.flatten()

    for i, scale in enumerate(all_scales):
        ax = axes[i]

        # Extract data for this scale
        active = np.array([d.get(scale, 0) for d in n_active_per_scale])
        total = np.array([d.get(scale, 0) for d in n_agents_per_scale])

        # Plot
        ax.plot(steps, total, 'b-', linewidth=2, label='Total', marker='o', markersize=3)
        ax.plot(steps, active, 'g-', linewidth=2, label='Active', marker='s', markersize=3)
        ax.fill_between(steps, 0, active, alpha=0.3, color='green')

        ax.set_ylabel('Agent Count', fontsize=11)
        ax.set_title(f'Scale {scale}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Mark condensation events
        if 'n_condensations' in history:
            condensations = np.array(history['n_condensations'])
            emergence_steps = steps[condensations > 0]

            for step in emergence_steps:
                ax.axvline(step, color='orange', alpha=0.5, linestyle='--', linewidth=1)

    axes[-1].set_xlabel('Step', fontsize=12)

    plt.suptitle('Agent Breakdown Per Scale', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved scale breakdown to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_update_statistics(history: Dict, output_path: Optional[Path] = None):
    """
    Plot update and prior synchronization statistics.
    """
    steps = np.array(history['step'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # (1) Updates applied
    if 'updates_applied' in history:
        updates = np.array(history['updates_applied'])
        ax1.plot(steps, updates, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_ylabel('Updates Applied', fontsize=12)
        ax1.set_title('Gradient Updates Per Step', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No update data available',
                ha='center', va='center', transform=ax1.transAxes)

    # (2) Priors updated (top-down coupling)
    if 'priors_updated' in history:
        priors = np.array(history['priors_updated'])
        ax2.plot(steps, priors, 'r-', linewidth=2, marker='s', markersize=3)
        ax2.set_ylabel('Priors Synchronized', fontsize=12)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_title('Top-Down Prior Updates', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No prior update data available',
                ha='center', va='center', transform=ax2.transAxes)

    # Mark condensations
    if 'n_condensations' in history:
        condensations = np.array(history['n_condensations'])
        emergence_steps = steps[condensations > 0]

        for ax in [ax1, ax2]:
            for step in emergence_steps:
                ax.axvline(step, color='orange', alpha=0.5, linestyle='--', linewidth=1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved update statistics to {output_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Multi-Scale Analysis Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hierarchical_analysis_suite.py --run-dir _results/_playground
  python hierarchical_analysis_suite.py --run-dir _results/_playground --all
  python hierarchical_analysis_suite.py --run-dir _results/_playground --plot overview
        """
    )

    parser.add_argument('--run-dir', type=str, required=True,
                       help='Path to run directory containing hierarchical_history files')
    parser.add_argument('--plot', type=str, choices=['overview', 'condensation', 'scales', 'updates'],
                       help='Generate specific plot (default: overview)')
    parser.add_argument('--all', action='store_true',
                       help='Generate all plots')
    parser.add_argument('--save', action='store_true',
                       help='Save plots instead of displaying')

    args = parser.parse_args()

    # Load data
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ Directory not found: {run_dir}")
        return

    history = load_hierarchical_history(run_dir)
    if history is None:
        print("\n❌ No hierarchical history data found!")
        print("\nExpected files:")
        print("  - hierarchical_history.pkl  (preferred)")
        print("  - hierarchical_history.npz  (fallback)")
        return

    # Print summary
    print_summary(history, run_dir)

    # Determine which plots to generate
    if args.all:
        plots_to_generate = ['overview', 'condensation', 'scales', 'updates']
    elif args.plot:
        plots_to_generate = [args.plot]
    else:
        plots_to_generate = ['overview']  # Default

    # Generate plots
    for plot_name in plots_to_generate:
        if args.save:
            output_path = run_dir / f"{plot_name}_analysis.png"
        else:
            output_path = None

        if plot_name == 'overview':
            plot_overview(history, output_path)
        elif plot_name == 'condensation':
            plot_condensation_timeline(history, output_path)
        elif plot_name == 'scales':
            plot_scale_breakdown(history, output_path)
        elif plot_name == 'updates':
            plot_update_statistics(history, output_path)

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
