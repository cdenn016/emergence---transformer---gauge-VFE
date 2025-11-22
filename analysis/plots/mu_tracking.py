"""
Mu Center Tracking Plots
=========================

Plotting functions for visualizing mu center evolution over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any


def plot_mu_summary(mu_tracker: Any, save_path: Path):
    """
    Plot summary of mu center tracking.

    Args:
        mu_tracker: MuCenterTracking object
        save_path: Path to save figure
    """
    if not hasattr(mu_tracker, 'steps') or not mu_tracker.steps:
        print("No mu tracking data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Mu Center Tracking', fontsize=16, fontweight='bold')

    steps = mu_tracker.steps
    n_agents = len(mu_tracker.mu_norms) if hasattr(mu_tracker, 'mu_norms') else 0

    # Mu norms over time
    ax = axes[0, 0]
    if hasattr(mu_tracker, 'mu_norms'):
        for i, norms in enumerate(mu_tracker.mu_norms):
            if norms:
                ax.plot(steps[:len(norms)], norms, alpha=0.7, label=f'Agent {i}')
        ax.set_title('Mu Norm Evolution')
        ax.set_ylabel('||μ(center)||')
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
        if n_agents < 10:
            ax.legend(fontsize=8)

    # Mean and variance of norms
    ax = axes[0, 1]
    if hasattr(mu_tracker, 'mu_norms'):
        all_norms = []
        for step_idx in range(len(steps)):
            step_norms = [norms[step_idx] if step_idx < len(norms) else 0
                         for norms in mu_tracker.mu_norms]
            all_norms.append(step_norms)

        mean_norms = [np.mean(n) for n in all_norms]
        std_norms = [np.std(n) for n in all_norms]

        ax.plot(steps, mean_norms, 'b-', linewidth=2, label='Mean')
        ax.fill_between(steps,
                        np.array(mean_norms) - np.array(std_norms),
                        np.array(mean_norms) + np.array(std_norms),
                        alpha=0.3, label='±1σ')
        ax.set_title('Mean Mu Norm (±1σ)')
        ax.set_ylabel('||μ||')
        ax.set_xlabel('Step')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Distribution at final step
    ax = axes[1, 0]
    if hasattr(mu_tracker, 'mu_norms'):
        final_norms = [norms[-1] if norms else 0 for norms in mu_tracker.mu_norms]
        ax.hist(final_norms, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(final_norms), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(final_norms):.3f}')
        ax.set_title('Final Mu Norm Distribution')
        ax.set_xlabel('||μ(center)||')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Info text
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""
    Mu Center Tracking Summary

    Number of agents: {n_agents}
    Training steps: {len(steps)}

    Final statistics:
    """
    if hasattr(mu_tracker, 'mu_norms'):
        final_norms = [norms[-1] if norms else 0 for norms in mu_tracker.mu_norms]
        info_text += f"""  Mean norm: {np.mean(final_norms):.4f}
      Std norm:  {np.std(final_norms):.4f}
      Min norm:  {np.min(final_norms):.4f}
      Max norm:  {np.max(final_norms):.4f}
    """
    ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved mu tracking plot: {save_path}")
