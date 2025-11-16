"""
Publication Figure Generator
============================

Creates publication-ready figures and tables from training metrics.

Generates:
1. 2x2 panel figure (publication_figures.pdf):
   - A) Validation BPC comparison (FFN modes)
   - B) Free energy decomposition (theory validation)
   - C) Training convergence (stability)
   - D) Attention statistics (β vs KL)

2. LaTeX table (results_table.tex):
   - Final metrics for all FFN modes
   - Formatted for direct copy-paste into paper

3. Summary JSON (publication_summary.json):
   - Numerical results for text

Usage:
    # After running ablation study:
    python plot_publication_figures.py

    # Or specify directory:
    python plot_publication_figures.py --checkpoint_dir checkpoints_publication

Output:
    - publication_figures.pdf (2x2 panel, high DPI)
    - results_table.tex (LaTeX table)
    - publication_summary.json (numerical summary)

Author: Designed for minimal publishable claim
Date: November 2025
"""

import csv
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import rcParams

    # Publication-quality settings
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    rcParams['figure.titlesize'] = 14
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 4

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - install with: pip install matplotlib")


# FFN mode configuration
FFN_MODES = {
    'learned': {
        'label': 'Learned MLP',
        'color': '#2E86AB',  # Blue
        'linestyle': '-',
        'marker': 'o',
    },
    'variational_approx': {
        'label': 'Variational (Approx)',
        'color': '#A23B72',  # Purple
        'linestyle': '--',
        'marker': 's',
    },
    'variational_full': {
        'label': 'Variational (Full)',
        'color': '#F18F01',  # Orange
        'linestyle': '-.',
        'marker': '^',
    },
    'variational_gradient_engine': {
        'label': 'Variational (Gradient Engine)',
        'color': '#C73E1D',  # Red
        'linestyle': '-',
        'marker': 'D',
    },
}


def load_metrics_csv(csv_path: Path) -> Dict[str, List]:
    """Load training metrics from CSV file."""
    metrics = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse step
            if 'step' in row:
                metrics['steps'].append(int(row['step']))

            # Parse all numeric fields
            for key, value in row.items():
                if key == 'step':
                    continue

                # Try to parse as float
                try:
                    if value and value not in ['', 'None', 'nan']:
                        metrics[key].append(float(value))
                    else:
                        metrics[key].append(None)
                except (ValueError, TypeError):
                    metrics[key].append(None)

    return dict(metrics)


def find_metrics_files(checkpoint_dir: Path) -> Dict[str, Path]:
    """Find metrics.csv for each FFN mode."""
    found_metrics = {}

    for mode in FFN_MODES.keys():
        mode_dir = checkpoint_dir / f'ffn_{mode}'
        metrics_path = mode_dir / 'metrics.csv'

        if metrics_path.exists():
            found_metrics[mode] = metrics_path
            print(f"  ✓ Found {mode}: {metrics_path}")
        else:
            print(f"  ✗ Missing {mode}: {metrics_path}")

    return found_metrics


def get_final_value(values: List, default=None):
    """Get final non-None value from list."""
    for v in reversed(values):
        if v is not None:
            return v
    return default


def get_best_value(values: List, minimize=True, default=None):
    """Get best (min or max) non-None value from list."""
    valid = [v for v in values if v is not None]
    if not valid:
        return default
    return min(valid) if minimize else max(valid)


def create_publication_figure(
    all_metrics: Dict[str, Dict],
    output_path: Path,
):
    """Create 2x2 publication-ready figure panel."""

    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create figures - matplotlib not installed")
        return

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # =========================================================================
    # Panel A: Validation BPC Comparison
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    for mode, metrics in all_metrics.items():
        if mode not in FFN_MODES:
            continue

        style = FFN_MODES[mode]
        steps = metrics.get('steps', [])
        val_bpc = metrics.get('val_bpc', [])

        # Filter valid values
        valid_data = [(s, v) for s, v in zip(steps, val_bpc) if v is not None]
        if valid_data:
            val_steps, val_values = zip(*valid_data)
            ax_a.plot(
                val_steps, val_values,
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markevery=max(1, len(val_steps) // 10),
                alpha=0.85,
            )

    ax_a.set_xlabel('Training Step')
    ax_a.set_ylabel('Bits Per Character')
    ax_a.set_title('A) Validation Performance', fontweight='bold', loc='left')
    ax_a.legend(framealpha=0.95, loc='best')
    ax_a.grid(True, alpha=0.3, linestyle='--')

    # =========================================================================
    # Panel B: Free Energy Components
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    # Show free energy decomposition for one mode (preferably variational)
    # Pick first available variational mode, or learned as fallback
    demo_mode = None
    for mode in ['variational_gradient_engine', 'variational_full', 'variational_approx', 'learned']:
        if mode in all_metrics:
            demo_mode = mode
            break

    if demo_mode:
        metrics = all_metrics[demo_mode]
        steps = metrics.get('steps', [])

        # Plot components
        components = [
            ('train_loss_ce', 'Cross-Entropy (obs)', '#2E86AB', '-'),
            ('train_loss_belief_align', 'β (Belief Align)', '#C73E1D', '--'),
            ('train_loss_self_consistency', 'α (Self-Consistency)', '#F18F01', '-.'),
            ('train_loss_model_align', 'γ (Model Align)', '#6A4C93', ':'),
        ]

        for key, label, color, linestyle in components:
            values = metrics.get(key, [])
            valid_data = [(s, v) for s, v in zip(steps, values) if v is not None and v > 1e-6]

            if valid_data:
                comp_steps, comp_values = zip(*valid_data)
                ax_b.plot(
                    comp_steps, comp_values,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.85,
                )

        mode_label = FFN_MODES[demo_mode]['label']
        ax_b.set_xlabel('Training Step')
        ax_b.set_ylabel('Loss Component')
        ax_b.set_title(f'B) Free Energy: {mode_label}', fontweight='bold', loc='left')
        ax_b.legend(framealpha=0.95, loc='best')
        ax_b.grid(True, alpha=0.3, linestyle='--')
        ax_b.set_yscale('log')

    # =========================================================================
    # Panel C: Training Loss Convergence
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    for mode, metrics in all_metrics.items():
        if mode not in FFN_MODES:
            continue

        style = FFN_MODES[mode]
        steps = metrics.get('steps', [])
        train_loss = metrics.get('train_loss_total', [])

        # Filter valid values
        valid_data = [(s, v) for s, v in zip(steps, train_loss) if v is not None]
        if valid_data:
            train_steps, train_values = zip(*valid_data)
            ax_c.plot(
                train_steps, train_values,
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                alpha=0.7,
            )

    ax_c.set_xlabel('Training Step')
    ax_c.set_ylabel('Total Loss (Free Energy)')
    ax_c.set_title('C) Training Convergence', fontweight='bold', loc='left')
    ax_c.legend(framealpha=0.95, loc='best')
    ax_c.grid(True, alpha=0.3, linestyle='--')

    # =========================================================================
    # Panel D: Validation Perplexity (Alternative Performance Metric)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])

    for mode, metrics in all_metrics.items():
        if mode not in FFN_MODES:
            continue

        style = FFN_MODES[mode]
        steps = metrics.get('steps', [])
        val_ppl = metrics.get('val_ppl', [])

        # Filter valid values
        valid_data = [(s, v) for s, v in zip(steps, val_ppl) if v is not None]
        if valid_data:
            val_steps, val_values = zip(*valid_data)
            ax_d.plot(
                val_steps, val_values,
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markevery=max(1, len(val_steps) // 10),
                alpha=0.85,
            )

    ax_d.set_xlabel('Training Step')
    ax_d.set_ylabel('Perplexity')
    ax_d.set_title('D) Validation Perplexity', fontweight='bold', loc='left')
    ax_d.legend(framealpha=0.95, loc='best')
    ax_d.grid(True, alpha=0.3, linestyle='--')

    # Add random baseline reference
    vocab_size = 100  # From config_publication.py
    ax_d.axhline(vocab_size, color='gray', linestyle=':', linewidth=1,
                 alpha=0.5, label=f'Random baseline (vocab={vocab_size})')
    ax_d.legend(framealpha=0.95, loc='best')

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n📊 Publication figure saved: {output_path}")

    # Also save PNG version
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    print(f"📊 PNG version saved: {png_path}")

    plt.close()


def generate_latex_table(
    all_metrics: Dict[str, Dict],
    output_path: Path,
):
    """Generate LaTeX table of final results."""

    # Collect final metrics for each mode
    table_data = []

    for mode in ['learned', 'variational_approx', 'variational_full', 'variational_gradient_engine']:
        if mode not in all_metrics:
            continue

        metrics = all_metrics[mode]
        style = FFN_MODES[mode]

        row = {
            'mode': style['label'],
            'val_bpc': get_best_value(metrics.get('val_bpc', []), minimize=True),
            'val_ppl': get_best_value(metrics.get('val_ppl', []), minimize=True),
            'final_train_loss': get_final_value(metrics.get('train_loss_total', [])),
            'beta_final': get_final_value(metrics.get('train_loss_belief_align', []), default=0.0),
            'alpha_final': get_final_value(metrics.get('train_loss_self_consistency', []), default=0.0),
        }

        table_data.append(row)

    # Generate LaTeX
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Performance comparison of FFN modes on character-level language modeling.}")
    latex.append(r"\label{tab:results}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\hline")
    latex.append(r"FFN Mode & Val BPC $\downarrow$ & Val PPL $\downarrow$ & $\beta$ (final) & $\alpha$ (final) \\")
    latex.append(r"\hline")

    for row in table_data:
        mode = row['mode']
        bpc = f"{row['val_bpc']:.3f}" if row['val_bpc'] is not None else "---"
        ppl = f"{row['val_ppl']:.2f}" if row['val_ppl'] is not None else "---"
        beta = f"{row['beta_final']:.4f}" if row['beta_final'] is not None else "0.0000"
        alpha = f"{row['alpha_final']:.4f}" if row['alpha_final'] is not None else "0.0000"

        latex.append(f"{mode} & {bpc} & {ppl} & {beta} & {alpha} \\\\")

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"\n📄 LaTeX table saved: {output_path}")
    print("\nTable preview:")
    print('\n'.join(latex))


def generate_summary_json(
    all_metrics: Dict[str, Dict],
    output_path: Path,
):
    """Generate JSON summary of numerical results."""

    summary = {}

    for mode, metrics in all_metrics.items():
        if mode not in FFN_MODES:
            continue

        summary[mode] = {
            'label': FFN_MODES[mode]['label'],
            'final_metrics': {
                'train_loss': get_final_value(metrics.get('train_loss_total', [])),
                'val_loss': get_best_value(metrics.get('val_loss', []), minimize=True),
                'val_bpc': get_best_value(metrics.get('val_bpc', []), minimize=True),
                'val_ppl': get_best_value(metrics.get('val_ppl', []), minimize=True),
            },
            'free_energy_components': {
                'cross_entropy': get_final_value(metrics.get('train_loss_ce', [])),
                'belief_align': get_final_value(metrics.get('train_loss_belief_align', []), default=0.0),
                'self_consistency': get_final_value(metrics.get('train_loss_self_consistency', []), default=0.0),
                'model_align': get_final_value(metrics.get('train_loss_model_align', []), default=0.0),
            },
            'training_info': {
                'total_steps': len(metrics.get('steps', [])),
                'final_step': get_final_value(metrics.get('steps', []), default=0),
            },
        }

    # Calculate relative performance vs baseline
    if 'learned' in summary and 'variational_gradient_engine' in summary:
        learned_bpc = summary['learned']['final_metrics']['val_bpc']
        var_bpc = summary['variational_gradient_engine']['final_metrics']['val_bpc']

        if learned_bpc is not None and var_bpc is not None:
            relative_gap = ((var_bpc - learned_bpc) / learned_bpc) * 100
            summary['analysis'] = {
                'variational_vs_learned_gap_percent': relative_gap,
                'interpretation': f"Variational is {relative_gap:.1f}% {'worse' if relative_gap > 0 else 'better'} than learned baseline"
            }

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Summary JSON saved: {output_path}")

    # Print key results
    print("\n" + "="*70)
    print("KEY RESULTS")
    print("="*70)

    for mode, data in summary.items():
        if mode == 'analysis':
            continue
        print(f"\n{data['label']}:")
        print(f"  Val BPC:  {data['final_metrics']['val_bpc']:.3f}" if data['final_metrics']['val_bpc'] else "  Val BPC: N/A")
        print(f"  Val PPL:  {data['final_metrics']['val_ppl']:.2f}" if data['final_metrics']['val_ppl'] else "  Val PPL: N/A")
        if data['free_energy_components']['belief_align'] > 1e-6:
            print(f"  β (final): {data['free_energy_components']['belief_align']:.4f}")

    if 'analysis' in summary:
        print(f"\n{summary['analysis']['interpretation']}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready figures from training metrics'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints_publication',
        help='Directory containing FFN mode subdirectories with metrics.csv'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save output files'
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PUBLICATION FIGURE GENERATOR")
    print("="*70)
    print(f"\nSearching for metrics in: {checkpoint_dir}")

    # Find all metrics files
    metrics_files = find_metrics_files(checkpoint_dir)

    if not metrics_files:
        print("\n❌ No metrics files found!")
        print(f"   Expected structure: {checkpoint_dir}/ffn_<mode>/metrics.csv")
        print("\n   Run training first:")
        print("   python transformer/train_publication.py --run_ablation")
        return

    print(f"\n✓ Found {len(metrics_files)} metrics file(s)")

    # Load all metrics
    all_metrics = {}
    for mode, path in metrics_files.items():
        print(f"\nLoading {mode}...")
        all_metrics[mode] = load_metrics_csv(path)
        print(f"  Steps: {len(all_metrics[mode].get('steps', []))}")

    # Generate outputs
    print("\n" + "="*70)
    print("GENERATING PUBLICATION OUTPUTS")
    print("="*70)

    # 1. Publication figure (2x2 panel)
    fig_path = output_dir / 'publication_figures.pdf'
    create_publication_figure(all_metrics, fig_path)

    # 2. LaTeX table
    table_path = output_dir / 'results_table.tex'
    generate_latex_table(all_metrics, table_path)

    # 3. Summary JSON
    summary_path = output_dir / 'publication_summary.json'
    generate_summary_json(all_metrics, summary_path)

    print("\n" + "="*70)
    print("✓ PUBLICATION OUTPUTS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. {fig_path} (2x2 panel, high-DPI PDF)")
    print(f"  2. {fig_path.with_suffix('.png')} (PNG version)")
    print(f"  3. {table_path} (LaTeX table)")
    print(f"  4. {summary_path} (JSON summary)")
    print("\nNext steps:")
    print("  - Include publication_figures.pdf in manuscript")
    print("  - Copy results_table.tex into LaTeX document")
    print("  - Use publication_summary.json for text references")
    print("="*70)


if __name__ == '__main__':
    main()