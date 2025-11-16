"""
Plot Training History
=====================

Visualize training metrics from CSV history file.

Usage:
    # CLICK TO RUN - auto-finds most recent metrics.csv
    python plot_training.py

    # Or specify a specific file
    python plot_training.py checkpoints_publication/ffn_learned/metrics.csv

Output:
    - training_plots.png (all metrics)
    - Text summary in console
"""

import csv
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - install with: pip install matplotlib")


def load_history(csv_path: Path):
    """Load training history from CSV."""
    history = {
        'steps': [],
        'train_loss': [],
        'train_ce_loss': [],
        'train_loss_belief_align': [],
        'train_loss_self_consistency': [],
        'train_loss_model_align': [],
        'train_perplexity': [],
        'train_bpc': [],
        'val_loss': [],
        'val_perplexity': [],
        'val_bpc': [],
        'beta_mean': [],
        'kl_mean': [],
        'mu_lr': [],
        'sigma_lr': [],
        'phi_lr': [],
        'ffn_lr': [],
        'grad_norm_total': [],
        'grad_norm_mu': [],
        'grad_norm_ffn': [],
        'step_time': [],
        'tokens_per_sec': [],
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history['steps'].append(int(row['step']))

            # Helper to safely parse float
            def get_float(key, default=None):
                return float(row[key]) if row.get(key) and row[key] not in ['', 'None'] else default

            # Core losses
            history['train_loss'].append(get_float('train_loss_total', get_float('train_loss')))
            history['train_ce_loss'].append(get_float('train_loss_ce', get_float('train_ce_loss')))
            history['train_loss_belief_align'].append(get_float('train_loss_belief_align', 0))
            history['train_loss_self_consistency'].append(get_float('train_loss_self_consistency', 0))
            history['train_loss_model_align'].append(get_float('train_loss_model_align', 0))

            # Metrics
            history['train_perplexity'].append(get_float('train_ppl', get_float('train_perplexity')))
            history['train_bpc'].append(get_float('train_bpc'))
            history['val_loss'].append(get_float('val_loss'))
            history['val_perplexity'].append(get_float('val_ppl', get_float('val_perplexity')))
            history['val_bpc'].append(get_float('val_bpc'))

            # Attention
            history['beta_mean'].append(get_float('beta_mean'))
            history['kl_mean'].append(get_float('kl_mean'))

            # Learning rates
            history['mu_lr'].append(get_float('mu_lr'))
            history['sigma_lr'].append(get_float('sigma_lr'))
            history['phi_lr'].append(get_float('phi_lr'))
            history['ffn_lr'].append(get_float('ffn_lr'))

            # Gradients
            history['grad_norm_total'].append(get_float('grad_norm_total'))
            history['grad_norm_mu'].append(get_float('grad_norm_mu'))
            history['grad_norm_ffn'].append(get_float('grad_norm_ffn'))

            # Performance
            history['step_time'].append(get_float('step_time'))
            history['tokens_per_sec'].append(get_float('tokens_per_sec'))

    return history


def print_summary(history):
    """Print text summary of training."""
    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    steps = history['steps']
    train_loss = [x for x in history['train_loss'] if x is not None]
    val_loss = [x for x in history['val_loss'] if x is not None]
    val_ppl = [x for x in history['val_perplexity'] if x is not None]
    val_bpc = [x for x in history['val_bpc'] if x is not None]
    step_times = [x for x in history['step_time'] if x is not None]

    print(f"Total steps: {len(steps)}")
    print(f"Final step: {steps[-1] if steps else 0}")
    print()

    if train_loss:
        print(f"Training Loss:")
        print(f"  Initial: {train_loss[0]:.4f}")
        print(f"  Final:   {train_loss[-1]:.4f}")
        print(f"  Best:    {min(train_loss):.4f}")
        print()

    if val_loss:
        print(f"Validation Loss:")
        print(f"  Best: {min(val_loss):.4f}")
        print()

    if val_ppl:
        print(f"Validation Perplexity:")
        print(f"  Best: {min(val_ppl):.2f}")
        print()

    if val_bpc:
        print(f"Validation BPC:")
        print(f"  Best: {min(val_bpc):.3f}")
        print()

    # Free energy components
    belief_align = [x for x in history['train_loss_belief_align'] if x is not None and x > 0]
    self_consist = [x for x in history['train_loss_self_consistency'] if x is not None and x > 0]
    model_align = [x for x in history['train_loss_model_align'] if x is not None and x > 0]

    if belief_align or self_consist or model_align:
        print(f"Free Energy Components (final):")
        if belief_align:
            print(f"  β (Belief Align):     {belief_align[-1]:.4f}")
        if self_consist:
            print(f"  α (Self-Consistency): {self_consist[-1]:.4f}")
        if model_align:
            print(f"  γ (Model Align):      {model_align[-1]:.4f}")
        print()

    # Gradient norms
    grad_norms = [x for x in history['grad_norm_total'] if x is not None]
    if grad_norms:
        print(f"Gradient Norms:")
        print(f"  Average: {sum(grad_norms)/len(grad_norms):.3f}")
        print(f"  Final:   {grad_norms[-1]:.3f}")
        print()

    if step_times:
        avg_time = sum(step_times) / len(step_times)
        print(f"Step Time:")
        print(f"  Average: {avg_time:.2f} seconds")
        print(f"  Total: {sum(step_times)/60:.1f} minutes")
        print()

    # Throughput
    tokens_per_sec = [x for x in history['tokens_per_sec'] if x is not None]
    if tokens_per_sec:
        avg_throughput = sum(tokens_per_sec) / len(tokens_per_sec)
        print(f"Throughput:")
        print(f"  Average: {avg_throughput:.0f} tokens/second")
        print()

    print("="*70)


def plot_history(history, output_path: Path):
    """Create plots of training history."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create plots - matplotlib not installed")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('Training History', fontsize=16)

    steps = history['steps']

    # Helper to filter non-None values
    def filter_vals(vals):
        return [(s, v) for s, v in zip(steps, vals) if v is not None]

    # 1. Loss curves
    ax = axes[0, 0]
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    ax.plot(steps, train_loss, label='Train Loss', alpha=0.7)

    val_data = filter_vals(val_loss)
    if val_data:
        val_steps, val_values = zip(*val_data)
        ax.plot(val_steps, val_values, 'o-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Free Energy Components
    ax = axes[0, 1]
    belief_align = history['train_loss_belief_align']
    self_consistency = history['train_loss_self_consistency']
    model_align = history['train_loss_model_align']

    belief_data = filter_vals(belief_align)
    self_data = filter_vals(self_consistency)
    model_data = filter_vals(model_align)

    if belief_data:
        b_steps, b_vals = zip(*belief_data)
        ax.plot(b_steps, b_vals, label='β (Belief Align)', alpha=0.7)
    if self_data:
        s_steps, s_vals = zip(*self_data)
        ax.plot(s_steps, s_vals, label='α (Self-Consistency)', alpha=0.7)
    if model_data:
        m_steps, m_vals = zip(*model_data)
        ax.plot(m_steps, m_vals, label='γ (Model Align)', alpha=0.7)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss Component')
    ax.set_title('Free Energy Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. BPC (Bits-Per-Character)
    ax = axes[1, 0]
    train_bpc = history['train_bpc']
    val_bpc = history['val_bpc']

    train_bpc_data = filter_vals(train_bpc)
    val_bpc_data = filter_vals(val_bpc)

    if train_bpc_data:
        t_steps, t_vals = zip(*train_bpc_data)
        ax.plot(t_steps, t_vals, label='Train BPC', alpha=0.7)
    if val_bpc_data:
        v_steps, v_vals = zip(*val_bpc_data)
        ax.plot(v_steps, v_vals, 'o-', label='Val BPC', linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Bits-Per-Character')
    ax.set_title('Bits-Per-Character (lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Gradient Norms
    ax = axes[1, 1]
    grad_total = history['grad_norm_total']
    grad_mu = history['grad_norm_mu']
    grad_ffn = history['grad_norm_ffn']

    total_data = filter_vals(grad_total)
    mu_data = filter_vals(grad_mu)
    ffn_data = filter_vals(grad_ffn)

    if total_data:
        t_steps, t_vals = zip(*total_data)
        ax.plot(t_steps, t_vals, label='Total', alpha=0.7)
    if mu_data:
        m_steps, m_vals = zip(*mu_data)
        ax.plot(m_steps, m_vals, label='μ (Embedding)', alpha=0.7)
    if ffn_data:
        f_steps, f_vals = zip(*ffn_data)
        ax.plot(f_steps, f_vals, label='FFN', alpha=0.7)

    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 5. Learning Rates
    ax = axes[2, 0]
    mu_lr = history['mu_lr']
    sigma_lr = history['sigma_lr']
    phi_lr = history['phi_lr']
    ffn_lr = history['ffn_lr']

    mu_lr_data = filter_vals(mu_lr)
    sigma_lr_data = filter_vals(sigma_lr)
    phi_lr_data = filter_vals(phi_lr)
    ffn_lr_data = filter_vals(ffn_lr)

    if mu_lr_data:
        m_steps, m_vals = zip(*mu_lr_data)
        ax.plot(m_steps, m_vals, label='μ LR', alpha=0.7)
    if sigma_lr_data:
        s_steps, s_vals = zip(*sigma_lr_data)
        ax.plot(s_steps, s_vals, label='σ LR', alpha=0.7)
    if phi_lr_data:
        p_steps, p_vals = zip(*phi_lr_data)
        ax.plot(p_steps, p_vals, label='φ LR', alpha=0.7)
    if ffn_lr_data:
        f_steps, f_vals = zip(*ffn_lr_data)
        ax.plot(f_steps, f_vals, label='FFN LR', alpha=0.7)

    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Performance
    ax = axes[2, 1]
    tokens_per_sec = history['tokens_per_sec']

    tokens_data = filter_vals(tokens_per_sec)
    if tokens_data:
        t_steps, t_vals = zip(*tokens_data)
        ax.plot(t_steps, t_vals, alpha=0.5, label='Tokens/sec')

        # Add moving average
        window = min(10, len(t_vals) // 10)
        if window > 1:
            moving_avg = []
            for i in range(len(t_vals)):
                start = max(0, i - window + 1)
                moving_avg.append(sum(t_vals[start:i+1]) / (i - start + 1))
            ax.plot(t_steps, moving_avg, linewidth=2, label=f'{window}-step MA')

        ax.set_xlabel('Step')
        ax.set_ylabel('Tokens/second')
        ax.set_title('Training Throughput')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {output_path}")

    # Also try to show if in interactive mode
    try:
        plt.show()
    except:
        pass


def find_most_recent_metrics():
    """Auto-find the most recent metrics.csv file."""
    # Search for all metrics.csv files in checkpoints directories
    checkpoint_dirs = [
        Path('checkpoints_publication'),
        Path('checkpoints'),
        Path('.'),
    ]

    csv_files = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            csv_files.extend(checkpoint_dir.rglob('metrics.csv'))

    if not csv_files:
        return None

    # Return most recently modified
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description='Plot training history')
    parser.add_argument('csv_file', type=str, nargs='?', default=None,
                        help='Path to metrics.csv (optional - auto-finds most recent if not provided)')
    parser.add_argument('--output', type=str, default='training_plots.png',
                        help='Output plot filename')

    args = parser.parse_args()

    # Auto-find if not specified
    if args.csv_file is None:
        print("🔍 Auto-searching for most recent metrics.csv...")
        csv_path = find_most_recent_metrics()
        if csv_path is None:
            print("❌ No metrics.csv files found in checkpoints directories")
            print("   Run training first or specify path manually:")
            print("   python plot_training.py path/to/metrics.csv")
            return
        print(f"✓ Found: {csv_path}\n")
    else:
        csv_path = Path(args.csv_file)

    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return

    # Load history
    print(f"Loading history from: {csv_path}")
    history = load_history(csv_path)

    # Print summary
    print_summary(history)

    # Create plots
    output_path = csv_path.parent / args.output
    plot_history(history, output_path)


if __name__ == '__main__':
    main()