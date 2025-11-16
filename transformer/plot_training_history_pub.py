"""
Publication-Quality Figure Generator
====================================

Generate publication-ready figures from gauge-theoretic transformer training metrics.
Produces high-quality visualizations suitable for JMLR, NeurIPS, Nature Machine Intelligence.

JUST CLICK RUN! All configuration is below.

Output:
    figures/
        fig1_training_dynamics.pdf      - Main training curves
        fig2_free_energy_components.pdf - α, β, γ component analysis  
        fig3_convergence_analysis.pdf   - Natural gradient convergence
        fig4_attention_dynamics.pdf     - KL divergence and β evolution
        fig5_ablation_comparison.pdf    - FFN mode comparison (if multiple)
        statistics_report.txt           - Detailed statistical analysis
        
Author: Publication Figure Suite
Date: November 2025
"""

# ============================================================================
# CONFIGURATION - EDIT THESE AND CLICK RUN
# ============================================================================

# Input data location
METRICS_FILE = 'checkpoints_publication/ffn_variational_gradient_engine/metrics.csv'  
# Or set to None to auto-detect most recent

# For ablation comparison (set to directory containing multiple experiments)
ABLATION_DIR = 'checkpoints_publication'  # Set to None to skip ablation
CREATE_ABLATION = True  # Set True to create comparison figure if multiple experiments exist

# Output configuration
OUTPUT_DIR = 'figures'
FIGURE_FORMAT = 'png'  # 'pdf', 'png', or 'svg'
DPI = 300  # Resolution for raster formats

# Figure selection (set False to skip specific figures)
CREATE_TRAINING_DYNAMICS = True
CREATE_FREE_ENERGY = True
CREATE_CONVERGENCE = True
CREATE_ATTENTION = True
CREATE_STATISTICS = True

# Style preferences
USE_LATEX_FONTS = True  # Use Computer Modern fonts (LaTeX style)
COLOR_SCHEME = 'default'  # 'default', 'colorblind', or 'grayscale'

# ============================================================================
# END CONFIGURATION
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AUTO-DETECTION
# ============================================================================

def auto_detect_metrics():
    """Auto-detect most recent metrics file."""
    search_paths = [
        Path('checkpoints_publication'),
        Path('checkpoints'),
        Path('.'),
    ]
    
    all_metrics = []
    for base_path in search_paths:
        if base_path.exists():
            all_metrics.extend(base_path.rglob('metrics.csv'))
    
    if not all_metrics:
        return None
    
    # Return most recently modified
    return max(all_metrics, key=lambda p: p.stat().st_mtime)


# ============================================================================
# COLOR SCHEMES
# ============================================================================

def get_color_scheme(scheme_name='default'):
    """Get color palette based on scheme name."""
    
    if scheme_name == 'colorblind':
        return {
            'train': '#0173B2',      # Blue
            'val': '#DE8F05',        # Orange
            'alpha': '#029E73',      # Green
            'beta': '#CC78BC',       # Light purple
            'gamma': '#ECE133',      # Yellow
            'grad': '#56B4E9',       # Light blue
            'lr': '#F0E442',         # Light yellow
            'attention': '#949494',  # Gray
        }
    elif scheme_name == 'grayscale':
        return {
            'train': '#000000',      
            'val': '#404040',        
            'alpha': '#606060',      
            'beta': '#808080',       
            'gamma': '#A0A0A0',      
            'grad': '#C0C0C0',       
            'lr': '#D0D0D0',         
            'attention': '#707070',  
        }
    else:  # default
        return {
            'train': '#2E86AB',      # Deep blue
            'val': '#A23B72',        # Burgundy
            'alpha': '#F18F01',      # Orange (self-consistency)
            'beta': '#C73E1D',       # Red (belief alignment)
            'gamma': '#6B4C9D',      # Purple (model alignment)
            'grad': '#2D7E2D',       # Green
            'lr': '#B8336A',         # Pink
            'attention': '#4A5859',  # Dark gray
        }


# ============================================================================
# PUBLICATION STYLE CONFIGURATION
# ============================================================================

def setup_publication_style(use_latex=True, color_scheme='default'):
    """Configure matplotlib for publication-quality figures."""
    
    # Use seaborn paper style as base
    sns.set_style("whitegrid", {
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.3,
    })
    
    # Font configuration
    if use_latex:
        font_family = 'serif'
        font_serif = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
    else:
        font_family = 'sans-serif'
        font_serif = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # Configure matplotlib for publication
    rcParams.update({
        # Figure
        'figure.dpi': 100,  # Display DPI
        'savefig.dpi': DPI,
        'savefig.format': FIGURE_FORMAT,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font
        'font.family': font_family,
        'font.serif': font_serif,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Axes
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        
        # Ticks
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'legend.borderpad': 0.5,
        
        # Math text
        'mathtext.fontset': 'cm' if use_latex else 'dejavusans',
    })
    
    return get_color_scheme(color_scheme)


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess metrics from CSV."""
    df = pd.read_csv(csv_path)
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(0)
    
    # Compute additional metrics if needed
    if 'train_bpc' not in df.columns and 'train_loss_ce' in df.columns:
        df['train_bpc'] = df['train_loss_ce'] / np.log(2)
    
    if 'val_bpc' not in df.columns and 'val_ce' in df.columns:
        df['val_bpc'] = df['val_ce'] / np.log(2)
        
    print(f"  Loaded {len(df)} steps from {csv_path.name}")
        
    return df


def smooth_curve(y: np.ndarray, window_size: int = None) -> np.ndarray:
    """Apply Savitzky-Golay smoothing for cleaner curves."""
    # For small datasets, don't smooth
    if len(y) < 10:
        return y
    
    if window_size is None:
        window_size = min(21, max(5, len(y) // 10))  # At least 5
        if window_size % 2 == 0:
            window_size += 1
    
    # Ensure polynomial order is less than window size
    polyorder = min(3, window_size - 1)
    
    if len(y) < window_size:
        return y
        
    return savgol_filter(y, window_size, polyorder)


# ============================================================================
# FIGURE 1: MAIN TRAINING DYNAMICS
# ============================================================================

def create_training_dynamics_figure(df: pd.DataFrame, colors: Dict, output_dir: Path):
    """Main training curves showing loss, perplexity, and BPC."""
    
    print("  Creating training dynamics figure...")
    
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['step'], df['train_loss_total'], color=colors['train'], 
             alpha=0.7, linewidth=1, label='Train')
    
    # Add validation points
    val_mask = df['val_loss'].notna() & (df['val_loss'] > 0)
    if val_mask.any():
        ax1.scatter(df.loc[val_mask, 'step'], df.loc[val_mask, 'val_loss'],
                   color=colors['val'], s=30, alpha=0.8, marker='o', 
                   edgecolors='white', linewidth=0.5, label='Validation', zorder=5)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('(a) Variational Free Energy', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Cross-Entropy Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['step'], df['train_loss_ce'], color=colors['train'],
             alpha=0.7, linewidth=1, label='Train CE')
    
    if val_mask.any():
        ax2.scatter(df.loc[val_mask, 'step'], df.loc[val_mask, 'val_ce'],
                   color=colors['val'], s=30, alpha=0.8, marker='o',
                   edgecolors='white', linewidth=0.5, label='Val CE', zorder=5)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_title('(b) Prediction Loss', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Perplexity
    ax3 = fig.add_subplot(gs[1, 0])
    train_ppl = df['train_ppl'].values
    train_ppl = np.clip(train_ppl, 1, 1000)  # Clip for visualization
    
    ax3.semilogy(df['step'], train_ppl, color=colors['train'],
                 alpha=0.7, linewidth=1, label='Train')
    
    val_ppl_mask = df['val_ppl'].notna() & (df['val_ppl'] > 0)
    if val_ppl_mask.any():
        ax3.scatter(df.loc[val_ppl_mask, 'step'], df.loc[val_ppl_mask, 'val_ppl'],
                   color=colors['val'], s=30, alpha=0.8, marker='o',
                   edgecolors='white', linewidth=0.5, label='Validation', zorder=5)
    
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Perplexity (log scale)')
    ax3.set_title('(c) Perplexity', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Bits-Per-Character
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['step'], df['train_bpc'], color=colors['train'],
             alpha=0.7, linewidth=1, label='Train')
    
    val_bpc_mask = df['val_bpc'].notna() & (df['val_bpc'] > 0)
    if val_bpc_mask.any():
        ax4.scatter(df.loc[val_bpc_mask, 'step'], df.loc[val_bpc_mask, 'val_bpc'],
                   color=colors['val'], s=30, alpha=0.8, marker='o',
                   edgecolors='white', linewidth=0.5, label='Validation', zorder=5)
    
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Bits-Per-Character')
    ax4.set_title('(d) Information Content', fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Add figure caption
    fig.text(0.5, 0.01, 
             'Figure 1: Training dynamics of the gauge-theoretic transformer showing convergence\n'
             'of variational free energy and predictive performance metrics.',
             ha='center', fontsize=9, style='italic')
    
    # Save
    output_path = output_dir / f'fig1_training_dynamics.{FIGURE_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


# ============================================================================
# FIGURE 2: FREE ENERGY COMPONENT ANALYSIS
# ============================================================================

def create_free_energy_figure(df: pd.DataFrame, colors: Dict, output_dir: Path):
    """Analyze individual components of the variational free energy."""
    
    print("  Creating free energy components figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Get component data
    belief_align = df['train_loss_belief_align'].values
    self_consistency = df['train_loss_self_consistency'].values
    model_align = df['train_loss_model_align'].values
    ce_loss = df['train_loss_ce'].values
    
    # Subplot 1: All Components
    ax = axes[0, 0]
    ax.plot(df['step'], smooth_curve(belief_align), color=colors['beta'],
            label=r'$\beta$ (Belief Alignment)', linewidth=1.5, alpha=0.8)
    ax.plot(df['step'], smooth_curve(self_consistency), color=colors['alpha'],
            label=r'$\alpha$ (Self-Consistency)', linewidth=1.5, alpha=0.8)
    ax.plot(df['step'], smooth_curve(model_align), color=colors['gamma'],
            label=r'$\gamma$ (Model Alignment)', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Component')
    ax.set_title('(a) Free Energy Components', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Relative Contributions
    ax = axes[0, 1]
    total_fe = belief_align + self_consistency + model_align + ce_loss
    
    # Compute relative contributions
    rel_beta = belief_align / (total_fe + 1e-8) * 100
    rel_alpha = self_consistency / (total_fe + 1e-8) * 100
    rel_gamma = model_align / (total_fe + 1e-8) * 100
    rel_ce = ce_loss / (total_fe + 1e-8) * 100
    
    # Stack plot
    ax.fill_between(df['step'], 0, rel_ce, color=colors['train'], alpha=0.7, label='CE Loss')
    ax.fill_between(df['step'], rel_ce, rel_ce + rel_beta, color=colors['beta'], alpha=0.7, label=r'$\beta$')
    ax.fill_between(df['step'], rel_ce + rel_beta, rel_ce + rel_beta + rel_alpha, 
                    color=colors['alpha'], alpha=0.7, label=r'$\alpha$')
    ax.fill_between(df['step'], rel_ce + rel_beta + rel_alpha, 100,
                    color=colors['gamma'], alpha=0.7, label=r'$\gamma$')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Relative Contribution (%)')
    ax.set_title('(b) Component Proportions', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Component Evolution (normalized)
    ax = axes[1, 0]
    
    # Normalize components to [0, 1] for comparison
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 0:
            return (x - x_min) / (x_max - x_min)
        return x
    
    ax.plot(df['step'], normalize(smooth_curve(belief_align)), color=colors['beta'],
            label=r'$\beta$ (normalized)', linewidth=1.5, alpha=0.8)
    ax.plot(df['step'], normalize(smooth_curve(self_consistency)), color=colors['alpha'],
            label=r'$\alpha$ (normalized)', linewidth=1.5, alpha=0.8)
    ax.plot(df['step'], normalize(smooth_curve(model_align)), color=colors['gamma'],
            label=r'$\gamma$ (normalized)', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Normalized Value')
    ax.set_title('(c) Normalized Evolution', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Phase Diagram
    ax = axes[1, 1]
    
    # Create 2D phase plot (beta vs alpha)
    scatter = ax.scatter(belief_align[::5], self_consistency[::5],
                        c=df['step'].values[::5], cmap='viridis',
                        s=10, alpha=0.6)
    
    # Add start and end markers
    ax.scatter(belief_align[0], self_consistency[0], color='green',
              s=100, marker='o', edgecolors='black', linewidth=1, label='Start', zorder=5)
    ax.scatter(belief_align[-1], self_consistency[-1], color='red',
              s=100, marker='s', edgecolors='black', linewidth=1, label='End', zorder=5)
    
    ax.set_xlabel(r'$\beta$ (Belief Alignment)')
    ax.set_ylabel(r'$\alpha$ (Self-Consistency)')
    ax.set_title('(d) Free Energy Phase Space', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Step', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Add figure caption
    fig.text(0.5, -0.02, 
             'Figure 2: Decomposition of variational free energy showing the evolution and\n'
             'relative contributions of gauge-theoretic components during training.',
             ha='center', fontsize=9, style='italic')
    
    # Save
    output_path = output_dir / f'fig2_free_energy_components.{FIGURE_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


# ============================================================================
# FIGURE 3: NATURAL GRADIENT CONVERGENCE
# ============================================================================

def create_convergence_figure(df: pd.DataFrame, colors: Dict, output_dir: Path):
    """Analyze convergence properties and gradient dynamics."""
    
    print("  Creating convergence analysis figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Subplot 1: Gradient Norms
    ax = axes[0, 0]
    ax.semilogy(df['step'], smooth_curve(df['grad_norm_total']), 
                color=colors['grad'], linewidth=1.5, alpha=0.8, label='Total')
    ax.semilogy(df['step'], smooth_curve(df['grad_norm_mu']),
                color=colors['beta'], linewidth=1.5, alpha=0.8, label=r'$\mu$ (Embeddings)')
    ax.semilogy(df['step'], smooth_curve(df['grad_norm_ffn']),
                color=colors['alpha'], linewidth=1.5, alpha=0.8, label='FFN')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm (log scale)')
    ax.set_title('(a) Natural Gradient Norms', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Learning Rate Schedules
    ax = axes[0, 1]
    ax.plot(df['step'], df['mu_lr'], color=colors['beta'],
            linewidth=1.5, alpha=0.8, label=r'$\mu$ LR')
    ax.plot(df['step'], df['sigma_lr'], color=colors['alpha'],
            linewidth=1.5, alpha=0.8, label=r'$\sigma$ LR')
    ax.plot(df['step'], df['phi_lr'], color=colors['gamma'],
            linewidth=1.5, alpha=0.8, label=r'$\phi$ LR')
    ax.plot(df['step'], df['ffn_lr'], color=colors['lr'],
            linewidth=1.5, alpha=0.8, label='FFN LR')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('(b) Adaptive Learning Rates', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Convergence Rate Analysis
    ax = axes[1, 0]
    
    # Compute convergence rate (change in loss per step)
    loss_diff = np.diff(df['train_loss_total'].values)
    loss_diff_smooth = smooth_curve(np.abs(loss_diff))
    
    ax.semilogy(df['step'].values[1:], loss_diff_smooth,
                color=colors['train'], linewidth=1.5, alpha=0.8)
    
    # Add exponential fit
    valid_idx = loss_diff_smooth > 0
    if np.sum(valid_idx) > 10:
        steps_fit = df['step'].values[1:][valid_idx]
        loss_fit = loss_diff_smooth[valid_idx]
        
        # Fit exponential decay
        z = np.polyfit(steps_fit, np.log(loss_fit + 1e-10), 1)
        p = np.poly1d(z)
        ax.plot(steps_fit, np.exp(p(steps_fit)), '--',
                color='red', linewidth=1, alpha=0.7, label=f'Exp. fit (rate={-z[0]:.3e})')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('|∆Loss| (log scale)')
    ax.set_title('(c) Convergence Rate', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Efficiency Metrics
    ax = axes[1, 1]
    
    # Throughput
    ax2 = ax.twinx()
    
    # Loss reduction per second
    time_cumsum = np.cumsum(df['step_time'].values)
    loss_reduction = df['train_loss_total'].values[0] - df['train_loss_total'].values
    efficiency = loss_reduction / (time_cumsum + 1e-8)
    
    ax.plot(df['step'], smooth_curve(efficiency), color=colors['grad'],
            linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Reduction / Second', color=colors['grad'])
    ax.tick_params(axis='y', labelcolor=colors['grad'])
    
    # Tokens per second on secondary axis
    ax2.plot(df['step'], smooth_curve(df['tokens_per_sec']), 
             color=colors['lr'], linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Tokens / Second', color=colors['lr'])
    ax2.tick_params(axis='y', labelcolor=colors['lr'])
    
    ax.set_title('(d) Training Efficiency', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add figure caption
    fig.text(0.5, -0.02,
             'Figure 3: Convergence analysis showing natural gradient dynamics, adaptive learning\n'
             'rates, and computational efficiency of the gauge-theoretic optimization.',
             ha='center', fontsize=9, style='italic')
    
    # Save
    output_path = output_dir / f'fig3_convergence_analysis.{FIGURE_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


# ============================================================================
# FIGURE 4: ATTENTION MECHANISM DYNAMICS
# ============================================================================

def create_attention_figure(df: pd.DataFrame, colors: Dict, output_dir: Path):
    """Visualize attention mechanism through KL divergence dynamics."""
    
    print("  Creating attention dynamics figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Subplot 1: KL Divergence Evolution
    ax = axes[0, 0]
    kl_mean = smooth_curve(df['kl_mean'].values)
    ax.plot(df['step'], kl_mean, color=colors['attention'],
            linewidth=1.5, alpha=0.8)
    
    # Add confidence band if std available
    if 'kl_std' in df.columns and df['kl_std'].sum() > 0:
        kl_std = df['kl_std'].values
        ax.fill_between(df['step'], kl_mean - kl_std, kl_mean + kl_std,
                       color=colors['attention'], alpha=0.2)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('(a) Belief Transport KL[q||Ω[q]]', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Attention Weight Distribution
    ax = axes[0, 1]
    beta_mean = smooth_curve(df['beta_mean'].values)
    ax.plot(df['step'], beta_mean, color=colors['beta'],
            linewidth=1.5, alpha=0.8)
    
    # Add horizontal line at softmax baseline
    ax.axhline(y=0.03125, color='gray', linestyle='--', alpha=0.5,
              label='Uniform (1/32)')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel(r'Mean $\beta_{ij}$')
    ax.set_title('(b) Attention Weight Evolution', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Phase Portrait
    ax = axes[1, 0]
    
    # Create phase plot of KL vs beta
    scatter = ax.scatter(df['kl_mean'].values[::5], df['beta_mean'].values[::5],
                        c=df['step'].values[::5], cmap='plasma',
                        s=10, alpha=0.6)
    
    # Add trajectory line
    ax.plot(df['kl_mean'].values, df['beta_mean'].values,
           color='gray', linewidth=0.5, alpha=0.3)
    
    # Mark start and end
    ax.scatter(df['kl_mean'].iloc[0], df['beta_mean'].iloc[0],
              color='green', s=100, marker='o', edgecolors='black',
              linewidth=1, label='Start', zorder=5)
    ax.scatter(df['kl_mean'].iloc[-1], df['beta_mean'].iloc[-1],
              color='red', s=100, marker='s', edgecolors='black',
              linewidth=1, label='End', zorder=5)
    
    ax.set_xlabel('KL Divergence')
    ax.set_ylabel(r'Mean $\beta_{ij}$')
    ax.set_title('(c) Attention Phase Space', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Step', rotation=270, labelpad=15)
    
    # Subplot 4: Information Flow
    ax = axes[1, 1]
    
    # Compute information metrics
    entropy_reduction = -df['train_loss_ce'].values
    belief_alignment = -df['train_loss_belief_align'].values
    
    # Normalize for visualization
    entropy_norm = (entropy_reduction - entropy_reduction.min()) / (entropy_reduction.max() - entropy_reduction.min() + 1e-8)
    belief_norm = (belief_alignment - belief_alignment.min()) / (belief_alignment.max() - belief_alignment.min() + 1e-8)
    
    ax.plot(df['step'], smooth_curve(entropy_norm), color=colors['train'],
            linewidth=1.5, alpha=0.8, label='Entropy Reduction')
    ax.plot(df['step'], smooth_curve(belief_norm), color=colors['beta'],
            linewidth=1.5, alpha=0.8, label='Belief Alignment')
    
    # Add correlation annotation
    corr = np.corrcoef(entropy_norm, belief_norm)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
           transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Normalized Information')
    ax.set_title('(d) Information Geometry', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add figure caption
    fig.text(0.5, -0.02,
             'Figure 4: Attention mechanism dynamics showing KL divergence-based transport,\n'
             'weight evolution, and information-geometric properties of belief alignment.',
             ha='center', fontsize=9, style='italic')
    
    # Save
    output_path = output_dir / f'fig4_attention_dynamics.{FIGURE_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


# ============================================================================
# FIGURE 5: ABLATION STUDY COMPARISON
# ============================================================================

def create_ablation_figure(metrics_dict: Dict[str, pd.DataFrame], colors: Dict, output_dir: Path):
    """Compare different FFN modes (learned vs variational)."""
    
    print("  Creating ablation comparison figure...")
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Define style for each mode
    mode_styles = {
        'learned': {'color': '#2E86AB', 'linestyle': '-', 'label': 'Learned (baseline)'},
        'variational_approx': {'color': '#F18F01', 'linestyle': '--', 'label': 'Variational (1st order)'},
        'variational_full': {'color': '#C73E1D', 'linestyle': '-.', 'label': 'Variational (full)'},
        'variational_gradient_engine': {'color': '#6B4C9D', 'linestyle': ':', 'label': 'Gradient Engine'}
    }
    
    # Subplot 1: Training Loss Comparison
    ax = axes[0, 0]
    for mode, df in metrics_dict.items():
        style = mode_styles.get(mode, {})
        ax.plot(df['step'], smooth_curve(df['train_loss_total']),
               color=style.get('color', 'gray'),
               linestyle=style.get('linestyle', '-'),
               linewidth=1.5, alpha=0.8,
               label=style.get('label', mode))
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('(a) Training Loss', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Validation Perplexity
    ax = axes[0, 1]
    for mode, df in metrics_dict.items():
        style = mode_styles.get(mode, {})
        val_mask = df['val_ppl'].notna() & (df['val_ppl'] > 0)
        if val_mask.any():
            ax.semilogy(df.loc[val_mask, 'step'], df.loc[val_mask, 'val_ppl'],
                       color=style.get('color', 'gray'),
                       marker='o', markersize=4,
                       linestyle=style.get('linestyle', '-'),
                       linewidth=1.5, alpha=0.8,
                       label=style.get('label', mode))
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Perplexity (log)')
    ax.set_title('(b) Perplexity Comparison', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Convergence Speed
    ax = axes[0, 2]
    for mode, df in metrics_dict.items():
        style = mode_styles.get(mode, {})
        # Compute relative improvement
        initial_loss = df['train_loss_total'].iloc[0]
        relative_loss = (df['train_loss_total'] / initial_loss) * 100
        ax.plot(df['step'], smooth_curve(relative_loss),
               color=style.get('color', 'gray'),
               linestyle=style.get('linestyle', '-'),
               linewidth=1.5, alpha=0.8,
               label=style.get('label', mode))
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Relative Loss (%)')
    ax.set_title('(c) Convergence Speed', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Free Energy Components Comparison
    ax = axes[1, 0]
    
    # Bar plot of final component values
    modes = []
    alpha_vals = []
    beta_vals = []
    gamma_vals = []
    
    for mode, df in metrics_dict.items():
        modes.append(mode_styles.get(mode, {}).get('label', mode))
        alpha_vals.append(df['train_loss_self_consistency'].iloc[-1])
        beta_vals.append(df['train_loss_belief_align'].iloc[-1])
        gamma_vals.append(df['train_loss_model_align'].iloc[-1])
    
    x = np.arange(len(modes))
    width = 0.25
    
    ax.bar(x - width, alpha_vals, width, label=r'$\alpha$', color=colors['alpha'], alpha=0.8)
    ax.bar(x, beta_vals, width, label=r'$\beta$', color=colors['beta'], alpha=0.8)
    ax.bar(x + width, gamma_vals, width, label=r'$\gamma$', color=colors['gamma'], alpha=0.8)
    
    ax.set_ylabel('Final Component Value')
    ax.set_title('(d) Final Free Energy Components', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 5: Gradient Norm Comparison
    ax = axes[1, 1]
    for mode, df in metrics_dict.items():
        style = mode_styles.get(mode, {})
        ax.semilogy(df['step'], smooth_curve(df['grad_norm_total']),
                   color=style.get('color', 'gray'),
                   linestyle=style.get('linestyle', '-'),
                   linewidth=1.5, alpha=0.8,
                   label=style.get('label', mode))
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm (log)')
    ax.set_title('(e) Gradient Dynamics', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Subplot 6: Final Performance Table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    # Create performance table
    table_data = []
    headers = ['Mode', 'Val PPL', 'Val BPC', 'Time (s)']
    
    for mode, df in metrics_dict.items():
        val_mask = df['val_ppl'].notna() & (df['val_ppl'] > 0)
        if val_mask.any():
            final_ppl = df.loc[val_mask, 'val_ppl'].iloc[-1]
            final_bpc = df.loc[val_mask, 'val_bpc'].iloc[-1]
        else:
            final_ppl = np.nan
            final_bpc = np.nan
        
        total_time = df['step_time'].sum()
        
        mode_name = mode_styles.get(mode, {}).get('label', mode).replace(' (baseline)', '')
        table_data.append([
            mode_name[:20],  # Truncate long names
            f'{final_ppl:.1f}' if not np.isnan(final_ppl) else 'N/A',
            f'{final_bpc:.3f}' if not np.isnan(final_bpc) else 'N/A',
            f'{total_time:.0f}'
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.4, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    ax.set_title('(f) Performance Summary', fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Add figure caption
    fig.text(0.5, -0.02,
             'Figure 5: Ablation study comparing learned MLP baseline with variational active\n'
             'inference implementations, demonstrating comparable performance with explicit inference.',
             ha='center', fontsize=9, style='italic')
    
    # Save
    output_path = output_dir / f'fig5_ablation_comparison.{FIGURE_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def generate_statistics_report(df: pd.DataFrame, output_dir: Path):
    """Generate detailed statistical analysis report."""
    
    print("  Generating statistics report...")
    
    report = []
    report.append("="*70)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("="*70)
    
    # Training Summary
    report.append("\n1. TRAINING SUMMARY")
    report.append("-"*40)
    report.append(f"Total Steps: {len(df)}")
    report.append(f"Total Time: {df['step_time'].sum()/3600:.2f} hours")
    report.append(f"Avg Step Time: {df['step_time'].mean():.2f} seconds")
    report.append(f"Avg Throughput: {df['tokens_per_sec'].mean():.0f} tokens/sec")
    
    # Loss Statistics
    report.append("\n2. LOSS STATISTICS")
    report.append("-"*40)
    
    initial_loss = df['train_loss_total'].iloc[0]
    final_loss = df['train_loss_total'].iloc[-1]
    best_loss = df['train_loss_total'].min()
    
    report.append(f"Initial Loss: {initial_loss:.4f}")
    report.append(f"Final Loss: {final_loss:.4f}")
    report.append(f"Best Loss: {best_loss:.4f}")
    report.append(f"Improvement: {(1 - final_loss/initial_loss)*100:.1f}%")
    
    # Validation Performance
    val_mask = df['val_ppl'].notna() & (df['val_ppl'] > 0)
    if val_mask.any():
        report.append("\n3. VALIDATION PERFORMANCE")
        report.append("-"*40)
        
        best_val_ppl = df.loc[val_mask, 'val_ppl'].min()
        final_val_ppl = df.loc[val_mask, 'val_ppl'].iloc[-1]
        best_val_bpc = df.loc[val_mask, 'val_bpc'].min()
        
        report.append(f"Best Val PPL: {best_val_ppl:.2f}")
        report.append(f"Final Val PPL: {final_val_ppl:.2f}")
        report.append(f"Best Val BPC: {best_val_bpc:.3f}")
        
        # vs random baseline (assuming vocab size ~256)
        random_ppl = 256
        improvement = random_ppl / best_val_ppl
        report.append(f"\nVs Random Baseline:")
        report.append(f"  Random PPL: {random_ppl}")
        report.append(f"  Improvement: {improvement:.1f}x")
    
    # Component Analysis
    report.append("\n4. FREE ENERGY COMPONENTS")
    report.append("-"*40)
    
    components = {
        'CE Loss': df['train_loss_ce'],
        'α (Self-Consistency)': df['train_loss_self_consistency'],
        'β (Belief Align)': df['train_loss_belief_align'],
        'γ (Model Align)': df['train_loss_model_align']
    }
    
    for name, values in components.items():
        if values.sum() > 0:
            report.append(f"\n{name}:")
            report.append(f"  Initial: {values.iloc[0]:.4f}")
            report.append(f"  Final: {values.iloc[-1]:.4f}")
            report.append(f"  Mean: {values.mean():.4f}")
            report.append(f"  Std: {values.std():.4f}")
    
    # Convergence Analysis
    report.append("\n5. CONVERGENCE ANALYSIS")
    report.append("-"*40)
    
    # Compute convergence rate
    loss_log = np.log(df['train_loss_total'].values + 1e-8)
    steps = df['step'].values
    
    # Fit exponential decay to last half of training
    mid_point = len(steps) // 2
    z = np.polyfit(steps[mid_point:], loss_log[mid_point:], 1)
    convergence_rate = -z[0]
    
    report.append(f"Convergence Rate: {convergence_rate:.3e} (exponential)")
    
    # Steps to 90% improvement
    loss_90 = initial_loss * 0.1
    steps_90 = np.argmax(df['train_loss_total'].values < loss_90)
    if steps_90 > 0:
        report.append(f"Steps to 90% improvement: {steps_90}")
    
    # Gradient Statistics
    report.append("\n6. GRADIENT DYNAMICS")
    report.append("-"*40)
    
    report.append(f"Mean Total Gradient Norm: {df['grad_norm_total'].mean():.4f}")
    report.append(f"Max Total Gradient Norm: {df['grad_norm_total'].max():.4f}")
    report.append(f"Final Total Gradient Norm: {df['grad_norm_total'].iloc[-1]:.4f}")
    
    # Attention Statistics
    report.append("\n7. ATTENTION MECHANISM")
    report.append("-"*40)
    
    report.append(f"Mean KL Divergence: {df['kl_mean'].mean():.4f}")
    report.append(f"Final KL Divergence: {df['kl_mean'].iloc[-1]:.4f}")
    report.append(f"Mean Beta: {df['beta_mean'].mean():.6f}")
    report.append(f"Final Beta: {df['beta_mean'].iloc[-1]:.6f}")
    
    report.append("\n" + "="*70)
    
    # Save report (with UTF-8 encoding for Greek letters)
    report_path = output_dir / 'statistics_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"    ✓ Saved: {report_path}")
    
    return '\n'.join(report)


# ============================================================================
# MAIN EXECUTION  
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PUBLICATION FIGURE GENERATOR")
    print("="*70)
    
    # Setup
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}/")
    print(f"Figure format: {FIGURE_FORMAT}")
    print(f"DPI: {DPI}")
    print(f"Color scheme: {COLOR_SCHEME}")
    
    # Configure style
    print("\n🎨 Setting up publication style...")
    colors = setup_publication_style(USE_LATEX_FONTS, COLOR_SCHEME)
    
    # Determine input
    if METRICS_FILE:
        input_path = Path(METRICS_FILE)
        if not input_path.exists():
            print(f"❌ File not found: {input_path}")
            print("   Please check METRICS_FILE configuration at top of script")
            return
    else:
        print("\n🔍 Auto-detecting metrics file...")
        input_path = auto_detect_metrics()
        if input_path is None:
            print("❌ No metrics.csv files found")
            print("   Run training first or set METRICS_FILE at top of script")
            return
        print(f"   Found: {input_path}")
    
    # Single experiment figures
    if input_path.is_file():
        print(f"\n📊 Loading metrics from: {input_path}")
        df = load_metrics(input_path)
        
        print("\n📈 Generating figures...")
        
        if CREATE_TRAINING_DYNAMICS:
            create_training_dynamics_figure(df, colors, output_dir)
            
        if CREATE_FREE_ENERGY:
            create_free_energy_figure(df, colors, output_dir)
            
        if CREATE_CONVERGENCE:
            create_convergence_figure(df, colors, output_dir)
            
        if CREATE_ATTENTION:
            create_attention_figure(df, colors, output_dir)
        
        if CREATE_STATISTICS:
            stats_report = generate_statistics_report(df, output_dir)
            print("\n" + stats_report)
    
    # Ablation comparison if requested
    if CREATE_ABLATION and ABLATION_DIR:
        ablation_path = Path(ABLATION_DIR)
        if ablation_path.exists() and ablation_path.is_dir():
            print(f"\n📁 Loading experiments for ablation from: {ablation_path}")
            
            metrics_dict = {}
            for subdir in ablation_path.iterdir():
                if subdir.is_dir() and subdir.name.startswith('ffn_'):
                    metrics_file = subdir / 'metrics.csv'
                    if metrics_file.exists():
                        mode = subdir.name.replace('ffn_', '')
                        print(f"  ✓ Found {mode}: {metrics_file}")
                        metrics_dict[mode] = load_metrics(metrics_file)
            
            if len(metrics_dict) > 1:
                print("\n📈 Generating ablation comparison figure...")
                create_ablation_figure(metrics_dict, colors, output_dir)
            else:
                print("  ⚠️  Need at least 2 experiments for ablation comparison")
    
    print(f"\n✅ All figures saved to: {output_dir}/")
    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print("\nFigures ready for publication submission!")
    print("Remember to check figure guidelines for your target venue:")
    print("  - JMLR: Prefer vector formats (PDF/EPS)")
    print("  - NeurIPS: Max 8 pages + references")
    print("  - Nature MI: Extended Data figures allowed")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()