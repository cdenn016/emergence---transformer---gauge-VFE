# -*- coding: utf-8 -*-
"""
ENHANCED Analysis / Visualization Suite for Multi-Agent Runs
============================================================

Comprehensive analysis toolkit with all requested features:
- Separate energy component plots
- Gauge field phi(c) visualization
- Covariance field Sigma(c) analysis
- Phi norm and gradient analysis
- KL divergence spatial fields
- Complete validation reporting

Usage
-----
    python analysis_suite.py --run-dir results/playground

Author: Chris
Date: November 2025
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt



# =============================================================================
# Helpers: Loading
# =============================================================================


def load_history(run_dir: Path):
    """Load training history from pkl (preferred) or npz."""
    pkl_path = run_dir / "history.pkl"
    npz_path = run_dir / "history.npz"

    # PREFER pkl because it has mu_tracker!
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            history = pickle.load(f)
        print(f"✓ Loaded history from {pkl_path}")
        
        # Check if it has mu_tracker
        mu_tracker = get_mu_tracker(history)
        if mu_tracker is not None:
            print(f"  ✓ Mu tracking data available: {len(mu_tracker.steps)} steps")
        else:
            print("  ⚠️  No mu tracking data in history")
        
        return history
    
    # Fallback to npz (but this won't have mu_tracker)
    if npz_path.exists():
        data = np.load(npz_path)
        history = {k: data[k] for k in data.files}
        print(f"✓ Loaded history from {npz_path}")
        print("  ⚠️  NPZ format doesn't include mu_tracker - use PKL for full data")
        return history

    print("⚠️  No history.(npz|pkl) found in run directory.")
    return None


def get_mu_tracker(history):
    """
    Extract mu_tracker from history regardless of format (dict or object).

    Args:
        history: Either a dict with 'mu_tracker' key or object with mu_tracker attribute

    Returns:
        MuCenterTracking instance or None
    """
    if history is None:
        return None

    # Dict format (hierarchical training)
    if isinstance(history, dict):
        return history.get('mu_tracker', None)

    # Object format (standard training)
    if hasattr(history, 'mu_tracker'):
        return history.mu_tracker

    return None


def normalize_history(history):
    """
    Convert TrainingHistory object to dict format for plotting.

    This allows plot functions to work with both pkl and npz formats.
    """
    if history is None:
        return None

    # If it's already a dict, return as-is
    if isinstance(history, dict):
        return history

    # If it's a TrainingHistory object, convert to dict
    if hasattr(history, 'steps'):
        hist_dict = {
            "step": history.steps if hasattr(history, 'steps') else [],
            "total": history.total_energy if hasattr(history, 'total_energy') else [],
            "self": history.self_energy if hasattr(history, 'self_energy') else [],
            "belief_align": history.belief_align if hasattr(history, 'belief_align') else [],
            "prior_align": history.prior_align if hasattr(history, 'prior_align') else [],
            "observations": history.observations if hasattr(history, 'observations') else [],
            "grad_norm_mu_q": history.grad_norm_mu_q if hasattr(history, 'grad_norm_mu_q') else [],
            "grad_norm_Sigma_q": history.grad_norm_Sigma_q if hasattr(history, 'grad_norm_Sigma_q') else [],
            "grad_norm_phi": history.grad_norm_phi if hasattr(history, 'grad_norm_phi') else [],
        }
        return hist_dict

    return None





def load_system(run_dir: Path):
    """Load final MultiAgentSystem from pickle."""
    state_path = run_dir / "final_state.pkl"
    if not state_path.exists():
        print("⚠ No final_state.pkl found in run directory.")
        return None

    with open(state_path, "rb") as f:
        system = pickle.load(f)
    print(f"✓ Loaded final system from {state_path}")
    return system


# =============================================================================
# Helpers: Geometry
# =============================================================================

def _get_spatial_shape_from_system(system):
    """Infer spatial shape from first agent's support or base_manifold."""
    a0 = system.agents[0]
    if hasattr(a0, "support") and a0.support is not None:
        shape = getattr(a0.support, "base_shape", None)
        if shape is None:
            shape = getattr(a0.support, "mask", np.array([])).shape
        return tuple(shape)
    if hasattr(a0, "base_manifold"):
        return tuple(a0.base_manifold.shape)
    if hasattr(a0, "support_shape"):
        return tuple(a0.support_shape)
    return ()


def _pick_reference_agent(system):
    """Pick a reference agent i that actually has neighbors."""
    for i in range(system.n_agents):
        neighbors = system.get_neighbors(i)
        if neighbors:
            return i
    return 0


# =============================================================================
# NEW: SEPARATE Energy Component Plots
# =============================================================================

def plot_energy_components_separate(history, out_dir: Path):
    """
    Plot each energy component in its own subplot for clarity.
    
    Creates a grid showing:
    - Total energy
    - Self energy  
    - Belief alignment
    - Prior alignment
    - Observations (if present)
    - Gauge smoothness (if present)
    """
    if history is None or "step" not in history or "total" not in history:
        print("⚠ History missing 'step' or 'total' — skipping energy plots.")
        return

    steps = history["step"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine grid layout
    components = ['total', 'self', 'belief_align', 'prior_align']
    if 'observations' in history and np.any(np.array(history['observations']) != 0):
        components.append('observations')
    if 'gauge_smooth' in history and np.any(np.array(history['gauge_smooth']) != 0):
        components.append('gauge_smooth')
    
    n_plots = len(components)
    ncols = 2
    nrows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Labels for pretty printing
    labels = {
        'total': 'Total Energy',
        'self': 'Self Energy (KL(q||p))',
        'belief_align': 'Belief Alignment',
        'prior_align': 'Prior Alignment',
        'observations': 'Observation Likelihood',
        'gauge_smooth': 'Gauge Smoothness'
    }
    
    for idx, key in enumerate(components):
        ax = axes[idx]
        if key in history and len(history[key]) == len(steps):
            data = history[key]
            ax.plot(steps, data, linewidth=2, color='C0' if key=='total' else 'C1')
            ax.set_xlabel("Step")
            ax.set_ylabel("Energy")
            ax.set_title(labels.get(key, key))
            ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    path = out_dir / "energy_components_separate.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")
    
    # Also save combined plot for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(steps, history["total"], label="Total", linewidth=2.5, color='black')
    
    for key, label in [
        ("self", "Self"),
        ("belief_align", "Belief"),
        ("prior_align", "Prior"),
        ("observations", "Obs"),
        ("gauge_smooth", "Gauge")
    ]:
        if key in history and len(history[key]) == len(steps):
            plt.plot(steps, history[key], "--", label=label, alpha=0.7)
    
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy Evolution (Combined)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "energy_components_combined.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


# =============================================================================
# NEW: Gauge Field phi(c) Visualization
# =============================================================================

def plot_phi_fields(system, out_dir: Path):
    """
    Visualize gauge fields phi(c) for all agents.
    
    Shows:
    - Individual phi components (phi_x, phi_y, phi_z)
    - Phi norm ||phi(c)||
    - Comparison across agents
    """
    if system is None:
        return
    
    shape = _get_spatial_shape_from_system(system)
    ndim = len(shape)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if ndim == 0:
        # 0D: bar plot of phi magnitudes
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Individual components
        ax = axes[0]
        agents = range(len(system.agents))
        width = 0.25
        x = np.arange(len(agents))
        
        for comp_idx, comp_name in enumerate(['φ_x', 'φ_y', 'φ_z']):
            values = [system.agents[i].phi[comp_idx] for i in agents]
            ax.bar(x + comp_idx*width, values, width, label=comp_name, alpha=0.7)
        
        ax.set_xlabel('Agent')
        ax.set_ylabel('φ component')
        ax.set_title('Gauge Field Components (0D)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(agents)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Norms
        ax = axes[1]
        norms = [np.linalg.norm(system.agents[i].phi) for i in agents]
        ax.bar(agents, norms, color='purple', alpha=0.6)
        ax.axhline(np.pi, color='red', linestyle='--', label='π (branch cut)')
        ax.set_xlabel('Agent')
        ax.set_ylabel('||φ||')
        ax.set_title('Gauge Field Norms (0D)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        path = out_dir / "phi_fields_0d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
        return
    
    if ndim == 1:
        # 1D: line plots
        x = np.arange(shape[0])
                
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Component plots
        for comp_idx, (ax, comp_name) in enumerate(zip(axes.flat[:3], ['φ_x', 'φ_y', 'φ_z'])):
            for i, agent in enumerate(system.agents):
                phi_comp = agent.phi[..., comp_idx]
                mask = agent.support.mask_continuous if hasattr(agent, 'support') else None
                if mask is not None:
                    ax.plot(x, phi_comp, label=f'Agent {i}', alpha=0.7)
                    ax.fill_between(x, phi_comp.min(), phi_comp.max(), 
                                   where=mask>0.5, alpha=0.1)
                else:
                    ax.plot(x, phi_comp, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel(comp_name)
            ax.set_title(f'{comp_name}(c) Fields')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Norm plot
        ax = axes.flat[3]
        for i, agent in enumerate(system.agents):
            phi_norm = np.linalg.norm(agent.phi, axis=-1)
            ax.plot(x, phi_norm, label=f'Agent {i}', linewidth=2, alpha=0.7)
        ax.axhline(np.pi, color='red', linestyle='--', linewidth=2, label='π (branch cut)')
        ax.set_xlabel('Position')
        ax.set_ylabel('||φ(c)||')
        ax.set_title('Gauge Field Norms')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        path = out_dir / "phi_fields_1d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
    
    elif ndim == 2:
        # 2D: heatmaps
        H, W = shape
                
        # Per-agent component plots
        for i, agent in enumerate(system.agents):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Three components
            for comp_idx, (ax, comp_name) in enumerate(zip(axes.flat[:3], ['φ_x', 'φ_y', 'φ_z'])):
                phi_comp = agent.phi[..., comp_idx].reshape(H, W)
                im = ax.imshow(phi_comp, origin='lower', cmap='RdBu_r')
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f'{comp_name}(c) - Agent {i}')
            
            # Norm
            ax = axes.flat[3]
            phi_norm = np.linalg.norm(agent.phi, axis=-1).reshape(H, W)
            im = ax.imshow(phi_norm, origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax, fraction=0.046, label='||φ||')
          #  ax.contour(phi_norm, levels=[np.pi * 0.9], colors='red', linewidths=2)
            ax.set_title(f'||φ(c)|| - Agent {i}')
            
            plt.tight_layout()
            path = out_dir / f"phi_fields_2d_agent{i}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"✓ Saved {path}")

from mpl_toolkits.mplot3d.art3d import Line3DCollection
# =============================================================================
# NEW: Covariance Field Sigma(c) Analysis
# =============================================================================

def plot_sigma_fields(system, out_dir: Path):
    """
    Visualize covariance fields Sigma(c).
    
    Shows:
    - Eigenvalue spectrum across space
    - Determinant (generalized variance)
    - Trace (total variance)
    - Condition number (numerical stability)
    """
    if system is None:
        return
    
    shape = _get_spatial_shape_from_system(system)
    ndim = len(shape)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if ndim == 0:
        # 0D: eigenvalue bars
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax_idx, (ax, which) in enumerate(zip(axes, ['q', 'p'])):
            agents = range(len(system.agents))
            Sigma_key = 'Sigma_q' if which == 'q' else 'Sigma_p'
            
            eigs_all = []
            for i in agents:
                Sigma = getattr(system.agents[i], Sigma_key)
                eigs = np.linalg.eigvalsh(Sigma)
                eigs_all.append(eigs)
            
            eigs_all = np.array(eigs_all)  # (n_agents, K)
            K = eigs_all.shape[1]
            
            x = np.arange(len(agents))
            width = 0.8 / K
            
            for k in range(K):
                ax.bar(x + k*width, eigs_all[:, k], width, label=f'λ_{k}', alpha=0.7)
            
            ax.set_xlabel('Agent')
            ax.set_ylabel('Eigenvalue')
            ax.set_title(f'Σ_{which} Eigenvalues (0D)')
            ax.set_xticks(x + width*(K-1)/2)
            ax.set_xticklabels(agents)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        path = out_dir / "sigma_fields_0d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
        return
    
    if ndim == 1:
        x = np.arange(shape[0])
        
        for which in ['q', 'p']:
            Sigma_key = 'Sigma_q' if which == 'q' else 'Sigma_p'
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Min eigenvalue (stability)
            ax = axes[0, 0]
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                eigs = np.linalg.eigvalsh(Sigma)  # (*S, K)
                min_eig = eigs[..., 0]
                ax.semilogy(x, min_eig, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel('Min eigenvalue')
            ax.set_title(f'Σ_{which}: Minimum Eigenvalue (Stability)')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Max eigenvalue (scale)
            ax = axes[0, 1]
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                eigs = np.linalg.eigvalsh(Sigma)
                max_eig = eigs[..., -1]
                ax.semilogy(x, max_eig, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel('Max eigenvalue')
            ax.set_title(f'Σ_{which}: Maximum Eigenvalue (Scale)')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Determinant (volume)
            ax = axes[1, 0]
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                det_Sigma = np.linalg.det(Sigma)
                det_Sigma = np.maximum(det_Sigma, 1e-20)  # Avoid log(0)
                ax.semilogy(x, det_Sigma, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel('det(Σ)')
            ax.set_title(f'Σ_{which}: Determinant (Volume)')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Condition number (stability)
            ax = axes[1, 1]
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                eigs = np.linalg.eigvalsh(Sigma)
                cond = eigs[..., -1] / np.maximum(eigs[..., 0], 1e-20)
                ax.semilogy(x, cond, label=f'Agent {i}', alpha=0.7)
            ax.set_xlabel('Position')
            ax.set_ylabel('κ(Σ)')
            ax.set_title(f'Σ_{which}: Condition Number')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            path = out_dir / f"sigma_fields_1d_{which}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"✓ Saved {path}")
    
    elif ndim == 2:
        H, W = shape
        
        for which in ['q', 'p']:
            Sigma_key = 'Sigma_q' if which == 'q' else 'Sigma_p'
            
            for i, agent in enumerate(system.agents):
                Sigma = getattr(agent, Sigma_key)
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Min eigenvalue
                ax = axes[0, 0]
                eigs = np.linalg.eigvalsh(Sigma).reshape(H, W, -1)
                min_eig = eigs[..., 0]
                im = ax.imshow(np.log10(min_eig), origin='lower', cmap='viridis')
                plt.colorbar(im, ax=ax, label='log10(λ_min)')
                ax.set_title(f'Min Eigenvalue - Agent {i}')
                
                # Max eigenvalue
                ax = axes[0, 1]
                max_eig = eigs[..., -1]
                im = ax.imshow(np.log10(max_eig), origin='lower', cmap='plasma')
                plt.colorbar(im, ax=ax, label='log10(λ_max)')
                ax.set_title(f'Max Eigenvalue - Agent {i}')
                
                # Determinant
                ax = axes[1, 0]
                det_Sigma = np.linalg.det(Sigma).reshape(H, W)
                det_Sigma = np.maximum(det_Sigma, 1e-20)
                im = ax.imshow(np.log10(det_Sigma), origin='lower', cmap='coolwarm')
                plt.colorbar(im, ax=ax, label='log10(det)')
                ax.set_title(f'Determinant - Agent {i}')
                
                # Condition number
                ax = axes[1, 1]
                cond = max_eig / np.maximum(min_eig, 1e-20)
                im = ax.imshow(np.log10(cond), origin='lower', cmap='hot')
                plt.colorbar(im, ax=ax, label='log10(κ)')
                ax.set_title(f'Condition Number - Agent {i}')
                
                plt.tight_layout()
                path = out_dir / f"sigma_fields_2d_{which}_agent{i}.png"
                plt.savefig(path, dpi=150)
                plt.close()
                print(f"✓ Saved {path}")


# =============================================================================
# Existing: Support masks
# =============================================================================

def plot_supports(system, out_dir: Path):
    """Visualize agent supports in 1D or 2D."""
    if system is None:
        return

    shape = _get_spatial_shape_from_system(system)
    ndim = len(shape)

    if ndim == 0:
        print("Base manifold is 0D — supports are trivial. Skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    for agent in system.agents:
        if hasattr(agent, "support") and agent.support is not None:
            masks.append(agent.support.mask.astype(float))
        else:
            masks.append(np.ones(shape, dtype=float))

    masks = np.stack(masks, axis=0)

    if ndim == 1:
        plt.figure(figsize=(8, 4))
        plt.imshow(masks, aspect="auto", origin="lower")
        plt.colorbar(label="support (1 = active)")
        plt.xlabel("position")
        plt.ylabel("agent")
        plt.yticks(range(system.n_agents))
        plt.title("Agent supports (1D)")
        plt.tight_layout()
        path = out_dir / "supports_1d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

    elif ndim == 2:
        H, W = shape

        active_count = masks.sum(axis=0)
        plt.figure(figsize=(5, 4))
        im = plt.imshow(active_count, origin="lower")
        plt.colorbar(im, label="# active agents")
        plt.title("Number of active agents per location")
        plt.tight_layout()
        path = out_dir / "supports_2d_counts.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

        n_agents = system.n_agents
        cols = min(3, n_agents)
        rows = int(np.ceil(n_agents / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))
        for i in range(n_agents):
            ax = plt.subplot(rows, cols, i + 1)
            im = ax.imshow(masks[i], origin="lower")
            ax.set_title(f"Agent {i} support")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        path = out_dir / "supports_2d_agents.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")


# =============================================================================
# Existing: Overlap matrix
# =============================================================================

def compute_overlap_matrix(system):
    """Compute overlap fractions."""
    n = system.n_agents
    O = np.zeros((n, n), dtype=float)

    if not hasattr(system, "get_overlap_fraction"):
        return None

    for i in range(n):
        for j in range(n):
            if i == j:
                O[i, j] = 1.0
            else:
                try:
                    O[i, j] = system.get_overlap_fraction(i, j)
                except Exception:
                    O[i, j] = 0.0
    return O


def plot_overlap_matrix(system, out_dir: Path):
    """Plot heatmap of pairwise overlap fractions."""
    if system is None:
        return

    O = compute_overlap_matrix(system)
    if O is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    im = plt.imshow(O, vmin=0.0, vmax=1.0, origin="lower")
    plt.colorbar(im, label="|C_i ∩ C_j| / |C_i|")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title("Agent overlap matrix")
    plt.xticks(range(system.n_agents))
    plt.yticks(range(system.n_agents))
    plt.tight_layout()
    path = out_dir / "overlap_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


# =============================================================================
# Existing: mu fields
# =============================================================================

def plot_mu_fields(system, out_dir: Path):
    """Visualize μ_q fields for agents."""
    if system is None:
        return

    shape = _get_spatial_shape_from_system(system)
    ndim = len(shape)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_fields = []
    for agent in system.agents:
        mu = getattr(agent, "mu_q", None)
        if mu is None:
            mu_fields.append(None)
        elif mu.ndim == 1:
            mu_fields.append(mu[0:1])
        else:
            mu_fields.append(mu[..., 0])

    if ndim == 0:
        values = [mf[0] if mf is not None and mf.size > 0 else np.nan
                  for mf in mu_fields]
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(values)), values)
        plt.xlabel("Agent")
        plt.ylabel("μ_q (first dim)")
        plt.title("0D μ_q per agent")
        plt.tight_layout()
        path = out_dir / "mu_fields_0d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
        return

    if ndim == 1:
        x = np.arange(shape[0])
        plt.figure(figsize=(8, 4))
        for i, mf in enumerate(mu_fields):
            if mf is None:
                continue
            plt.plot(x, mf, label=f"Agent {i}", alpha=0.8)
        plt.xlabel("Position")
        plt.ylabel("μ_q[..., 0]")
        plt.title("Belief mean μ_q (first latent dim, 1D)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = out_dir / "mu_fields_1d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")
        return

    if ndim == 2:
        H, W = shape
        n_agents = system.n_agents
        cols = min(3, n_agents)
        rows = int(np.ceil(n_agents / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))
        for i, mf in enumerate(mu_fields):
            if mf is None:
                continue
            ax = plt.subplot(rows, cols, i + 1)
            im = ax.imshow(mf.reshape(H, W), origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"μ_q[...,0] – Agent {i}")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        path = out_dir / "mu_fields_2d_agents.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")


# =============================================================================
# Existing: Softmax weights
# =============================================================================

def plot_softmax_weights(system, out_dir: Path, agent_idx: int = None, mode: str = "belief"):
    """Visualize spatial softmax weights β_ij(c) or γ_ij(c).

    For visualization, we mask by agent i's support χ_i so that weights
    are zero outside the agent's domain. This does NOT affect the physics
    or training, only the plots.
    """
    if system is None:
        return

    if agent_idx is None:
        agent_idx = _pick_reference_agent(system)

    shape = _get_spatial_shape_from_system(system)
    ndim = len(shape)

    neighbors = system.get_neighbors(agent_idx)
    if not neighbors:
        return

    # Compute raw β/γ fields (defined on full base grid)
    weights = system.compute_softmax_weights(agent_idx, mode=mode)
    if not weights:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Only keep neighbors for which we have weights
    js = sorted([j for j in neighbors if j in weights])
    if not js:
        return

    # Agent i's continuous support mask χ_i(c), shape (*S,)
    agent_i = system.agents[agent_idx]
    chi_i = np.asarray(agent_i.support.mask_continuous, dtype=float)

    if ndim == 1:
        # 1D base manifold
        x = np.arange(shape[0])

        plt.figure(figsize=(8, 4))
        for j in js:
            wj = np.asarray(weights[j], dtype=float)  # shape (S,)
            # Mask for visualization: zero outside agent i
            wj_vis = wj * chi_i
            plt.plot(x, wj_vis, label=f"j={j}", alpha=0.9)

        plt.xlabel("Position")
        plt.ylabel(f"{'β' if mode == 'belief' else 'γ'}_ij(x)")
        plt.title(f"{mode.capitalize()} softmax weights for agent i={agent_idx}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fname = f"{'beta' if mode == 'belief' else 'gamma'}_agent{agent_idx}_{mode}_1d.png"
        path = out_dir / fname
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

    elif ndim == 2:
        # 2D base manifold
        H, W = shape

        # Reshape χ_i to 2D
        chi_i_2d = chi_i.reshape(H, W)

        n_nb = len(js)
        cols = min(3, n_nb)
        rows = int(np.ceil(n_nb / cols))
        plt.figure(figsize=(4 * cols, 4 * rows))

        for k, j in enumerate(js):
            ax = plt.subplot(rows, cols, k + 1)

            field = np.asarray(weights[j], dtype=float).reshape(H, W)
            # Mask for visualization: zero outside agent i
            field_vis = field * chi_i_2d

            im = ax.imshow(field_vis, origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{'β' if mode == 'belief' else 'γ'}_{{{agent_idx},{j}}}(c)")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fname = f"{'beta' if mode == 'belief' else 'gamma'}_agent{agent_idx}_{mode}_2d_agents.png"
        path = out_dir / fname
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"✓ Saved {path}")

    else:
        print(f"⚠️ plot_softmax_weights: ndim={ndim} not supported yet.")

from pathlib import Path


# =============================================================================
# ADD THESE FUNCTIONS TO analysis_suite.py
# =============================================================================

def plot_mu_norm_trajectories(history, out_dir: Path):
    """
    Plot ||μ(center)|| over time for all agents.
    
    Key diagnostic:
    - Vacuum theory: All agents should have same norm (gauge orbit)
    - With observations: Norms diverge (symmetry breaking)
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        print("⚠️  No mu tracking data available")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = np.array(tracker.steps)
    norms = tracker.get_all_norms()  # (n_agents, n_steps)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Individual trajectories
    ax = axes[0]
    for i in range(norms.shape[0]):
        ax.plot(steps, norms[i], label=f'Agent {i}', alpha=0.7, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('||μ(center)||')
    ax.set_title('Belief Mean Norms at Agent Centers')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    # Right: Statistics
    ax = axes[1]
    mean_norm = np.mean(norms, axis=0)
    std_norm = np.std(norms, axis=0)
    
    ax.plot(steps, mean_norm, 'k-', linewidth=2.5, label='Mean')
    ax.fill_between(steps, mean_norm - std_norm, mean_norm + std_norm, 
                     alpha=0.3, color='gray', label='±1 std')
    ax.set_xlabel('Step')
    ax.set_ylabel('||μ||')
    ax.set_title('Norm Statistics Across Agents')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = out_dir / "mu_norm_trajectories.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {path}")


def plot_mu_component_trajectories(history, out_dir: Path, agent_idx: int = 0):
    """
    Plot individual μ components over time for one agent.
    
    Shows how the agent moves through latent space.
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = np.array(tracker.steps)
    components = tracker.get_component_array(agent_idx)  # (n_steps, K)
    K = components.shape[1]
    
    # Plot each component
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for k in range(K):
        ax.plot(steps, components[:, k], label=f'μ[{k}]', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('μ component value')
    ax.set_title(f'Belief Mean Components at Center - Agent {agent_idx}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    path = out_dir / f"mu_components_agent{agent_idx}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


def plot_norm_variance_evolution(history, out_dir: Path):
    """
    Plot variance of ||μ|| across agents over time.
    
    KEY SYMMETRY DIAGNOSTIC:
    - Var(||μ||) ≈ 0: Gauge symmetry preserved (vacuum theory)
    - Var(||μ||) > 0: Symmetry broken (observations active)
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = np.array(tracker.steps)
    var_norm = tracker.compute_norm_variance()
    mean_norm = tracker.compute_mean_norm()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Variance (symmetry measure)
    ax = axes[0]
    ax.semilogy(steps, var_norm, 'r-', linewidth=2.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Var(||μ||)')
    ax.set_title('Norm Variance Across Agents (Symmetry Measure)')
    ax.grid(alpha=0.3)
    
    # Add symmetry interpretation
    if len(var_norm) > 0:
        final_var = var_norm[-1]
        if final_var < 1e-3:
            sym_text = "Gauge symmetry preserved"
            color = 'green'
        elif final_var < 0.1:
            sym_text = "Weak symmetry breaking"
            color = 'orange'
        else:
            sym_text = "Strong symmetry breaking"
            color = 'red'
        
        ax.text(0.95, 0.95, sym_text, 
               transform=ax.transAxes,
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
               fontsize=12, fontweight='bold')
    
    # Bottom: Mean norm
    ax = axes[1]
    ax.plot(steps, mean_norm, 'b-', linewidth=2.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean ||μ||')
    ax.set_title('Average Norm Across Agents')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = out_dir / "mu_norm_variance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


def plot_mu_phase_space(history, out_dir: Path, dims: tuple = (0, 1)):
    """
    Plot μ trajectories in 2D phase space projection.
    
    Shows how agents explore the latent space manifold.
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dim_x, dim_y = dims
    n_agents = len(tracker.mu_components)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot trajectory for each agent
    for i in range(n_agents):
        components = tracker.get_component_array(i)  # (n_steps, K)
        
        if components.shape[1] <= max(dims):
            continue
        
        x = components[:, dim_x]
        y = components[:, dim_y]
        
        # Plot trajectory with color gradient
        ax.plot(x, y, alpha=0.6, linewidth=1.5, label=f'Agent {i}')
        
        # Mark start and end
        ax.scatter(x[0], y[0], marker='o', s=100, alpha=0.8, edgecolors='black')
        ax.scatter(x[-1], y[-1], marker='s', s=100, alpha=0.8, edgecolors='black')
    
    ax.set_xlabel(f'μ[{dim_x}]')
    ax.set_ylabel(f'μ[{dim_y}]')
    ax.set_title(f'Belief Mean Trajectories (dims {dim_x}, {dim_y})')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    path = out_dir / f"mu_phase_space_dims{dim_x}_{dim_y}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved {path}")


def plot_mu_summary_report(history, out_dir: Path):
    """
    Generate comprehensive summary report of μ tracking.
    
    Combines all key diagnostics in one multi-panel figure.
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = np.array(tracker.steps)
    norms = tracker.get_all_norms()
    var_norm = tracker.compute_norm_variance()
    mean_norm = tracker.compute_mean_norm()
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # (1) Norm trajectories
    ax = fig.add_subplot(gs[0, :])
    for i in range(norms.shape[0]):
        ax.plot(steps, norms[i], label=f'Agent {i}', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('||μ||')
    ax.set_title('Norm Trajectories')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    ax.grid(alpha=0.3)
    
    # (2) Variance (symmetry)
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(steps, var_norm, 'r-', linewidth=2.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Var(||μ||)')
    ax.set_title('Symmetry Measure')
    ax.grid(alpha=0.3)
    
    # (3) Mean norm
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, mean_norm, 'b-', linewidth=2.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean ||μ||')
    ax.set_title('Average Norm')
    ax.grid(alpha=0.3)
    
    # (4) Phase space (first 2 dims)
    ax = fig.add_subplot(gs[2, 0])
    for i in range(min(norms.shape[0], 8)):  # Max 8 agents
        components = tracker.get_component_array(i)
        if components.shape[1] >= 2:
            ax.plot(components[:, 0], components[:, 1], alpha=0.6, label=f'Agent {i}')
            ax.scatter(components[0, 0], components[0, 1], marker='o', s=50, alpha=0.8)
    ax.set_xlabel('μ[0]')
    ax.set_ylabel('μ[1]')
    ax.set_title('Phase Space (dims 0,1)')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # (5) Final statistics table
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    
    # Compute statistics
    final_var = var_norm[-1] if len(var_norm) > 0 else 0
    final_mean = mean_norm[-1] if len(mean_norm) > 0 else 0
       
    stats_text = f"""
    FINAL STATISTICS
    ================
    
    Mean ||μ||:     {final_mean:.4f}
    Var(||μ||):     {final_var:.4e}
    
    Symmetry Status:
    {_get_symmetry_status(final_var)}
    
    Norm Range:
    Min: {norms[:, -1].min():.4f}
    Max: {norms[:, -1].max():.4f}
    """
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('μ(center) Tracking Summary', fontsize=16, fontweight='bold')
    
    path = out_dir / "mu_tracking_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {path}")


def _get_symmetry_status(var_norm: float) -> str:
    """Classify symmetry breaking based on norm variance."""
    if var_norm < 1e-4:
        return "✓ Gauge symmetry preserved"
    elif var_norm < 1e-2:
        return "~ Weak symmetry breaking"
    else:
        return "✗ Strong symmetry breaking"



from pathlib import Path

from pathlib import Path

def plot_mu_gauge_orbit(history, out_dir: Path):
    r"""
    Visualize normalized $\mu$-trajectories as gauge orbits.

    This plot shows how each agent's latent mean vector $\mu_i(t)$ moves on
    the **unit circle** (for $K=2$) or **unit sphere** (for $K\geq 3$) after
    per-step normalization:

    \[
        \hat{\mu}_i(t)
        \;=\;
        \frac{\mu_i(t)}{\lVert \mu_i(t) \rVert_2}
        \quad\in\quad
        \mathbb{S}^{K-1}.
    \]
    """
    # -------------------------------------------------------------------------
    # Safety checks and early exits
    # -------------------------------------------------------------------------
    tracker = get_mu_tracker(history)
    if tracker is None:
        print("⚠️  No mu tracking data available — skipping gauge orbit plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Use first agent to infer latent dimension K
    components0 = tracker.get_component_array(0)  # (n_steps, K)
    if components0.size == 0:
        print("⚠️  Empty μ tracking data — skipping gauge orbit plot.")
        return

    K = components0.shape[1]

    import numpy as np
    import matplotlib.pyplot as plt

    if K < 2:
        print("⚠️  Latent dim K < 2 — no meaningful gauge orbit to plot.")
        return

    # Determine number of agents from tracker
    n_agents = (
        len(tracker.agent_indices)
        if hasattr(tracker, "agent_indices")
        else tracker.n_agents
    )

    # -------------------------------------------------------------------------
    # Case 1: 2D latent space → unit circle
    # -------------------------------------------------------------------------
    if K == 2:
        fig, ax = plt.subplots(figsize=(6, 6))

        # Reference unit circle
        theta = np.linspace(0.0, 2.0 * np.pi, 512)
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="Unit circle"
        )

        # Agent trajectories
        for i in range(n_agents):
            comps = tracker.get_component_array(i)  # (n_steps, 2)

            # Normalize onto unit circle
            norms = np.linalg.norm(comps, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            unit = comps / norms  # (n_steps, 2)

            # Continuous trajectory
            ax.plot(
                unit[:, 0],
                unit[:, 1],
                alpha=0.9,
                linewidth=1.5,
                label=f"Agent {i}"
            )

            # Start and end markers
            ax.scatter(
                unit[0, 0],
                unit[0, 1],
                s=40,
                marker="o",
                alpha=0.95
            )
            ax.scatter(
                unit[-1, 0],
                unit[-1, 1],
                s=40,
                marker="x",
                alpha=0.95
            )

        ax.set_xlabel(r"$\hat{\mu}_0$")
        ax.set_ylabel(r"$\hat{\mu}_1$")
        ax.set_title(r"Gauge Orbit on Unit Circle ($\mu$ normalized)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

        path = out_dir / "mu_gauge_orbit_circle.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✓ Saved {path}")



    # -------------------------------------------------------------------------
    # Case 2: K ≥ 3 → project to first 3 dims on unit sphere
    # -------------------------------------------------------------------------
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # ---------------------------------------------------------------------
        # 1. Camera / view direction
        # ---------------------------------------------------------------------
        elev = 25
        azim = -55
        ax.view_init(elev=elev, azim=azim)

        elev_rad = np.deg2rad(elev)
        azim_rad = np.deg2rad(azim)
        view_dir = np.array([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ], dtype=float)
        view_dir /= np.linalg.norm(view_dir)

        # ---------------------------------------------------------------------
        # 2. Opaque shaded unit sphere (no transparency → no color tint)
        # ---------------------------------------------------------------------
        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 60)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Simple shading via light source
        nx, ny, nz = x, y, z
        light_dir = np.array([1.0, -1.0, 1.5], dtype=float)
        light_dir /= np.linalg.norm(light_dir)

        shade = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
        shade = np.clip(shade, 0.0, 1.0)
        shade = (shade - shade.min()) / (shade.max() - shade.min() + 1e-12)
        shade = 0.25 + 0.75 * shade  # ambient + directional

        base_color = np.array([0.80, 0.83, 0.95], dtype=float)
        sphere_colors = base_color[None, None, :] * shade[..., None]

        ax.plot_surface(
            x, y, z,
            facecolors=sphere_colors,
            rstride=1, cstride=1,
            linewidth=0.3,
            edgecolor=(0.7, 0.7, 0.8),
            antialiased=True,
            alpha=0.5,      # OPAQUE: no tinting of lines
            shade=False,
        )

        # ---------------------------------------------------------------------
        # 2. Agent trajectories: per-agent color, explicit front/back segments
        # ---------------------------------------------------------------------
        base_cmap = plt.get_cmap("tab10")
        agent_colors = [base_cmap(i % base_cmap.N) for i in range(n_agents)]

        path_radius = 1.015  # slightly outside the unit sphere

        for i in range(n_agents):
            comps = tracker.get_component_array(i)  # (n_steps, K)

            norms = np.linalg.norm(comps, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            unit = comps / norms                     # on S^K-1

            # points we actually plot: just outside the sphere
            xyz = path_radius * unit[:, :3]          # (n_steps, 3)


            n_steps = xyz.shape[0]
            if n_steps < 2:
                continue

            segments = np.stack([xyz[:-1], xyz[1:]], axis=1)

            # Front vs back wrt camera
            front_point_mask = (xyz @ view_dir) > 0
            front_seg_mask = front_point_mask[:-1] & front_point_mask[1:]
            back_seg_mask = ~front_seg_mask

            color = agent_colors[i]

            # Back segments: faint, dashed
            if np.any(back_seg_mask):
                seg_back = segments[back_seg_mask]
                lc_back = Line3DCollection(
                    seg_back,
                    colors=[color],
                    linewidths=1.5,
                    alpha=0.65,
                    linestyles="solid",
                )
                ax.add_collection3d(lc_back)

            # Front segments: solid, fully on top of everything (no front wireframe)
            if np.any(front_seg_mask):
                seg_front = segments[front_seg_mask]
                lc_front = Line3DCollection(
                    seg_front,
                    colors=[color],
                    linewidths=1.5,
                    alpha=1.0,
                    linestyles="solid",
                )
                ax.add_collection3d(lc_front)

            # Markers
            ax.scatter(
                xyz[0, 0], xyz[0, 1], xyz[0, 2],
                s=75, marker="o", color=color, edgecolor="k",
                linewidths=0.8, depthshade=False,
                label=f"Agent {i}" if i == 0 else None,
            )
            ax.scatter(
                xyz[-1, 0], xyz[-1, 1], xyz[-1, 2],
                s=75, marker="X", color=color, edgecolor="k",
                linewidths=0.8, depthshade=False,
            )

        # ---------------------------------------------------------------------
        # 3. Axes, layout
        # ---------------------------------------------------------------------
        ax.set_xlabel(r"$\hat{\mu}_0$")
        ax.set_ylabel(r"$\hat{\mu}_1$")
        ax.set_zlabel(r"$\hat{\mu}_2$")
        ax.set_title(r"Gauge Orbits on Unit Sphere ($\mu$ normalized)")

        max_range = np.array([
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min(),
        ]).max() / 2.0
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_box_aspect((1, 1, 1))

        ax.grid(alpha=0.1)
        ax.legend(loc="upper left", fontsize=8, frameon=False)

        path = out_dir / "mu_gauge_orbit_solid_sphere.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✓ Saved {path}")




def plot_mu_gauge_orbit_projections(history, out_dir: Path):
    """
    2D projections of normalized μ-trajectories: (μ0, μ1), (μ0, μ2), (μ1, μ2).

    This complements the 3D gauge-orbit visualization by showing unambiguous
    planar projections of the same normalized trajectories.
    """
    tracker = get_mu_tracker(history)
    if tracker is None:
        print("⚠️  No mu tracking data available — skipping orbit projections.")
        return

    components0 = tracker.get_component_array(0)
    if components0.size == 0:
        print("⚠️  Empty μ tracking data — skipping orbit projections.")
        return

    K = components0.shape[1]
    if K < 3:
        print("⚠️  Need K ≥ 3 for 3-panel projections — skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import matplotlib.pyplot as plt

    n_agents = (
        len(tracker.agent_indices)
        if hasattr(tracker, "agent_indices")
        else tracker.n_agents
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    titles = [
        r"Projection $(\hat{\mu}_0, \hat{\mu}_1)$",
        r"Projection $(\hat{\mu}_0, \hat{\mu}_2)$",
        r"Projection $(\hat{\mu}_1, \hat{\mu}_2)$",
    ]

    cmap = plt.get_cmap("viridis")

    for ax, (i0, i1), title in zip(axes, pairs, titles):
        for a in range(n_agents):
            comps = tracker.get_component_array(a)  # (n_steps, K)

            norms = np.linalg.norm(comps, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            unit = comps / norms

            xy = unit[:, [i0, i1]]
            n_steps = xy.shape[0]
            t = np.linspace(0.0, 1.0, n_steps)

            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=1.5,
                alpha=0.9,
                label=f"Agent {a}"
            )
            ax.scatter(
                xy[0, 0],
                xy[0, 1],
                s=30,
                marker="o",
                alpha=0.95
            )
            ax.scatter(
                xy[-1, 0],
                xy[-1, 1],
                s=30,
                marker="X",
                alpha=0.95
            )

        ax.set_xlabel(rf"$\hat{{\mu}}_{i0}$".replace("i0", str(i0)))
        ax.set_ylabel(rf"$\hat{{\mu}}_{i1}$".replace("i1", str(i1)))
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8, frameon=False)

    path = out_dir / "mu_gauge_orbit_projections.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✓ ploted projections {path}")



# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced analysis / visualization for multi-agent runs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="_results/_playground",
        help="Path to run directory (contains history.* and final_state.pkl).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"✗ Run directory does not exist: {run_dir}")
        return

    print("\n" + "=" * 70)
    print(f"ENHANCED ANALYSIS SUITE – {run_dir}")
    print("=" * 70)

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    
    # Load data
    history = load_history(run_dir)
    history_dict = normalize_history(history)   # TrainingHistory -> dict
    system = load_system(run_dir)
    # 🔥 MIGRATE TO CHOLESKY if needed
    
    
    # Generate all plots
    print("\nGenerating visualizations...")

    # ------------------------------------------------------------------
    # Energy analysis: use dict-style history
    # ------------------------------------------------------------------
    if history_dict is not None:
        plot_energy_components_separate(history_dict, out_dir)

    # Geometric structure
    #plot_overlap_matrix(system, out_dir)
    #plot_supports(system, out_dir)

    # Field analysis
    #plot_mu_fields(system, out_dir)
    #plot_phi_fields(system, out_dir)
    #plot_sigma_fields(system, out_dir)

    # ------------------------------------------------------------------
    # ✨ Mu tracking plots: use original TrainingHistory (needs mu_tracker)
    # ------------------------------------------------------------------
    if history is not None:
        mu_dir = out_dir / "mu_tracking"
        mu_dir.mkdir(exist_ok=True)

        print("\n[Mu Center Tracking]")
        # All of these expect history.mu_tracker, so pass `history`, not `history_dict`
        plot_mu_norm_trajectories(history, mu_dir)
        plot_norm_variance_evolution(history, mu_dir)
        plot_mu_summary_report(history, mu_dir)
        plot_mu_gauge_orbit(history, mu_dir)
        plot_mu_gauge_orbit_projections(history,mu_dir)
        # Components for first few agents
        for i in range(min(3, system.n_agents if system else 1)):
            plot_mu_component_trajectories(history, mu_dir, agent_idx=i)

        # Phase space projections
        plot_mu_phase_space(history, mu_dir, dims=(0, 1))
        if system and system.agents[0].config.K >= 3:
            plot_mu_phase_space(history, mu_dir, dims=(0, 2))

    # Softmax weights
    if system is not None:
        beta_dir = out_dir / "softmax"
        plot_softmax_weights(system, beta_dir, agent_idx=None, mode="belief")
        plot_softmax_weights(system, beta_dir, agent_idx=None, mode="prior")

    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Figures saved in: {out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()