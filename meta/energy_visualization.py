"""
Energy Landscape and Thermodynamics Visualization

Visualizes the energy decomposition, non-equilibrium dynamics, and thermodynamic
properties of the hierarchical meta-agent system.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from meta.participatory_diagnostics import ParticipatoryDiagnostics


class EnergyVisualizer:
    """Visualize energy landscapes and thermodynamic properties."""

    def __init__(self, diagnostics: ParticipatoryDiagnostics):
        """
        Initialize with diagnostics tracker.

        Args:
            diagnostics: ParticipatoryDiagnostics instance that has been tracking evolution
        """
        self.diagnostics = diagnostics

    def plot_energy_landscape(self, figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot multi-scale energy decomposition over time.

        Shows stacked contributions from:
        - E_self (intrinsic energy)
        - E_belief_align (lateral coherence)
        - E_prior_align (vertical alignment)

        Args:
            figsize: Figure size
        """
        if not self.diagnostics.energy_snapshots:
            raise ValueError("No energy snapshots recorded. Run diagnostics during evolution.")

        # Extract data
        times = [snap['time'] for snap in self.diagnostics.energy_snapshots]
        scale_energy = defaultdict(lambda: {'self': [], 'belief': [], 'prior': [], 'total': []})

        for snap in self.diagnostics.energy_snapshots:
            for scale, energy_data in snap['by_scale'].items():
                scale_energy[scale]['self'].append(energy_data['E_self'])
                scale_energy[scale]['belief'].append(energy_data['E_belief_align'])
                scale_energy[scale]['prior'].append(energy_data['E_prior_align'])
                scale_energy[scale]['total'].append(energy_data['E_total'])

        max_scale = max(scale_energy.keys())

        # Create figure with subplots for each scale
        n_scales = len(scale_energy)
        fig, axes = plt.subplots(n_scales, 1, figsize=figsize, sharex=True)

        if n_scales == 1:
            axes = [axes]

        colors = {'self': '#FF6B6B', 'belief': '#4ECDC4', 'prior': '#45B7D1'}

        for idx, (scale, energies) in enumerate(sorted(scale_energy.items())):
            ax = axes[idx]

            # Stack plot for energy components
            ax.fill_between(times, 0, energies['self'],
                           label='E_self', alpha=0.8, color=colors['self'])
            ax.fill_between(times, energies['self'],
                           np.array(energies['self']) + np.array(energies['belief']),
                           label='E_belief_align', alpha=0.8, color=colors['belief'])
            ax.fill_between(times,
                           np.array(energies['self']) + np.array(energies['belief']),
                           energies['total'],
                           label='E_prior_align', alpha=0.8, color=colors['prior'])

            # Total energy line
            ax.plot(times, energies['total'], 'k-', linewidth=2, label='Total')

            ax.set_ylabel(f'Energy\n(ζ={scale})', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9, ncol=4)
            ax.grid(alpha=0.3, axis='y')

        axes[-1].set_xlabel('Time', fontsize=12)
        fig.suptitle('Multi-Scale Energy Landscape Decomposition', fontsize=16, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_energy_flow(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot energy flux (rate of change) across scales.

        Args:
            figsize: Figure size
        """
        if len(self.diagnostics.energy_snapshots) < 2:
            raise ValueError("Need at least 2 snapshots to compute energy flux")

        # Compute energy flux
        times = []
        scale_flux = defaultdict(list)

        for i in range(1, len(self.diagnostics.energy_snapshots)):
            snap_prev = self.diagnostics.energy_snapshots[i - 1]
            snap_curr = self.diagnostics.energy_snapshots[i]

            dt = snap_curr['time'] - snap_prev['time']
            if dt == 0:
                continue

            times.append(snap_curr['time'])

            for scale in snap_curr['by_scale'].keys():
                if scale in snap_prev['by_scale']:
                    E_prev = snap_prev['by_scale'][scale]['E_total']
                    E_curr = snap_curr['by_scale'][scale]['E_total']
                    flux = (E_curr - E_prev) / dt
                    scale_flux[scale].append(flux)
                else:
                    scale_flux[scale].append(0)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot flux for each scale
        colors = plt.cm.viridis(np.linspace(0, 1, len(scale_flux)))

        for (scale, flux), color in zip(sorted(scale_flux.items()), colors):
            ax.plot(times[:len(flux)], flux, marker='o', label=f'ζ={scale}',
                   color=color, linewidth=2, alpha=0.8)

        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Energy Flux (dE/dt)', fontsize=12)
        ax.set_title('Energy Flow Across Scales', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_prior_evolution(self, figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Plot prior change history (KL divergence from previous prior).

        Shows top-down information flow.

        Args:
            figsize: Figure size
        """
        if not self.diagnostics.prior_changes:
            raise ValueError("No prior changes recorded")

        # Organize by scale and agent
        scale_agent_changes = defaultdict(lambda: defaultdict(list))
        scale_agent_times = defaultdict(lambda: defaultdict(list))

        for change in self.diagnostics.prior_changes:
            scale = change['scale']
            agent_id = change['local_index']
            scale_agent_changes[scale][agent_id].append(change['kl_change'])
            scale_agent_times[scale][agent_id].append(change['time'])

        # Create subplots for each scale
        scales = sorted(scale_agent_changes.keys())
        n_scales = len(scales)

        fig, axes = plt.subplots(n_scales, 1, figsize=figsize, sharex=True)

        if n_scales == 1:
            axes = [axes]

        for idx, scale in enumerate(scales):
            ax = axes[idx]

            # Plot each agent's prior changes
            for agent_id, changes in scale_agent_changes[scale].items():
                times = scale_agent_times[scale][agent_id]
                ax.plot(times, changes, marker='o', alpha=0.6,
                       label=f'Agent {agent_id}', linewidth=1.5)

            # Average line
            all_times = []
            all_changes = []
            for agent_id in scale_agent_changes[scale].keys():
                all_times.extend(scale_agent_times[scale][agent_id])
                all_changes.extend(scale_agent_changes[scale][agent_id])

            if all_times:
                # Compute moving average
                unique_times = sorted(set(all_times))
                avg_changes = []
                for t in unique_times:
                    changes_at_t = [all_changes[i] for i, tt in enumerate(all_times) if tt == t]
                    avg_changes.append(np.mean(changes_at_t))

                ax.plot(unique_times, avg_changes, 'k-', linewidth=3,
                       label='Average', alpha=0.8)

            ax.set_ylabel(f'KL Change\n(ζ={scale})', fontsize=11, fontweight='bold')
            ax.set_yscale('log')
            ax.legend(loc='upper right', fontsize=8, ncol=3)
            ax.grid(alpha=0.3, which='both')

        axes[-1].set_xlabel('Time', fontsize=12)
        fig.suptitle('Prior Evolution (Top-Down Information Flow)', fontsize=16, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_non_equilibrium_indicators(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot non-equilibrium indicators dashboard.

        Shows:
        - Energy variance
        - Gradient variance
        - Information flux
        - Equilibrium score

        Args:
            figsize: Figure size
        """
        if not self.diagnostics.energy_snapshots:
            raise ValueError("No snapshots recorded")

        times = [snap['time'] for snap in self.diagnostics.energy_snapshots]

        # Extract indicators
        energy_vars = []
        gradient_vars = []
        info_fluxes = []

        for snap in self.diagnostics.energy_snapshots:
            # Energy variance across scales
            energies = [scale_data['E_total'] for scale_data in snap['by_scale'].values()]
            energy_vars.append(np.var(energies) if len(energies) > 1 else 0)

            # For now, use placeholder for gradient variance and info flux
            # These would need to be tracked in diagnostics
            gradient_vars.append(0)
            info_fluxes.append(0)

        # Compute equilibrium score (0 = equilibrium, 1 = far from equilibrium)
        # Based on energy variance
        max_var = max(energy_vars) if energy_vars else 1
        eq_scores = [1 - np.exp(-v / (max_var + 1e-8)) for v in energy_vars]

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Energy variance
        axes[0].plot(times, energy_vars, 'o-', color='#E74C3C', linewidth=2)
        axes[0].set_ylabel('Energy\nVariance', fontsize=11, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].fill_between(times, 0, energy_vars, alpha=0.3, color='#E74C3C')

        # Gradient variance
        axes[1].plot(times, gradient_vars, 'o-', color='#3498DB', linewidth=2)
        axes[1].set_ylabel('Gradient\nVariance', fontsize=11, fontweight='bold')
        axes[1].grid(alpha=0.3)

        # Information flux
        axes[2].plot(times, info_fluxes, 'o-', color='#2ECC71', linewidth=2)
        axes[2].set_ylabel('Information\nFlux', fontsize=11, fontweight='bold')
        axes[2].grid(alpha=0.3)

        # Equilibrium score
        axes[3].plot(times, eq_scores, 'o-', color='#9B59B6', linewidth=2)
        axes[3].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Threshold')
        axes[3].fill_between(times, 0, eq_scores, alpha=0.3, color='#9B59B6')
        axes[3].set_ylabel('Non-Equilibrium\nScore', fontsize=11, fontweight='bold')
        axes[3].set_xlabel('Time', fontsize=12)
        axes[3].legend(fontsize=9)
        axes[3].grid(alpha=0.3)

        fig.suptitle('Non-Equilibrium Dynamics Indicators', fontsize=16, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_energy_per_agent(self,
                             scale: int = 0,
                             snapshot_idx: int = -1,
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot energy decomposition for individual agents at a specific scale.

        Args:
            scale: Which scale to visualize
            snapshot_idx: Which snapshot (-1 for latest)
            figsize: Figure size
        """
        if not self.diagnostics.energy_snapshots:
            raise ValueError("No energy snapshots recorded")

        snapshot = self.diagnostics.energy_snapshots[snapshot_idx]

        if 'by_agent' not in snapshot:
            raise ValueError("Agent-level energy data not available")

        # Filter agents at target scale
        agents_at_scale = [(idx, data) for idx, data in snapshot['by_agent'].items()
                          if idx[0] == scale]  # idx is (scale, local_index)

        if not agents_at_scale:
            raise ValueError(f"No agents at scale {scale}")

        # Sort by local index
        agents_at_scale.sort(key=lambda x: x[0][1])

        agent_ids = [f"A{idx[1]}" for idx, _ in agents_at_scale]
        E_self = [data['E_self'] for _, data in agents_at_scale]
        E_belief = [data['E_belief_align'] for _, data in agents_at_scale]
        E_prior = [data['E_prior_align'] for _, data in agents_at_scale]

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(agent_ids))
        width = 0.25

        ax.bar(x - width, E_self, width, label='E_self', color='#FF6B6B', alpha=0.8)
        ax.bar(x, E_belief, width, label='E_belief_align', color='#4ECDC4', alpha=0.8)
        ax.bar(x + width, E_prior, width, label='E_prior_align', color='#45B7D1', alpha=0.8)

        ax.set_xlabel('Agent', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title(f'Energy Decomposition per Agent (Scale {scale}, t={snapshot["time"]})',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_ids)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_interactive_energy_3d(self, scale: int = 0) -> Optional[go.Figure]:
        """
        Create 3D interactive energy landscape using Plotly.

        Shows energy evolution over time and agent space.

        Args:
            scale: Which scale to visualize

        Returns:
            Plotly figure or None if plotly not available
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Install with: pip install plotly")
            return None

        if not self.diagnostics.energy_snapshots:
            raise ValueError("No energy snapshots recorded")

        # Collect data
        times = []
        agent_energies = defaultdict(list)

        for snap in self.diagnostics.energy_snapshots:
            if 'by_agent' not in snap:
                continue

            times.append(snap['time'])

            # Get all agents at this scale
            for (s, local_idx), data in snap['by_agent'].items():
                if s == scale:
                    agent_energies[local_idx].append(data['E_total'])

        if not agent_energies:
            raise ValueError(f"No agent-level data at scale {scale}")

        # Create meshgrid
        agent_indices = sorted(agent_energies.keys())
        Z = np.zeros((len(times), len(agent_indices)))

        for i, agent_idx in enumerate(agent_indices):
            energies = agent_energies[agent_idx]
            Z[:len(energies), i] = energies

        X, Y = np.meshgrid(agent_indices, times)

        # Create 3D surface
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])

        fig.update_layout(
            title=f'Energy Landscape Evolution (Scale {scale})',
            scene=dict(
                xaxis_title='Agent Index',
                yaxis_title='Time',
                zaxis_title='Energy'
            ),
            width=900,
            height=700
        )

        return fig

    def create_energy_report(self, output_dir: str = './energy_analysis') -> Dict[str, str]:
        """
        Generate complete energy analysis report.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        saved_files = {}

        print("Generating energy visualizations...")

        try:
            fig = self.plot_energy_landscape()
            path = output_path / 'energy_landscape.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['energy_landscape'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved energy landscape to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate energy landscape: {e}")

        try:
            fig = self.plot_energy_flow()
            path = output_path / 'energy_flow.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['energy_flow'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved energy flow to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate energy flow: {e}")

        try:
            fig = self.plot_prior_evolution()
            path = output_path / 'prior_evolution.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['prior_evolution'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved prior evolution to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate prior evolution: {e}")

        try:
            fig = self.plot_non_equilibrium_indicators()
            path = output_path / 'non_equilibrium_indicators.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['non_equilibrium'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved non-equilibrium indicators to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate non-equilibrium indicators: {e}")

        try:
            fig = self.plot_energy_per_agent(scale=0)
            path = output_path / 'energy_per_agent.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['energy_per_agent'] = str(path)
            plt.close(fig)
            print(f"  ✓ Saved energy per agent to {path}")
        except Exception as e:
            print(f"  ✗ Failed to generate energy per agent: {e}")

        print(f"\nEnergy analysis complete! Generated {len(saved_files)} outputs in {output_dir}")
        return saved_files
