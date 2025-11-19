# -*- coding: utf-8 -*-
"""
Phase Transition Scanner for Participatory Dynamics
====================================================

Scans parameter space to find critical points in the cultural/hierarchical
tension model where the system undergoes phase transitions.

Primary scan: LAMBDA_PRIOR_ALIGN (cultural authority pressure)
Critical point observed around λ_prior ≈ 1.7

Order parameters tracked:
1. Cultural conformity: ⟨KL(q||p)⟩ - drops at transition
2. Belief diversity: Var(μ_q) - collapses above critical point
3. Prior certainty: det(Σ_p) - diverges (→ 0) above transition
4. Energy stability: Max/std of total energy - blows up above transition
5. Hierarchical depth: Max scale reached
6. Emergence rate: Number of condensation events

Usage:
    python phase_transition_scanner.py --param LAMBDA_PRIOR_ALIGN --range 1.0 2.0 --points 20

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

# Import simulation components
from geometry.geometry_base import BaseManifold, TopologyType
from config import AgentConfig, SystemConfig, MaskConfig
from meta.emergence import MultiScaleSystem
from meta.hierarchical_evolution import HierarchicalEvolutionEngine, HierarchicalConfig
from meta.consensus import ConsensusDetector
from math_utils.generators import generate_so3_generators
from gradients.gradient_engine import compute_natural_gradients
from free_energy_clean import compute_total_free_energy


@dataclass
class OrderParameters:
    """Order parameters characterizing system state."""
    param_value: float

    # Conformity metrics
    mean_kl_q_p: float  # ⟨KL(q||p)⟩ - cultural conformity
    std_kl_q_p: float

    # Diversity metrics
    belief_variance: float  # Var(μ_q) - belief diversity
    prior_variance: float   # Var(μ_p) - prior diversity

    # Certainty metrics
    mean_prior_det: float  # Mean det(Σ_p) - prior certainty
    min_prior_det: float   # Min det(Σ_p) - strongest certainty

    # Energy metrics
    final_energy: float
    max_energy: float
    energy_std: float
    energy_trend: float  # Linear fit slope

    # Structural metrics
    max_scale: int
    n_emergence_events: int
    final_n_agents: int

    # Stability
    converged: bool  # Did simulation complete without exploding?
    error_message: str = ""


def compute_order_parameters(history: Dict, system, param_value: float) -> OrderParameters:
    """
    Extract order parameters from simulation results.

    Args:
        history: Simulation history dict
        system: Final MultiScaleSystem state
        param_value: Parameter value that was scanned

    Returns:
        OrderParameters instance
    """
    try:
        # Energy metrics
        energies = np.array(history['total'])
        final_energy = energies[-1] if len(energies) > 0 else np.nan
        max_energy = np.max(energies) if len(energies) > 0 else np.nan
        energy_std = np.std(energies) if len(energies) > 1 else np.nan

        # Energy trend (linear fit)
        if len(energies) > 2:
            steps = np.arange(len(energies))
            energy_trend = np.polyfit(steps, energies, 1)[0]
        else:
            energy_trend = np.nan

        # Structural metrics
        max_scale = history.get('n_scales', [0])[-1] if 'n_scales' in history else 0
        n_emergence_events = len(history.get('emergence_events', []))
        final_n_agents = history.get('n_active_agents', [0])[-1] if 'n_active_agents' in history else 0

        # Compute conformity and diversity from system state
        all_agents = system.get_all_active_agents() if system else []

        if len(all_agents) > 0:
            # KL(q||p) for each agent (conformity)
            kl_q_p_values = []
            belief_mus = []
            prior_mus = []
            prior_dets = []

            for agent in all_agents:
                # KL divergence between belief and prior
                from math_utils.numerical_utils import kl_gaussian
                kl = kl_gaussian(agent.mu_q, agent.Sigma_q, agent.mu_p, agent.Sigma_p)
                kl_q_p_values.append(kl)

                belief_mus.append(agent.mu_q)
                prior_mus.append(agent.mu_p)
                prior_dets.append(np.linalg.det(agent.Sigma_p))

            mean_kl_q_p = np.mean(kl_q_p_values)
            std_kl_q_p = np.std(kl_q_p_values)

            # Belief diversity = variance of μ_q across agents
            belief_mus = np.array(belief_mus)
            belief_variance = np.var(belief_mus)

            # Prior diversity = variance of μ_p across agents
            prior_mus = np.array(prior_mus)
            prior_variance = np.var(prior_mus)

            # Prior certainty metrics
            mean_prior_det = np.mean(prior_dets)
            min_prior_det = np.min(prior_dets)
        else:
            mean_kl_q_p = np.nan
            std_kl_q_p = np.nan
            belief_variance = np.nan
            prior_variance = np.nan
            mean_prior_det = np.nan
            min_prior_det = np.nan

        converged = True
        error_message = ""

    except Exception as e:
        # Simulation failed or exploded
        mean_kl_q_p = np.nan
        std_kl_q_p = np.nan
        belief_variance = np.nan
        prior_variance = np.nan
        mean_prior_det = np.nan
        min_prior_det = np.nan
        final_energy = np.nan
        max_energy = np.nan
        energy_std = np.nan
        energy_trend = np.nan
        max_scale = 0
        n_emergence_events = 0
        final_n_agents = 0
        converged = False
        error_message = str(e)

    return OrderParameters(
        param_value=param_value,
        mean_kl_q_p=mean_kl_q_p,
        std_kl_q_p=std_kl_q_p,
        belief_variance=belief_variance,
        prior_variance=prior_variance,
        mean_prior_det=mean_prior_det,
        min_prior_det=min_prior_det,
        final_energy=final_energy,
        max_energy=max_energy,
        energy_std=energy_std,
        energy_trend=energy_trend,
        max_scale=max_scale,
        n_emergence_events=n_emergence_events,
        final_n_agents=final_n_agents,
        converged=converged,
        error_message=error_message
    )


def run_single_simulation(param_name: str, param_value: float, n_steps: int = 100,
                          n_agents: int = 12, K_latent: int = 3, seed: int = 42) -> tuple:
    """
    Run a single simulation with specified parameter value.

    Args:
        param_name: Name of parameter to vary
        param_value: Value of the parameter
        n_steps: Number of training steps
        n_agents: Number of base agents
        K_latent: Latent dimension
        seed: Random seed

    Returns:
        (history, system, success)
    """
    np.random.seed(seed)

    try:
        # Fixed configuration (from simulation_suite.py)
        LAMBDA_SELF = 3.0
        LAMBDA_BELIEF_ALIGN = 2.0
        LAMBDA_PRIOR_ALIGN = 1.68  # Default, will be overridden if scanning this param
        LAMBDA_OBS = 0.0
        LAMBDA_PHI = 0.0

        # Override the scanned parameter
        if param_name == 'LAMBDA_PRIOR_ALIGN':
            LAMBDA_PRIOR_ALIGN = param_value
        elif param_name == 'LAMBDA_SELF':
            LAMBDA_SELF = param_value
        elif param_name == 'LAMBDA_BELIEF_ALIGN':
            LAMBDA_BELIEF_ALIGN = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        # Create manifold and agents
        manifold = BaseManifold(shape=(), topology=TopologyType.PERIODIC)

        system_cfg = SystemConfig(
            lambda_self=LAMBDA_SELF,
            lambda_belief_align=LAMBDA_BELIEF_ALIGN,
            lambda_prior_align=LAMBDA_PRIOR_ALIGN,
            lambda_obs=LAMBDA_OBS,
            lambda_phi=LAMBDA_PHI,
            kappa_beta=1.0,
            kappa_gamma=1.0,
            identical_priors="off"
        )

        agent_cfg = AgentConfig(
            K=K_latent,
            observation_noise=0.1
        )

        # Create multi-scale system
        multi_scale_system = MultiScaleSystem(
            manifold,
            max_emergence_levels=3,
            max_meta_membership=10,
            max_total_agents=1000
        )
        multi_scale_system.system_config = system_cfg

        # Create base agents
        generators = generate_so3_generators(K_latent)

        # Create supports for point manifolds (required for gradient computation)
        # For 0D manifolds, use SupportRegionSmooth (matches simulation_suite.py)
        from agent.masking import SupportRegionSmooth
        mask_cfg = MaskConfig()
        supports = []
        for _ in range(n_agents):
            support = SupportRegionSmooth(
                mask_binary=np.array(True),  # 0D: single point, always True
                base_shape=(),
                config=mask_cfg
            )
            # CRITICAL: Add base_manifold attribute for gradient engine compatibility
            support.base_manifold = manifold
            supports.append(support)

        for i in range(n_agents):
            # Create hierarchical agent directly (no need for regular Agent)
            hier_agent = multi_scale_system.add_base_agent(agent_cfg, agent_id=f"agent_{i}")

            # Attach support (CRITICAL for gradient computation)
            hier_agent.support = supports[i]

            # Initialize fields with random values
            rng = np.random.default_rng(seed + i)
            hier_agent.generators = generators

            # Random initialization for beliefs and priors
            hier_agent.mu_q = rng.standard_normal(K_latent) * 0.1
            hier_agent.Sigma_q = np.eye(K_latent)
            hier_agent.mu_p = rng.standard_normal(K_latent) * 0.1  # Diverse priors
            hier_agent.Sigma_p = np.eye(K_latent)

            # Random gauge frame (element of SO(3))
            # Generate random rotation: φ ~ Uniform on S² × [0, π]
            from math_utils.so3_frechet import so3_exp
            random_axis = rng.standard_normal(3)
            random_axis = random_axis / np.linalg.norm(random_axis)  # Normalize
            random_angle = rng.uniform(0, np.pi)
            random_phi = random_axis * random_angle
            hier_agent.gauge.phi = so3_exp(random_phi)

        # Hierarchical evolution config
        hier_config = HierarchicalConfig(
            enable_top_down_priors=True,
            enable_bottom_up_obs=False,
            enable_timescale_filtering=False,
            consensus_check_interval=5,
            consensus_kl_threshold=0.05,
            min_cluster_size=2,
            deactivate_constituents=False,
            lr_mu_q=0.08,
            lr_sigma_q=0.001,
            lr_mu_p=0.2,
            lr_sigma_p=0.01,
            lr_phi=0.1
        )

        engine = HierarchicalEvolutionEngine(
            multi_scale_system, hier_config
        )

        # Training loop (simplified - no monitoring to speed up)
        history = {
            'step': [],
            'total': [],
            'n_scales': [],
            'n_active_agents': [],
            'n_condensations': [],
            'emergence_events': []
        }

        # Gradient adapter (from simulation_suite.py)
        class _GradientSystemAdapter:
            """
            Minimal adapter to make MultiScaleSystem compatible with gradient engine.

            Provides the interface needed by compute_natural_gradients WITHOUT
            re-initializing agents (which would corrupt their state).

            CRITICAL: Must respect spatial overlaps to match standard training!
            """
            def __init__(self, agents_list, system_config):
                from math_utils.transport import compute_transport
                import numpy as np

                self.agents = agents_list  # List of active agents
                self.config = system_config  # System configuration
                self.n_agents = len(agents_list)
                self._compute_transport = compute_transport

                # For point manifolds, all agents overlap (no spatial separation)
                # Check if we have a point manifold by checking first agent
                is_point_manifold = False
                if len(agents_list) > 0:
                    agent = agents_list[0]
                    if hasattr(agent, 'base_manifold') and hasattr(agent.base_manifold, 'shape'):
                        is_point_manifold = (agent.base_manifold.shape == ())

                # Compute overlap relationships once (lightweight check)
                # This ensures gradient computation matches standard training
                self._overlaps = {}

                if is_point_manifold:
                    # Point manifold: all agents overlap (exist at same abstract point)
                    for i in range(self.n_agents):
                        for j in range(self.n_agents):
                            if i != j:
                                self._overlaps[(i, j)] = True
                else:
                    # Spatial manifold: check actual overlaps
                    overlap_threshold = 1e-3

                    for i in range(self.n_agents):
                        for j in range(self.n_agents):
                            if i == j:
                                continue

                            agent_i = agents_list[i]
                            agent_j = agents_list[j]

                            # Check if both have supports
                            if not (hasattr(agent_i, 'support') and hasattr(agent_j, 'support')):
                                # No support info - assume overlap
                                self._overlaps[(i, j)] = True
                                continue

                            if agent_i.support is None or agent_j.support is None:
                                # Missing support - assume overlap
                                self._overlaps[(i, j)] = True
                                continue

                            # Get masks (try both mask_continuous and chi_weight)
                            chi_i = getattr(agent_i.support, 'mask_continuous',
                                           getattr(agent_i.support, 'chi_weight', None))
                            chi_j = getattr(agent_j.support, 'mask_continuous',
                                           getattr(agent_j.support, 'chi_weight', None))

                            if chi_i is None or chi_j is None:
                                # No mask - assume overlap
                                self._overlaps[(i, j)] = True
                                continue

                            # CRITICAL: Match MultiAgentSystem's two-check overlap logic
                            # Check 1: Upper bound (product of maxes)
                            max_overlap = np.max(chi_i) * np.max(chi_j)
                            if max_overlap < overlap_threshold:
                                self._overlaps[(i, j)] = False
                                continue

                            # Check 2: Actual overlap (max of products)
                            chi_ij = chi_i * chi_j  # Element-wise product
                            has_overlap = np.max(chi_ij) >= overlap_threshold
                            self._overlaps[(i, j)] = has_overlap

            def get_neighbors(self, agent_idx: int):
                """Return agents that spatially overlap (matches MultiAgentSystem behavior)."""
                neighbors = []
                for j in range(self.n_agents):
                    # CRITICAL: Default to False (no overlap) like MultiAgentSystem.has_overlap
                    if j != agent_idx and self._overlaps.get((agent_idx, j), False):
                        neighbors.append(j)
                return neighbors

            def compute_transport_ij(self, i: int, j: int):
                """Compute transport operator Ω_ij = exp(φ_i) exp(-φ_j)."""
                agent_i = self.agents[i]
                agent_j = self.agents[j]
                return self._compute_transport(
                    agent_i.gauge.phi,
                    agent_j.gauge.phi,
                    agent_i.generators,
                    validate=False
                )

        for step in range(n_steps):
            active_agents = multi_scale_system.get_all_active_agents()
            if len(active_agents) == 0:
                break

            temp_system = _GradientSystemAdapter(active_agents, multi_scale_system.system_config)

            # Compute energy
            energies = compute_total_free_energy(temp_system)
            total_energy = energies.total

            # Check for explosion
            if not np.isfinite(total_energy) or total_energy > 1000:
                print(f"  ✗ Simulation exploded at step {step} (E={total_energy:.2f})")
                return history, multi_scale_system, False

            # Evolve
            def compute_grads_with_adapter(system):
                return compute_natural_gradients(temp_system)

            metrics = engine.evolve_step(
                learning_rate=0.08,
                compute_gradients_fn=compute_grads_with_adapter
            )

            # Record
            n_scales = len(metrics.get('n_active', {}))
            total_active = sum(metrics.get('n_active', {}).values())

            history['step'].append(step)
            history['total'].append(total_energy)
            history['n_scales'].append(n_scales)
            history['n_active_agents'].append(total_active)
            history['n_condensations'].append(metrics.get('n_condensations', 0))

            if metrics.get('n_condensations', 0) > 0:
                event = {
                    'step': step,
                    'n_condensations': metrics['n_condensations'],
                    'n_scales': n_scales
                }
                history['emergence_events'].append(event)

        return history, multi_scale_system, True

    except Exception as e:
        import traceback
        print(f"  ✗ Error: {e}")
        print("  Full traceback:")
        traceback.print_exc()
        return {}, None, False


def scan_parameter(param_name: str, param_values: np.ndarray,
                   n_steps: int = 100, output_dir: Path = None) -> List[OrderParameters]:
    """
    Scan across parameter range and collect order parameters.

    Args:
        param_name: Parameter to scan
        param_values: Array of parameter values to test
        n_steps: Number of simulation steps per run
        output_dir: Directory to save results

    Returns:
        List of OrderParameters for each parameter value
    """
    results = []

    print(f"\n{'='*70}")
    print(f"PHASE TRANSITION SCAN: {param_name}")
    print(f"{'='*70}")
    print(f"Range: [{param_values.min():.3f}, {param_values.max():.3f}]")
    print(f"Points: {len(param_values)}")
    print(f"Steps per run: {n_steps}")
    print()

    for i, value in enumerate(param_values):
        print(f"[{i+1}/{len(param_values)}] {param_name} = {value:.4f} ... ", end="", flush=True)

        history, system, success = run_single_simulation(
            param_name=param_name,
            param_value=value,
            n_steps=n_steps,
            seed=42 + i  # Different seed for each run
        )

        if success:
            order_params = compute_order_parameters(history, system, value)
            print(f"✓ E={order_params.final_energy:.4f}, scales={order_params.max_scale}")
        else:
            # Create failed order parameters
            order_params = OrderParameters(
                param_value=value,
                mean_kl_q_p=np.nan,
                std_kl_q_p=np.nan,
                belief_variance=np.nan,
                prior_variance=np.nan,
                mean_prior_det=np.nan,
                min_prior_det=np.nan,
                final_energy=np.nan,
                max_energy=np.nan,
                energy_std=np.nan,
                energy_trend=np.nan,
                max_scale=0,
                n_emergence_events=0,
                final_n_agents=0,
                converged=False,
                error_message="Simulation failed"
            )
            print("✗ FAILED")

        results.append(order_params)

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"scan_{param_name}.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Results saved to {results_path}")

    return results


def plot_phase_diagram(results: List[OrderParameters], param_name: str, output_dir: Path):
    """
    Plot phase diagram showing order parameters vs. control parameter.

    Args:
        results: List of OrderParameters
        param_name: Name of scanned parameter
        output_dir: Directory to save plots
    """
    param_values = np.array([r.param_value for r in results])
    converged = np.array([r.converged for r in results])

    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Phase Transition Diagram: {param_name}', fontsize=16, fontweight='bold')

    # 1. Cultural Conformity (KL(q||p))
    ax = axes[0, 0]
    kl_q_p = np.array([r.mean_kl_q_p for r in results])
    ax.plot(param_values[converged], kl_q_p[converged], 'o-', linewidth=2, markersize=6)
    ax.axvline(1.7, color='red', linestyle='--', alpha=0.5, label='Critical point')
    ax.set_xlabel(param_name)
    ax.set_ylabel('⟨KL(q||p)⟩')
    ax.set_title('Cultural Conformity')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Belief Diversity
    ax = axes[0, 1]
    belief_var = np.array([r.belief_variance for r in results])
    ax.plot(param_values[converged], belief_var[converged], 'o-', linewidth=2, markersize=6, color='green')
    ax.axvline(1.7, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Var(μ_q)')
    ax.set_title('Belief Diversity (Collapse at Transition)')
    ax.grid(True, alpha=0.3)

    # 3. Prior Certainty (det(Σ_p))
    ax = axes[1, 0]
    prior_det = np.array([r.mean_prior_det for r in results])
    ax.semilogy(param_values[converged], prior_det[converged], 'o-', linewidth=2, markersize=6, color='purple')
    ax.axvline(1.7, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Mean det(Σ_p)')
    ax.set_title('Prior Certainty (Diverges at Transition)')
    ax.grid(True, alpha=0.3)

    # 4. Energy Stability
    ax = axes[1, 1]
    max_energy = np.array([r.max_energy for r in results])
    ax.plot(param_values[converged], max_energy[converged], 'o-', linewidth=2, markersize=6, color='red')
    ax.axvline(1.7, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Max Energy')
    ax.set_title('Energy Scale (Blows Up at Transition)')
    ax.grid(True, alpha=0.3)

    # 5. Hierarchical Structure
    ax = axes[2, 0]
    max_scale = np.array([r.max_scale for r in results])
    n_emergence = np.array([r.n_emergence_events for r in results])
    ax.plot(param_values, max_scale, 'o-', linewidth=2, markersize=6, label='Max Scale', color='blue')
    ax.plot(param_values, n_emergence, 's-', linewidth=2, markersize=6, label='# Emergences', color='orange')
    ax.axvline(1.7, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Count')
    ax.set_title('Hierarchical Structure')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Convergence Map
    ax = axes[2, 1]
    colors = ['red' if not c else 'green' for c in converged]
    ax.scatter(param_values, [1]*len(param_values), c=colors, s=100, alpha=0.6)
    ax.axvline(1.7, color='red', linestyle='--', alpha=0.5, label='Critical point')
    ax.set_xlabel(param_name)
    ax.set_yticks([])
    ax.set_title('Simulation Stability (green=converged, red=failed)')
    ax.set_ylim([0.5, 1.5])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    plt.tight_layout()

    # Save
    output_path = output_dir / f'phase_diagram_{param_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Phase diagram saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scan parameter space to find phase transitions in participatory dynamics."
    )
    parser.add_argument(
        "--param",
        type=str,
        default="LAMBDA_PRIOR_ALIGN",
        choices=["LAMBDA_PRIOR_ALIGN", "LAMBDA_SELF", "LAMBDA_BELIEF_ALIGN"],
        help="Parameter to scan"
    )
    parser.add_argument(
        "--min",
        type=float,
        default=1.0,
        help="Minimum parameter value"
    )
    parser.add_argument(
        "--max",
        type=float,
        default=2.0,
        help="Maximum parameter value"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=20,
        help="Number of points to sample"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps per run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="_results/phase_scan",
        help="Directory to save results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Generate parameter values
    param_values = np.linspace(args.min, args.max, args.points)

    # Run scan
    results = scan_parameter(
        param_name=args.param,
        param_values=param_values,
        n_steps=args.steps,
        output_dir=output_dir
    )

    # Plot phase diagram
    plot_phase_diagram(results, args.param, output_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    n_converged = sum(r.converged for r in results)
    print(f"Converged simulations: {n_converged}/{len(results)}")

    # Find critical point (where simulations start failing)
    critical_idx = None
    for i, r in enumerate(results):
        if not r.converged:
            critical_idx = i
            break

    if critical_idx is not None and critical_idx > 0:
        critical_lower = results[critical_idx - 1].param_value
        critical_upper = results[critical_idx].param_value
        print(f"Critical point estimate: {args.param} ∈ [{critical_lower:.4f}, {critical_upper:.4f}]")
    else:
        print("No critical point found in scanned range")

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
