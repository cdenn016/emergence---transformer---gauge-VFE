# -*- coding: utf-8 -*-
"""
Hierarchical Evolution Loop for Multi-Scale Gauge System
=========================================================

Main evolution loop integrating:
1. Cross-scale prior updates (top-down)
2. Cross-scale observation generation (bottom-up)
3. Gradient computation for all scales
4. Timescale-separated updates
5. Automatic consensus detection and meta-agent formation

This is the core dynamics engine for the hierarchical gauge-theoretic system.

Author: Chris & Christine
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from meta.emergence import MultiScaleSystem, HierarchicalAgent
from retraction import retract_spd
from gradients.gauge_fields import retract_to_principal_ball


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical evolution."""

    # Cross-scale dynamics
    enable_top_down_priors: bool = True
    enable_bottom_up_obs: bool = True

    # Timescale separation
    enable_timescale_filtering: bool = True
    info_change_metric: str = "fisher_metric"  # "fisher_metric", "kl_divergence", or "gradient_norm"

    # Consensus detection
    consensus_check_interval: int = 10  # Steps between consensus checks
    consensus_kl_threshold: float = 0.01
    min_cluster_size: int = 2

    # Observation likelihood for meta-agents
    lambda_obs_meta: float = 1.0  # Weight for constituent-based observations

    # Learning rates (must match Trainer for consistent behavior!)
    lr_mu_q: float = 0.1
    lr_sigma_q: float = 0.001  # Much smaller than mu!
    lr_mu_p: float = 0.1
    lr_sigma_p: float = 0.001  # Much smaller than mu!
    lr_phi: float = 0.1


class HierarchicalEvolutionEngine:
    """
    Evolution engine for hierarchical multi-scale system.

    Implements the full dynamics:
    1. Update priors from parents (top-down)
    2. Compute gradients (using existing gradient_engine)
    3. Apply updates with timescale filtering
    4. Check for consensus and form new meta-agents
    """

    def __init__(self,
                 system: MultiScaleSystem,
                 config: Optional[HierarchicalConfig] = None):
        """
        Initialize hierarchical evolution engine.

        Args:
            system: MultiScaleSystem to evolve
            config: Evolution configuration
        """
        self.system = system
        self.config = config or HierarchicalConfig()

        # Evolution state
        self.step_count = 0
        self.condensation_history = []

        # Metrics tracking
        self.metrics_history = {
            'step': [],
            'n_agents_per_scale': [],
            'n_active_per_scale': [],
            'priors_updated': [],
            'updates_applied': [],
            'condensations': []
        }

    def evolve_step(self,
                   learning_rate: float = 0.01,
                   compute_gradients_fn: Optional[callable] = None) -> Dict:
        """
        Perform one step of hierarchical evolution.

        Full algorithm:
        1. Update priors from meta-agents (top-down)
        2. Compute gradients for all active agents
        3. Apply updates with timescale filtering
        4. Check for consensus (periodically)

        Args:
            learning_rate: Base learning rate for updates
            compute_gradients_fn: Function to compute gradients
                                 Should accept (system) and return List[AgentGradients]

        Returns:
            metrics: Dict with step metrics
        """
        metrics = {
            'step': self.step_count,
            'n_agents': {},
            'n_active': {},
            'n_priors_updated': 0,
            'n_updates_applied': 0,
            'n_condensations': 0,
            'info_changes': [],
        }

        # =====================================================================
        # Phase 1: Prior Updates (Hierarchical + Self-Referential)
        # =====================================================================
        if self.config.enable_top_down_priors:
            update_info = self.system.update_cross_scale_priors()
            metrics['n_priors_from_parent'] = update_info['from_parent']
            metrics['n_priors_from_global'] = update_info['from_global']
            metrics['n_priors_updated'] = update_info['total']

        # =====================================================================
        # Phase 2: Compute Gradients
        # =====================================================================
        if compute_gradients_fn is None:
            # Use default gradient computation (requires gradient_engine)
            try:
                from gradients.gradient_engine import compute_natural_gradients

                # Create temporary wrapper for compatibility
                class SystemWrapper:
                    def __init__(self, multiscale_system):
                        self.agents = multiscale_system.get_all_active_agents()
                        self.n_agents = len(self.agents)
                        self.config = self.agents[0].config if self.agents else None

                    def get_neighbors(self, agent_idx):
                        """Return all active agents for cross-scale coupling."""
                        return list(range(self.n_agents))

                wrapper = SystemWrapper(self.system)
                gradients = compute_natural_gradients(wrapper)

            except ImportError:
                print("[Warning] gradient_engine not available, skipping gradient computation")
                gradients = None
        else:
            gradients = compute_gradients_fn(self.system)

        # =====================================================================
        # Phase 3: Apply Updates with Timescale Filtering
        # =====================================================================
        if gradients is not None:
            n_applied = self._apply_filtered_updates(
                gradients,
                learning_rate,
                metrics
            )
            metrics['n_updates_applied'] = n_applied

            # CRITICAL: Match Trainer post-update operations!
            # These ensure consistency between hierarchical and standard training

            # (1) Re-enforce identical priors if lock mode (matches Trainer line 174-176)
            if hasattr(self.system, 'system_config'):
                if getattr(self.system.system_config, "identical_priors", "off") == "lock":
                    # Apply shared prior to all scale-0 agents
                    base_agents = self.system.agents.get(0, [])
                    if len(base_agents) > 0:
                        mu_p_avg = sum(a.mu_p for a in base_agents) / len(base_agents)
                        Sigma_p_avg = sum(a.Sigma_p for a in base_agents) / len(base_agents)
                        for a in base_agents:
                            a.mu_p = mu_p_avg.copy()
                            a.Sigma_p = Sigma_p_avg.copy()
                            if hasattr(a, 'invalidate_caches'):
                                a.invalidate_caches()

        # =====================================================================
        # Phase 4: Consensus Detection (Periodic)
        # =====================================================================
        if self.step_count % self.config.consensus_check_interval == 0:
            new_condensations = self._check_and_condense_all_scales()
            metrics['n_condensations'] = len(new_condensations)
            self.condensation_history.extend(new_condensations)

        # =====================================================================
        # Record Metrics
        # =====================================================================
        for scale in self.system.agents.keys():
            agents_at_scale = self.system.agents[scale]
            metrics['n_agents'][scale] = len(agents_at_scale)
            metrics['n_active'][scale] = sum(1 for a in agents_at_scale if a.is_active)

        self._record_metrics(metrics)
        self.step_count += 1
        self.system.current_time = self.step_count

        return metrics

    def _apply_filtered_updates(self,
                                gradients: List,
                                learning_rate: float,
                                metrics: Dict) -> int:
        """
        Apply gradient updates with timescale filtering.

        Only updates agents when accumulated information exceeds threshold.

        Args:
            gradients: List of AgentGradients for all active agents
            learning_rate: Learning rate
            metrics: Metrics dict to update

        Returns:
            Number of agents actually updated
        """
        active_agents = self.system.get_all_active_agents()
        n_applied = 0

        # CRITICAL: Only apply timescale filtering when hierarchy exists!
        # Before meta-agents form, all agents should update at every step
        # (matching standard training behavior)
        has_hierarchy = self.system.max_scale() > 0
        use_filtering = self.config.enable_timescale_filtering and has_hierarchy

        for agent, grad in zip(active_agents, gradients):
            # Compute information change from gradient
            delta_info = self._compute_info_change(agent, grad)
            metrics['info_changes'].append(delta_info)

            # Check if agent should update (timescale filtering)
            if use_filtering:
                should_update = agent.should_update(delta_info)
            else:
                should_update = True

            if should_update:
                # Apply natural gradient update
                self._apply_single_update(agent, grad, learning_rate)
                n_applied += 1

        return n_applied

    def _compute_info_change(self, agent: HierarchicalAgent, grad) -> float:
        """
        Compute information change for timescale filtering.

        Uses Fisher information metric on statistical manifold:
        ΔI² = δμᵀ Σ⁻¹ δμ + tr(Σ⁻¹ δΣ Σ⁻¹ δΣ)

        This respects the natural Riemannian geometry and gauge structure.

        Args:
            agent: Agent being updated
            grad: Gradient object

        Returns:
            Information change estimate (in nats, convert to bits via /log(2))
        """
        if self.config.info_change_metric == "fisher_metric":
            # Fisher information metric (gauge-aware)
            info_sq = 0.0

            # Mean contribution: δμᵀ Σ⁻¹ δμ
            if grad.delta_mu_q is not None:
                try:
                    Sigma_inv = np.linalg.inv(agent.Sigma_q + 1e-6 * np.eye(agent.K))
                    info_sq += grad.delta_mu_q @ Sigma_inv @ grad.delta_mu_q
                except np.linalg.LinAlgError:
                    # Fallback to norm if singular
                    info_sq += np.sum(grad.delta_mu_q ** 2)

            # Covariance contribution: tr(Σ⁻¹ δΣ Σ⁻¹ δΣ)
            if grad.delta_Sigma_q is not None:
                try:
                    Sigma_inv = np.linalg.inv(agent.Sigma_q + 1e-6 * np.eye(agent.K))
                    M = Sigma_inv @ grad.delta_Sigma_q
                    info_sq += np.trace(M @ M)
                except np.linalg.LinAlgError:
                    info_sq += np.sum(grad.delta_Sigma_q ** 2) / agent.K

            # Convert to nats
            info_nats = np.sqrt(max(info_sq, 0.0))

            # Convert to bits
            info_change = info_nats / np.log(2)

        elif self.config.info_change_metric == "kl_divergence":
            # KL(q_old || q_new) - exact information change
            # q_new = q_old - learning_rate * grad
            # For small updates: KL ≈ 1/2 Fisher metric
            # So use Fisher metric as fast approximation
            info_sq = 0.0

            if grad.delta_mu_q is not None:
                try:
                    Sigma_inv = np.linalg.inv(agent.Sigma_q + 1e-6 * np.eye(agent.K))
                    info_sq += grad.delta_mu_q @ Sigma_inv @ grad.delta_mu_q
                except:
                    info_sq += np.sum(grad.delta_mu_q ** 2)

            if grad.delta_Sigma_q is not None:
                try:
                    Sigma_inv = np.linalg.inv(agent.Sigma_q + 1e-6 * np.eye(agent.K))
                    M = Sigma_inv @ grad.delta_Sigma_q
                    info_sq += np.trace(M @ M) / 2
                except:
                    info_sq += np.sum(grad.delta_Sigma_q ** 2) / (2 * agent.K)

            info_change = np.sqrt(info_sq) / np.log(2)  # bits

        elif self.config.info_change_metric == "gradient_norm":
            # Fallback: simple gradient norm (not gauge-aware)
            delta_mu = np.linalg.norm(grad.delta_mu_q) if grad.delta_mu_q is not None else 0.0
            delta_Sigma = np.linalg.norm(grad.delta_Sigma_q) if grad.delta_Sigma_q is not None else 0.0
            info_change = np.log2(1.0 + delta_mu + delta_Sigma)

        else:
            info_change = 1.0

        return float(info_change)

    def _apply_single_update(self,
                            agent: HierarchicalAgent,
                            grad,
                            learning_rate: float):
        """
        Apply gradient update to a single agent using SPD-aware retractions.

        CRITICAL: Must use SPD retraction for covariance matrices to ensure
        they remain positive-definite. Naive Euclidean updates can leave the
        SPD manifold and cause numerical blow-up.

        IMPORTANT: Uses different learning rates for different parameters
        from config (lr_mu_q, lr_sigma_q, etc.) to match Trainer behavior!

        Args:
            agent: Agent to update
            grad: Gradient object with delta_mu_q, delta_Sigma_q, delta_phi
            learning_rate: DEPRECATED - using config learning rates instead
        """
        # CRITICAL: Natural gradients are ALREADY negated (descent directions)
        # So we use ADDITION: param_new = param + lr * delta
        # See math_utils/fisher_metric.py:146 - δμ = -Σ ∇_μ (negated!)

        # Update belief mean
        if grad.delta_mu_q is not None:
            agent.mu_q = agent.mu_q + self.config.lr_mu_q * grad.delta_mu_q

        # Update belief covariance (SPD manifold - use retraction!)
        # CRITICAL: Uses lr_sigma_q (much smaller than lr_mu_q!)
        if grad.delta_Sigma_q is not None:
            Sigma_q_new = retract_spd(
                agent.Sigma_q,
                grad.delta_Sigma_q,
                step_size=self.config.lr_sigma_q,
                trust_region=None,
                max_condition=None
            )
            agent.Sigma_q = Sigma_q_new.astype(np.float32)

        # Priors are NEVER updated via gradients in hierarchical system!
        # They come from either:
        # 1. Parent meta-agents (regular hierarchy)
        # 2. Global state (self-referential closure at top)
        #
        # Only update priors if top-down flow is disabled (non-hierarchical mode)
        if not self.config.enable_top_down_priors:
            if grad.delta_mu_p is not None:
                agent.mu_p = agent.mu_p + self.config.lr_mu_p * grad.delta_mu_p

            if grad.delta_Sigma_p is not None:
                Sigma_p_new = retract_spd(
                    agent.Sigma_p,
                    grad.delta_Sigma_p,
                    step_size=self.config.lr_sigma_p,
                    trust_region=None,
                    max_condition=None
                )
                agent.Sigma_p = Sigma_p_new.astype(np.float32)

        # Update gauge field (SO(3) manifold - use retraction!)
        if grad.delta_phi is not None:
            phi_new = agent.gauge.phi + self.config.lr_phi * grad.delta_phi
            agent.gauge.phi = retract_to_principal_ball(
                phi_new,
                margin=1e-2,
                mode='mod2pi'
            )

        # 🔥 CRITICAL: Re-enforce support constraints after all updates
        # (ensures fields remain zero outside support region)
        if hasattr(agent, 'enforce_support_constraints'):
            agent.enforce_support_constraints()

        # Invalidate any cached computations that depend on parameters
        if hasattr(agent, 'invalidate_caches'):
            agent.invalidate_caches()

    def _check_and_condense_all_scales(self) -> List:
        """
        Check all scales for consensus and condense.

        Returns:
            List of newly formed meta-agents
        """
        new_meta_agents = []

        # Check each scale (except max scale)
        for scale in sorted(self.system.agents.keys()):
            active_at_scale = self.system.get_active_agents_at_scale(scale)

            if len(active_at_scale) < self.config.min_cluster_size:
                continue

            # Detect and condense
            new_agents = self.system.auto_detect_and_condense(
                scale=scale,
                kl_threshold=self.config.consensus_kl_threshold,
                min_cluster_size=self.config.min_cluster_size
            )

            new_meta_agents.extend(new_agents)

        return new_meta_agents

    def _record_metrics(self, metrics: Dict):
        """Record metrics to history."""
        self.metrics_history['step'].append(metrics['step'])
        self.metrics_history['n_agents_per_scale'].append(metrics['n_agents'].copy())
        self.metrics_history['n_active_per_scale'].append(metrics['n_active'].copy())
        self.metrics_history['priors_updated'].append(metrics['n_priors_updated'])
        self.metrics_history['updates_applied'].append(metrics['n_updates_applied'])
        self.metrics_history['condensations'].append(metrics['n_condensations'])

    def evolve(self,
              n_steps: int,
              learning_rate: float = 0.01,
              compute_gradients_fn: Optional[callable] = None,
              verbose: bool = True) -> Dict:
        """
        Run hierarchical evolution for multiple steps.

        Args:
            n_steps: Number of evolution steps
            learning_rate: Learning rate for updates
            compute_gradients_fn: Custom gradient computation function
            verbose: Print progress

        Returns:
            Final metrics and history
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"HIERARCHICAL EVOLUTION: {n_steps} steps")
            print(f"{'='*70}\n")
            print(self.system.summary())
            print()

        for step in range(n_steps):
            metrics = self.evolve_step(learning_rate, compute_gradients_fn)

            if verbose and (step % 10 == 0 or metrics['n_condensations'] > 0):
                self._print_step_summary(metrics)

        if verbose:
            print(f"\n{'='*70}")
            print("Evolution complete")
            print(f"{'='*70}\n")
            print(self.system.summary())

        return {
            'metrics_history': self.metrics_history,
            'final_state': self.system.summary(),
            'condensation_events': self.system.condensation_events
        }

    def _print_step_summary(self, metrics: Dict):
        """Print concise step summary."""
        scales_str = ', '.join([
            f"ζ{s}:{metrics['n_active'][s]}"
            for s in sorted(metrics['n_active'].keys())
        ])

        info_mean = np.mean(metrics['info_changes']) if metrics['info_changes'] else 0.0

        msg = f"Step {metrics['step']:4d}: [{scales_str}] "
        msg += f"updated={metrics['n_updates_applied']:3d} "

        # Show prior update breakdown
        if 'n_priors_from_global' in metrics and metrics['n_priors_from_global'] > 0:
            msg += f"priors={metrics['n_priors_updated']:2d}(↻{metrics['n_priors_from_global']}) "
        else:
            msg += f"priors={metrics['n_priors_updated']:2d} "

        msg += f"ΔI={info_mean:.3f}"

        if metrics['n_condensations'] > 0:
            msg += f" ⚡ CONDENSED {metrics['n_condensations']} clusters!"

        print(msg)


# =============================================================================
# Convenience Functions
# =============================================================================

def evolve_hierarchical_system(system: MultiScaleSystem,
                               n_steps: int,
                               learning_rate: float = 0.01,
                               config: Optional[HierarchicalConfig] = None,
                               verbose: bool = True) -> Dict:
    """
    Convenience function to evolve a hierarchical system.

    Args:
        system: MultiScaleSystem to evolve
        n_steps: Number of steps
        learning_rate: Learning rate
        config: Evolution configuration
        verbose: Print progress

    Returns:
        Evolution results with metrics history

    Example:
        >>> system = MultiScaleSystem(base_manifold)
        >>> # Add base agents...
        >>> results = evolve_hierarchical_system(system, n_steps=100)
    """
    engine = HierarchicalEvolutionEngine(system, config)
    return engine.evolve(n_steps, learning_rate, verbose=verbose)


def create_and_evolve_demo(n_base_agents: int = 12,
                           n_steps: int = 50,
                           K: int = 3) -> Dict:
    """
    Create and evolve a demo hierarchical system.

    Args:
        n_base_agents: Number of base agents to create
        n_steps: Evolution steps
        K: Latent dimension

    Returns:
        Evolution results
    """
    from geometry.geometry_base import BaseManifold, TopologyType
    from config import AgentConfig

    # Create base manifold (0D transformers)
    base_manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)

    # Create system
    system = MultiScaleSystem(base_manifold)

    # Add base agents
    agent_config = AgentConfig(
        K=K,
        spatial_shape=(),
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.5
    )

    print(f"Creating {n_base_agents} base agents...")
    for i in range(n_base_agents):
        system.add_base_agent(agent_config, agent_id=f"agent_{i}")

    # Configure evolution
    config = HierarchicalConfig(
        enable_top_down_priors=True,
        enable_bottom_up_obs=True,
        enable_timescale_filtering=True,
        consensus_check_interval=10,
        consensus_kl_threshold=0.05
    )

    # Evolve
    results = evolve_hierarchical_system(
        system,
        n_steps=n_steps,
        learning_rate=0.01,
        config=config,
        verbose=True
    )

    return results
