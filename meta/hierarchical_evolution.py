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
from update_engine import GradientApplier


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical evolution."""

    # Cross-scale dynamics
    enable_top_down_priors: bool = True
    enable_bottom_up_obs: bool = True

    # Ouroboros: Multi-scale hyperprior tower (non-Markovian prior propagation)
    enable_hyperprior_tower: bool = False  # Receive priors from ALL levels above, not just parent
    max_hyperprior_depth: int = 3          # How many ancestral levels to include (1=standard Markov)
    hyperprior_decay: float = 0.3          # Exponential decay with scale distance: λ_k = λ * decay^k

    # Timescale separation
    enable_timescale_filtering: bool = True
    info_change_metric: str = "fisher_metric"  # "fisher_metric", "kl_divergence", or "gradient_norm"

    # Consensus detection
    consensus_check_interval: int = 10  # Steps between consensus checks
    consensus_kl_threshold: float = 0.01
    min_cluster_size: int = 2

    # Condensation behavior
    deactivate_constituents: bool = False  # True = categorical, False = continuous flow
    # False: Constituents keep evolving (continuous renormalization flow) ← DEFAULT
    # True:  Constituents freeze when condensed (discrete emergence, for efficiency)

    # Observation likelihood for meta-agents
    lambda_obs_meta: float = 1.0  # Weight for constituent-based observations

    # Learning rates (must match Trainer for consistent behavior!)
    lr_mu_q: float = 0.1
    lr_sigma_q: float = 0.001  # Much smaller than mu!
    lr_mu_p: float = 0.1
    lr_sigma_p: float = 0.001  # Much smaller than mu!
    lr_phi: float = 0.1

    # Adaptive learning rates (1-bit discretization)
    enable_adaptive_lr: bool = False  # Adapt lr so each step changes F by ~1 bit
    target_free_energy_change_bits: float = 1.0  # Target ΔF per step (in bits)


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
                 config: Optional[HierarchicalConfig] = None,
                 participatory_monitor=None,
                 diagnostics=None):
        """
        Initialize hierarchical evolution engine.

        Args:
            system: MultiScaleSystem to evolve
            config: Evolution configuration
            participatory_monitor: Optional ParticipatoryMonitor for validation
            diagnostics: Optional ParticipatoryDiagnostics for detailed tracking
        """
        self.system = system
        self.config = config or HierarchicalConfig()
        self.monitor = participatory_monitor
        self.diagnostics = diagnostics

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
        # Phase 1: Prior Updates (Hierarchical + Self-Referential + Ouroboros Tower)
        # =====================================================================
        if self.config.enable_top_down_priors:
            update_info = self.system.update_cross_scale_priors(
                enable_tower=self.config.enable_hyperprior_tower,
                max_depth=self.config.max_hyperprior_depth,
                decay=self.config.hyperprior_decay
            )
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
            # Adaptive learning rate (1-bit discretization)
            if self.config.enable_adaptive_lr:
                adapted_lr = self._compute_adaptive_lr(
                    gradients,
                    learning_rate,
                    self.config.target_free_energy_change_bits
                )
                metrics['lr_adapted'] = adapted_lr
                metrics['lr_base'] = learning_rate
            else:
                adapted_lr = learning_rate

            n_applied = self._apply_filtered_updates(
                gradients,
                adapted_lr,
                metrics
            )
            metrics['n_updates_applied'] = n_applied

            # CRITICAL: Match Trainer post-update operations!
            # Using shared GradientApplier ensures identical behavior

            # (1) Re-enforce identical priors if lock or init_copy mode (matches Trainer)
            if hasattr(self.system, 'system_config'):
                identical_priors_mode = getattr(self.system.system_config, "identical_priors", "off")
                if identical_priors_mode in ("lock", "init_copy"):
                    GradientApplier.apply_identical_priors_lock_to_scale(self.system, scale=0)

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

        # =====================================================================
        # Participatory Monitor (if enabled)
        # =====================================================================
        if self.monitor is not None:
            self.monitor.take_snapshot(self.step_count)

        # =====================================================================
        # Diagnostics (if enabled)
        # =====================================================================
        if self.diagnostics is not None:
            self.diagnostics.record_snapshot(self.step_count)

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
                # Apply natural gradient update using shared GradientApplier
                GradientApplier.apply_updates([agent], [grad], self.config)
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

    # NOTE: _apply_single_update() method removed - now using GradientApplier.apply_updates()
    # See update_engine.py for the shared update logic used by both Trainer and
    # HierarchicalEvolutionEngine. This ensures mathematical consistency.

    def _check_and_condense_all_scales(self) -> List:
        """
        Check all scales for consensus and condense.

        IMPORTANT: With continuous flow, newly formed meta-agents should NOT
        be immediately checked for condensation in the same step. This prevents
        cascading condensations where the same underlying agents create
        meta-agents at multiple scales in one step.

        Returns:
            List of newly formed meta-agents
        """
        new_meta_agents = []

        # DIAGNOSTIC: Show detector status
        print(f"\n[Step {self.step_count}] 🔍 Consensus check (interval={self.config.consensus_check_interval})")

        # Track scales to check BEFORE any condensations happen
        # This prevents cascading: if scale 3 forms a meta at scale 4,
        # we don't immediately check scale 4 in the same step
        scales_to_check = sorted(self.system.agents.keys())

        # Check each scale (except max scale)
        for scale in scales_to_check:
            all_active = self.system.get_active_agents_at_scale(scale, free_only=False)
            free_active = self.system.get_active_agents_at_scale(scale, free_only=True)
            total_at_scale = len(self.system.agents[scale])

            # Show free/total (free = not in a meta-agent)
            if len(free_active) < len(all_active):
                print(f"  Scale {scale}: {len(free_active)} free / {len(all_active)} active / {total_at_scale} total", end="")
            else:
                print(f"  Scale {scale}: {len(all_active)}/{total_at_scale} active", end="")

            if len(free_active) < self.config.min_cluster_size:
                print(f" → SKIP (need >={self.config.min_cluster_size} free agents)")
                continue

            print(f" → checking consensus...")

            # Detect and condense
            new_agents = self.system.auto_detect_and_condense(
                scale=scale,
                kl_threshold=self.config.consensus_kl_threshold,
                min_cluster_size=self.config.min_cluster_size,
                deactivate_constituents=self.config.deactivate_constituents
            )

            if new_agents:
                print(f"    ✨ Formed {len(new_agents)} meta-agent(s)!")
            else:
                print(f"    → No consensus (KL > {self.config.consensus_kl_threshold})")

            new_meta_agents.extend(new_agents)

        if not new_meta_agents:
            print(f"  Result: No condensations this step")
        else:
            print(f"  Result: {len(new_meta_agents)} meta-agent(s) formed across {len(set(m.scale for m in new_meta_agents))} scale(s)")

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

    def _compute_adaptive_lr(self,
                            gradients: List,
                            base_lr: float,
                            target_bits: float) -> float:
        """
        Compute adaptive learning rate so ΔF ≈ target_bits.

        For gradient descent on variational free energy F:
            ΔF ≈ -lr · ||∇F||²  (to first order)

        To achieve ΔF ≈ target_bits · ln(2) [converting bits to nats]:
            lr = target_bits · ln(2) / ||∇F||²

        Args:
            gradients: List of AgentGradients
            base_lr: Base learning rate (used as max cap and fallback)
            target_bits: Target free energy change in bits

        Returns:
            Adapted learning rate
        """
        if not gradients:
            return base_lr

        # Compute total gradient norm² across all agents
        total_grad_norm_sq = 0.0

        for grad in gradients:
            # mu_q gradient contribution
            if hasattr(grad, 'mu_q') and grad.mu_q is not None:
                total_grad_norm_sq += np.sum(grad.mu_q ** 2)

            # Sigma_q gradient contribution (Frobenius norm squared)
            if hasattr(grad, 'Sigma_q') and grad.Sigma_q is not None:
                total_grad_norm_sq += np.sum(grad.Sigma_q ** 2)

            # mu_p gradient contribution
            if hasattr(grad, 'mu_p') and grad.mu_p is not None:
                total_grad_norm_sq += np.sum(grad.mu_p ** 2)

            # Sigma_p gradient contribution
            if hasattr(grad, 'Sigma_p') and grad.Sigma_p is not None:
                total_grad_norm_sq += np.sum(grad.Sigma_p ** 2)

            # phi gradient contribution
            if hasattr(grad, 'phi') and grad.phi is not None:
                total_grad_norm_sq += np.sum(grad.phi ** 2)

        # Safety check
        if total_grad_norm_sq < 1e-12:
            # Gradients are essentially zero, use base lr
            return base_lr

        # Compute adapted learning rate
        # ΔF ≈ -lr · ||∇F||², so lr = target_ΔF / ||∇F||²
        target_delta_F_nats = target_bits * np.log(2)  # Convert bits to nats
        adapted_lr = target_delta_F_nats / total_grad_norm_sq

        # Cap at 10x base_lr (safety limit) and don't go below 0.01x base_lr
        max_lr = 10 * base_lr
        min_lr = 0.01 * base_lr
        adapted_lr = np.clip(adapted_lr, min_lr, max_lr)

        return adapted_lr


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
