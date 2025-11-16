# -*- coding: utf-8 -*-
"""
Playground Experiment Runner with Support Masking
==================================================

Enhanced simulation suite with:
- Smooth support boundaries (Gaussian decay)
- Large covariance outside support
- Overlap thresholding
- Comprehensive diagnostics

Author: Chris
Date: November 2025
"""
# --- High-level run info ---
EXPERIMENT_NAME        = "_playground"
EXPERIMENT_DESCRIPTION = "Multi-agent with smooth support boundaries"
OUTPUT_DIR             = "_results"
SEED                   = 2


import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from config import AgentConfig, SystemConfig, TrainingConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem

from agent.trainer import Trainer, TrainingHistory

from geometry.geometry_base import (
    BaseManifold,
    TopologyType,
    SupportPatterns,
)

from agent.masking import (
    SupportRegionSmooth,
    SupportPatternsSmooth,
    MaskConfig,
    FieldEnforcer,
)

from data_utils.masking_diagnostics import (
    generate_masking_report,
    plot_agent_support_masks,
    plot_overlap_masks,
    plot_covariance_eigenvalues,
    plot_mean_field_norms,
    )


# =============================================================================
# USER CONFIG SECTION — EDIT THIS ONLY
# =============================================================================


# --- Base manifold / spatial geometry ---
SPATIAL_SHAPE          = ()
MANIFOLD_TOPOLOGY      = "periodic"


# --- Training loop ---
N_STEPS                = 50

# --- Agents & latent space ---
N_AGENTS               = 5        # 5 agents to see them emerge into meta-agent
K_LATENT               = 11

D_X                    = 5        #observation dimension

CONNECTION_TYPE        = 'flat'   #['flat', 'random', 'constant'] = 'flat'
USE_CONNECTION         =  False


# --- Hierarchical Emergence (NEW!) ---
ENABLE_EMERGENCE       = True     # Enable automatic meta-agent formation
CONSENSUS_THRESHOLD    = 0.05     # KL threshold for epistemic death
CONSENSUS_CHECK_INTERVAL = 5      # Check for consensus every N steps
MIN_CLUSTER_SIZE       = 2        # Minimum agents to form meta-agent
ENABLE_CROSS_SCALE_PRIORS = False  # Top-down prior propagation (DISABLED for now)
ENABLE_TIMESCALE_SEP   = False     # Timescale separation (DISABLED for now)
INFO_METRIC            = "fisher_metric"  # Information change metric


# --- Energy weights ---
LAMBDA_SELF            = 1      # Weak self-coupling (allows consensus)
LAMBDA_BELIEF_ALIGN    = 1     # STRONG belief alignment (encourages consensus)
LAMBDA_PRIOR_ALIGN     = 0     # Strong prior alignment
LAMBDA_OBS             = 0        # No observations (pure alignment dynamics)
LAMBDA_PHI             = 0     # Small gauge coupling



KAPPA_BETA             = 1     # Low temperature (sharp attention)
KAPPA_GAMMA            = 1

identical_priors = IDENTICAL_PRIORS = "init_copy"    #lock, off, init_copy


LR_MU_Q                = 0.1
LR_SIGMA_Q             = 0.001
LR_MU_P                = 0.1
LR_SIGMA_P             = 0.001
LR_PHI                 = 0.1

# --- Agent support geometry ---
SUPPORT_PATTERN        = "point"    #point   circles_2d full (for 1d)
AGENT_PLACEMENT_2D     = "center"  # "center", "random", or "grid"

# For 2D circular supports
AGENT_RADIUS           = 5  # Fixed radius (used for "center" and "grid" placement)
 

# Random placement parameters (only used if AGENT_PLACEMENT_2D = "random")
RANDOM_RADIUS_RANGE    = (4, 10)  # (min_radius, max_radius) - None uses AGENT_RADIUS for all
 


MU_SCALE        = 0.5
SIGMA_SCALE     = 1.0
PHI_SCALE       = 0.1
MEAN_SMOOTHNESS = 1                             




# Mask type: how the support boundary behaves
#   "hard"     : Step function (χ = 1 inside, 0 outside)
#   "smooth"   : Smooth transition using tanh
#   "gaussian" : Gaussian decay exp(-r²/2σ²)
MASK_TYPE                    = "gaussian"

# Overlap thresholding
OVERLAP_THRESHOLD            = 1e-1     # Ignore overlaps where χ_i·χ_j < this value
MIN_MASK_FOR_NORMAL_COV      = 1e-1  # Below this, use large Σ

# Gaussian mask parameters (only used if MASK_TYPE = "gaussian")
GAUSSIAN_SIGMA               = 1.0 / np.sqrt(-2 * np.log(OVERLAP_THRESHOLD))        # σ relative to radius (0.3 = gentle decay)
GAUSSIAN_CUTOFF_SIGMA        = 3  # Hard cutoff at N*σ

# Smooth mask parameters (only used if MASK_TYPE = "smooth")
SMOOTH_WIDTH                 = 0.1  # Transition width relative to radius

# Covariance outside support
# Step 1: Use smooth initialization
COVARIANCE_STRATEGY          = "smooth"  # Gaussian-filtered Cholesky factors
MIN_MASK_FOR_NORMAL_COV      = 1e-3  # Transition centered at χ=0.1
# - Lower value (0.01): transition very close to support edge
# - Higher value (0.5): transition deep into mask
OUTSIDE_COV_SCALE            = 1e3      # Scale for diagonal Σ outside support
USE_SMOOTH_COV_TRANSITION    = False  # Interpolate Σ at boundaries





IDENTICAL_PRIORS_SOURCE = "first"   # first or mean


# ⚡ ADD THESE NEW OBSERVATION PARAMETERS
# =============================================================================
# OBSERVATION MODEL PARAMETERS
# =============================================================================
# Moderate observation parameters


OBS_BIAS_SCALE             = 0.5              # Enough to break symmetry
OBS_NOISE_SCALE            = 1             # Measurement noise
OBS_W_SCALE                = 0.5                 # Observation matrix scale
OBS_R_SCALE                = 1.0                 # ✅ INCREASED - more tolerance
OBS_GROUND_TRUTH_AMPLITUDE = 0.5  # ✅ REDUCED - closer to init scale
OBS_GROUND_TRUTH_MODES     = 3        # Number of sinusoidal modes



LOG_EVERY  = 1


INTERVAL_OVERLAP_FRACTION =0.25   #for 1D


# --- ✨ NEW: Diagnostic options ---
RUN_INITIAL_DIAGNOSTICS = False   # Validate masking after initialization
RUN_FINAL_DIAGNOSTICS   = False   # Validate masking after training
SAVE_DIAGNOSTIC_PLOTS   = True   # Save mask/overlap/covariance plots
SAVE_DIAGNOSTIC_REPORT  = False   # Save text diagnostic report

GRID_LAYOUT             = None
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _topology_from_string(name: str) -> TopologyType:
    """Map simple string to TopologyType enum."""
    name = name.lower()
    if name == "periodic":
        return TopologyType.PERIODIC
    elif name == "open":
        return TopologyType.OPEN
    elif name == "sphere":
        return TopologyType.SPHERE
    elif name == "hyperbolic":
        return TopologyType.HYPERBOLIC
    else:
        raise ValueError(f"Unknown topology '{name}'")


def build_manifold():
    """Create BaseManifold from SPATIAL_SHAPE + topology."""
    manifold = BaseManifold(
        shape=SPATIAL_SHAPE,
        topology=_topology_from_string(MANIFOLD_TOPOLOGY),
    )
   
    return manifold



def create_mask_config() -> MaskConfig:
    """Create MaskConfig from user parameters."""
    return MaskConfig(
        mask_type=MASK_TYPE,
        smooth_width=SMOOTH_WIDTH,
        gaussian_sigma=GAUSSIAN_SIGMA,
        gaussian_cutoff_sigma=GAUSSIAN_CUTOFF_SIGMA,
        overlap_threshold=OVERLAP_THRESHOLD,
        min_mask_for_normal_cov=MIN_MASK_FOR_NORMAL_COV,
        outside_cov_scale=OUTSIDE_COV_SCALE,
        use_smooth_cov_transition=USE_SMOOTH_COV_TRANSITION,
    )





def build_supports(manifold, rng: np.random.Generator):
    """
    Build SupportRegionSmooth objects with proper masking.
    
    Returns list of smooth support regions.
    """
    ndim = manifold.ndim
    supports = []
    mask_config = create_mask_config()
    
   

    if ndim == 0:
        # Single point – everybody lives on the same point
        # (Smooth masking doesn't apply to 0D)
        # Directly create SupportRegionSmooth for 0D
        supports = [
            SupportRegionSmooth(
                mask_binary=np.array(True),  # 0D: single point, always True
                base_shape=(),
                config=mask_config
            ) for _ in range(N_AGENTS)
        ]

    elif ndim == 1:
        n_points = manifold.shape[0]
        if SUPPORT_PATTERN == "full":
            # Full support
            basic_supports = [SupportPatterns.full(manifold) for _ in range(N_AGENTS)]
        
        elif SUPPORT_PATTERN == "intervals_1d":
            # Overlapping intervals
            base_width = n_points / N_AGENTS
            overlap = int(base_width * INTERVAL_OVERLAP_FRACTION)
            
            basic_supports = []
            for i in range(N_AGENTS):
                start = int(max(0, round(i * base_width) - overlap // 2))
                end   = int(min(n_points, round((i + 1) * base_width) + overlap // 2))
                support = SupportPatterns.interval(manifold, start=start, end=end)
                basic_supports.append(support)
        else:
            raise ValueError(f"Unsupported SUPPORT_PATTERN '{SUPPORT_PATTERN}' for 1D")
        
        # Convert to smooth supports
        supports = [
            SupportRegionSmooth(
                mask_binary=s.mask,
                base_shape=s.base_shape,
                config=mask_config
            ) for s in basic_supports
        ]

    elif ndim == 2:
        H, W = manifold.shape
        if SUPPORT_PATTERN == "full":
            # Full support
            basic_supports = [SupportPatterns.full(manifold) for _ in range(N_AGENTS)]
            supports = [
                SupportRegionSmooth(
                    mask_binary=s.mask,
                    base_shape=s.base_shape,
                    config=mask_config
                ) for s in basic_supports
            ]

        elif SUPPORT_PATTERN == "circles_2d":
            # Use SupportPatternsSmooth.circle directly (already smooth!)
            
            if AGENT_PLACEMENT_2D == "center":
                # All agents at center
                center = (H // 2, W // 2)
                for _ in range(N_AGENTS):
                    support = SupportPatternsSmooth.circle(
                        manifold_shape=manifold.shape,
                        center=center,
                        radius=AGENT_RADIUS,
                        config=mask_config
                    )
                    supports.append(support)
            
            elif AGENT_PLACEMENT_2D == "random":
                # Random centers and optionally random radii
                print("  Random placement: centers from uniform distribution")
                
                if RANDOM_RADIUS_RANGE is not None:
                    min_r, max_r = RANDOM_RADIUS_RANGE
                    print(f"  Random radii: uniform in [{min_r}, {max_r}]")
                    use_random_radii = True
                else:
                    print(f"  Fixed radius: {AGENT_RADIUS}")
                    use_random_radii = False
                
                for i in range(N_AGENTS):
                    # Random center (uniform over entire grid)
                    cy = rng.uniform(0, H)
                    cx = rng.uniform(0, W)
                    
                    # Random or fixed radius
                    if use_random_radii:
                        radius = rng.uniform(min_r, max_r)
                    else:
                        radius = AGENT_RADIUS
                    
                    support = SupportPatternsSmooth.circle(
                        manifold_shape=manifold.shape,
                        center=(cy, cx),
                        radius=radius,
                        config=mask_config
                    )
                    supports.append(support)
                    
                    if i < 2:  # Show first 3 for debugging
                        print(f"    Agent {i}: center=({cy:.1f},{cx:.1f}), radius={radius:.1f}")
                
                if N_AGENTS > 2:
                    print(f"    ... ({N_AGENTS-3} more agents)")
            
            
            else:
                raise ValueError(f"Unknown AGENT_PLACEMENT_2D '{AGENT_PLACEMENT_2D}'")
        else:
            raise ValueError(f"Unsupported SUPPORT_PATTERN '{SUPPORT_PATTERN}' for 2D")
    
    else:
        # For higher dimensions, use full support
        print(f"  ndim={ndim} > 2: using full support for all agents")
        basic_supports = [SupportPatterns.full(manifold) for _ in range(N_AGENTS)]
        supports = [
            SupportRegionSmooth(
                mask_binary=s.mask,
                base_shape=s.base_shape,
                config=mask_config
            ) for s in basic_supports
        ]
    
    return supports


def build_agents(manifold, supports, rng: np.random.Generator):
    """
    Create Agent objects with proper support enforcement.
    
    CRITICAL: Agents must be initialized with supports BEFORE field initialization.
    """
   
    
    # Create mask config for agents
    mask_config = create_mask_config()
    
    # AgentConfig with mask config
    agent_cfg = AgentConfig(
        spatial_shape=SPATIAL_SHAPE,
        K=K_LATENT,
        
        mu_scale=MU_SCALE,
        sigma_scale=SIGMA_SCALE,
        phi_scale=PHI_SCALE,
        mean_smoothness_scale = MEAN_SMOOTHNESS
        
    )
    
    # Add mask config to agent config
    agent_cfg.mask_config = mask_config
  
    
    agents = []
    for i in range(N_AGENTS):
        agent = Agent(
            agent_id=i,
            config=agent_cfg,
            rng=rng,
            base_manifold=manifold
        )
        
        # ✨ CRITICAL: Attach support BEFORE field initialization
        agent.support = supports[i]
        
        # Re-initialize fields with support enforcement
        # (This uses the patched versions that enforce support)
        agent._initialize_belief_cholesky()   # NEW
        agent._initialize_prior_cholesky()    # NEW
        agent._initialize_gauge()
        
        # Update geometry
        agent.geometry.support = supports[i]
        agent.geometry.n_active = supports[i].n_active
        
        agents.append(agent)
    
    return agents




def build_system(agents, rng: np.random.Generator):
    """Create MultiAgentSystem or MultiScaleSystem with masking support."""
    print(f"\n{'='*70}")
    print("SYSTEM")
    print(f"{'='*70}")

    # ⚡ NEW: Create SystemConfig with ALL parameters (no more setattr hacks!)
    system_cfg = SystemConfig(
        # Energy weights
        lambda_self=LAMBDA_SELF,
        lambda_belief_align=LAMBDA_BELIEF_ALIGN,
        lambda_prior_align=LAMBDA_PRIOR_ALIGN,
        lambda_obs=LAMBDA_OBS,
        lambda_phi=LAMBDA_PHI,


        identical_priors = IDENTICAL_PRIORS,
        identical_priors_source = IDENTICAL_PRIORS_SOURCE,

        # Softmax temps
        kappa_beta=KAPPA_BETA,
        kappa_gamma=KAPPA_GAMMA,

        # Overlap
        overlap_threshold=OVERLAP_THRESHOLD,

        # Connection
        use_connection=USE_CONNECTION,
        connection_init_mode=CONNECTION_TYPE,

        # ⚡ NEW: Observation parameters (no more setattr!)
        D_x=D_X,
        obs_W_scale=OBS_W_SCALE,
        obs_R_scale=OBS_R_SCALE,
        obs_noise_scale=OBS_NOISE_SCALE,
        obs_bias_scale=OBS_BIAS_SCALE,
        obs_ground_truth_modes=OBS_GROUND_TRUTH_MODES,
        obs_ground_truth_amplitude=OBS_GROUND_TRUTH_AMPLITUDE,
        seed=int(rng.integers(0, 2**31)),
    )

    # Add mask config
    system_cfg.mask_config = create_mask_config()

    # ⚡ HIERARCHICAL EMERGENCE: Create MultiScaleSystem if enabled
    if ENABLE_EMERGENCE:
        from meta.emergence import MultiScaleSystem
        from math_utils.generators import generate_so3_generators

        print("  Mode: HIERARCHICAL (emergence enabled)")
        print(f"  Consensus threshold: {CONSENSUS_THRESHOLD}")
        print(f"  Min cluster size: {MIN_CLUSTER_SIZE}")

        # Create multi-scale system
        manifold = agents[0].base_manifold  # All agents share same manifold
        system = MultiScaleSystem(manifold)
        system.system_config = system_cfg

        # Add agents as base agents (scale 0)
        generators = generate_so3_generators(K_LATENT)
        for agent in agents:
            # Convert regular Agent to HierarchicalAgent at scale 0
            h_agent = system.add_base_agent(agent.config, agent_id=agent.agent_id)
            h_agent.support = agent.support
            h_agent.generators = generators
            # Copy state from original agent
            h_agent.mu_q = agent.mu_q.copy()
            h_agent.Sigma_q = agent.Sigma_q.copy()
            h_agent.mu_p = agent.mu_p.copy()
            h_agent.Sigma_p = agent.Sigma_p.copy()
            if hasattr(agent, 'gauge'):
                h_agent.gauge.phi = agent.gauge.phi.copy()

        # Apply identical priors if configured (matches MultiAgentSystem behavior)
        if system_cfg.identical_priors in ("init_copy", "lock"):
            base_agents = system.agents[0]  # Scale 0 agents
            if len(base_agents) > 0:
                # CRITICAL: Respect identical_priors_source like MultiAgentSystem does!
                if system_cfg.identical_priors_source == "mean":
                    # Average across all base agents
                    mu_p_shared = sum(a.mu_p for a in base_agents) / len(base_agents)
                    L_p_shared = sum(a.L_p for a in base_agents) / len(base_agents)
                else:
                    # Use first agent's prior (default behavior)
                    mu_p_shared = base_agents[0].mu_p.copy()
                    L_p_shared = base_agents[0].L_p.copy()

                # Apply to all base agents (set L_p, not Sigma_p, to match MultiAgentSystem!)
                for a in base_agents:
                    a.mu_p = mu_p_shared.copy()
                    a.L_p = L_p_shared.copy()
                    if hasattr(a, 'invalidate_caches'):
                        a.invalidate_caches()

        print(f"  Created MultiScaleSystem with {len(system.agents[0])} base agents")
    else:
        print("  Mode: STANDARD (no emergence)")
        # Create standard system
        system = MultiAgentSystem(agents, system_cfg)

        # ⚡ SIMPLIFIED: Just call ensure_observation_model() - no more setattr!
        if system.config.has_observations:
            system.ensure_observation_model()

    return system


def run_initial_diagnostics(system, output_dir: Path):
    """Run validation checks after system initialization."""
    if not RUN_INITIAL_DIAGNOSTICS:
        return

    
    # Save diagnostic report to file
    if SAVE_DIAGNOSTIC_REPORT:
        report_path = output_dir / "diagnostic_report_initial.txt"
        import sys
        from io import StringIO
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        
        #generate_masking_report(system, gradients=None)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Save to file with UTF-8 encoding
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(captured.getvalue())
 
    # Save diagnostic plots
    if SAVE_DIAGNOSTIC_PLOTS and system.agents[0].base_manifold.ndim == 2:
        # Support masks
        fig_path = output_dir / "diagnostic_support_masks.png"
        plot_agent_support_masks(system.agents, save_path=str(fig_path))
        
        # Overlap masks (only if there are overlaps)
        if len(system.overlap_masks) > 0:
            fig_path = output_dir / "diagnostic_overlap_masks.png"
            plot_overlap_masks(system, save_path=str(fig_path))
        else:
            print("  ⚠️  No overlaps detected - skipping overlap plot")
            print("     Consider: increasing AGENT_RADIUS or decreasing OVERLAP_THRESHOLD")
        
        # Covariance eigenvalues for first agent
        fig_path = output_dir / "diagnostic_covariance_eigenvalues_agent0.png"
        plot_covariance_eigenvalues(system.agents[0], which='q', save_path=str(fig_path))
        
        # Mean field norms for first agent
        fig_path = output_dir / "diagnostic_mean_field_norms_agent0.png"
        plot_mean_field_norms(system.agents[0], which='q', save_path=str(fig_path))
        




# =============================================================================
# Gradient System Adapter (for hierarchical training compatibility)
# =============================================================================

class _GradientSystemAdapter:
    """
    Minimal adapter to make MultiScaleSystem compatible with gradient engine.

    Provides the interface needed by compute_natural_gradients WITHOUT
    re-initializing agents (which would corrupt their state).

    CRITICAL: Must respect spatial overlaps to match standard training!
    """
    def __init__(self, agents_list, system_config):
        from math_utils.so3_utils import compute_transport
        import numpy as np

        self.agents = agents_list  # List of active agents
        self.config = system_config  # System configuration
        self.n_agents = len(agents_list)
        self._compute_transport = compute_transport

        # Compute overlap relationships once (lightweight check)
        # This ensures gradient computation matches standard training
        self._overlaps = {}
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


def run_training(system, output_dir: Path):
    """Run Trainer and save history."""
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = TrainingConfig(
        n_steps=N_STEPS,
        log_every=LOG_EVERY,
        save_history=True,
        save_checkpoints=True,
        lr_mu_q= LR_MU_Q,
        lr_sigma_q = LR_SIGMA_Q,
        lr_mu_p = LR_MU_P,
        lr_sigma_p = LR_SIGMA_P,
        lr_phi=LR_PHI,
        checkpoint_every=1,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    
    print(f"  Steps      : {training_cfg.n_steps}")
    print(f"  Log every  : {training_cfg.log_every}")
    print()

    trainer = Trainer(system, training_cfg)

    history = trainer.train()

    # Normalize history to a plain dict for saving/plotting
    if isinstance(history, TrainingHistory):
        hist_dict = {
            "step": history.steps,
            "total": history.total_energy,
            "self": history.self_energy,
            "belief_align": history.belief_align,
            "prior_align": history.prior_align,
            "observations": history.observations,
            "grad_norm_mu_q": history.grad_norm_mu_q,
            "grad_norm_Sigma_q": history.grad_norm_Sigma_q,
            "grad_norm_phi": history.grad_norm_phi,
        }
    else:
        # Backwards-compat: if trainer ever returns a dict again
        hist_dict = history



    hist_path = output_dir / "history.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    print("✓ Saved history.pkl")

    arrays = {}
    for k, v in hist_dict.items():
        try:
            arrays[k] = np.array(v)
        except Exception:
                # Skip anything that can't be turned into a clean array
            pass
    npz_path = output_dir / "history.npz"
    np.savez(npz_path, **arrays)
    print("✓ Saved history.npz")


    # Save plots
    if "step" in hist_dict and "total" in hist_dict:
        plt.figure(figsize=(10, 6))
        steps = hist_dict["step"]
            
            # Total energy
        plt.plot(steps, hist_dict["total"], linewidth=2.5, label="Total", color='black')
            
            # Components
        if "self" in hist_dict:
                plt.plot(steps, hist_dict["self"], "--", alpha=0.7, label="Self")
        if "belief_align" in hist_dict:
                plt.plot(steps, hist_dict["belief_align"], "--", alpha=0.7, label="Belief align")
        if "prior_align" in hist_dict:
                plt.plot(steps, hist_dict["prior_align"], "--", alpha=0.7, label="Prior align")
        if "observations" in hist_dict and max(hist_dict["observations"]) > 0:
                plt.plot(steps, hist_dict["observations"], "--", alpha=0.7, label="Observations")
            
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Energy", fontsize=12)
        plt.legend()
        plt.tight_layout()
            
        fig_path = output_dir / "energy_evolution.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved energy_evolution.png")


    # Save final system state
    state_path = output_dir / "final_state.pkl"
    with open(state_path, 'wb') as f:
            # Remove cache before pickling
        if hasattr(system, '_transport_cache'):
                cache_stats = system._transport_cache.get_stats()
                print(f"Temporarily removing cache for pickle: {cache_stats}")
                
                # Store original method
                original_method = system._original_compute_transport_ij
                cache = system._transport_cache
                
                # Remove cache
                system.compute_transport_ij = original_method
                del system._original_compute_transport_ij
                del system._transport_cache
            
            # Now pickle works!
        pickle.dump(system, f)
            
            # Restore cache after pickling
        if cache is not None:
                from math_utils.transport_cache import add_cache_to_system
                add_cache_to_system(system, max_size=cache.max_size)
                print("Cache restored after pickling")
    print("✓ Saved final_state.pkl")

    return history


def run_hierarchical_training(multi_scale_system, output_dir: Path):
    """
    Run training with hierarchical emergence enabled.

    Agents form meta-agents automatically through consensus detection.
    """
    from meta.hierarchical_evolution import HierarchicalEvolutionEngine, HierarchicalConfig
    from gradients.gradient_engine import compute_natural_gradients

    print(f"\n{'='*70}")
    print("HIERARCHICAL TRAINING WITH EMERGENCE")
    print(f"{'='*70}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create hierarchical config (match Trainer learning rates!)
    hier_config = HierarchicalConfig(
        enable_top_down_priors=ENABLE_CROSS_SCALE_PRIORS,
        enable_bottom_up_obs=True,
        enable_timescale_filtering=ENABLE_TIMESCALE_SEP,
        info_change_metric=INFO_METRIC,
        consensus_check_interval=CONSENSUS_CHECK_INTERVAL,
        consensus_kl_threshold=CONSENSUS_THRESHOLD,
        min_cluster_size=MIN_CLUSTER_SIZE,
        # Learning rates (match Trainer exactly!)
        lr_mu_q=LR_MU_Q,
        lr_sigma_q=LR_SIGMA_Q,
        lr_mu_p=LR_MU_P,
        lr_sigma_p=LR_SIGMA_P,
        lr_phi=LR_PHI
    )

    # Create evolution engine (detector created internally)
    engine = HierarchicalEvolutionEngine(multi_scale_system, hier_config)

    print(f"  Steps              : {N_STEPS}")
    print(f"  Consensus check    : every {CONSENSUS_CHECK_INTERVAL} steps")
    print(f"  Consensus threshold: {CONSENSUS_THRESHOLD}")
    print(f"  Timescale sep      : {ENABLE_TIMESCALE_SEP}")
    print(f"  Cross-scale priors : {ENABLE_CROSS_SCALE_PRIORS}")
    print()

    # Storage for history
    history = {
        'step': [],
        'total_energy': [],
        'n_scales': [],
        'n_active_agents': [],
        'n_condensations': [],
        'emergence_events': []
    }

    # NOTE: _GradientSystemAdapter now defined at module level (line 591)
    # so it can be imported for testing

    def compute_gradients_fn(system):
        """Wrapper to compute gradients for all active agents."""
        active_agents = system.get_all_active_agents()
        if len(active_agents) == 0:
            return []

        # Create minimal adapter that provides ONLY what gradient engine needs:
        # - system.agents (list)
        # - system.config (SystemConfig)
        # - system.n_agents (int)
        # WITHOUT re-initializing/corrupting agent state!
        temp_system = _GradientSystemAdapter(active_agents, system.system_config)

        # Compute gradients (returns List[AgentGradients] in same order)
        gradients = compute_natural_gradients(temp_system)

        # Return list directly (already in correct order)
        return gradients

    # Training loop
    print("Training with emergence enabled...")
    print("-" * 70)

    # Reuse adapter to avoid recreating overlaps
    from free_energy_clean import compute_total_free_energy

    # 🚀 CACHE: Create adapter once, reuse across steps (only update agent list)
    # DISABLED FOR DEBUGGING - recreate every step
    for step in range(N_STEPS):
        # Get active agents
        active_agents = multi_scale_system.get_all_active_agents()
        if len(active_agents) == 0:
            break

        # Create fresh adapter every step (for debugging)
        temp_system = _GradientSystemAdapter(active_agents, multi_scale_system.system_config)

        # Compute energy BEFORE updates (like Trainer does)
        energies = compute_total_free_energy(temp_system)
        total_energy = energies.total

        # Wrapper that reuses the adapter we just created
        def compute_grads_with_adapter(system):
            return compute_natural_gradients(temp_system)

        # Evolve one step with hierarchical dynamics
        metrics = engine.evolve_step(
            learning_rate=LR_MU_Q,
            compute_gradients_fn=compute_grads_with_adapter
        )

        # Convert dict metrics to scalars
        n_scales = len(metrics.get('n_active', {}))
        total_active = sum(metrics.get('n_active', {}).values())

        # Record metrics
        history['step'].append(step)
        history['total_energy'].append(total_energy)
        history['n_scales'].append(n_scales)
        history['n_active_agents'].append(total_active)
        history['n_condensations'].append(metrics.get('n_condensations', 0))

        # Check for emergence events
        if metrics.get('n_condensations', 0) > 0:
            event = {
                'step': step,
                'n_condensations': metrics['n_condensations'],
                'n_scales': n_scales
            }
            history['emergence_events'].append(event)

            print(f"\n🌟 EMERGENCE at step {step}!")
            print(f"   {metrics['n_condensations']} new meta-agent(s) formed")
            print(f"   Total scales: {n_scales}")
            print(f"   Active agents: {total_active}")

        # Periodic logging
        if step % LOG_EVERY == 0:
            print(f"Step {step:4d} | "
                  f"Energy: {total_energy:.4f} | "
                  f"Scales: {n_scales} | "
                  f"Active: {total_active}")

    print("-" * 70)
    print("✓ Training complete")

    # Save history
    hist_path = output_dir / "hierarchical_history.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    print("✓ Saved hierarchical_history.pkl")

    # Save as npz
    arrays = {k: np.array(v) for k, v in history.items() if k != 'emergence_events'}
    npz_path = output_dir / "hierarchical_history.npz"
    np.savez(npz_path, **arrays)
    print("✓ Saved hierarchical_history.npz")

    # Plot emergence
    if history['step']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Energy evolution with emergence events
        ax1.plot(history['step'], history['total_energy'], 'b-', linewidth=2, label='Energy')
        for event in history['emergence_events']:
            ax1.axvline(x=event['step'], color='red', alpha=0.3, linestyle='--', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Total Energy')
        ax1.set_title('Energy Evolution (red lines = emergence events)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Hierarchical structure evolution
        ax2.plot(history['step'], history['n_scales'], 'g-', linewidth=2, marker='o', label='# Scales')
        ax2.plot(history['step'], history['n_active_agents'], 'b-', linewidth=2, marker='s', label='# Active Agents')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Count')
        ax2.set_title('Hierarchical Structure Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / "emergence_evolution.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved emergence_evolution.png")

    # Print final summary
    print(f"\n{'='*70}")
    print("EMERGENCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total emergence events: {len(history['emergence_events'])}")
    print(f"Final # scales: {history['n_scales'][-1] if history['n_scales'] else 1}")
    print(f"Final # active agents: {history['n_active_agents'][-1] if history['n_active_agents'] else 0}")
    print(f"{'='*70}\n")

    # Print system summary
    from meta.emergence import analyze_hierarchical_structure
    structure = analyze_hierarchical_structure(multi_scale_system)
    print("\nFinal Hierarchical Structure:")
    print(f"  Max scale: {structure['n_scales'] - 1}")
    for scale in range(structure['n_scales']):
        n_total = structure['agents_per_scale'].get(scale, 0)
        n_active = structure['active_per_scale'].get(scale, 0)
        print(f"  Scale {scale}: {n_active}/{n_total} active agents")

    return history


def run_final_diagnostics(system, output_dir: Path):
    """Run validation checks after training."""
    if not RUN_FINAL_DIAGNOSTICS:
        return
    
    print(f"\n{'='*70}")
    print("FINAL DIAGNOSTICS")
    print(f"{'='*70}")
    from gradients.gradient_engine import compute_natural_gradients
    
    
    # Compute final gradients for validation
    gradients = compute_natural_gradients(system)
    
    # Generate full diagnostic report
    generate_masking_report(system, gradients=gradients)
    
    # Save diagnostic report to file
    if SAVE_DIAGNOSTIC_REPORT:
        report_path = output_dir / "diagnostic_report_final.txt"
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        
        generate_masking_report(system, gradients=gradients)
        
        sys.stdout = old_stdout
        
        # Save with UTF-8 encoding
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(captured.getvalue())
        
        print(f"✓ Saved final diagnostic report: {report_path}")
    
    # Save final diagnostic plots
    if SAVE_DIAGNOSTIC_PLOTS and system.agents[0].base_manifold.ndim == 2:
        # Final covariance eigenvalues
        fig_path = output_dir / "diagnostic_covariance_eigenvalues_final_agent0.png"
        plot_covariance_eigenvalues(system.agents[0], which='q', save_path=str(fig_path))
        
        # Final mean field norms
        fig_path = output_dir / "diagnostic_mean_field_norms_final_agent0.png"
        plot_mean_field_norms(system.agents[0], which='q', save_path=str(fig_path))
        
        # Final overlaps (only if there are any)
        if len(system.overlap_masks) > 0:
            fig_path = output_dir / "diagnostic_overlap_masks_final.png"
            plot_overlap_masks(system, save_path=str(fig_path))
        
        print("✓ Saved final diagnostic plots")







def save_config_file(output_dir: Path):
    """Save configuration snapshot for reproducibility."""
    
    cfg_path = output_dir / "config_simulation.txt"
    with open(cfg_path, "w") as f:
        f.write("# Simulation Configuration\n")
        f.write(f"# {'='*60}\n\n")
        
        # ... existing sections ...
        
        f.write("[Energy]\n")
        f.write(f"LAMBDA_SELF          = {LAMBDA_SELF}\n")
        f.write(f"LAMBDA_BELIEF_ALIGN  = {LAMBDA_BELIEF_ALIGN}\n")
        f.write(f"LAMBDA_PRIOR_ALIGN   = {LAMBDA_PRIOR_ALIGN}\n")
        f.write(f"LAMBDA_OBS           = {LAMBDA_OBS}\n")
        
        f.write(f"KAPPA_BETA           = {KAPPA_BETA}\n")
        f.write(f"KAPPA_GAMMA          = {KAPPA_GAMMA}\n\n")
        
        # ⚡ ADD THIS NEW SECTION
        f.write("[Observations]\n")
        f.write(f"D_X                  = {D_X}\n")
        f.write(f"OBS_BIAS_SCALE       = {OBS_BIAS_SCALE}\n")
        f.write(f"OBS_NOISE_SCALE      = {OBS_NOISE_SCALE}\n")
        f.write(f"OBS_W_SCALE          = {OBS_W_SCALE}\n")
        f.write(f"OBS_R_SCALE          = {OBS_R_SCALE}\n")
        f.write(f"OBS_GROUND_TRUTH_MODES = {OBS_GROUND_TRUTH_MODES}\n")
        f.write(f"OBS_GROUND_TRUTH_AMPLITUDE = {OBS_GROUND_TRUTH_AMPLITUDE}\n\n")
        f.write("# Simulation Configuration\n")
        f.write(f"# {'='*60}\n\n")
        
        f.write("[Experiment]\n")
        f.write(f"EXPERIMENT_NAME      = {EXPERIMENT_NAME}\n")
        f.write(f"EXPERIMENT_DESC      = {EXPERIMENT_DESCRIPTION}\n")
        f.write(f"SEED                 = {SEED}\n\n")
        
        f.write("[Manifold]\n")
        f.write(f"SPATIAL_SHAPE        = {SPATIAL_SHAPE}\n")
        f.write(f"MANIFOLD_TOPOLOGY    = {MANIFOLD_TOPOLOGY}\n\n")
        
        f.write("[Agents]\n")
        f.write(f"N_AGENTS             = {N_AGENTS}\n")
        f.write(f"K_LATENT             = {K_LATENT}\n")
        
        f.write(f"MU_SCALE             = {MU_SCALE}\n")
        f.write(f"SIGMA_SCALE          = {SIGMA_SCALE}\n")
        f.write(f"PHI_SCALE            = {PHI_SCALE}\n\n")
        
        f.write("[Support]\n")
        f.write(f"SUPPORT_PATTERN      = {SUPPORT_PATTERN}\n")
        if len(SPATIAL_SHAPE) == 2 and SUPPORT_PATTERN == 'circles_2d':
            f.write(f"AGENT_RADIUS         = {AGENT_RADIUS}\n")
            f.write(f"AGENT_PLACEMENT_2D   = {AGENT_PLACEMENT_2D}\n")
            if AGENT_PLACEMENT_2D == "random":
                f.write(f"RANDOM_RADIUS_RANGE  = {RANDOM_RADIUS_RANGE}\n")
            
                
        f.write("\n")
        
        f.write("[Masking]\n")
        f.write(f"MASK_TYPE            = {MASK_TYPE}\n")
        f.write(f"GAUSSIAN_SIGMA       = {GAUSSIAN_SIGMA}\n")
        f.write(f"GAUSSIAN_CUTOFF_SIGMA = {GAUSSIAN_CUTOFF_SIGMA}\n")
        f.write(f"SMOOTH_WIDTH         = {SMOOTH_WIDTH}\n")
        f.write(f"OVERLAP_THRESHOLD    = {OVERLAP_THRESHOLD}\n")
        f.write(f"MIN_MASK_FOR_NORMAL_COV = {MIN_MASK_FOR_NORMAL_COV}\n")
        f.write(f"OUTSIDE_COV_SCALE    = {OUTSIDE_COV_SCALE}\n")
        f.write(f"USE_SMOOTH_COV_TRANSITION = {USE_SMOOTH_COV_TRANSITION}\n\n")
        
        f.write("[Energy]\n")
        f.write(f"LAMBDA_SELF          = {LAMBDA_SELF}\n")
        f.write(f"LAMBDA_BELIEF_ALIGN  = {LAMBDA_BELIEF_ALIGN}\n")
        f.write(f"LAMBDA_PRIOR_ALIGN   = {LAMBDA_PRIOR_ALIGN}\n")
        f.write(f"LAMBDA_OBS           = {LAMBDA_OBS}\n")
        
        f.write(f"KAPPA_BETA           = {KAPPA_BETA}\n")
        f.write(f"KAPPA_GAMMA          = {KAPPA_GAMMA}\n\n")
        
        f.write("[Training]\n")
        f.write(f"N_STEPS              = {N_STEPS}\n")
        f.write(f"LOG_EVERY            = {LOG_EVERY}\n")
        f.write(f"LR_MU_Q               = {LR_MU_Q}\n")
        f.write(f"LR_SIGMA_Q             = {LR_SIGMA_Q}\n")
        f.write(f"LR_MU_P                = {LR_MU_P}\n")
        f.write(f"LR_SIGMA_P             = {LR_SIGMA_P}\n")
        f.write(f"LR_PHI               = {LR_PHI}\n\n")
        
        f.write("[Diagnostics]\n")
        f.write(f"RUN_INITIAL_DIAGNOSTICS = {RUN_INITIAL_DIAGNOSTICS}\n")
        f.write(f"RUN_FINAL_DIAGNOSTICS   = {RUN_FINAL_DIAGNOSTICS}\n")
        f.write(f"SAVE_DIAGNOSTIC_PLOTS   = {SAVE_DIAGNOSTIC_PLOTS}\n")
        f.write(f"SAVE_DIAGNOSTIC_REPORT  = {SAVE_DIAGNOSTIC_REPORT}\n")
    
    print(f"✓ Saved config snapshot: {cfg_path}")





# =============================================================================
# MAIN
# =============================================================================

def main():
    # Set global seed
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    output_dir = Path(OUTPUT_DIR) / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("\n" + "=" * 70)
    print("MULTI-AGENT SIMULATION WITH SUPPORT MASKING")
    print("=" * 70)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Output     : {output_dir}")
    print("=" * 70)

    # Build components
    manifold = build_manifold()
    supports = build_supports(manifold, rng)
    agents   = build_agents(manifold, supports, rng)
    system   = build_system(agents, rng)
    
    # Initial diagnostics
    if not ENABLE_EMERGENCE:  # Skip for hierarchical system (different structure)
        run_initial_diagnostics(system, output_dir)

    # Save config
    save_config_file(output_dir)

    # Train (hierarchical or standard)
    if ENABLE_EMERGENCE:
        history = run_hierarchical_training(system, output_dir)
    else:
        history = run_training(system, output_dir)

    # Final diagnostics
    if not ENABLE_EMERGENCE:  # Skip for hierarchical system (different structure)
        run_final_diagnostics(system, output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("✓ SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")

    if ENABLE_EMERGENCE:
        # Hierarchical history is a dict
        if history['total_energy']:
            print(f"Final energy: {history['total_energy'][-1]:.4f}")
            print(f"Emergence events: {len(history['emergence_events'])}")
            print(f"Final scales: {history['n_scales'][-1]}")
    else:
        # Standard history is TrainingHistory object
        if hasattr(history, 'total_energy') and history.total_energy:
            print(f"Final energy: {history.total_energy[-1]:.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()