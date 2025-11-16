# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:38:52 2025

@author: chris and christine
"""

"""
Publication Proof-of-Principle Configuration
=============================================

Minimal viable experiment for academic publication demonstrating:
1. Variational FFN works - inference via gradient descent performs comparably to learned MLP
2. Architecture is trainable - converges to reasonable performance
3. Theoretical framework is sound - gauge-invariant inference + active inference
4. Novel contribution - interpretable, principled transformers

Configuration for character-level language modeling on WikiText-2:
    - N=32 (long enough for patterns, short enough to train quickly)
    - vocab_size ≤ 256 (ASCII characters)
    - 1000-2000 steps (full convergence)
    - Three FFN modes for ablation study

Expected Results:
    - Random baseline: PPL = vocab_size (~100)
    - Learned FFN: PPL = 15-30
    - Variational_approx: PPL = 18-35 (within 15-20%)
    - Variational_full: PPL = 16-32 (within 10-15%)

Author: Designed for minimal publishable claim
Date: November 2025
"""

# =============================================================================
# Publication Proof-of-Principle Configuration
# =============================================================================

PUBLICATION_CONFIG = {
    # Model architecture (minimal but meaningful)
    'vocab_size': 256,        # Will be set by actual char vocab (typically ~100 chars in WikiText-2)
    'embed_dim': 11,          # K=11 (ODD - required for SO(3) irreps!)
    'n_layers': 3,            # Depth for non-trivial learning
    'hidden_dim': 44,         # 4×embed_dim
    'max_seq_len': 32,        # N=32 (key: enough for patterns!)

    # Gauge transformer parameters
    'kappa_beta': 1.0,
    'epsilon': 1e-8,
    'pos_encoding_mode': 'learned',
    'evolve_sigma': False,  # Auto-enabled for variational_gradient_engine mode
    'evolve_phi': False,    # Keep simple for publication
    'tie_embeddings': True,

    # Attention pattern (full for small N=32)
    'attention_pattern': 'full',
    'attention_window': 32,
    'attention_global_tokens': 0,

    # Variational FFN parameters (will be varied in ablation study)
    'ffn_mode': 'variational_gradient_engine',        # Default: will be overridden in ablation
    'ffn_alpha': 0.2,            # Prior weight (balanced)
    'ffn_tau_eff': 1.0,           # Temperature
    'ffn_kappa': 1.0,             # Softmax temperature
    'ffn_n_iterations': 1,        # Single inference step per forward pass
    'ffn_learnable_lr': True,     # Learn step size for variational descent

    # Sparse variational inference (full for N=32)
    'ffn_pattern': 'full',
    'ffn_window': 32,

    # Training (optimized for convergence)
    'batch_size': 8,             # Larger batches for stability
    'max_steps': 100,             # Adjusted for ~2 hour runtime

    # Natural gradient learning rates (balanced for fast convergence)
    'mu_lr': 0.05,                # Belief means
    'sigma_lr': 0.003,            # Belief covariances
    'phi_lr': 0.05,               # Gauge transformations
    'ffn_lr': 0.05,              # FFN parameters (if learned mode)

    'warmup_steps': 5,          # Gradual warmup for stability

    # Free energy weights (balanced gauge-theoretic learning)
    'alpha': 0.2,                # Self-consistency regularization
    'beta': 1,                  # Belief alignment (key gauge term)
    'lambda_gamma': 0,          # Model alignment (disabled)
    'kappa_gamma': 1.0,           # Temperature for γ_ij coupling

    # Regularization (light for small model)
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging (frequent for publication plots)
    'log_interval': 1,
    'eval_interval': 5,          # Eval every 50 steps
    'checkpoint_interval': 20,
    'patience': 3,               # Early stopping patience

    # Irrep structure (for K=11)
    'irrep_spec': [
        ('ℓ0', 5, 1),    # 5 dimensions (scalars)
        ('ℓ1', 2, 3),    # 6 dimensions (vectors)
        # Total: 5 + 6 = 11 ✓
    ],
}

