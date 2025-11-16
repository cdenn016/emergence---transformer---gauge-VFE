"""
Gauge Fields on SO(3) Principal Bundle
======================================

Representation and geometry of gauge fields φ(c) for active inference agents.

Mathematical Framework:
----------------------
Each agent carries a gauge field φ: C → so(3) over its support region.

**Axis-Angle Representation:**
    φ(c) ∈ ℝ³ where ||φ|| encodes rotation angle, φ/||φ|| encodes axis
    
**Lie Group Element:**
    g(c) = exp(φ(c)) ∈ SO(3)
    
**Transport Operator:**
    Ω_ij(c) = g_i(c) · g_j(c)^{-1} = exp(φ_i(c)) · exp(-φ_j(c))

**Principal Ball:**
    Valid φ must satisfy ||φ(c)|| < π - margin to avoid branch cuts

**Natural Gradient:**
    Updates on φ must respect SO(3) geometry via right-Jacobian J(φ)

Author: Clean Rebuild
Date: November 2025
"""

import numpy as np
from typing import Optional, Tuple, Literal

# =============================================================================
# Gauge Field Container
# =============================================================================

class GaugeField:
    """
    Container for agent's gauge field φ(c) over spatial support.
    
    Attributes:
        phi: Axis-angle field, shape (*S, 3)
        support_shape: Spatial dimensions tuple
        K: Latent dimension
    
    Examples:
        >>> # 1D chain
        >>> field_1d = GaugeField.zeros(shape=(100,), K=3)
        >>> field_1d.phi.shape
        (100, 3)
        
        >>> # 2D grid
        >>> field_2d = GaugeField.zeros(shape=(32, 32), K=5)
        >>> field_2d.phi.shape
        (32, 32, 3)
    """
    
    def __init__(
        self,
        phi: np.ndarray,
        K: int,
        *,
        validate: bool = True,
        margin: float = 1e-2,
    ):
        """
        Initialize gauge field from axis-angle array.
        
        Args:
            phi: Axis-angle field, shape (*S, 3)
            K: Latent dimension
            validate: If True, check principal ball constraint
            margin: Safety margin from branch cut (π - margin)
        """
        self.phi = np.asarray(phi, dtype=np.float32, order='C')
        self.K = int(K)
        
        if self.phi.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3 (so(3)), got {self.phi.shape[-1]}")
        
        self.support_shape = self.phi.shape[:-1]
        
         

    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], K: int) -> 'GaugeField':
        """Create identity gauge field (φ = 0 everywhere)."""
        phi = np.zeros(shape + (3,), dtype=np.float32)
        return cls(phi, K, validate=False)
    
    @classmethod
    def random(
        cls,
        shape: Tuple[int, ...],
        K: int,
        *,
        scale: float = 0.5,
        seed: Optional[int] = None,
    ) -> 'GaugeField':
        """
        Create random gauge field with controlled magnitude.
        
        Args:
            shape: Spatial shape
            K: Latent dimension
            scale: Maximum rotation angle
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        phi = np.random.randn(*shape, 3).astype(np.float32)
        norms = np.linalg.norm(phi, axis=-1, keepdims=True)
        phi = phi / np.maximum(norms, 1e-8) * (scale * np.random.rand(*shape, 1))
        
        return cls(phi, K, validate=True, margin=0.1)
    
    def copy(self) -> 'GaugeField':
        """Create a copy of this gauge field."""
        return GaugeField(self.phi.copy(), self.K, validate=False)
    
    def __repr__(self) -> str:
        return (
            f"GaugeField(shape={self.support_shape}, K={self.K}, "
            f"||φ||_max={np.max(np.linalg.norm(self.phi, axis=-1)):.4f})"
        )




# =============================================================================
# Principal Ball Retraction
# =============================================================================

def retract_to_principal_ball(
    phi: np.ndarray,
    *,
    margin: float = 1e-2,
    mode: Literal['mod2pi', 'project'] = 'mod2pi',
) -> np.ndarray:
    """
    Retract gauge field to principal ball ||φ|| < π - margin.
    
    Two modes:
        - 'mod2pi': Wrap to [0, 2π) with antipodal flip
        - 'project': Radial projection (simpler but discontinuous)
    
    Args:
        phi: Axis-angle field, shape (*S, 3)
        margin: Safety margin from branch cut
        mode: Retraction method
    
    Returns:
        phi_retracted: Shape (*S, 3), satisfies ||φ|| < π - margin
    
    Examples:
        >>> phi = np.array([[3.5, 0.0, 0.0]])  # θ ≈ 3.5 > π
        >>> phi_ret = retract_to_principal_ball(phi)
        >>> np.linalg.norm(phi_ret) < np.pi
        True
    """
    phi = np.asarray(phi, dtype=np.float64)
    
    if phi.shape[-1] != 3:
        raise ValueError(f"Expected shape (*S, 3), got {phi.shape}")
    
    # Compute norms
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)  # (*S, 1)
    
    # ✅ FIX: Add epsilon BEFORE division to avoid 0/0
    theta_safe = np.maximum(theta, 1e-12)  # Never actually zero
    
    # Compute normalized axis (safe now)
    axis = phi / theta_safe  # No warning!
    
    # Handle true zero-norm case: set to arbitrary unit vector
    # (These points won't matter since theta=0 means identity anyway)
    is_zero = (theta < 1e-12)[..., 0]
    if np.any(is_zero):
        axis[is_zero] = np.array([1.0, 0.0, 0.0])
    
    # Threshold
    r_max = float(np.pi - margin)
    
    if mode == 'mod2pi':
        # ========== Modulo 2π with antipodal flip ==========
        two_pi = 2.0 * np.pi
        
        # Wrap to [0, 2π)
        theta_wrapped = np.remainder(theta[..., 0], two_pi)
        
        # Flip axis if θ > π (antipodal symmetry)
        flip = theta_wrapped > np.pi
        theta_final = np.where(flip, two_pi - theta_wrapped, theta_wrapped)
        axis_final = np.where(flip[..., None], -axis, axis)
        
        # Clamp to safety margin
        theta_final = np.minimum(theta_final, r_max)
        
        phi_new = axis_final * theta_final[..., None]
    
    elif mode == 'project':
        # ========== Radial projection ==========
        # Only scale down if exceeds limit
        exceeds = theta[..., 0] > r_max
        scale = np.ones_like(theta[..., 0])
        scale[exceeds] = r_max / theta_safe[exceeds, 0]
        
        phi_new = phi * scale[..., None]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return phi_new.astype(np.float32, copy=False)





def grad_kl_wrt_phi_i(
    mu_i, sigma_i,
    mu_j, sigma_j,                 
    phi_i, phi_j,                  
    Omega_ij,
    generators,
    exp_phi_i,
    exp_phi_j,
    mu_j_t,                        
    sigma_j_t_inv,                 
    eps=1e-8,
    exp_neg_phi_j=None,
    Q_all=None,
):
        
    # ---- cast to f64 for internal math ----
    mu_i   = np.asarray(mu_i,   np.float64)
    mu_j_t = np.asarray(mu_j_t, np.float64)

    Sigma_i = sanitize_sigma(np.asarray(sigma_i, np.float64), eps=eps)
   
    # Q_all → (..., a, K, K)
    if Q_all is None:
        Q_all = d_exp_exact(np.asarray(phi_i, np.float64), generators)
    
    Q = np.stack([np.asarray(x, np.float64) for x in Q_all], axis=-3)

    # exp(-phi_j) broadcast to Q lead dims
    if exp_neg_phi_j is None:
        exp_neg_phi_j = safe_omega_inv(np.asarray(exp_phi_j, np.float64))
   
    # Ω and Ω^{-1}
    Om   = np.asarray(Omega_ij, np.float64)
    Oinv = safe_omega_inv(Om).astype(np.float64, copy=False)

    # Sanity on provided precision
    Sj_inv = np.asarray(sigma_j_t_inv, np.float64)
    
    Sj_inv = 0.5 * (Sj_inv + np.swapaxes(Sj_inv, -1, -2))
    
    # Core pieces
    delta_mu = (mu_i - mu_j_t)                     # (..., K)
    mu_j_src = np.einsum("...ij,...j->...i", Oinv, mu_j_t, optimize=True)  # (..., K)

    # dΩ/dφ_i^a = Q[a] @ e^{-φ_j}
    Q = np.einsum("...aik,...kj->...aij", Q, exp_neg_phi_j, optimize=True)
    Q_T = np.swapaxes(Q, -1, -2)
   
    # dΩ Ω^{-1}
    Q_tilde   = np.einsum("...aik,...kj->...aij", Q, Oinv, optimize=True)  
    Qtilde_T  = np.swapaxes(Q_tilde, -1, -2)
    
    # ---- precision (trace) term:  -1/2 * ⟨ A, Q̃ Σ_i + Σ_i Q̃^T ⟩
    t1 = np.einsum("...ij,...ajk,...ki->...a", Sj_inv, Q_tilde,  Sigma_i, optimize=True)
    t2 = np.einsum("...aij,...jk, ...ki->...a", Qtilde_T, Sj_inv, Sigma_i, optimize=True)
    trace_term = -0.5 * (t1 + t2)
    
    # mean term
    tmp1 = np.einsum("...aij,...j->...ai", Q_T,    delta_mu, optimize=True)
    tmp2 = np.einsum("...ij,...aj->...ai", Sj_inv, tmp1,     optimize=True)
    mean_term = -np.einsum("...i,...ai->...a", mu_j_src, tmp2, optimize=True)
    
    # ---- Mahalanobis (through ∂A) term: +1/2 * Δμ^T ( A Q̃ + Q̃^T A ) Δμ
    v1 = np.einsum("...ij,...j->...i", Sj_inv, delta_mu, optimize=True)          # A Δμ
    part1 = np.einsum("...aij,...j->...ai", Qtilde_T, v1, optimize=True)         # Q̃^T A Δμ
    v2    = np.einsum("...aij,...j->...ai", Q_tilde,  delta_mu, optimize=True)   # Q̃ Δμ
    part2 = np.einsum("...ij,...aj->...ai", Sj_inv,   v2, optimize=True)         # A Q̃ Δμ
    mahal_term = 0.5 * np.einsum("...i,...ai->...a", delta_mu, part1 + part2, optimize=True)

    grad = (trace_term + mean_term + mahal_term)

    return grad



def grad_kl_wrt_phi_j(
    mu_i, sigma_i,
    mu_j, sigma_j,                  # not used directly; kept for parity
    phi_i, phi_j,                   # not used directly here; we use exp(phi_*)
    Omega_ij,
    generators,
    exp_phi_i,
    exp_phi_j,
    mu_j_t,                         # μ_j' = Ω_ij μ_j
    sigma_j_t_inv,                  # Σ_j'^(-1) (already pushed)
    eps=1e-8,
    exp_neg_phi_j=None,
    R_all=None,                     # d exp(φ_j)/d φ_j^b (list/array of matrices); if None, we compute
):
    """
    ∂/∂φ_j KL(q_i || Ω_ij q_j). Mirrors grad_kl_wrt_phi_i but with:
      dΩ/dφ_j^b = - Ω  * ( R_j[b] @ e^{-φ_j} ).
    We work with U := dΩ and Ũ := (dΩ) Ω^{-1} = - Ω R̃ Ω^{-1}.
    """
    import numpy as np
    mu_i   = np.asarray(mu_i,   np.float64)
    mu_j_t = np.asarray(mu_j_t, np.float64)

    # sanitize Σ_i only once here
    Sigma_i = sanitize_sigma(np.asarray(sigma_i, np.float64), eps=eps)

    Om   = np.asarray(Omega_ij, np.float64)
    Oinv = safe_omega_inv(Om).astype(np.float64, copy=False)

    Sj_inv = np.asarray(sigma_j_t_inv, np.float64)
    Sj_inv = 0.5 * (Sj_inv + np.swapaxes(Sj_inv, -1, -2))

    # Δμ and μ_j in source coords
    delta_mu = (mu_i - mu_j_t)                                     # (..., K)
    mu_j_src = np.einsum("...ij,...j->...i", Oinv, mu_j_t, optimize=True)

    # exp(-φ_j)
    if exp_neg_phi_j is None:
        exp_neg_phi_j = safe_omega_inv(np.asarray(exp_phi_j, np.float64))

    # R_all: d exp(φ_j)/d φ_j^b  → stack as (..., b, K, K)
    if R_all is None:
        R_all = d_exp_exact(np.asarray(phi_j, np.float64), generators)
    R = np.stack([np.asarray(x, np.float64) for x in R_all], axis=-3)   # (..., b, K, K)

    # R̃_b = R_b @ e^{-φ_j}
    R_tilde = np.einsum("...bik,...kj->...bij", R, exp_neg_phi_j, optimize=True)  # (..., b, K, K)

    # U := dΩ = - Ω R̃ ;   U^T;  and Ũ := (dΩ) Ω^{-1} = - Ω R̃ Ω^{-1}
    U       = -np.einsum("...ik,...bkj->...bij", Om, R_tilde, optimize=True)      # (..., b, K, K)
    U_T     = np.swapaxes(U, -1, -2)
    U_tilde = -np.einsum("...ik,...bkj,...jl->...bil", Om, R_tilde, Oinv, optimize=True)
    Utilde_T= np.swapaxes(U_tilde, -1, -2)

    # -------- precision (trace) term:  -1/2 ⟨ A, Ũ Σ_i + Σ_i Ũ^T ⟩
    t1 = np.einsum("...ij,...bij,...jk->...b", Sj_inv, U_tilde, Sigma_i, optimize=True)
    t2 = np.einsum("...bij,...jk,...ki->...b", Utilde_T, Sj_inv, Sigma_i, optimize=True)
    trace_term = -0.5 * (t1 + t2)

    # -------- mean term (through dμ' = U μ_j_src)
    # tmp1_bi = U^T_bij Δμ_j'^j
    tmp1 = np.einsum("...bij,...j->...bi", U_T, delta_mu, optimize=True)
    # tmp2_bi = A_{ik} tmp1_bk
    tmp2 = np.einsum("...ij,...bj->...bi", Sj_inv, tmp1, optimize=True)
    mean_term = -np.einsum("...i,...bi->...b", mu_j_src, tmp2, optimize=True)

    # -------- Mahalanobis term: +1/2 Δμ^T ( A Ũ + Ũ^T A ) Δμ
    v1 = np.einsum("...ij,...j->...i", Sj_inv, delta_mu, optimize=True)            # A Δμ
    part1 = np.einsum("...bij,...j->...bi", Utilde_T, v1, optimize=True)           # Ũ^T A Δμ
    v2    = np.einsum("...bij,...j->...bi", U_tilde,  delta_mu, optimize=True)     # Ũ Δμ
    part2 = np.einsum("...ij,...bj->...bi", Sj_inv,   v2, optimize=True)           # A Ũ Δμ
    mahal_term = 0.5 * np.einsum("...i,...bi->...b", delta_mu, part1 + part2, optimize=True)

    grad = (trace_term + mean_term + mahal_term)    # (..., b)
    return grad


