# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 09:53:21 2025

@author: chris and christine
"""
import numpy as np
from typing import Optional


def compute_observation_gradients(system, agent):
    """
    ∂/∂θ of -∫ χ_i χ_obs E_q[log p(o|x)].
    For p(o|x) = N(o | Cx, R):
        ∂/∂μ_q :  C^T R^{-1} (o - Cμ_q)   with a minus sign from the energy
        ∂/∂Σ_q : -½ C^T R^{-1} C          (constant wrt space)
    Weighted pointwise by λ_obs · χ_i(c) · χ_obs(c).
    """
    spatial_shape = agent.base_manifold.shape
    K = agent.K
    grad_mu_q  = np.zeros((*spatial_shape, K), dtype=np.float32)
    grad_Sigma_q = np.zeros((*spatial_shape, K, K), dtype=np.float32)

    if not getattr(agent, "has_observation_field", False) or not agent.has_observation_field:
        from gradients.gradient_engine import AgentGradients  # keep your import style
        return AgentGradients(
            grad_mu_q=grad_mu_q,
            grad_Sigma_q=grad_Sigma_q,
            grad_mu_p=np.zeros_like(grad_mu_q),
            grad_Sigma_p=np.zeros_like(grad_Sigma_q),
            grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32),
        )

    C = agent.C_obs.astype(np.float64)
    R = agent.R_obs.astype(np.float64)
    R_inv = np.linalg.inv(R + 1e-8 * np.eye(R.shape[0], dtype=R.dtype))
    lam = float(getattr(system.config, "lambda_obs", 1.0))

    chi_i = np.asarray(agent.support.chi_weight, dtype=np.float64)    # (*S,) or scalar
    chi_obs = np.asarray(agent.obs_mask, dtype=np.float64)            # (*S,) or scalar
    w = lam * (chi_i * chi_obs)                                       # weights

    mu_q = np.asarray(agent.mu_q, dtype=np.float64)                   # (*S, K) or (K,)
    o    = np.asarray(agent.obs_values, dtype=np.float64)             # (*S, D) or (D,)

    CT_Rinv = C.T @ R_inv
    CT_Rinv_C = CT_Rinv @ C   # (K, K) constant

    if mu_q.ndim == 1:
        innov = o - C @ mu_q
        gmu = -(CT_Rinv @ innov) * w                                  # (K,)
        gS  = -0.5 * CT_Rinv_C * w                                    # (K,K) scaled by scalar w
        grad_mu_q = gmu.astype(np.float32)
        grad_Sigma_q = gS.astype(np.float32)
    else:
        innov = o - np.einsum('dk,...k->...d', C, mu_q, optimize=True)   # (*S, D)
        gmu = -np.einsum('kd,de,...e->...k', C.T, R_inv, innov, optimize=True)  # (*S, K)
        gS  = -0.5 * CT_Rinv_C                                         # (K,K)
        # apply spatial weights
        grad_mu_q = (w[..., None] * gmu).astype(np.float32)
        grad_Sigma_q = (w[..., None, None] * gS).astype(np.float32)

    from gradients.gradient_engine import AgentGradients
    return AgentGradients(
        grad_mu_q=grad_mu_q,
        grad_Sigma_q=grad_Sigma_q,
        grad_mu_p=np.zeros_like(grad_mu_q),
        grad_Sigma_p=np.zeros_like(grad_Sigma_q),
        grad_phi=np.zeros((*spatial_shape, 3), dtype=np.float32),
    )


def gaussian_subset_mask(base_shape, *, support_chi, center=None,
                         frac_radius=0.5, cutoff_sigma=3.0) -> np.ndarray:
    """
    Build χ_obs(c) as a Gaussian blob subset of the agent's support χ_i(c).

    Args:
        base_shape: tuple (*S,)
        support_chi: χ_i(c) ∈ [0,1], shape (*S,) or scalar for 0D
        center: optional float coordinates in each dim; if None use χ_i-weighted centroid
        frac_radius: subset radius as a fraction of agent's effective radius
        cutoff_sigma: hard cutoff at Nσ

    Returns:
        mask_obs: χ_obs(c) ∈ [0,1], shape (*S,) (or scalar for 0D)
    """
    if len(base_shape) == 0:
        # 0D: the "subset" is just the single point
        return np.array(1.0, dtype=np.float32)

    support_chi = np.asarray(support_chi, dtype=np.float64)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in base_shape],
                        indexing='ij')
    # χ-weighted centroid for robustness
    if center is None:
        w = support_chi / (support_chi.sum() + 1e-12)
        center = tuple((g * w).sum() for g in grids)

    # effective radius from χ mass (isotropic heuristic)
    area = support_chi.sum()
    # convert |C_i| to an isotropic 'radius' in d dims
    d = len(base_shape)
    # volume of d-ball: V_d R^d ~ area  →  R_eff ~ (area / V_d)^(1/d)
    # V_d for d=1..3: 2, π, 4π/3 (we’ll just use a simple constant heuristic)
    Vd = {1: 2.0, 2: np.pi, 3: 4.0*np.pi/3.0}.get(d, 2.0*np.pi)  # ok heuristic
    R_eff = (max(area, 1e-12) / Vd) ** (1.0 / max(d,1))
    sigma = max(1e-6, frac_radius * R_eff)  # width of the subset

    # distance
    r2 = 0.0
    for g, c in zip(grids, center):
        r2 = r2 + (g - c) ** 2

    mask = np.exp(-0.5 * r2 / (sigma**2))
    # hard cutoff
    mask[np.sqrt(r2) > cutoff_sigma * sigma] = 0.0
    # ensure it's a subset: gate by χ_i
    mask = mask * support_chi
    return mask.astype(np.float32)




def setup_smooth_observations(self,
                              *,
                              frac_radius: float = 0.5,
                              cutoff_sigma: float = 3.0,
                              noise_scale: Optional[float] = None,
                              bias_scale: Optional[float] = None,
                              seed: Optional[int] = None) -> None:
    """
    Build smooth observation fields for all agents on subsets of their supports.

    - χ_obs,i(c) = Gaussian blob subset of χ_i(c)
    - o_i(c) = C x*(c) + b_i + ε(c) on χ_obs,i(c), zeros elsewhere
    """
    rng = (np.random.default_rng(seed) if seed is not None
           else self.config.get_obs_rng())
    K = self.agents[0].K
    D = getattr(self.config, "D_x", K)

    # Shared observation model (can be per-agent if you prefer)
    W = rng.normal(0.0, getattr(self.config, "obs_W_scale", 0.5), size=(D, K)).astype(np.float32)
    R_scale = getattr(self.config, "obs_R_scale", 0.3)
    R = (R_scale ** 2) * np.eye(D, dtype=np.float32)

    # Ground-truth latent field x*(c) (dimension-agnostic)
    shape = self.agents[0].base_manifold.shape
    if len(shape) == 0:
        grid = [np.array([0.0])]
    else:
        grid = [np.linspace(0, 2*np.pi, n, endpoint=False, dtype=np.float32) for n in shape]
    grids = np.meshgrid(*grid, indexing='ij') if len(shape) > 0 else []
    modes = getattr(self.config, "obs_ground_truth_modes", 3)
    amp   = getattr(self.config, "obs_ground_truth_amplitude", 1.0)
    # x*(c) has shape (*S, K)
    x_true = np.zeros(shape + (K,), dtype=np.float32)
    for k in range(K):
        # sum a couple sin/cos modes across dims:
        acc = 0.0
        for m in range(1, modes+1):
            term = 0.0
            for ax, g in enumerate(grids):
                term = term + np.sin(m*g + 0.1*k) + np.cos(0.5*m*g - 0.2*k)
            acc = acc + term
        x_true[..., k] = (amp / max(1, modes)) * acc

    # Build per-agent obs fields
    for agent in self.agents:
        chi_i = agent.support.chi_weight  # (*S,) or scalar for 0D
        mask_obs = gaussian_subset_mask(agent.base_manifold.shape,
                                        support_chi=chi_i,
                                        center=None,
                                        frac_radius=frac_radius,
                                        cutoff_sigma=cutoff_sigma)

        bias_std = bias_scale if (bias_scale is not None) else getattr(self.config, "obs_bias_scale", 0.3)
        noise_std = noise_scale if (noise_scale is not None) else getattr(self.config, "obs_noise_scale", 0.2)

        b_i = rng.normal(0.0, bias_std, size=(D,)).astype(np.float32)

        if len(shape) == 0:
            # 0D
            o_val = (W @ x_true + b_i).astype(np.float32)
            o_val = o_val + rng.normal(0.0, noise_std, size=(D,)).astype(np.float32)
        else:
            # (*S, D)
            o_val = (x_true @ W.T) + b_i  # broadcast bias
            o_val = o_val + rng.normal(0.0, noise_std, size=o_val.shape).astype(np.float32)

        # Optionally zero outside χ_obs (not strictly necessary, we weight below)
        if len(shape) > 0:
            o_val = (mask_obs[..., None] * o_val).astype(np.float32)

        agent.set_observation_field(C=W, R=R, mask=mask_obs, values=o_val, bias=b_i)

    # Keep a copy on the system if you like:
    self.W_obs, self.R_obs = W, R
    self.x_true = x_true



def setup_smooth_observations(self,
                              *,
                              frac_radius: float = 0.5,
                              cutoff_sigma: float = 3.0,
                              noise_scale: Optional[float] = None,
                              bias_scale: Optional[float] = None,
                              seed: Optional[int] = None) -> None:
    """
    Build smooth observation fields for all agents on subsets of their supports.

    - χ_obs,i(c) = Gaussian blob subset of χ_i(c)
    - o_i(c) = C x*(c) + b_i + ε(c) on χ_obs,i(c), zeros elsewhere
    """
    rng = (np.random.default_rng(seed) if seed is not None
           else self.config.get_obs_rng())
    K = self.agents[0].K
    D = getattr(self.config, "D_x", K)

    # Shared observation model (can be per-agent if you prefer)
    W = rng.normal(0.0, getattr(self.config, "obs_W_scale", 0.5), size=(D, K)).astype(np.float32)
    R_scale = getattr(self.config, "obs_R_scale", 0.3)
    R = (R_scale ** 2) * np.eye(D, dtype=np.float32)

    # Ground-truth latent field x*(c) (dimension-agnostic)
    shape = self.agents[0].base_manifold.shape
    if len(shape) == 0:
        grid = [np.array([0.0])]
    else:
        grid = [np.linspace(0, 2*np.pi, n, endpoint=False, dtype=np.float32) for n in shape]
    grids = np.meshgrid(*grid, indexing='ij') if len(shape) > 0 else []
    modes = getattr(self.config, "obs_ground_truth_modes", 3)
    amp   = getattr(self.config, "obs_ground_truth_amplitude", 1.0)
    # x*(c) has shape (*S, K)
    x_true = np.zeros(shape + (K,), dtype=np.float32)
    for k in range(K):
        # sum a couple sin/cos modes across dims:
        acc = 0.0
        for m in range(1, modes+1):
            term = 0.0
            for ax, g in enumerate(grids):
                term = term + np.sin(m*g + 0.1*k) + np.cos(0.5*m*g - 0.2*k)
            acc = acc + term
        x_true[..., k] = (amp / max(1, modes)) * acc

    # Build per-agent obs fields
    for agent in self.agents:
        chi_i = agent.support.chi_weight  # (*S,) or scalar for 0D
        mask_obs = gaussian_subset_mask(agent.base_manifold.shape,
                                        support_chi=chi_i,
                                        center=None,
                                        frac_radius=frac_radius,
                                        cutoff_sigma=cutoff_sigma)

        bias_std = bias_scale if (bias_scale is not None) else getattr(self.config, "obs_bias_scale", 0.3)
        noise_std = noise_scale if (noise_scale is not None) else getattr(self.config, "obs_noise_scale", 0.2)

        b_i = rng.normal(0.0, bias_std, size=(D,)).astype(np.float32)

        if len(shape) == 0:
            # 0D
            o_val = (W @ x_true + b_i).astype(np.float32)
            o_val = o_val + rng.normal(0.0, noise_std, size=(D,)).astype(np.float32)
        else:
            # (*S, D)
            o_val = (x_true @ W.T) + b_i  # broadcast bias
            o_val = o_val + rng.normal(0.0, noise_std, size=o_val.shape).astype(np.float32)

        # Optionally zero outside χ_obs (not strictly necessary, we weight below)
        if len(shape) > 0:
            o_val = (mask_obs[..., None] * o_val).astype(np.float32)

        agent.set_observation_field(C=W, R=R, mask=mask_obs, values=o_val, bias=b_i)

    # Keep a copy on the system if you like:
    self.W_obs, self.R_obs = W, R
    self.x_true = x_true

def compute_observation_energy(self, system) -> float:
    """
    -E_q[log p(o|x)] integrated with χ_i(c)·χ_obs(c)
    p(o|x) = N(o | Cx, R) with x ~ q_i(c) = N(μ_q, Σ_q).
    """
    if not self.has_observation_field or system.config.lambda_obs == 0.0:
        return 0.0

    C = self.C_obs
    R = self.R_obs
    # precompute R^{-1}, log|2πR|
    D = R.shape[0]
    R_inv = np.linalg.inv(R + 1e-8*np.eye(D, dtype=R.dtype))
    const = 0.5 * np.log(np.linalg.det(2*np.pi*R + 1e-8*np.eye(D))).astype(np.float64)

    chi = np.asarray(self.support.chi_weight, dtype=np.float64)       # (*S,) or scalar
    chi_obs = np.asarray(self.obs_mask, dtype=np.float64)             # (*S,) or scalar

    weight = chi * chi_obs                                            # (*S,) or scalar

    mu = np.asarray(self.mu_q, dtype=np.float64)                      # (*S, K) or (K,)
    Sigma = np.asarray(self.Sigma_q, dtype=np.float64)                # (*S, K,K) or (K,K)
    o = np.asarray(self.obs_values, dtype=np.float64)                 # (*S, D) or (D,)

    # E_q[ (o - Cx)(o - Cx)^T ] = (o - Cμ)(o - Cμ)^T + C Σ C^T
    if mu.ndim == 1:
        innov = o - C @ mu
        term = innov @ (R_inv @ innov) + np.trace(R_inv @ (C @ Sigma @ C.T))
        nll = const + 0.5 * term
        total = float(system.config.lambda_obs) * float(weight) * float(nll)
        return total

    # ND case (vectorized over *S)
    C_Sigma_CT = np.einsum('dk,...kl,el->...de', C, Sigma, C, optimize=True)   # (*S, D,D)
    innov = o - np.einsum('dk,...k->...d', C, mu, optimize=True)               # (*S, D)

    # innov^T R^{-1} innov
    quad = np.einsum('...d,de,...e->...', innov, R_inv, innov, optimize=True)  # (*S,)
    tr_term = np.einsum('de,...ed->...', R_inv, C_Sigma_CT, optimize=True)     # (*S,)
    nll = const + 0.5 * (quad + tr_term)                                       # (*S,)

    energy = float(system.config.lambda_obs) * np.sum(weight * nll)
    return float(energy)



