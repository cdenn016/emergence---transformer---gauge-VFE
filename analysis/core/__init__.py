"""
Core Analysis Utilities
========================

Data loading, preprocessing, and geometry helpers.
"""

from .loaders import (
    load_history,
    load_system,
    get_mu_tracker,
    filter_history_steps,
    filter_mu_tracker,
    normalize_history,
    DEFAULT_SKIP_STEPS,
)

from .geometry import (
    get_spatial_shape_from_system,
    pick_reference_agent,
    get_ndim_from_shape,
)

__all__ = [
    # Loaders
    'load_history',
    'load_system',
    'get_mu_tracker',
    'filter_history_steps',
    'filter_mu_tracker',
    'normalize_history',
    'DEFAULT_SKIP_STEPS',
    # Geometry
    'get_spatial_shape_from_system',
    'pick_reference_agent',
    'get_ndim_from_shape',
]
