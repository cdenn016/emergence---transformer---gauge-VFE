#!/usr/bin/env python3
"""
Compare energy computation between Path A (standard) and Path B (hierarchical).
"""
import numpy as np
from pathlib import Path

# Import simulation suite components
from simulation_suite import (
    build_manifold,
    build_supports,
    build_agents,
    build_system,
    _GradientSystemAdapter,
    SEED,
    SYSTEM_CONFIG
)

from free_energy_clean import compute_total_free_energy
from meta.multi_scale_system import MultiScaleSystem

def compare_paths():
    """Compare energy between standard system and hierarchical adapter."""

    print("=" * 70)
    print("ENERGY PATH COMPARISON")
    print("=" * 70)
    print("\nSetting up shared components with identical seed...")

    # Use same seed for both paths
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Build components
    manifold = build_manifold()
    supports = build_supports(manifold, rng)
    agents = build_agents(manifold, supports, rng)
    system_std = build_system(agents, rng)

    print(f"  Created {system_std.n_agents} agents")
    print(f"  Overlap pairs stored: {len(system_std.overlap_masks)}")

    # ========================================================================
    # PATH A: Standard system (MultiAgentSystem)
    # ========================================================================
    print(f"\n{'='*70}")
    print("PATH A: Standard MultiAgentSystem (enable_emergence=False)")
    print(f"{'='*70}")

    energies_std = compute_total_free_energy(system_std)

    print(f"  Self energy:      {energies_std.self_energy:12.6f}")
    print(f"  Belief align:     {energies_std.belief_align:12.6f}")
    print(f"  Prior align:      {energies_std.prior_align:12.6f}")
    print(f"  Observations:     {energies_std.observations:12.6f}")
    print(f"  {'-'*70}")
    print(f"  TOTAL:            {energies_std.total:12.6f}")

    # ========================================================================
    # PATH B: Hierarchical adapter (_GradientSystemAdapter)
    # ========================================================================
    print(f"\n{'='*70}")
    print("PATH B: Hierarchical via Adapter (enable_emergence=True, no meta-agents)")
    print(f"{'='*70}")

    # Create MultiScaleSystem wrapper (what run_hierarchical_training uses)
    multi_scale = MultiScaleSystem(system_std, SYSTEM_CONFIG)

    # Get active agents (should be all base agents initially)
    active_agents = multi_scale.get_all_active_agents()
    print(f"  Active agents: {len(active_agents)}")

    # Create adapter (what the hierarchical path uses)
    adapter = _GradientSystemAdapter(active_agents, SYSTEM_CONFIG)
    print(f"  Adapter n_agents: {adapter.n_agents}")
    print(f"  Adapter overlap pairs: {len(adapter._overlaps)}")

    energies_hier = compute_total_free_energy(adapter)

    print(f"  Self energy:      {energies_hier.self_energy:12.6f}")
    print(f"  Belief align:     {energies_hier.belief_align:12.6f}")
    print(f"  Prior align:      {energies_hier.prior_align:12.6f}")
    print(f"  Observations:     {energies_hier.observations:12.6f}")
    print(f"  {'-'*70}")
    print(f"  TOTAL:            {energies_hier.total:12.6f}")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    print(f"\n  Component-wise ratios (A/B):")
    components = [
        ('Self energy', energies_std.self_energy, energies_hier.self_energy),
        ('Belief align', energies_std.belief_align, energies_hier.belief_align),
        ('Prior align', energies_std.prior_align, energies_hier.prior_align),
        ('Observations', energies_std.observations, energies_hier.observations),
        ('TOTAL', energies_std.total, energies_hier.total),
    ]

    for name, val_a, val_b in components:
        if abs(val_b) > 1e-10:
            ratio = val_a / val_b
            diff = val_a - val_b
            match = "✓" if abs(ratio - 1.0) < 0.01 else "✗"
            print(f"    {match} {name:15s}: {ratio:8.4f}x  (diff: {diff:+10.6f})")
        else:
            match = "✓" if abs(val_a) < 1e-10 else "✗"
            print(f"    {match} {name:15s}: N/A (both zero)")

    # Check neighbor counts
    print(f"\n  Neighbor count comparison:")
    for i in range(system_std.n_agents):
        neighbors_std = system_std.get_neighbors(i)
        neighbors_hier = adapter.get_neighbors(i)
        match = "✓" if neighbors_std == neighbors_hier else "✗"
        print(f"    {match} Agent {i}: std={len(neighbors_std)}, hier={len(neighbors_hier)}")

    # Verdict
    print(f"\n{'='*70}")
    total_ratio = energies_std.total / energies_hier.total if energies_hier.total != 0 else float('inf')

    if abs(total_ratio - 1.0) < 0.01:
        print("✅ PASS: Energies match!")
    else:
        print(f"❌ FAIL: Energy mismatch!")
        print(f"   Ratio: {total_ratio:.4f}x")
        if abs(total_ratio - 2.0) < 0.1:
            print(f"   ⚠️  Approximately 2x - likely double-counting in Path A")
        elif abs(total_ratio - 0.5) < 0.1:
            print(f"   ⚠️  Approximately 0.5x - likely double-counting in Path B")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    compare_paths()
