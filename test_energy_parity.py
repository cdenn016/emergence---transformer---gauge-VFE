#!/usr/bin/env python3
"""
Minimal test to compare energy computation between standard and hierarchical paths.
"""

import numpy as np
import sys
from pathlib import Path

# Use the existing simulation_suite setup
sys.path.insert(0, str(Path(__file__).parent))

from agent.system import MultiAgentSystem
from free_energy_clean import compute_total_free_energy

# Import adapter from simulation_suite
from simulation_suite import _GradientSystemAdapter

def test_energy_parity():
    """Test if standard system and adapter give same energy."""

    print("="*70)
    print("ENERGY PARITY TEST")
    print("="*70)

    # Use simulation_suite's existing agent creation
    from simulation_suite import (
        create_base_manifold,
        build_system,
        SYSTEM_CONFIG,
        AGENT_CONFIG
    )

    # Create system using standard path
    print("\n[1] Creating standard system...")
    manifold = create_base_manifold()
    system_std = build_system(manifold)

    print(f"    n_agents: {system_std.n_agents}")
    print(f"    len(agents): {len(system_std.agents)}")
    print(f"    overlaps stored: {len(system_std.overlap_masks)}")

    # Compute energy via standard path
    print("\n[2] Computing energy (standard path)...")
    energy_std = compute_total_free_energy(system_std)

    print(f"\n    Standard Path Energies:")
    print(f"      Self:         {energy_std.self_energy:12.6f}")
    print(f"      Belief align: {energy_std.belief_align:12.6f}")
    print(f"      Prior align:  {energy_std.prior_align:12.6f}")
    print(f"      Observations: {energy_std.observations:12.6f}")
    print(f"      ------------------------------------")
    print(f"      TOTAL:        {energy_std.total:12.6f}")

    # Create adapter using same agents
    print("\n[3] Creating adapter (hierarchical path)...")
    adapter = _GradientSystemAdapter(system_std.agents, SYSTEM_CONFIG)

    print(f"    n_agents: {adapter.n_agents}")
    print(f"    len(agents): {len(adapter.agents)}")
    print(f"    overlaps stored: {len(adapter._overlaps)}")

    # Compute energy via adapter
    print("\n[4] Computing energy (adapter/hierarchical path)...")
    energy_hier = compute_total_free_energy(adapter)

    print(f"\n    Adapter Path Energies:")
    print(f"      Self:         {energy_hier.self_energy:12.6f}")
    print(f"      Belief align: {energy_hier.belief_align:12.6f}")
    print(f"      Prior align:  {energy_hier.prior_align:12.6f}")
    print(f"      Observations: {energy_hier.observations:12.6f}")
    print(f"      ------------------------------------")
    print(f"      TOTAL:        {energy_hier.total:12.6f}")

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    ratio = energy_std.total / energy_hier.total if energy_hier.total != 0 else float('inf')
    diff = energy_std.total - energy_hier.total

    print(f"\n  Standard total:     {energy_std.total:12.6f}")
    print(f"  Hierarchical total: {energy_hier.total:12.6f}")
    print(f"  Difference:         {diff:12.6f}")
    print(f"  Ratio (std/hier):   {ratio:12.6f}x")

    # Component-wise comparison
    print(f"\n  Component ratios:")
    for component in ['self_energy', 'belief_align', 'prior_align', 'observations']:
        std_val = getattr(energy_std, component)
        hier_val = getattr(energy_hier, component)
        if hier_val != 0:
            ratio_comp = std_val / hier_val
            print(f"    {component:15s}: {ratio_comp:8.4f}x")
        else:
            print(f"    {component:15s}: N/A (hier=0)")

    # Neighbor comparison
    print(f"\n  Neighbor counts:")
    for i in range(system_std.n_agents):
        neighbors_std = system_std.get_neighbors(i)
        neighbors_hier = adapter.get_neighbors(i)
        print(f"    Agent {i}: std={len(neighbors_std)}, hier={len(neighbors_hier)}")
        if neighbors_std != neighbors_hier:
            print(f"      ⚠️  MISMATCH! std={neighbors_std}, hier={neighbors_hier}")

    # Verdict
    print("\n" + "="*70)
    if abs(diff) < 1e-6:
        print("✅ PASS: Energies match!")
    else:
        print(f"❌ FAIL: Energy mismatch! Difference = {diff:.6f}")
        if abs(ratio - 2.0) < 0.01:
            print("   ⚠️  EXACTLY 2x - likely double-counting bug!")
    print("="*70)

if __name__ == "__main__":
    test_energy_parity()
