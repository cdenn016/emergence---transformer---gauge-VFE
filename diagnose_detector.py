#!/usr/bin/env python3
"""
Diagnostic script to debug meta-agent detector timing issues.

This script adds detailed logging to show:
1. When consensus detector runs
2. How many agents are active at each scale
3. What KL divergences are between agents
4. Why detection succeeds or fails
"""

import numpy as np
import sys
from pathlib import Path

# Add instrumentation to hierarchical_evolution.py
def patch_detector_logging():
    """Patch the detector to add verbose logging."""
    import meta.hierarchical_evolution as hev

    original_check_and_condense = hev.HierarchicalEvolutionEngine._check_and_condense_all_scales

    def logged_check_and_condense(self):
        """Wrapper with detailed logging."""
        print(f"\n{'='*70}")
        print(f"🔍 CONSENSUS CHECK at step {self.step_count}")
        print(f"{'='*70}")

        # Show active agents at each scale
        for scale in sorted(self.system.agents.keys()):
            all_agents = self.system.agents[scale]
            active_agents = self.system.get_active_agents_at_scale(scale)
            print(f"  Scale {scale}: {len(active_agents)}/{len(all_agents)} active agents")

            if len(active_agents) < self.config.min_cluster_size:
                print(f"    ⚠️  Too few active agents (need >= {self.config.min_cluster_size})")
                print(f"    → Skipping scale {scale}")
            else:
                print(f"    ✓ Enough agents, checking for consensus...")

                # Show pairwise KL divergences
                if len(active_agents) >= 2:
                    from math_utils.numerical_utils import kl_gaussian
                    from math_utils.transport import compute_transport

                    print(f"    Pairwise KL divergences (belief):")
                    for i in range(len(active_agents)):
                        for j in range(i+1, len(active_agents)):
                            agent_i = active_agents[i]
                            agent_j = active_agents[j]

                            # Compute transport
                            omega_ij = compute_transport(
                                agent_i.gauge.phi,
                                agent_j.gauge.phi,
                                agent_i.generators,
                                validate=False
                            )

                            # Transport j to i's frame
                            mu_j_transported = omega_ij @ agent_j.mu_q
                            Sigma_j_transported = omega_ij @ agent_j.Sigma_q @ omega_ij.T

                            # KL divergence
                            kl_ij = kl_gaussian(
                                agent_i.mu_q, agent_i.Sigma_q,
                                mu_j_transported, Sigma_j_transported
                            )

                            threshold = self.config.consensus_kl_threshold
                            status = "✓ CONSENSUS" if kl_ij < threshold else "✗ DIVERGED"
                            print(f"      Agent {i} ↔ {j}: KL={kl_ij:.6f} (threshold={threshold:.6f}) {status}")

        # Call original method
        result = original_check_and_condense(self)

        if result:
            print(f"\n  🌟 CONDENSED {len(result)} new meta-agent(s)!")
        else:
            print(f"\n  → No consensus found, no condensation")

        print(f"{'='*70}\n")
        return result

    hev.HierarchicalEvolutionEngine._check_and_condense_all_scales = logged_check_and_condense
    print("✓ Patched consensus detector with detailed logging")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("META-AGENT DETECTOR DIAGNOSTIC")
    print("="*70 + "\n")

    # Apply patches
    patch_detector_logging()

    # Now run the simulation
    print("Running simulation with diagnostic logging...\n")

    # Import and run simulation
    import simulation_runner
    simulation_runner.main()
