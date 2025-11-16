#!/usr/bin/env python3
"""
Hierarchical Evolution Text Report
===================================

Lightweight text-based analysis that works without numpy/matplotlib.
Useful for quick inspection of hierarchical evolution results.

Usage:
    python hierarchical_report.py _results/_playground
"""

import pickle
import sys
from pathlib import Path
from collections import defaultdict


def load_history(run_dir: Path):
    """Load hierarchical history from pkl file."""
    pkl_path = run_dir / "hierarchical_history.pkl"

    if not pkl_path.exists():
        print(f"❌ File not found: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        history = pickle.load(f)

    print(f"✓ Loaded: {pkl_path}\n")
    return history


def print_report(history: dict, run_dir: Path):
    """Generate text-based report of hierarchical evolution."""

    print("=" * 80)
    print(f" HIERARCHICAL EVOLUTION REPORT – {run_dir.name}")
    print("=" * 80)
    print()

    # Basic stats
    steps = history['step']
    n_steps = len(steps)

    print(f"📊 BASIC STATISTICS")
    print(f"   Total steps: {n_steps}")
    print(f"   Step range: {steps[0]} → {steps[-1]}")
    print()

    # Condensation events
    if 'condensations' in history or 'n_condensations' in history:
        cond_key = 'n_condensations' if 'n_condensations' in history else 'condensations'
        condensations = history[cond_key]
        total_cond = sum(condensations)
        emergence_steps = [steps[i] for i in range(len(steps)) if condensations[i] > 0]

        print(f"✨ CONDENSATION EVENTS")
        print(f"   Total condensations: {total_cond}")
        print(f"   Events at steps: {emergence_steps}")

        if emergence_steps:
            print()
            print(f"   Event details:")
            for step in emergence_steps:
                idx = steps.index(step)
                count = condensations[idx]
                print(f"      Step {step:3d}: {count} meta-agent(s) formed")
        print()

    # Scale evolution
    if 'n_scales' in history:
        n_scales = history['n_scales']
        max_scales = max(n_scales)

        print(f"🔬 SCALE EVOLUTION")
        print(f"   Initial scales: {n_scales[0]}")
        print(f"   Final scales: {n_scales[-1]}")
        print(f"   Maximum scales: {max_scales}")
        print()

    # Agent counts
    if 'n_active_agents' in history:
        n_active = history['n_active_agents']

        print(f"🤖 AGENT POPULATION")
        print(f"   Initial active: {n_active[0]}")
        print(f"   Final active: {n_active[-1]}")
        print(f"   Peak active: {max(n_active)}")
        print()

    # Per-scale breakdown
    if 'n_active_per_scale' in history and 'n_agents_per_scale' in history:
        active_per_scale = history['n_active_per_scale']
        total_per_scale = history['n_agents_per_scale']

        # Final state per scale
        final_active = active_per_scale[-1]
        final_total = total_per_scale[-1]

        print(f"📈 FINAL STATE PER SCALE")

        for scale in sorted(final_total.keys()):
            total = final_total.get(scale, 0)
            active = final_active.get(scale, 0)
            inactive = total - active

            bar_width = 40
            if total > 0:
                active_bar = int(bar_width * active / total)
                inactive_bar = bar_width - active_bar
                bar = "█" * active_bar + "░" * inactive_bar
            else:
                bar = "░" * bar_width

            print(f"   Scale {scale}: [{bar}] {active}/{total} active")

        print()

    # Energy
    if 'total_energy' in history:
        energy = history['total_energy']
        initial_energy = energy[0]
        final_energy = energy[-1]
        delta_energy = final_energy - initial_energy

        if initial_energy != 0:
            percent_change = (delta_energy / initial_energy) * 100
        else:
            percent_change = 0

        print(f"⚡ ENERGY")
        print(f"   Initial: {initial_energy:12.4f}")
        print(f"   Final:   {final_energy:12.4f}")
        print(f"   Change:  {delta_energy:12.4f} ({percent_change:+.1f}%)")
        print()

    # Update statistics
    if 'updates_applied' in history:
        updates = history['updates_applied']
        total_updates = sum(updates)
        avg_updates = total_updates / len(updates) if updates else 0

        print(f"🔄 GRADIENT UPDATES")
        print(f"   Total updates: {total_updates}")
        print(f"   Average per step: {avg_updates:.1f}")
        print()

    if 'priors_updated' in history:
        priors = history['priors_updated']
        total_priors = sum(priors)
        avg_priors = total_priors / len(priors) if priors else 0

        print(f"🎯 TOP-DOWN PRIOR SYNCHRONIZATION")
        print(f"   Total prior updates: {total_priors}")
        print(f"   Average per step: {avg_priors:.1f}")
        print()

    # Timeline
    print(f"📜 EVOLUTION TIMELINE")
    print(f"   {'Step':<6} {'Energy':<14} {'Scales':<8} {'Active':<8} {'Event':<20}")
    print(f"   {'-'*6} {'-'*14} {'-'*8} {'-'*8} {'-'*20}")

    # Show first 5, condensation events, and last 5
    indices_to_show = set()

    # First 5
    indices_to_show.update(range(min(5, n_steps)))

    # Last 5
    indices_to_show.update(range(max(0, n_steps - 5), n_steps))

    # Condensation events
    if 'condensations' in history or 'n_condensations' in history:
        cond_key = 'n_condensations' if 'n_condensations' in history else 'condensations'
        condensations = history[cond_key]
        for i in range(len(steps)):
            if condensations[i] > 0:
                # Include before and after
                indices_to_show.update(range(max(0, i-1), min(n_steps, i+2)))

    indices_to_show = sorted(indices_to_show)

    prev_idx = -2
    for i in indices_to_show:
        # Add separator for gaps
        if i > prev_idx + 1:
            print(f"   {'...':<6}")

        step = steps[i]

        energy_str = f"{history['total_energy'][i]:.4f}" if 'total_energy' in history else "N/A"
        scales_str = f"{history['n_scales'][i]}" if 'n_scales' in history else "N/A"
        active_str = f"{history['n_active_agents'][i]}" if 'n_active_agents' in history else "N/A"

        event_str = ""
        if 'condensations' in history or 'n_condensations' in history:
            cond_key = 'n_condensations' if 'n_condensations' in history else 'condensations'
            if history[cond_key][i] > 0:
                event_str = f"✨ {history[cond_key][i]} meta-agent(s)"

        print(f"   {step:<6} {energy_str:<14} {scales_str:<8} {active_str:<8} {event_str:<20}")

        prev_idx = i

    print()
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python hierarchical_report.py <run_dir>")
        print("Example: python hierarchical_report.py _results/_playground")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"❌ Directory not found: {run_dir}")
        sys.exit(1)

    history = load_history(run_dir)

    if history is None:
        print("\n❌ Could not load hierarchical_history.pkl")
        sys.exit(1)

    print_report(history, run_dir)


if __name__ == '__main__':
    main()
