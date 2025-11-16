# Meta-Agent Detector Issue: Diagnosis and Fix

## Problem Statement
Meta-agent detector only triggers at early steps (0, 5) but never at later steps (10+), even though alignment appears to plummet.

## Root Cause

The detector **IS running** at steps 10, 15, 20, etc. (the modulo timing logic is correct), but it finds **nothing to condense** because:

### Scenario 1: All agents condensed into single meta-agent
1. At step 0 or 5: All 5 base agents reach consensus
2. They condense into 1 meta-agent at scale 1
3. Constituent agents become inactive (`is_active = False`)
4. At step 10+:
   - **Scale 0**: 0 active agents (all condensed) → **SKIP** (< min_cluster_size)
   - **Scale 1**: 1 meta-agent → **SKIP** (< min_cluster_size = 2)
5. Detector runs but finds nothing

### Scenario 2: Agents diverge after initial consensus
1. At step 0 or 5: Agents condense into meta-agents
2. At step 10+: Meta-agents **diverge** (KL > threshold)
3. Detector runs, checks KL divergences, finds KL > threshold
4. No consensus → no condensation

## Key Insight
**The detector only triggers when agents are ALIGNED (KL < threshold), not when they're MISALIGNED.**

"Alignment plummets" means agents are diverging, which makes consensus **less likely**, not more likely. The detector is working correctly—it's just not finding consensus because agents are too far apart.

## The Fix
Added diagnostic logging to show:
1. ✅ **When detector runs** (confirms every N steps as expected)
2. ✅ **Active agent counts** at each scale (shows why scales are skipped)
3. ✅ **Pairwise KL divergences** (shows why consensus isn't found)
4. ✅ **Detection results** (condensed vs. no consensus)

### Changes Made

#### File: `meta/hierarchical_evolution.py`
- Added logging in `_check_and_condense_all_scales()`:
  - Shows step number and check interval
  - Displays active/total agents per scale
  - Explains why scales are skipped
  - Reports condensation results

#### File: `meta/emergence.py`
- Added logging in `auto_detect_and_condense()`:
  - Shows pairwise KL divergences between agents
  - Indicates which pairs pass/fail threshold
  - Makes detection logic transparent

## Example Output

```
[Step 10] 🔍 Consensus check (interval=5)
  Scale 0: 0/5 active → SKIP (need >=2)
  Scale 1: 1/1 active → SKIP (need >=2)
  Result: No condensations this step
```

or

```
[Step 10] 🔍 Consensus check (interval=5)
  Scale 0: 0/5 active → SKIP (need >=2)
  Scale 1: 3/3 active → checking consensus...
    Pairwise belief KL divergences (threshold=0.0500):
      0↔1: KL=0.234567 ✗
      0↔2: KL=0.187234 ✗
      1↔2: KL=0.298765 ✗
    → No consensus (KL > 0.05)
  Result: No condensations this step
```

## How to Use

Run your simulation with `ENABLE_EMERGENCE=True` and you'll now see detailed diagnostic output showing:
- Exactly when the detector runs
- Why it skips certain scales (too few active agents)
- Why it doesn't find consensus (KL values too high)

## Next Steps

If you want the detector to trigger at step 10+, you need to either:

1. **Lower the consensus threshold** (e.g., `CONSENSUS_THRESHOLD = 0.1` instead of `0.05`)
   - Agents don't need to be as closely aligned

2. **Change training dynamics** to promote consensus:
   - Increase `LAMBDA_BELIEF_ALIGN` (stronger alignment pressure)
   - Decrease learning rates (agents converge slower but more stably)

3. **Allow meta-agents to split** (not currently implemented):
   - Add logic to detect when a meta-agent's constituents diverge
   - Reactivate constituents when they become misaligned

## Theory Note

The detector looks for **epistemic death** (consensus), not misalignment. When "alignment plummets" (agents diverge), this actually makes condensation **less likely**, not more likely. The system is working as designed—meta-agents only form when there's strong consensus.
