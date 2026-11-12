# FRC 09 — Growth Learning

## Purpose

Test whether ordered relationship information can be reconstructed from successive cumulative growth states and used by the same linear classifier.

## Implementation

- Used eight carrier frequencies and all 28 pairwise relationships.
- Generated all 40,320 possible frequency introduction orders.
- Constructed eight cumulative relationship snapshots for every sequence.
- Recovered each relationship's introduction state by detecting its first appearance.
- Verified every recovered state against the known construction order.
- Compared original carrier, unordered relationship and growth-derived representations.
- Used the same least-squares linear classifier throughout.

## Results

| Representation | Features | Accuracy |
|---|---:|---:|
| Original carriers | 8 | 50.04% |
| Unordered relationships | 28 | 50.29% |
| Growth-derived representation | 64 | 100.00% |

All 40,320 growth sequences passed reconstruction verification. The classifier correctly classified all 12,096 independent test examples.

## What This Demonstrates

Cumulative growth snapshots can preserve sufficient information to reconstruct relationship introduction states.

The resulting ordered representation allowed the existing linear classifier to distinguish relationship formation order with perfect accuracy on the test set.

## Research Status

Successful controlled digital demonstration.

Growth snapshots were constructed from known introduction sequences. Automatic waveform-based recovery across all generated sequences has not yet been tested.

## Next Step

**FRC 10 — Boolean Learning:** expand the learning experiments to multiple Boolean operations, testing how the pairwise-enhanced representation handles AND, OR, XOR and NAND tasks.