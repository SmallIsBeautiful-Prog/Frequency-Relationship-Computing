# FRC 08 — Ordered Representation

## Purpose

Test whether preserving the introduction order of pairwise frequency relationships gives a linear classifier additional computational capability.

## Implementation

- Used eight carrier frequencies and all 28 pairwise relationships.
- Generated all 40,320 possible frequency introduction orders.
- Created an order-dependent classification task comparing when two designated relationships were introduced.
- Compared three representations:
  1. Original carrier amplitudes.
  2. Unordered pairwise relationship amplitudes.
  3. Ordered representation containing carrier amplitudes, relationship amplitudes and introduction positions.
- Used the same least-squares linear classifier for all representations.
- Training and testing used separate frequency permutations.

## Results

| Representation | Features | Accuracy |
|---|---:|---:|
| Original carriers | 8 | 50.05% |
| Unordered relationships | 28 | 49.82% |
| Ordered relationships | 64 | 100.00% |

The ordered representation correctly classified all 12,096 test examples.

## What This Demonstrates

Preserving relationship introduction positions provides information that is unavailable in the final unordered relationship set.

The same linear classifier achieved perfect classification when supplied with the ordered representation.

This demonstrates that growth history can become a useful computational feature without changing the underlying classifier.

## Research Status

Successful controlled demonstration.

Introduction positions were explicitly supplied from the known growth sequence. The experiment does not yet demonstrate independent reconstruction of order from waveform measurements or general sequence understanding.

## Next Step

**FRC 09 — Growth Learning:** develop a learning process that uses successive growth states and their recovered relationships to construct and learn ordered representations.