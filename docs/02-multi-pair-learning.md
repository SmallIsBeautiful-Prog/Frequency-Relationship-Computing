# FRC 02 — Multi-Pair Learning

## Purpose

Investigate whether one linear learning architecture can simultaneously learn multiple tasks using several pairwise frequency relationships.

## Implementation

- Four carrier frequencies: 12,000, 11,900, 11,770 and 11,610 Hz.
- Six unique pairwise difference-frequency relationships.
- Square-law processing extracts the interaction components.
- Two simultaneous outputs learn XOR(A, B) and XOR(C, D).
- Compared original carrier features against all four carriers plus six pairwise features.
- Used ordinary least squares with 2,240 training examples and 960 held-out test examples.

## Results

| Classifier | Output 1 | Output 2 | Both correct |
|---|---:|---:|---:|
| Original features | 49.27% | 50.00% | 24.48% |
| Pairwise-enhanced | 100.00% | 100.00% | 100.00% |

The enhanced classifier correctly predicted both outputs on all 960 test examples.

## What this demonstrates

Multiple pairwise relationships can be supplied to the same linear learning architecture to solve two simultaneous XOR tasks.

The learned weights strongly favoured the relevant A–B and C–D relationships, while assigning near-zero weights to the other four relationships.

The learning method remained unchanged. The expanded feature representation supplied the additional computational information.

## Research status

**Demonstrated:** Simultaneous learning of two pairwise XOR tasks using four carriers and six interaction features in a controlled digital experiment.

**Not yet demonstrated:** Generalisation to unfamiliar frequency configurations, automatic discovery of arbitrary relationships or general-purpose reasoning.

## Next step

Develop a configurable frequency representation that automatically constructs and measures all pairwise relationships as the number of carrier frequencies increases.