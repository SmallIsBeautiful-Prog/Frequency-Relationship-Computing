# FRC 01 — Original XOR Learning

## Purpose

Demonstrate that pairwise frequency relationships can extend the computational capability of a simple linear classifier.

## Implementation

- Two binary inputs represented by 12,000 Hz and 11,900 Hz carriers.
- Combined the waveforms and applied a square-law transformation.
- Extracted the 100 Hz difference-frequency component as the pairwise feature.
- Compared a least-squares linear classifier using original features (A, B) against one using enhanced features (A, B, A×B).
- Generated 800 examples, with 560 for training and 240 for independent testing.

## Results

| Classifier | Test accuracy |
|---|---:|
| Original features (A, B) | 49.58% |
| Pairwise-enhanced (A, B, A×B) | 100.00% |

The enhanced classifier correctly classified all 240 test examples.

## What this demonstrates

The pairwise interaction feature allows a linear classifier to solve XOR, a problem that cannot be separated using the two original features alone.

The learning algorithm remained unchanged. The improvement came from expanding its input representation.

## Research status

**Demonstrated:** Pairwise feature expansion enables a simple linear classifier to learn XOR in this controlled digital experiment.

**Not yet demonstrated:** General-purpose reasoning, higher-order relationships, symbolic learning or temporal order representation.

## Next step

Extend the experiment to multiple frequency pairs and determine whether one linear classifier can learn several pairwise relationships simultaneously.
