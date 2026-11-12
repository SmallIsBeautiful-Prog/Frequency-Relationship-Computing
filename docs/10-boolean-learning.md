# FRC 10 — Boolean Learning

## Purpose

Test whether a single linear classifier can learn multiple Boolean operations simultaneously using pairwise-enhanced frequency representations.

## Implementation

- Eight carrier frequencies.
- Twenty-eight pairwise relationships.
- 51,200 examples covering all 256 possible binary input combinations.
- Four simultaneous Boolean outputs: AND, OR, XOR and NAND.
- Two feature representations:
  - Original: 8 carrier features.
  - Enhanced: 36 features, including 28 pairwise products.
- One multi-output least-squares linear classifier.
- Training and testing split: 70% / 30%.

## Results

| Operation | Original | Enhanced |
|---|---:|---:|
| AND | 100.00% | 100.00% |
| OR | 100.00% | 100.00% |
| XOR | 50.09% | 100.00% |
| NAND | 100.00% | 100.00% |
| Average | 87.52% | 100.00% |

Joint accuracy, requiring all four outputs to be correct:

- Original representation: 50.09%.
- Pairwise-enhanced representation: 100.00%.

The classifier identified the corresponding pairwise relationships as the strongest relationship features for each operation.

## What this demonstrates

A single linear classifier can learn several different Boolean operations simultaneously when supplied with an appropriate pairwise-enhanced representation.

AND, OR and NAND were already linearly separable using the original carrier features. XOR required the additional pairwise feature.

This demonstrates an expansion in the computational functions accessible to the same linear learning architecture through feature representation.

## Research status

Successful controlled digital experiment.

Pairwise products were calculated directly rather than extracted from a newly generated waveform. Their equivalence to square-law difference-frequency measurements was established in earlier experiments.

The results demonstrate Boolean learning under controlled conditions, not general-purpose reasoning or unrestricted logical computation.

## Next step

FRC 11 — Higher-Order Relationships: investigate whether three-way frequency relationships can extend the representation beyond pairwise interactions.