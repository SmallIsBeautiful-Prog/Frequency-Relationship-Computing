# FRC 11 — Higher-Order Relationships

## Purpose

Investigate whether third-order frequency relationships extend the computational capabilities of the existing linear classifier beyond pairwise interactions.

## Implementation

- Eight carrier frequencies.
- 28 pairwise relationships.
- 56 three-way relationships.
- 25,600 examples covering all 256 binary input combinations.
- Two simultaneous three-input parity tasks:
  - A XOR B XOR C
  - D XOR E XOR F
- Three feature representations:
  - Original: 8 carrier features.
  - Pairwise-enhanced: 36 features.
  - Third-order-enhanced: 92 features.
- Multi-output least-squares linear classifier.
- Training and testing split: 70% / 30%.

## Results

| Representation | Features | Average accuracy | Joint accuracy |
|---|---:|---:|---:|
| Original | 8 | 50.20% | 24.90% |
| Pairwise-enhanced | 36 | 50.35% | 25.69% |
| Third-order-enhanced | 92 | 100.00% | 100.00% |

Both three-input parity outputs achieved 100% test accuracy using the third-order representation.

The strongest learned three-way weights corresponded to the exact input triplets defining the two target operations.

## What this demonstrates

Pairwise feature expansion alone was insufficient for these three-input parity tasks.

Adding third-order products supplied the additional representation needed for the same linear classifier to learn both outputs perfectly.

This extends the demonstrated Boolean learning capability from pairwise to three-way interactions without changing the underlying learning algorithm.

## Research status

Successful controlled digital experiment.

Third-order products were explicitly constructed from binary carrier amplitudes. The experiment demonstrates the usefulness of higher-order polynomial features, not automatic discovery of arbitrary relationships or a physical three-frequency computation mechanism.

## Next step

FRC 12 — Compositional Learning: investigate whether previously constructed relationships can be combined into new features and reused to learn more complex functions.