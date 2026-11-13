# FRC 12 — Compositional Learning

## Purpose

Investigate whether higher-order frequency relationships can be constructed compositionally from lower-order relationships, and whether progressively higher-order features expand the capabilities of the same linear classifier.

## Implementation

The experiment uses eight binary input channels represented by eight carrier frequencies.

It constructs:

- 8 original carrier features.
- 28 pairwise relationships.
- 56 three-way relationships.
- 70 four-way relationships.

Three simultaneous classification tasks are evaluated:

1. Three-input parity: A XOR B XOR C.
2. Three-input parity: D XOR E XOR F.
3. Four-input parity: A XOR B XOR C XOR D.

Five feature representations are compared:

1. Original carrier features (8).
2. Pairwise-enhanced features (36).
3. Direct third-order features (92).
4. Composed third-order features (92).
5. Composed third- and fourth-order features (162).

All representations use the same multi-output least-squares linear classifier and the same training/test split.

## Results

The experiment generated 25,600 examples, with 17,920 used for training and 7,680 for testing.

| Feature representation | Average accuracy | Joint accuracy |
|---|---:|---:|
| Original carriers | 50.32% | 12.84% |
| Pairwise-enhanced | 48.50% | 15.72% |
| Direct third-order | 83.49% | 50.47% |
| Composed third-order | 83.49% | 50.47% |
| Composed third and fourth-order | **100.00%** | **100.00%** |

### Composition verification

Direct and compositionally constructed third-order features matched exactly:

- Mean absolute error: 0.000000
- Maximum absolute error: 0.000000
- Verification: PASS

### Classification findings

- Third-order features solved both three-input parity tasks with 100% accuracy.
- Third-order features alone did not solve the four-input parity task.
- Adding fourth-order features raised all three task accuracies to 100%.
- All three outputs were simultaneously correct for every test example using the highest-order representation.

## What This Demonstrates

FRC 12 demonstrates that progressively higher-order feature representations can extend the computational capabilities of an unchanged linear classifier.

Three-way relationships can be composed from pairwise features, while four-way relationships extend the representation sufficiently to solve four-input parity.

The experiment supports the use of compositional polynomial feature expansion as a way of increasing the expressive power of FRC.

## Research Status

**Successful controlled digital experiment.**

The results demonstrate mathematical feature composition and higher-order classification. They do not yet demonstrate physical higher-order frequency computation, automatic relationship discovery, or an advantage over conventional polynomial feature expansion.

Direct and composed third-order features are mathematically equivalent; their identical results verify the construction method rather than demonstrate a performance improvement.

## Next Step

**FRC 13 — Relationship Selection**

Investigate whether FRC can identify the relationships most useful for a given task, reducing unnecessary features while preserving classification performance.