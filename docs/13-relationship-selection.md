# FRC 13 — Relationship Selection

## Purpose

Investigate whether FRC can automatically identify the relationships most useful for individual classification tasks, reducing the number of features required without sacrificing accuracy.

## Implementation

The experiment uses eight binary input channels and constructs a full representation containing:

- 8 original carrier features.
- 28 pairwise relationships.
- 56 three-way relationships.
- 70 four-way relationships.

Three simultaneous classification tasks are evaluated:

1. A XOR B XOR C.
2. D XOR E XOR F.
3. A XOR B XOR C XOR D.

A least-squares linear classifier is trained using the full 162-feature representation.

Feature selection ranks relationships using learned model weights, then uses a separate validation dataset to identify smaller task-specific feature sets. The selected models are retrained and evaluated on an untouched test dataset.

## Results

The experiment generated 25,600 examples:

- Training: 15,360.
- Validation: 5,120.
- Testing: 5,120.

| Representation | Features | Average accuracy | Joint accuracy |
|---|---:|---:|---:|
| Original carriers | 8 | 50.20% | 13.30% |
| Pairwise-enhanced | 36 | 50.27% | 14.02% |
| Full fourth-order | 162 | 100.00% | 100.00% |
| Automatically selected | 21 unique | **100.00%** | **100.00%** |

### Task-specific feature selection

| Classification task | Selected features | Test accuracy |
|---|---:|---:|
| A XOR B XOR C | 7 | 100.00% |
| D XOR E XOR F | 7 | 100.00% |
| A XOR B XOR C XOR D | 15 | 100.00% |

The selected representations contained:

- Three-input parity: 3 original, 3 pairwise and 1 third-order feature.
- Four-input parity: 4 original, 6 pairwise, 4 third-order and 1 fourth-order feature.

The union of selected features contained 21 of the original 162 features, representing an **87.04% reduction**.

## What This Demonstrates

FRC 13 demonstrates that a rich relationship representation can be reduced to smaller task-specific feature sets while retaining classification performance.

The same linear classifier achieved 100% average and joint accuracy using only 21 unique features instead of all 162.

This introduces automatic relationship selection as a potential way to improve the efficiency and adaptability of FRC.

## Research Status

**Successful controlled digital experiment.**

Feature selection was guided by learned model coefficients and validation accuracy. The final test set was not used for feature selection.

The experiment demonstrates selection under controlled conditions, not independent discovery of relationship meaning. All binary input combinations were represented in the training, validation and test sets, with different amplitude variations.

The results do not yet establish generalisation to previously unseen input combinations or an advantage over conventional feature-selection methods.

## Next Step

**FRC 14 — Adaptive Representation**

Investigate whether FRC can progressively introduce higher-order relationships when simpler representations are insufficient for a classification task, rather than constructing the full feature space in advance.