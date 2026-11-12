# FRC 03 — Scalable Frequency Representation

## Purpose

Develop an automatically configurable frequency representation that generates all pairwise relationships from a supplied set of carrier frequencies.

## Implementation

- Automatically generated pairwise relationships for 4, 6 and 8 carriers.
- Applied square-law processing to expose difference-frequency components.
- Extracted all original carrier amplitudes and pairwise interaction amplitudes.
- Validated that carrier frequencies and pairwise difference frequencies were unique and aligned with FFT bins.
- Used the same ordinary least-squares classifier to learn two simultaneous tasks: XOR(A, B) and XOR(C, D).
- Compared original carrier features against the complete pairwise-enhanced representation.

## Results

| Carriers | Relationships | Features | Original joint accuracy | Enhanced joint accuracy |
|---:|---:|---:|---:|---:|
| 4 | 6 | 10 | 21.04% | 100.00% |
| 6 | 15 | 21 | 22.71% | 100.00% |
| 8 | 28 | 36 | 22.92% | 100.00% |

Each configuration used 1,120 training examples and 480 independent test examples.

The enhanced classifier achieved 100% joint accuracy in all three configurations.

## What this demonstrates

The frequency representation can automatically expand as additional carriers are introduced.

The number of unique pairwise relationships follows:

P = N(N − 1) / 2

The same linear learning method successfully used the expanded representation to solve both XOR tasks.

The learned weights consistently favoured the A–B and C–D relationships relevant to the respective outputs.

## Research status

**Demonstrated:** Automatic generation and extraction of pairwise features for 4, 6 and 8 carriers, with successful simultaneous learning of two XOR tasks.

**Not yet demonstrated:** Large-scale computational efficiency, transfer to unfamiliar frequency configurations or general-purpose reasoning.

## Next step

Investigate how the numerical weights of original features and pairwise relationships are learned, interpreted and compared within a consistent relationship-learning architecture.