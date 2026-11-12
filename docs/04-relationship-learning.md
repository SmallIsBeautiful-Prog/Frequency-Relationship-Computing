# FRC 04 — Relationship Learning

## Purpose

Develop a reusable linear learning component capable of learning from original frequency amplitudes and pairwise frequency relationships, while identifying which features contribute to a task.

## Implementation

- Eight frequency carriers and 28 automatically generated pairwise relationships.
- Two simultaneous XOR classification tasks.
- Reusable `LinearRelationshipModel` using least-squares training.
- Training-only feature standardisation.
- Learned feature-weight ranking.
- Individual relationship ablation: remove one relationship, retrain and evaluate.

## Results

| Measurement | Accuracy |
|---|---:|
| Baseline, output 1 | 49.79% |
| Baseline, output 2 | 53.33% |
| Baseline, both outputs correct | 27.29% |
| Expanded, output 1 | 100.00% |
| Expanded, output 2 | 100.00% |
| Expanded, both outputs correct | 100.00% |

The learned weights identified the expected relationships:

- A-B, 100 Hz: -0.860669
- C-D, 160 Hz: -0.829972

Removing A-B reduced joint accuracy to 51.04%. Removing C-D reduced it to 49.38%. Removing any of the other 26 relationships individually retained 100% joint accuracy.

## What this demonstrates

The reusable linear classifier can learn multiple tasks simultaneously and assign different importance to individual pairwise relationships.

Ablation testing confirms that the two task-defining relationships are necessary for the observed performance, while the other relationships are individually dispensable in this experiment.

## Research status

Successful controlled digital experiment. Results demonstrate selective use of pairwise features within this dataset. They do not establish generalisation to arbitrary frequency configurations or independent real-world data.

## Next step

FRC 05 — Growth States: construct successive frequency states and investigate whether cancellation can recover newly introduced components while preserving their order.