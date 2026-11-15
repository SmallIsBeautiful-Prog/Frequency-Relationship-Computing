# FRC14 — Adaptive Representation

## Purpose

FRC14 investigates whether Frequency Relationship Computing can automatically select the minimum feature representation required for different learning tasks.

Previous experiments demonstrated that higher-order relationships can represent increasingly complex patterns. However, using the highest available relationship order for every task may introduce unnecessary features.

This experiment introduces adaptive representation selection.

The objective is to determine whether each task can use a representation of an appropriate relationship order while maintaining classification accuracy.

The classifier remains a simple linear model. Adaptation occurs in the feature representation selected during model development, not inside the classifier.

## Implementation

FRC14 uses eight carrier-associated input channels:

```python
[12000, 11900, 11770, 11610, 11420, 11200, 10950, 10670]
```

The experiment constructs feature representations at four levels.

| Relationship order | Feature count | Description |
|---|---:|---|
| First order | 8 | Individual channels |
| Second order | 36 | Individual channels and pairwise relationships |
| Third order | 92 | First-, second-, and third-order features |
| Fourth order | 162 | Complete representation through fourth order |

### Adaptive task selection

Three classification tasks were used, each designed to require a different relationship order.

The experiment evaluates candidate representations using validation data and selects the minimum relationship order that satisfies the task's selection criterion.

The selected representation is then evaluated on separate test data.

This process allows each task to use a different feature representation without changing the classifier.

### Classifier architecture

The same simple linear least-squares classifier is used throughout the experiment:

\[
\hat{y}=w^Tx+b
\]

The classifier learns linear weights from the supplied features.

There are no neural networks, hidden nonlinear learning stages, or adaptive nonlinear classifiers.

The additional representational power comes from the explicitly calculated relationship features.

## Results

The adaptive selection process identified different minimum relationship orders for the three tasks.

| Classification task | Selected order | Selected features | Test accuracy |
|---|---:|---:|---:|
| XOR(A, B) | 2 | 36 | 100% |
| Parity(C, D, E) | 3 | 92 | 100% |
| Parity(E, F, G, H) | 4 | 162 | 100% |

All three tasks achieved 100% test accuracy using their selected representations.

### Feature efficiency

The experiment also compared the total number of task-specific feature connections required by the adaptive approach against using the full fourth-order representation for every task.

| Representation strategy | Total task-specific connections |
|---|---:|
| Fixed fourth-order representation | 486 |
| Adaptive representation | 290 |

The reduction was:

\[
\frac{486-290}{486}\times100
=40.33\%
\]

<badge color="success">40.33% fewer connections</badge>

The adaptive approach therefore reduced the total number of task-specific feature connections while maintaining the same reported test accuracy.

## What This Demonstrates

FRC14 demonstrates that different classification tasks can use different relationship orders within the same FRC architecture.

The experiment establishes several findings:

1. **Different tasks require different representations:** A second-order representation was sufficient for the XOR task, while the other tasks selected third- and fourth-order representations.
2. **Adaptive feature selection:** Validation-based selection identified the minimum qualifying representation order for each task.
3. **Linear classification remains unchanged:** All tasks used the same simple linear classifier.
4. **Reduced feature requirements:** Adaptive selection reduced the total task-specific feature connections by 40.33% compared with using fourth-order features for every task.
5. **Maintained test performance:** All three selected representations achieved 100% test accuracy.

An important distinction is that adaptation occurs during model development. The system selects a representation for each task; it does not dynamically determine the required relationship order for every individual input.

The tasks were deliberately constructed to require their corresponding parity orders. These results demonstrate successful selection under the experimental conditions, rather than proving that the method will always identify the minimum representation for arbitrary real-world tasks.

## Research Status

**Completed — Adaptive representation selection demonstrated.**

FRC14 successfully demonstrates that FRC can select different relationship orders for different tasks while retaining a fixed linear classifier.

The experiment achieved:

- 100% test accuracy across all three tasks.
- Three different selected relationship orders.
- A 40.33% reduction in task-specific feature connections.

The results support further investigation into representation efficiency and task-dependent relationship selection.

The experiment does not establish that adaptive selection will generalise to arbitrary tasks or datasets. Further testing with more varied tasks and unseen combinations would be needed.

## Next Step

**FRC15 — Direct File Representation**

The next experiment begins a new direction for FRC.

Rather than supplying predefined numerical channel inputs, FRC15 investigates converting actual digital file data into ordered eight-channel numerical frames.

This creates a potential pathway for applying FRC representations to ordinary files, including images, audio, and other binary data.

The initial objective is to establish a reversible mapping between file bytes and FRC channel representations before investigating relationship analysis on file-derived data.