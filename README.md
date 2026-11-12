# Frequency Relationship Computing (FRC)

**An experimental approach to expanding the computational capabilities of simple linear classifiers through frequency relationships.**

Frequency Relationship Computing explores whether information represented as frequencies, and the relationships formed between them, can provide increasingly expressive representations for learning and computation.

The central research idea is to retain a relatively simple learning mechanism while progressively expanding the information supplied to it.

Rather than continually increasing the complexity of the classifier, FRC investigates whether richer representations of frequency relationships, growth states and higher-order interactions can expand the range of tasks that the classifier can learn.

---

## Core concept

The current architecture investigates five main stages:

1. **Information encoding** — Represent binary and numerical information using individual frequency components.
2. **Relationship construction** — Establish relationships between frequency components.
3. **Feature expansion** — Use square-law processing and mathematical products to expose pairwise and higher-order interactions.
4. **Growth-state construction** — Preserve successive representations and recover newly introduced components and relationships.
5. **Learning** — Supply the resulting features to a consistent linear classifier that learns numerical weights and produces predictions.

The underlying classifier remains based on linear least-squares learning. The research focuses on expanding the representation supplied to it.

## Current development

<badge color="success">11 experiments completed</badge>

The first eleven experiments establish the current foundation of FRC.

### Foundation: Pairwise relationships and learning

| Experiment | Capability investigated | Main result |
|---|---|---|
| FRC 01 | Original XOR learning | Pairwise enhancement improved accuracy from 49.58% to 100%. |
| FRC 02 | Multiple simultaneous relationships | Two independent XOR outputs achieved 100% accuracy. |
| FRC 03 | Scalable frequency representation | 4, 6 and 8 carriers generated 6, 15 and 28 unique relationships respectively; all configurations achieved 100% on the tested tasks. |
| FRC 04 | Relationship learning and ablation | Identified the specific pairwise relationships required for two simultaneous XOR tasks. |

### Growth states and order

| Experiment | Capability investigated | Main result |
|---|---|---|
| FRC 05 | Sequential frequency growth states | Eight cumulative states constructed; every newly introduced component recovered exactly by subtraction. |
| FRC 06 | State cancellation | Recovered all 28 incremental pairwise relationships from squared-state differences. |
| FRC 07 | Incremental relationship records | Recorded all 28 relationships, their introduction states and their cumulative presence. |
| FRC 08 | Ordered representation | Achieved 100% accuracy on an order-dependent task using introduction-position features. |
| FRC 09 | Growth-derived learning | Reconstructed relationship introduction positions from cumulative presence snapshots; achieved 100% accuracy on the order-dependent task. |

### Computational expansion

| Experiment | Capability investigated | Main result |
|---|---|---|
| FRC 10 | Boolean learning | One multi-output linear model learned AND, OR, XOR and NAND simultaneously, achieving 100% accuracy on all four outputs. |
| FRC 11 | Higher-order relationships | Third-order features enabled 100% accuracy on two simultaneous three-input parity tasks. |

---

## Key experimental results

### FRC 01 — Pairwise-enhanced XOR

Two carrier frequencies:

- A: 12,000 Hz
- B: 11,900 Hz
- Difference frequency: 100 Hz

The combined waveform was squared to expose the pairwise interaction.

| Representation | Test accuracy |
|---|---:|
| Original features (A, B) | 49.58% |
| Pairwise-enhanced (A, B, A×B) | 100.00% |

This demonstrated that a pairwise interaction feature can make XOR learnable by a simple linear classifier in a controlled digital experiment.

### FRC 08–09 — Learning introduction order

Eight frequencies produce 28 unique pairwise relationships and 40,320 possible introduction orders.

The task was to predict whether one specified relationship was introduced before another.

| Representation | FRC 08 accuracy | FRC 09 accuracy |
|---|---:|---:|
| Original carrier features | 50.05% | 50.04% |
| Unordered relationships | 49.82% | 50.29% |
| Ordered / growth-derived features | 100.00% | 100.00% |

FRC 08 supplied introduction positions from the known growth order. FRC 09 reconstructed those positions from cumulative relationship-presence snapshots.

These experiments demonstrate the value of retaining growth history. They do not yet establish automatic recovery of every possible introduction order directly from measured waveforms.

### FRC 10 — Boolean learning

One classifier learned four different Boolean operations simultaneously.

| Operation | Original features | Pairwise-enhanced |
|---|---:|---:|
| AND | 100.00% | 100.00% |
| OR | 100.00% | 100.00% |
| XOR | 50.09% | 100.00% |
| NAND | 100.00% | 100.00% |
| Average | 87.52% | 100.00% |

The experiment demonstrated that pairwise enhancement supplied the additional representation required for XOR while preserving the ability to learn the other operations.

### FRC 11 — Higher-order relationships

Eight carriers generated:

- 8 original carrier features.
- 28 pairwise relationships.
- 56 three-way relationships.
- 92 total features in the third-order representation.

Two simultaneous tasks were tested:

- A XOR B XOR C
- D XOR E XOR F

| Representation | Features | Average accuracy | Joint accuracy |
|---|---:|---:|---:|
| Original | 8 | 50.20% | 24.90% |
| Pairwise-enhanced | 36 | 50.35% | 25.69% |
| Third-order-enhanced | 92 | 100.00% | 100.00% |

The experiment demonstrated that pairwise features alone were insufficient for these three-input parity tasks, while the inclusion of third-order products enabled the linear classifier to learn both outputs.

---

## Mathematical foundation

For a combined signal:

\[
x(t)=\sum_i A_i\sin(2\pi f_i t+\phi_i)
\]

Square-law processing produces:

\[
y(t)=x(t)^2
\]

The resulting waveform contains self-interaction terms and cross-interaction terms.

For two different frequency components, the cross terms include sum and difference frequencies. The difference-frequency amplitude is proportional to the product of the two original amplitudes.

This provides a signal-based route to pairwise relationship features.

For binary inputs, the corresponding mathematical feature expansion includes:

\[
\phi_{ij}=A_iA_j
\]

Higher-order representations extend this principle:

\[
\phi_{ijk}=A_iA_jA_k
\]

These products provide additional dimensions in which a linear classifier can learn relationships that are not linearly separable using the original inputs alone.

Direct arithmetic multiplication and signal-derived interaction measurements have been shown to agree in the controlled digital experiments performed so far.

---

## Current architecture

The current development direction is:

```text
Information
    |
    v
Frequency Encoding
    |
    v
Growth-State Construction
    |
    v
State Cancellation
    |
    v
Relationship Extraction
    |
    v
Pairwise and Higher-Order Features
    |
    v
Linear Classifier
    |
    v
Predictions
```

The architecture is being developed incrementally. Each new experiment investigates an additional capability while retaining the same underlying learning principle wherever possible.

---

## Project structure

```text
frequency-relationship-computing/
│
├── src/
│   ├── frc_01_xor.py
│   ├── frc_02_multi_pair.py
│   ├── frc_03_scalable_representation.py
│   ├── frc_04_relationship_learning.py
│   ├── frc_05_growth_states.py
│   ├── frc_06_state_cancellation.py
│   ├── frc_07_incremental_relationships.py
│   ├── frc_08_ordered_representation.py
│   ├── frc_09_growth_learning.py
│   ├── frc_10_boolean_learning.py
│   └── frc_11_higher_order.py
│
├── docs/
│   └── Individual experiment documentation
│
├── output/
│   └── Generated experimental results
│
├── README.md
├── LICENSE
└── requirements.txt
```

The structure will expand as additional experiments are completed. Generated output files may be selectively included where they provide useful supporting evidence.

---

## Requirements

- Python 3.10 or later
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment from the project root:

```bash
python src/frc_01_xor.py
```

On Windows:

```bat
py src\frc_01_xor.py
```

Each experiment has its own script and can be run independently.

---

## Development roadmap

The next research stages will investigate:

1. **Compositional learning** — Combine previously constructed relationships into new features.
2. **Relationship selection** — Identify which relationships are useful for a given task.
3. **Adaptive representations** — Expand representations according to task requirements.
4. **Symbolic encoding** — Represent symbolic concepts using frequencies and their relationships.
5. **Hierarchical learning** — Build increasingly complex representations from simpler learned relationships.
6. **Persistent knowledge** — Store, retrieve and reuse learned information.
7. **Context and transfer** — Investigate whether learned relationships can be reused across different tasks.
8. **Multimodal learning** — Explore numerical, audio and image representations.
9. **Generation** — Investigate whether the architecture can produce structured outputs and sequences.
10. **Scaling** — Evaluate larger learning systems and more demanding tasks.

These are research directions, not demonstrated capabilities. Each stage will be implemented and evaluated before being described as achieved.

---

## Research approach and limitations

FRC is an experimental research project.

The current results are primarily controlled digital demonstrations using known frequency components, synthetic inputs and explicitly constructed mathematical features.

Important distinctions:

- Square-law processing is a nonlinear transformation of a signal. Linear addition alone does not generate new difference-frequency tones.
- Directly calculated products are mathematically equivalent to the corresponding ideal pairwise interaction amplitudes under the tested conditions.
- Growth-state cancellation currently relies on controlled digital signals with consistent timing, amplitude and phase.
- Ordered representations depend on retaining growth history or reconstructing it from cumulative snapshots.
- Higher-order products are explicitly constructed features; their success does not demonstrate automatic discovery of arbitrary relationships.
- High accuracy on selected synthetic tasks does not establish general intelligence or superiority over conventional machine-learning approaches.
- No computational efficiency advantage has yet been demonstrated.

Future experiments should include independent datasets, appropriate baselines, noise and robustness testing, larger tasks and comparisons with conventional feature-engineering and learning methods.

The long-term objective is to investigate how far a simple linear learning mechanism can be extended through increasingly expressive frequency-based representations.

---

## Licence

Released under the MIT License. See [`LICENSE`](LICENSE).

Contributions, experimentation and independent evaluation are welcome.