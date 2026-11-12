# Frequency Relationship Computing (FRC)

**An experimental approach to expanding the computational capabilities of simple linear classifiers through frequency relationships.**

Frequency Relationship Computing explores whether information represented as frequencies, and the relationships formed between them, can provide increasingly expressive representations for learning and computation.

The central idea is to keep the learning mechanism relatively simple while progressively expanding the information supplied to it.

## Core concept

1. **Frequency representation** — Encode information using individual frequency components.
2. **Relationship construction** — Combine frequencies to establish pairwise relationships.
3. **Feature extraction** — Use square-law processing to expose interaction components, including difference frequencies.
4. **Learning** — Supply original frequency measurements and their relationships to a linear classifier.
5. **Progressive development** — Extend the representation to multiple relationships, sequential growth states, symbolic information and more complex tasks.

## Current development

### FRC 01 — Original XOR Learning

The first experiment uses two frequencies:

- Carrier A: 12,000 Hz
- Carrier B: 11,900 Hz
- Pairwise difference: 100 Hz

The combined waveform is squared to expose the pairwise interaction. A least-squares linear classifier is then trained using two representations:

| Representation | Test accuracy |
|---|---:|
| Original features (A, B) | 49.58% |
| Pairwise-enhanced (A, B, A×B) | 100.00% |

The experiment demonstrates that adding a pairwise interaction feature enables a simple linear classifier to solve XOR in this controlled digital test.

See [`docs/01-original-xor.md`](docs/01-original-xor.md).

## Project structure

```text
frequency-relationship-computing/
│
├── src/
│   └── frc_01_xor.py
│
├── docs/
│   └── 01-original-xor.md
│
├── README.md
├── LICENSE
└── requirements.txt
```

New scripts and documentation will be added as capabilities are developed and tested.

## Requirements

- Python 3.10 or later
- NumPy

Install the dependency:

```bash
pip install -r requirements.txt
```

Run the first experiment from the project root:

```bash
python src/frc_01_xor.py
```

On Windows, this can also be run using:

```bat
py src\frc_01_xor.py
```

## Development roadmap

The project will progressively investigate:

- Multiple simultaneous pairwise relationships.
- Scalable frequency representations.
- Sequential growth states and cancellation.
- Recovery and preservation of introduction order.
- Higher-order relationships.
- Symbolic and numerical information.
- Compositional learning, abstraction and reasoning.
- Memory, context and adaptive learning.
- Audio, image and multimodal applications.
- Generative capabilities.

These are research directions, not claimed capabilities. Each stage will be implemented and evaluated before being described as demonstrated.

## Research approach

FRC is an experimental research project. Results will be documented alongside the methods used to obtain them.

Experiments will aim to distinguish demonstrated capabilities from hypotheses, compare against appropriate conventional approaches, and identify limitations.

## Licence

Released under the MIT License. See [`LICENSE`](LICENSE).

Contributions, experimentation and independent evaluation are welcome.