# FRC17 — Relationship Analysis

## 1. Purpose

FRC17 investigates whether numerical representations derived directly from file bytes can be expanded into explicit mathematical relationships.

Building on FRC15 (Direct File Representation) and FRC16 (Representation Verification), this experiment introduces pairwise and higher-order relationship calculations within each eight-channel frame.

The objective is to examine the resulting relationship structures while preserving the ability to reconstruct the original file exactly.

## 2. Implementation

FRC17 reads file data as bytes and maps each byte into a numerical channel value between -1 and +1.

The conversion is:

\[
A_i = \frac{2b_i}{255}-1
\]

Where:

- \(b_i\) is the original byte value (0–255).
- \(A_i\) is the corresponding numerical channel value.
- Each frame contains up to eight channels.

The experiment then calculates explicit relationships using combinations of channel values.

### Relationship orders

| Order | Calculation | Features per complete frame |
|---|---|---:|
| First order | \(A_i\) | 8 |
| Second order | \(A_i A_j\) | 28 |
| Third order | \(A_i A_j A_k\) | 56 |
| Fourth order | \(A_i A_j A_k A_l\) | 70 |
| **Total** | | **162** |

Relationship calculations use combinations of distinct channels within each frame.

For incomplete frames, only relationships involving valid input bytes are calculated. Padded channels are excluded.

## 3. Experiment Configuration

- Input: Built-in sample text file.
- Original file size: 115 bytes.
- Channel count: 8.
- Frame count: 15.
- Complete frames: 14.
- Final incomplete frame: 3 valid bytes.
- Maximum relationship order: 4.
- Maximum features per complete frame: 162.
- Classifier: None.
- External dependencies: None.

The experiment uses Python's standard library.

## 4. Results

### 4.1 Frame representation

The 115-byte input was divided into 15 frames:

- 14 complete frames containing 8 bytes each.
- 1 final frame containing 3 valid bytes.

Each complete frame generated 162 relationship values.

The final three-byte frame generated:

| Relationship order | Valid features |
|---|---:|
| First order | 3 |
| Second order | 3 |
| Third order | 1 |
| Fourth order | 0 |
| **Total** | **7** |

### 4.2 Aggregate relationship counts

Across all 15 frames, the experiment produced:

| Relationship order | Total features |
|---|---:|
| First order | 115 |
| Second order | 395 |
| Third order | 785 |
| Fourth order | 980 |
| **Total** | **2,275** |

These counts reflect the number of valid relationships calculated from the original bytes.

### 4.3 Relationship value statistics

| Order | Minimum | Maximum | Mean absolute value |
|---|---:|---:|---:|
| First | -0.749020 | -0.050980 | 0.260904 |
| Second | 0.004598 | 0.561030 | 0.074080 |
| Third | -0.363020 | -0.000608 | 0.024788 |
| Fourth | 0.000083 | 0.223506 | 0.010036 |

The mean absolute relationship values decrease as relationship order increases.

This is an expected mathematical effect of multiplying values whose magnitudes are between zero and one. Repeated multiplication generally reduces magnitude.

It does not establish that higher-order relationships contain less information or have less potential value.

The sample contains ASCII text, meaning its byte values are below 128. Under the chosen mapping, these values become negative channel values. Consequently, the relationship signs alternate with order:

- First order: negative.
- Second order: positive.
- Third order: negative.
- Fourth order: positive.

This sign pattern is a consequence of the sample's numerical values and mapping, not evidence of a special text or frequency property.

## 5. Reconstruction Verification

The experiment also reconstructed the original file from its numerical frames.

**Original SHA-256:**

`330f96607229a18d2a313c756cd54ded1d78ac8a3d5cb7c7ff3f584e4185a36e`

**Reconstructed SHA-256:**

`330f96607229a18d2a313c756cd54ded1d78ac8a3d5cb7c7ff3f584e4185a36e`

**Result: PASS**

The reconstructed file matched the original byte-for-byte.

This confirms that the relationship analysis did not compromise the underlying file representation.

## 6. What This Demonstrates

FRC17 demonstrates that:

1. File bytes can be represented as ordered eight-channel numerical frames.
2. Explicit pairwise, third-order and fourth-order relationships can be calculated from those frames.
3. Incomplete frames can be handled without calculating relationships involving invalid padded channels.
4. Relationship counts can be verified mathematically against the number of valid input bytes.
5. The original file can still be reconstructed exactly after the analysis.

The experiment establishes a working method for expanding file-derived numerical representations into explicit within-frame relationships.

## 7. Limitations

Several important limitations remain:

- The experiment analyses relationships within individual frames only.
- Relationships between consecutive frames have not yet been investigated.
- The input is a small sample of ASCII text, not a representative collection of different file types.
- No classification, prediction or semantic recognition is performed.
- No physical frequency interaction or analogue computation is demonstrated.
- All data is processed in memory; streaming and large-file processing have not been tested.
- The experiment does not establish that the calculated relationships provide useful information for a particular task.

The numerical relationships are explicit mathematical products of the mapped byte values. Their practical value requires further testing.

## 8. Research Status

**FRC17 — Completed: Initial Relationship Analysis**

FRC17 successfully expanded file-derived numerical frames into first-, second-, third- and fourth-order relationships.

The 115-byte sample generated 2,275 valid relationship values across 15 frames. All 14 complete frames produced 162 features each, while the final three-byte frame produced 7 valid features.

Original-file reconstruction passed SHA-256 verification, confirming an exact byte-for-byte match.

This experiment establishes the initial within-frame relationship analysis capability. It does not yet demonstrate semantic understanding, classification performance or the usefulness of these relationships for different file types.

## 9. Position Within the FRC Research

FRC17 builds on the file-representation experiments:

| Experiment | Focus |
|---|---|
| FRC15 | Direct File Representation |
| FRC16 | Representation Verification |
| FRC17 | Relationship Analysis |

Together, these experiments establish an initial pipeline:

**File bytes → Numerical frames → Explicit relationships → Reconstruction verification**

Further experiments can investigate how this representation behaves across different file types and whether relationships between frames provide additional useful information.