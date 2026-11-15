# FRC15 — Direct File Representation

## Purpose

FRC15 introduces a new direction for Frequency Relationship Computing (FRC): representing actual digital files as ordered numerical channel frames.

Previous FRC experiments used predefined numerical inputs associated with carrier channels. This experiment investigates whether arbitrary file data can be converted directly into the same eight-channel representation, creating a bridge between ordinary digital files and the FRC architecture.

The objective is to establish a simple, reversible representation without generating physical waveforms, performing Fourier transforms, or introducing unnecessary processing.

## Implementation

FRC15 implements a direct byte-to-channel mapping.

The input file is read as a sequence of bytes. Consecutive groups of eight bytes are converted into eight-channel frames.

Each byte is mapped from its original integer range of 0–255 into a normalised numerical range of -1 to +1.

The mapping is:

\[
A_i = \frac{2b_i}{255}-1
\]

Where:

- \(b_i\) is the original byte value.
- \(A_i\) is the resulting channel value.
- Each frame contains eight channels.

The eight channels correspond to the eight byte positions within each consecutive block.

### Frame structure

Each frame contains:

- Channel A
- Channel B
- Channel C
- Channel D
- Channel E
- Channel F
- Channel G
- Channel H

The original frame order is preserved.

If the final group contains fewer than eight bytes, the remaining channels are padded with the representation of byte value zero. A `valid_bytes` field records how many channels contain actual input data.

This allows the reconstruction process to distinguish original data from padding.

### Reconstruction

The reverse mapping is:

\[
b_i = \operatorname{round}\left(\frac{A_i+1}{2}\times255\right)
\]

Only valid bytes are reconstructed from the final frame.

The reconstructed byte sequence can then be compared directly with the original file.

## Results

The initial experiment used a sample text file containing 110 bytes.

| Measurement | Result |
|---|---:|
| Original file size | 110 bytes |
| Channel count | 8 |
| Generated frames | 14 |
| Full frames | 13 |
| Final frame | 6 valid bytes |
| Representation range | -1 to +1 |
| Frame order | Preserved |

### Reconstruction verification

The original file and reconstructed file produced identical SHA-256 hashes.

**Original SHA-256:**

`c226ff42a76c93b5055855e44cc49bf53a37432b33703272be1b955bc6b22786`

**Reconstructed SHA-256:**

`c226ff42a76c93b5055855e44cc49bf53a37432b33703272be1b955bc6b22786`

<badge color="success">PASS — Byte-for-byte match</badge>

The reconstructed file was identical to the original 110-byte input.

## What This Demonstrates

FRC15 establishes that arbitrary digital file data can be represented as ordered eight-channel numerical frames and reconstructed exactly.

This is an important architectural step because the representation now accepts ordinary file data rather than requiring manually prepared numerical inputs.

The experiment also establishes several useful properties:

1. **Direct conversion:** File bytes can be mapped into FRC channels using a simple mathematical transformation.
2. **Preserved ordering:** The sequence of frames retains the original byte order.
3. **Reversibility:** The original data can be recovered from the representation.
4. **Frame boundaries:** Files that do not divide evenly into eight-byte groups can still be represented without losing information.
5. **Compatibility:** The resulting frames have the same eight-channel structure used by the existing FRC experiments.

Importantly, these channels are numerical representation dimensions. They are not yet physical sinusoidal signals or measured frequency components.

The mapping itself does not establish semantic understanding of the file contents. It provides a reversible numerical representation that can be investigated by subsequent FRC experiments.

## Research Status

**Completed — Initial direct file representation verified.**

FRC15 successfully demonstrates the conversion and exact reconstruction of a sample file.

The current implementation loads the input file and its generated frames into memory. Streaming and large-file handling have not yet been implemented.

Further verification is required across different file types, file sizes, and edge cases before treating the representation as generally validated.

No classification or learning performance is claimed by this experiment.

## Next Step

**FRC16 — Representation Verification**

The next experiment will test the reliability of the direct byte-to-channel mapping across a wider range of inputs, including:

- Empty files.
- Single-byte files.
- Files smaller than eight bytes.
- Files containing exactly eight bytes.
- Files with incomplete final frames.
- Larger text files.
- Binary image files such as PNG.

The objective is to establish that the representation and reconstruction process works consistently across different file structures.

Once verified, subsequent experiments can investigate applying FRC pairwise and higher-order relationship expansion to file-derived representations.