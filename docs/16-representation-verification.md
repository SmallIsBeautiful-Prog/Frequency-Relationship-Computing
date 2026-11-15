# FRC16 — Representation Verification

## Purpose

FRC16 investigates the reliability of the direct byte-to-channel representation introduced in FRC15.

FRC15 demonstrated that a sample file could be converted into ordered eight-channel numerical frames and reconstructed exactly.

FRC16 extends this verification across different file sizes, frame boundaries, and binary content.

The objective is to establish whether the representation consistently preserves original file data, including empty files, incomplete frames, and all possible byte values.

## Implementation

FRC16 uses the same direct byte-to-channel mapping established in FRC15.

Each byte is converted from the integer range 0–255 into a numerical channel value between -1 and +1.

\[
A_i = \frac{2b_i}{255}-1
\]

Consecutive groups of eight bytes form eight-channel frames.

Each frame contains eight numerical channels and a `valid_bytes` field identifying how many channels contain actual input data.

Incomplete frames are padded using the representation of byte value zero.

The reconstruction process uses the valid-byte count to exclude padding and recover the original byte sequence.

### Verification tests

Eight tests were performed:

1. Empty file.
2. Single-byte file.
3. File smaller than eight bytes.
4. File containing exactly eight bytes.
5. File with an incomplete final frame.
6. Larger text file.
7. Valid PNG image.
8. Exhaustive verification of all 256 possible byte values.

The PNG test used a programmatically generated valid 2 × 2 RGB image. Its structure was checked using the PNG signature and chunk CRC checksums.

Each file-representation test checked:

- Expected frame count.
- Eight channels per frame.
- Correct valid-byte counts.
- Channel values within the permitted range.
- Exact byte-for-byte reconstruction.
- Matching SHA-256 hashes.

The exhaustive byte test independently checked that every possible byte value could be converted and reconstructed correctly.

## Results

**All eight tests passed. No failures were recorded.**

| Test | Input size | Generated frames | Result |
|---|---:|---:|---|
| Empty file | 0 bytes | 0 | PASS |
| Single-byte file | 1 byte | 1 | PASS |
| Smaller than eight bytes | 5 bytes | 1 | PASS |
| Exactly eight bytes | 8 bytes | 1 | PASS |
| Incomplete final frame | 19 bytes | 3 | PASS |
| Larger text file | 58,000 bytes | 7,250 | PASS |
| Valid PNG image | 75 bytes | 10 | PASS |
| All byte values | 256 values | — | PASS |

### Overall verification

| Measurement | Result |
|---|---:|
| Tests completed | 8 |
| Tests passed | 8 |
| Tests failed | 0 |
| Byte values tested | 256 |
| Byte values reconstructed correctly | 256 |

<badge color="success">ALL TESTS PASSED</badge>

### Data integrity

Every file-representation test produced an exact reconstruction.

The SHA-256 comparison confirmed that the original and reconstructed byte sequences were identical in every case.

## What This Demonstrates

FRC16 provides further evidence that the direct byte-to-channel representation preserves arbitrary byte sequences under the tested conditions.

The experiment establishes several important properties:

1. **Boundary handling:** Empty files, partial frames, and complete frames are handled correctly.
2. **Complete byte coverage:** All 256 possible byte values can be converted and reconstructed without error.
3. **File-type independence:** Both text data and PNG binary data can be represented and reconstructed exactly.
4. **Preserved ordering:** Consecutive frames retain the original byte sequence.
5. **Padding management:** Incomplete frames do not introduce unwanted bytes during reconstruction.
6. **Data integrity:** SHA-256 comparisons confirm exact reconstruction across all file tests.

The larger text-file test is particularly useful because it verifies 7,250 consecutive frames rather than only a small number of frames.

The PNG test also confirms that the representation is not restricted to readable text.

However, this experiment tests binary representation and reconstruction, not image interpretation or semantic understanding.

## Research Status

**Completed — Direct representation verification passed.**

FRC16 successfully verified the byte-to-channel conversion across all eight test cases.

The results support the reliability of the representation for the tested input sizes and data types.

Several limitations remain:

- The PNG test used a small synthetic image rather than a collection of real-world images.
- Streaming and large-file processing have not yet been tested.
- The current implementation retains the input and generated frames in memory.
- Reduced numerical precision has not been evaluated.

No classification performance or semantic recognition capability is claimed.

## Next Step

**FRC17 — Relationship Analysis**

The next proposed investigation is to examine the relationships contained within file-derived eight-channel frames.

This would apply the existing FRC relationship expansion to the direct file representation, investigating:

- First-order channel values.
- Pairwise relationships.
- Third-order relationships.
- Fourth-order relationships.
- Differences between raw and expanded representations.

The existing simple linear classifier can remain unchanged if learning experiments are introduced.

The purpose would be to investigate whether explicit relationship expansion reveals useful structure in file-derived numerical data, without assuming that the byte mapping itself creates semantic meaning.