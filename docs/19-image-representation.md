# FRC19 — Image Representation Experiments

## 1. Objective

FRC19 investigates whether the existing Frequency Relationship Computing (FRC) direct file-representation architecture can be applied to image files.

The experiment takes an actual image file, reads its raw bytes, converts those bytes into ordered eight-channel numerical frames, and calculates explicit relationships up to fourth order.

The original image is then reconstructed to verify that the representation preserves the input data exactly.

This experiment establishes the initial connection between the FRC architecture and image data.

An important distinction is that FRC19 represents the image file itself. It does not yet decode or interpret individual pixels, identify objects, or perform visual classification.

## 2. Experimental Design

The experiment uses direct byte-to-channel mapping.

The input image is read as a sequence of bytes and divided into groups of eight.

Each byte is converted into a numerical channel value using:

\[
A_i = \frac{2b_i}{255}-1
\]

Where:

- \(b_i\) is the original byte value (0–255).
- \(A_i\) is the resulting numerical channel value (-1 to +1).

Each group of eight bytes forms one ordered frame.

If the final frame contains fewer than eight bytes, the remaining channels are padded. The number of valid bytes is retained so that padding is excluded from relationship calculations and reconstruction.

No DCT, FFT, hybrid representation or other frequency transformation is used.

## 3. Input Image

FRC19 initially uses a generated 16 × 16 RGB PNG image.

| Property | Result |
|---|---|
| Image format | PNG |
| Image dimensions | 16 × 16 |
| Original file size | 556 bytes |
| SHA-256 | `17313a19afe4b5f2d3cf1ba2cd891e7ee7deee13f5c940d041d4a9e9b05d256e` |

The script also accepts an externally supplied image file through a command-line argument.

The experiment identifies common image formats through their file signatures, but does not decode their visual contents.

## 4. Direct Byte-to-Channel Representation

The 556-byte image file was converted into eight-channel frames.

| Measurement | Result |
|---|---:|
| Channel count | 8 |
| Mapping range | -1 to +1 |
| Total frames | 70 |
| Complete frames | 69 |
| Partial frames | 1 |

The final frame contains four valid bytes.

The original byte sequence and frame ordering are preserved throughout the representation.

## 5. Explicit Relationship Expansion

The same explicit relationship expansion established in previous FRC experiments was applied to each frame.

The experiment calculates numerical products for relationship orders one through four.

| Relationship order | Features per complete frame |
|---|---:|
| First order | 8 |
| Second order | 28 |
| Third order | 56 |
| Fourth order | 70 |
| **Total** | **162** |

Incomplete frames calculate relationships only from their valid channels.

The experiment does not use a classifier or hidden nonlinear learning.

Relationship expansion remains central to the FRC architecture. It provides a more expressive representation by explicitly exposing pairwise and higher-order relationships.

The intended classifier remains relatively simple and strictly linear:

\[
\hat{y}=w^Tx+b
\]

The additional expressive capability is intended to come from the representation rather than from increasing the complexity of the classifier.

## 6. Experimental Results

### Aggregate relationship statistics

| Order | Feature count | Minimum | Maximum | Mean absolute value |
|---|---:|---:|---:|---:|
| First | 556 | -1.00000000 | 1.00000000 | 0.52224573 |
| Second | 1,938 | -0.98431373 | 1.00000000 | 0.28111495 |
| Third | 3,868 | -1.00000000 | 0.93847748 | 0.15574027 |
| Fourth | 4,831 | -0.89877739 | 1.00000000 | 0.08925180 |
| **Total** | **11,193** | | | |

The relationship counts are consistent with the number of complete and partial frames.

The 69 complete frames each produce 162 relationships.

The final four-byte frame produces:

- 4 first-order relationships.
- 6 second-order relationships.
- 4 third-order relationships.
- 1 fourth-order relationship.

This gives 11,193 valid relationship values across the entire image file.

### Relationship-value observations

Mean absolute relationship values decrease as relationship order increases.

This is consistent with the mathematical behaviour of products involving channel values in the range [-1,+1].

It does not establish that higher-order relationships are less informative or less useful.

The results demonstrate the relationship expansion process operating on image-file data, not the detection of visual features.

## 7. Reconstruction Verification

After relationship analysis, the original image was reconstructed from the numerical frames.

| Verification | Result |
|---|---|
| Original file size | 556 bytes |
| Reconstructed file size | 556 bytes |
| Byte-for-byte comparison | PASS |
| SHA-256 comparison | PASS |

Original SHA-256:

`17313a19afe4b5f2d3cf1ba2cd891e7ee7deee13f5c940d041d4a9e9b05d256e`

Reconstructed SHA-256:

`17313a19afe4b5f2d3cf1ba2cd891e7ee7deee13f5c940d041d4a9e9b05d256e`

Both hashes are identical.

The reconstructed image therefore preserves the original file exactly.

## 8. Interpretation

FRC19 establishes that the existing direct byte-to-channel architecture can process an actual image file.

The experiment demonstrates that:

1. Image-file bytes can be represented as ordered eight-channel numerical frames.
2. Explicit relationships up to fourth order can be calculated.
3. Incomplete frames can be handled without including padded channels in relationship calculations.
4. The original image can be reconstructed exactly.
5. The architecture does not require DCT or FFT transformations for this process.

However, the experiment does not establish image understanding or visual classification.

The PNG file contains headers, compressed image data and other file-level information. The numerical representation therefore reflects the encoded file structure rather than a direct representation of its individual visual elements.

## 9. Architectural Position

FRC19 retains the established FRC architecture:

**Original image bytes → Direct byte-to-channel mapping → Explicit relationship expansion → Simple linear classifier**

The classifier was not used in this experiment.

The purpose was to establish that image files can enter the existing representation architecture without introducing additional transformation stages.

Relationship expansion remains an essential part of FRC because it provides the expressive representation through which a simple linear classifier can extend its ability.

The next stage will investigate the visual contents of images rather than simply their encoded file bytes.

## 10. Limitations

- The experiment used one small, generated PNG image.
- Image pixels were not independently decoded or represented.
- No visual features or spatial relationships were calculated.
- No image classification was performed.
- No semantic understanding or object recognition was demonstrated.
- Large image datasets and streaming processing were not evaluated.
- The results do not establish that file-level relationships are useful for distinguishing image categories.

## 11. Conclusion

FRC19 successfully extends the direct byte-to-channel representation architecture to image files.

The experiment converted a 556-byte PNG image into 70 ordered eight-channel frames and calculated 11,193 explicit relationship values.

The original image was reconstructed exactly, with matching SHA-256 verification.

This confirms that the existing FRC representation can process image-file data while preserving the original information.

The next experiment will move beyond file-level representation and investigate whether the decoded visual contents of images can provide useful numerical relationships for classification.

**FRC19 status: Completed — PASS.**

**Next experiment: FRC20 — Cat vs Dog Classification.**