
"""
======================================================================
FREQUENCY RELATIONSHIP COMPUTING
FRC18: FREQUENCY-BASIS REPRESENTATION
======================================================================

PURPOSE

Compare two numerical representations of identical file data:

A. Direct byte-to-channel mapping.
B. Eight-point orthonormal Discrete Cosine Transform (DCT-II).

Both representations undergo the same explicit relationship
analysis using first-, second-, third- and fourth-order products.

The experiment also verifies that the DCT representation can
reconstruct the original file bytes.

IMPORTANT

- DCT coefficients are mathematical frequency-basis components.
- They are not physical frequencies or physical wave interactions.
- The DCT is a linear transformation.
- Relationship products are explicitly calculated and visible.
- No classifier or hidden nonlinear learning is used.

RUN

    py src\\frc_18_frequency_basis_representation.py

OPTIONAL FILE INPUT

    py src\\frc_18_frequency_basis_representation.py "path\\to\\file"

DEPENDENCIES

Python standard library only.
"""

import sys
import math
import hashlib

from pathlib import Path
from itertools import combinations


# ======================================================================
# 1. EXPERIMENT CONFIGURATION
# ======================================================================

CHANNEL_COUNT = 8

MAX_RELATIONSHIP_ORDER = 4

BUILT_IN_SAMPLE = (
    b"FRC18 Frequency Basis Representation Experiment. "
    b"This sample compares direct byte mapping with an "
    b"eight-point mathematical frequency basis."
)


# ======================================================================
# 2. FILE HANDLING
# ======================================================================

def load_input():
    """
    Load a supplied file or use the built-in sample.

    Returns:
        data: Original file bytes.
        source: Description of the input source.
    """

    if len(sys.argv) > 1:

        path = Path(sys.argv[1])

        if not path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

        data = path.read_bytes()

        source = str(path)

    else:

        data = BUILT_IN_SAMPLE

        source = "Built-in FRC18 sample"

    return data, source


def calculate_sha256(data):
    """Calculate the SHA-256 hash of byte data."""

    return hashlib.sha256(data).hexdigest()


# ======================================================================
# 3. DIRECT BYTE REPRESENTATION
# ======================================================================

def byte_to_channel(byte_value):
    """
    Convert a byte from the range 0-255
    into a numerical channel value from -1 to +1.
    """

    return (2.0 * byte_value / 255.0) - 1.0


def channel_to_byte(channel_value):
    """
    Convert a numerical channel value back into a byte.

    Rounding and clamping protect against small floating-point
    errors during reconstruction.
    """

    value = round(
        (channel_value + 1.0) * 255.0 / 2.0
    )

    return max(0, min(255, value))


def create_frames(data):
    """
    Divide file bytes into eight-channel frames.

    The final incomplete frame is padded using the numerical
    representation of byte zero.

    valid_bytes records the number of original bytes in each frame.
    """

    frames = []

    for start in range(0, len(data), CHANNEL_COUNT):

        block = data[start:start + CHANNEL_COUNT]

        valid_bytes = len(block)

        channels = [
            byte_to_channel(value)
            for value in block
        ]

        while len(channels) < CHANNEL_COUNT:

            channels.append(
                byte_to_channel(0)
            )

        frames.append({
            "channels": channels,
            "valid_bytes": valid_bytes
        })

    return frames


def reconstruct_direct_frames(frames):
    """
    Reconstruct original bytes from direct channel frames.
    """

    reconstructed = bytearray()

    for frame in frames:

        valid_bytes = frame["valid_bytes"]

        for value in frame["channels"][:valid_bytes]:

            reconstructed.append(
                channel_to_byte(value)
            )

    return bytes(reconstructed)


# ======================================================================
# 4. DISCRETE COSINE TRANSFORM
# ======================================================================

def dct_basis_scale(k, size):
    """
    Return the orthonormal DCT-II scaling factor.

    The first coefficient uses a different normalisation
    from the remaining coefficients.
    """

    if k == 0:

        return math.sqrt(1.0 / size)

    return math.sqrt(2.0 / size)


def dct_transform(values):
    """
    Calculate an orthonormal DCT-II.

    Input:
        Eight numerical channel values.

    Output:
        Eight mathematical frequency-basis coefficients.

    The transformation is linear and invertible.
    """

    size = len(values)

    coefficients = []

    for k in range(size):

        total = 0.0

        for n in range(size):

            angle = (
                math.pi / size
            ) * (n + 0.5) * k

            total += (
                values[n] * math.cos(angle)
            )

        coefficient = (
            dct_basis_scale(k, size) * total
        )

        coefficients.append(coefficient)

    return coefficients


def inverse_dct(coefficients):
    """
    Reconstruct numerical channel values from DCT coefficients.

    Uses the inverse of the orthonormal DCT-II.
    """

    size = len(coefficients)

    reconstructed = []

    for n in range(size):

        total = 0.0

        for k in range(size):

            angle = (
                math.pi / size
            ) * (n + 0.5) * k

            total += (
                dct_basis_scale(k, size)
                * coefficients[k]
                * math.cos(angle)
            )

        reconstructed.append(total)

    return reconstructed


# ======================================================================
# 5. FREQUENCY-BASIS FRAME GENERATION
# ======================================================================

def create_dct_frames(direct_frames):
    """
    Transform each direct frame into DCT coefficients.

    The original valid-byte count is retained for reconstruction.
    """

    frequency_frames = []

    for frame in direct_frames:

        coefficients = dct_transform(
            frame["channels"]
        )

        frequency_frames.append({
            "coefficients": coefficients,
            "valid_bytes": frame["valid_bytes"]
        })

    return frequency_frames


def reconstruct_dct_frames(frequency_frames):
    """
    Reconstruct the original file bytes from DCT frames.

    Each frame is inverse-transformed into its original
    eight numerical channel values.

    Only valid original bytes are converted back into bytes.
    """

    reconstructed = bytearray()

    for frame in frequency_frames:

        values = inverse_dct(
            frame["coefficients"]
        )

        valid_bytes = frame["valid_bytes"]

        for value in values[:valid_bytes]:

            reconstructed.append(
                channel_to_byte(value)
            )

    return bytes(reconstructed)


# ======================================================================
# 6. EXPLICIT RELATIONSHIP CALCULATION
# ======================================================================

def calculate_relationships(values):
    """
    Calculate explicit relationship products.

    First order:
        A

    Second order:
        A * B

    Third order:
        A * B * C

    Fourth order:
        A * B * C * D

    Only distinct channel indices are combined.

    All calculations are explicit and independently inspectable.
    """

    relationships = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    for order in range(
        1,
        MAX_RELATIONSHIP_ORDER + 1
    ):

        for indices in combinations(
            range(len(values)),
            order
        ):

            product = 1.0

            for index in indices:

                product *= values[index]

            relationships[order].append(product)

    return relationships


# ======================================================================
# 7. REPRESENTATION ANALYSIS
# ======================================================================

def analyse_representation(frames, value_key):
    """
    Analyse relationships in complete frames.

    Partial frames are excluded from the comparison.

    Reason:
        Padding an incomplete frame can influence all DCT
        coefficients, making its comparison with a direct
        representation less straightforward.

    Both paths therefore use exactly the same complete frames.
    """

    all_values = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    complete_frames = [

        frame for frame in frames

        if frame["valid_bytes"] == CHANNEL_COUNT

    ]

    for frame in complete_frames:

        relationships = calculate_relationships(
            frame[value_key]
        )

        for order in range(1, 5):

            all_values[order].extend(
                relationships[order]
            )

    return complete_frames, all_values


# ======================================================================
# 8. STATISTICS
# ======================================================================

def calculate_statistics(values):
    """
    Calculate basic statistics for relationship values.
    """

    if not values:

        return None

    return {
        "count": len(values),

        "minimum": min(values),

        "maximum": max(values),

        "mean_absolute": (
            sum(abs(value) for value in values)
            / len(values)
        )
    }


def print_relationship_statistics(title, values):
    """
    Display relationship statistics for all four orders.
    """

    print()
    print(title)

    print("-" * 82)

    print(
        f"{'Order':<12}"
        f"{'Count':>10}"
        f"{'Minimum':>16}"
        f"{'Maximum':>16}"
        f"{'Mean Absolute':>20}"
    )

    for order in range(1, 5):

        stats = calculate_statistics(
            values[order]
        )

        if stats is None:

            print(
                f"{order:<12}"
                f"{0:>10}"
            )

            continue

        print(
            f"{order:<12}"
            f"{stats['count']:>10}"
            f"{stats['minimum']:>16.6f}"
            f"{stats['maximum']:>16.6f}"
            f"{stats['mean_absolute']:>20.6f}"
        )

    total = sum(
        len(values[order])
        for order in range(1, 5)
    )

    print("-" * 82)

    print(
        f"Total relationship values: {total:,}"
    )


def compare_relationship_statistics(
    direct_relationships,
    frequency_relationships
):
    """
    Display a side-by-side comparison of mean absolute values.
    """

    print()
    print("DIRECT VS FREQUENCY-BASIS COMPARISON")

    print("-" * 82)

    print(
        f"{'Order':<12}"
        f"{'Direct Mean Abs.':>22}"
        f"{'DCT Mean Abs.':>22}"
        f"{'Difference':>22}"
    )

    for order in range(1, 5):

        direct_stats = calculate_statistics(
            direct_relationships[order]
        )

        dct_stats = calculate_statistics(
            frequency_relationships[order]
        )

        if direct_stats is None or dct_stats is None:
            continue

        direct_mean = direct_stats["mean_absolute"]

        dct_mean = dct_stats["mean_absolute"]

        difference = dct_mean - direct_mean

        print(
            f"{order:<12}"
            f"{direct_mean:>22.8f}"
            f"{dct_mean:>22.8f}"
            f"{difference:>22.8f}"
        )

    print("-" * 82)


# ======================================================================
# 9. RECONSTRUCTION VERIFICATION
# ======================================================================

def verify_reconstruction(original, reconstructed, title):
    """
    Compare original and reconstructed data.

    Uses both byte-for-byte comparison and SHA-256.
    """

    original_hash = calculate_sha256(original)

    reconstructed_hash = calculate_sha256(
        reconstructed
    )

    match = original == reconstructed

    print()
    print(title)

    print("-" * 72)

    print(
        f"Original SHA-256:      {original_hash}"
    )

    print(
        f"Reconstructed SHA-256: {reconstructed_hash}"
    )

    print(
        "Byte-for-byte match: "
        + ("PASS" if match else "FAIL")
    )

    return match


# ======================================================================
# 10. MAIN EXPERIMENT
# ======================================================================

def main():

    print("=" * 82)

    print("FREQUENCY RELATIONSHIP COMPUTING")

    print("FRC18: FREQUENCY-BASIS REPRESENTATION")

    print("=" * 82)

    # --------------------------------------------------------------
    # Load input
    # --------------------------------------------------------------

    original, source = load_input()

    print()
    print("INPUT FILE")

    print("-" * 82)

    print(f"Source: {source}")

    print(f"Original size: {len(original):,} bytes")

    print(
        f"SHA-256: {calculate_sha256(original)}"
    )

    # --------------------------------------------------------------
    # Create direct frames
    # --------------------------------------------------------------

    direct_frames = create_frames(original)

    complete_count = sum(

        1 for frame in direct_frames

        if frame["valid_bytes"] == CHANNEL_COUNT

    )

    partial_count = (
        len(direct_frames) - complete_count
    )

    print()
    print("FRAME CONFIGURATION")

    print("-" * 82)

    print(f"Channel count: {CHANNEL_COUNT}")

    print(f"Total frames: {len(direct_frames)}")

    print(f"Complete frames: {complete_count}")

    print(f"Partial frames: {partial_count}")

    # --------------------------------------------------------------
    # Create frequency-basis frames
    # --------------------------------------------------------------

    frequency_frames = create_dct_frames(
        direct_frames
    )

    print()
    print("REPRESENTATION CONFIGURATION")

    print("-" * 82)

    print(
        "Representation A: Direct byte-to-channel mapping"
    )

    print(
        "Representation B: Orthonormal eight-point DCT-II"
    )

    print("Coefficients per frame: 8")

    print(
        "Relationship orders: 1, 2, 3 and 4"
    )

    print("Classifier: None")

    # --------------------------------------------------------------
    # Display first frame comparison
    # --------------------------------------------------------------

    if direct_frames:

        print()
        print("FIRST FRAME COMPARISON")

        print("-" * 82)

        print("Direct channel values:")

        for index, value in enumerate(
            direct_frames[0]["channels"]
        ):

            print(
                f"  Channel {index}: {value: .8f}"
            )

        print()
        print("DCT frequency-basis coefficients:")

        for index, value in enumerate(
            frequency_frames[0]["coefficients"]
        ):

            print(
                f"  Coefficient {index}: {value: .8f}"
            )

    # --------------------------------------------------------------
    # Relationship analysis: direct representation
    # --------------------------------------------------------------

    direct_complete, direct_relationships = (
        analyse_representation(
            direct_frames,
            "channels"
        )
    )

    # --------------------------------------------------------------
    # Relationship analysis: frequency-basis representation
    # --------------------------------------------------------------

    frequency_complete, frequency_relationships = (
        analyse_representation(
            frequency_frames,
            "coefficients"
        )
    )

    # --------------------------------------------------------------
    # Display results
    # --------------------------------------------------------------

    print_relationship_statistics(
        "PATH A: DIRECT REPRESENTATION",
        direct_relationships
    )

    print_relationship_statistics(
        "PATH B: FREQUENCY-BASIS REPRESENTATION",
        frequency_relationships
    )

    compare_relationship_statistics(
        direct_relationships,
        frequency_relationships
    )

    # --------------------------------------------------------------
    # Verify direct representation reconstruction
    # --------------------------------------------------------------

    direct_reconstructed = reconstruct_direct_frames(
        direct_frames
    )

    direct_match = verify_reconstruction(
        original,
        direct_reconstructed,
        "DIRECT REPRESENTATION RECONSTRUCTION"
    )

    # --------------------------------------------------------------
    # Verify frequency-basis reconstruction
    # --------------------------------------------------------------

    frequency_reconstructed = reconstruct_dct_frames(
        frequency_frames
    )

    frequency_match = verify_reconstruction(
        original,
        frequency_reconstructed,
        "FREQUENCY-BASIS RECONSTRUCTION"
    )

    # --------------------------------------------------------------
    # Final experiment summary
    # --------------------------------------------------------------

    direct_total = sum(
        len(values)
        for values in direct_relationships.values()
    )

    frequency_total = sum(
        len(values)
        for values in frequency_relationships.values()
    )

    print()
    print("EXPERIMENT SUMMARY")

    print("-" * 82)

    print(
        f"Complete frames compared: {len(direct_complete)}"
    )

    print(
        "Direct relationship values: "
        f"{direct_total:,}"
    )

    print(
        "Frequency-basis relationship values: "
        f"{frequency_total:,}"
    )

    print(
        "Direct reconstruction: "
        + ("PASS" if direct_match else "FAIL")
    )

    print(
        "DCT reconstruction: "
        + ("PASS" if frequency_match else "FAIL")
    )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    print()
    print("FRC18 RESULT")

    if (
        direct_match
        and frequency_match
        and len(direct_complete) == len(frequency_complete)
    ):

        print("PASS: Both representations were analysed.")

        print("PASS: Direct reconstruction matched the original.")

        print("PASS: DCT reconstruction matched the original.")

    else:

        print(
            "FAIL: One or more verification checks require investigation."
        )

    print()
    print("=" * 82)

    print("FRC18 COMPLETE")

    print("=" * 82)


# ======================================================================
# 11. PROGRAM ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("EXPERIMENT ERROR")

        print("-" * 82)

        print(error)

        raise
