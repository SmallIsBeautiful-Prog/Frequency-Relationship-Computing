
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 17: RELATIONSHIP ANALYSIS

Purpose:
Apply explicit first-, second-, third-, and fourth-order
relationship expansion to direct file-derived channel frames.

The original byte sequence and frame order are preserved.

No classifier or hidden nonlinear learning is used.

Only standard Python libraries are required.
"""

import argparse
import hashlib
import itertools
from pathlib import Path
from statistics import mean


# ============================================================
# CONFIGURATION
# ============================================================

CHANNEL_COUNT = 8
MAX_ORDER = 4

CHANNEL_NAMES = "ABCDEFGH"

# Used when no input file is supplied.
DEFAULT_SAMPLE = (
    b"Frequency Relationship Computing - "
    b"FRC17 Direct File Relationship Analysis. "
    b"Testing explicit channel relationships."
)


# ============================================================
# BYTE / CHANNEL CONVERSION
# ============================================================

def byte_to_channel(byte_value):
    """Convert a byte (0-255) into a channel value (-1 to +1)."""

    return (byte_value * 2.0 / 255.0) - 1.0


def channel_to_byte(channel_value):
    """Convert a channel value back into its original byte."""

    value = round((channel_value + 1.0) * 255.0 / 2.0)

    return max(0, min(255, value))


# ============================================================
# DIRECT FILE REPRESENTATION
# ============================================================

def encode_file(file_data):
    """
    Convert file bytes into ordered eight-channel frames.

    Incomplete frames are padded with the representation
    of byte zero. The valid-byte count identifies real data.
    """

    frames = []

    for start in range(0, len(file_data), CHANNEL_COUNT):

        block = file_data[start:start + CHANNEL_COUNT]

        valid_bytes = len(block)

        channels = [
            byte_to_channel(value)
            for value in block
        ]

        while len(channels) < CHANNEL_COUNT:
            channels.append(byte_to_channel(0))

        frames.append({
            "channels": channels,
            "valid_bytes": valid_bytes
        })

    return frames


def decode_frames(frames):
    """Reconstruct the original byte sequence."""

    reconstructed = bytearray()

    for frame in frames:

        for channel in frame["channels"][:frame["valid_bytes"]]:
            reconstructed.append(channel_to_byte(channel))

    return bytes(reconstructed)


# ============================================================
# RELATIONSHIP GENERATION
# ============================================================

def generate_relationships(frame):
    """
    Generate explicit relationships up to fourth order.

    First order:
        Individual channel values.

    Second order:
        Products of two different channels.

    Third order:
        Products of three different channels.

    Fourth order:
        Products of four different channels.

    Relationships involving padded channels are marked None.
    """

    channels = frame["channels"]
    valid_bytes = frame["valid_bytes"]

    relationship_groups = {}

    for order in range(1, MAX_ORDER + 1):

        group = []

        for combination in itertools.combinations(
            range(CHANNEL_COUNT), order
        ):

            names = "".join(
                CHANNEL_NAMES[index]
                for index in combination
            )

            # Do not calculate a relationship using padding.
            if any(index >= valid_bytes for index in combination):

                group.append({
                    "name": names,
                    "value": None
                })

                continue

            product = 1.0

            for index in combination:
                product *= channels[index]

            group.append({
                "name": names,
                "value": product
            })

        relationship_groups[order] = group

    return relationship_groups


# ============================================================
# FEATURE COUNTS
# ============================================================

def count_observed_features(relationship_groups):

    counts = {}

    for order, group in relationship_groups.items():

        counts[order] = sum(
            item["value"] is not None
            for item in group
        )

    return counts


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def analyse_relationships(all_relationships):

    print("\nRELATIONSHIP STATISTICS")
    print("-" * 70)

    print(
        f"{'Order':<12}"
        f"{'Observed':>12}"
        f"{'Minimum':>14}"
        f"{'Maximum':>14}"
        f"{'Mean |value|':>18}"
    )

    print("-" * 70)

    for order in range(1, MAX_ORDER + 1):

        values = []

        for groups in all_relationships:

            for item in groups[order]:

                if item["value"] is not None:
                    values.append(item["value"])

        if not values:
            print(f"{order:<12}No observed values")
            continue

        minimum = min(values)
        maximum = max(values)
        mean_absolute = mean(abs(value) for value in values)

        print(
            f"{order:<12}"
            f"{len(values):>12}"
            f"{minimum:>14.6f}"
            f"{maximum:>14.6f}"
            f"{mean_absolute:>18.6f}"
        )


# ============================================================
# FRAME PREVIEW
# ============================================================

def display_frame_preview(frame, relationship_groups, frame_number):

    print(f"\nFRAME {frame_number}")
    print("-" * 70)

    print(f"Valid input bytes: {frame['valid_bytes']}")

    counts = count_observed_features(relationship_groups)

    for order in range(1, MAX_ORDER + 1):

        group = relationship_groups[order]

        observed = [
            item for item in group
            if item["value"] is not None
        ]

        print(
            f"Order {order}: "
            f"{counts[order]} observed relationships "
            f"out of {len(group)} possible"
        )

        # Display first-order and pairwise values in detail.
        if order <= 2:

            for item in observed[:12]:

                print(
                    f"  {item['name']}: "
                    f"{item['value']:.6f}"
                )

            if len(observed) > 12:
                print(
                    f"  ... {len(observed) - 12} additional relationships"
                )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="FRC17 Direct File Relationship Analysis"
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Optional path to an input file"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 17: RELATIONSHIP ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    if args.file:

        input_path = Path(args.file)

        if not input_path.is_file():
            print(f"\nERROR: Input file not found: {input_path}")
            return

        file_data = input_path.read_bytes()
        source_name = str(input_path)

    else:

        file_data = DEFAULT_SAMPLE
        source_name = "Built-in FRC17 sample"

    original_hash = hashlib.sha256(file_data).hexdigest()

    print("\nINPUT FILE")
    print("-" * 70)
    print(f"Source: {source_name}")
    print(f"Original size: {len(file_data)} bytes")
    print(f"SHA-256: {original_hash}")

    # --------------------------------------------------------
    # Generate direct representation
    # --------------------------------------------------------

    frames = encode_file(file_data)

    print("\nDIRECT REPRESENTATION")
    print("-" * 70)
    print(f"Channel count: {CHANNEL_COUNT}")
    print(f"Frame count: {len(frames)}")
    print("Mapping: byte [0,255] -> channel [-1,+1]")
    print("Frame order: preserved")

    # --------------------------------------------------------
    # Generate relationships
    # --------------------------------------------------------

    all_relationships = []

    for frame in frames:

        groups = generate_relationships(frame)

        all_relationships.append(groups)

    print("\nRELATIONSHIP CONFIGURATION")
    print("-" * 70)

    for order in range(1, MAX_ORDER + 1):

        possible = sum(
            1 for _ in itertools.combinations(
                range(CHANNEL_COUNT), order
            )
        )

        print(
            f"Order {order}: {possible} possible features per full frame"
        )

    print("Total: 162 possible features per complete frame")

    # --------------------------------------------------------
    # Frame-level analysis
    # --------------------------------------------------------

    print("\nFRAME ANALYSIS")
    print("-" * 70)

    for index, (frame, groups) in enumerate(
        zip(frames, all_relationships)
    ):

        counts = count_observed_features(groups)

        print(
            f"Frame {index + 1}: "
            f"{frame['valid_bytes']} valid bytes | "
            f"{sum(counts.values())} observed features"
        )

    # --------------------------------------------------------
    # Detailed preview
    # --------------------------------------------------------

    if frames:

        print("\nRELATIONSHIP PREVIEW")
        print("=" * 70)

        display_count = min(2, len(frames))

        for index in range(display_count):

            display_frame_preview(
                frames[index],
                all_relationships[index],
                index + 1
            )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    analyse_relationships(all_relationships)

    # --------------------------------------------------------
    # Reconstruction verification
    # --------------------------------------------------------

    reconstructed = decode_frames(frames)

    reconstructed_hash = hashlib.sha256(
        reconstructed
    ).hexdigest()

    reconstruction_ok = reconstructed == file_data

    print("\nRECONSTRUCTION VERIFICATION")
    print("-" * 70)
    print(f"Original SHA-256:      {original_hash}")
    print(f"Reconstructed SHA-256: {reconstructed_hash}")
    print(
        "Byte-for-byte match:   "
        f"{'PASS' if reconstruction_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FRC17 FINAL RESULTS")
    print("=" * 70)

    if reconstruction_ok:
        print("Representation reconstruction: PASS")
    else:
        print("Representation reconstruction: FAIL")

    print(
        f"Frames analysed: {len(frames)}"
    )

    print(
        "Relationship orders analysed: "
        f"{MAX_ORDER}"
    )

    if reconstruction_ok:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")

    print("=" * 70)


if __name__ == "__main__":
    main()
