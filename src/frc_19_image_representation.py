
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC19: IMAGE REPRESENTATION EXPERIMENTS

Purpose:
    Read an image file as raw bytes.
    Map bytes into ordered eight-channel numerical frames.
    Calculate explicit relationships of orders 1-4.
    Reconstruct the original image.
    Verify exact byte preservation.

Usage:
    py src\frc_19_image_representation.py

    py src\frc_19_image_representation.py "path\to\image.png"

Dependencies:
    Python standard library only.

Important:
    This experiment represents the image FILE.
    It does not yet decode pixels or perform image recognition.
"""

from pathlib import Path
from itertools import combinations
import hashlib
import math
import struct
import sys
import zlib
import binascii


# ============================================================
# CONFIGURATION
# ============================================================

CHANNEL_COUNT = 8
MAX_RELATIONSHIP_ORDER = 4

OUTPUT_DIR = Path("output")
SAMPLE_IMAGE = OUTPUT_DIR / "frc_19_sample_image.png"


# ============================================================
# IMAGE SAMPLE GENERATION
# ============================================================

def create_png_chunk(chunk_type, data):
    """Create a PNG chunk with its CRC checksum."""

    chunk = chunk_type + data
    crc = binascii.crc32(chunk) & 0xFFFFFFFF

    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", crc)
    )


def create_sample_png(path):
    """
    Generate a small valid RGB PNG using the standard library.

    The sample is created only if it does not already exist.
    """

    width = 16
    height = 16

    raw_image = bytearray()

    for y in range(height):
        raw_image.append(0)  # PNG filter type: None

        for x in range(width):
            red = int(x * 255 / (width - 1))
            green = int(y * 255 / (height - 1))
            blue = (red + green) // 2

            raw_image.extend([red, green, blue])

    png_data = bytearray()

    png_data.extend(b"\x89PNG\r\n\x1a\n")

    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        8,  # Bit depth
        2,  # RGB colour type
        0,
        0,
        0
    )

    png_data.extend(create_png_chunk(b"IHDR", ihdr))
    png_data.extend(
        create_png_chunk(b"IDAT", zlib.compress(bytes(raw_image)))
    )
    png_data.extend(create_png_chunk(b"IEND", b""))

    path.write_bytes(png_data)

    return width, height


# ============================================================
# IMAGE INFORMATION
# ============================================================

def detect_image_format(data, path):
    """Identify common image formats from their file signatures."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"

    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"

    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "GIF"

    if data.startswith(b"BM"):
        return "BMP"

    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "WEBP"

    return path.suffix.upper().lstrip(".") or "UNKNOWN"


def get_png_dimensions(data):
    """Read dimensions from a PNG IHDR header when available."""

    png_signature = b"\x89PNG\r\n\x1a\n"

    if not data.startswith(png_signature):
        return None

    if len(data) < 24:
        return None

    if data[12:16] != b"IHDR":
        return None

    width, height = struct.unpack(">II", data[16:24])

    return width, height


# ============================================================
# BYTE / CHANNEL CONVERSION
# ============================================================

def byte_to_channel(byte_value):
    """Convert an unsigned byte into the range [-1, +1]."""

    return (byte_value * 2.0 / 255.0) - 1.0


def channel_to_byte(channel_value):
    """Reconstruct an unsigned byte from a channel value."""

    result = round((channel_value + 1.0) * 255.0 / 2.0)

    return max(0, min(255, result))


# ============================================================
# DIRECT IMAGE REPRESENTATION
# ============================================================

def encode_image(data):
    """
    Divide image bytes into ordered frames of eight channels.

    The final frame may contain fewer than eight valid bytes.
    Padding is used only to complete the channel structure.
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
            channels.append(byte_to_channel(0))

        frames.append({
            "channels": channels,
            "valid_bytes": valid_bytes
        })

    return frames


# ============================================================
# IMAGE RECONSTRUCTION
# ============================================================

def decode_frames(frames):
    """Reconstruct the original byte sequence."""

    reconstructed = bytearray()

    for frame in frames:

        valid_bytes = frame["valid_bytes"]

        for channel in frame["channels"][:valid_bytes]:
            reconstructed.append(channel_to_byte(channel))

    return bytes(reconstructed)


# ============================================================
# RELATIONSHIP ANALYSIS
# ============================================================

def analyse_relationships(frames):
    """
    Calculate explicit within-frame relationships.

    Orders:
        1 = Individual channels
        2 = Pairwise products
        3 = Triplet products
        4 = Quadruplet products

    Incomplete frames use only their valid channels.
    """

    statistics = {
        order: {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "absolute_total": 0.0
        }
        for order in range(1, MAX_RELATIONSHIP_ORDER + 1)
    }

    preview_count = min(2, len(frames))

    for frame_index, frame in enumerate(frames):

        valid_count = frame["valid_bytes"]
        channels = frame["channels"][:valid_count]

        frame_counts = {}

        for order in range(1, MAX_RELATIONSHIP_ORDER + 1):

            count = 0

            for indexes in combinations(range(valid_count), order):

                value = math.prod(channels[index] for index in indexes)

                stats = statistics[order]

                stats["count"] += 1
                stats["absolute_total"] += abs(value)

                if stats["minimum"] is None or value < stats["minimum"]:
                    stats["minimum"] = value

                if stats["maximum"] is None or value > stats["maximum"]:
                    stats["maximum"] = value

                count += 1

            frame_counts[order] = count

        if frame_index < preview_count:

            print(f"\nFRAME {frame_index + 1}")
            print("-" * 60)

            print(f"Valid bytes: {valid_count}")

            print("Channel values:")

            for index, value in enumerate(channels):
                print(f"  Channel {index + 1}: {value:.8f}")

            print("Relationship counts:")

            for order, count in frame_counts.items():
                print(f"  Order {order}: {count}")

    return statistics


# ============================================================
# HASHING
# ============================================================

def calculate_sha256(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# OUTPUT FILE MANAGEMENT
# ============================================================

def get_unique_output_path(original_path):
    """Find an output filename without overwriting an existing file."""

    stem = original_path.stem
    suffix = original_path.suffix

    candidate = OUTPUT_DIR / f"frc_19_reconstructed_{stem}{suffix}"

    counter = 1

    while candidate.exists():

        candidate = OUTPUT_DIR / (
            f"frc_19_reconstructed_{stem}_{counter}{suffix}"
        )

        counter += 1

    return candidate


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 78)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC19: IMAGE REPRESENTATION EXPERIMENTS")
    print("=" * 78)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        input_path = Path(sys.argv[1])

        if not input_path.is_file():
            print(f"\nERROR: Input file not found: {input_path}")
            return

    else:

        input_path = SAMPLE_IMAGE

        if not input_path.exists():

            print("\nCreating default sample image...")

            width, height = create_sample_png(input_path)

            print(f"Sample dimensions: {width} x {height}")

        else:

            print("\nUsing existing sample image.")

    original_data = input_path.read_bytes()

    if not original_data:
        print("\nERROR: Image file is empty.")
        return

    # --------------------------------------------------------
    # INPUT INFORMATION
    # --------------------------------------------------------

    image_format = detect_image_format(original_data, input_path)
    dimensions = get_png_dimensions(original_data)

    print("\nINPUT IMAGE")
    print("-" * 78)

    print(f"File: {input_path}")
    print(f"Format: {image_format}")
    print(f"File size: {len(original_data):,} bytes")

    if dimensions:
        print(f"PNG dimensions: {dimensions[0]} x {dimensions[1]}")
    else:
        print("Dimensions: Not inspected in this experiment")

    original_hash = calculate_sha256(original_data)

    print(f"SHA-256: {original_hash}")

    # --------------------------------------------------------
    # REPRESENTATION
    # --------------------------------------------------------

    print("\nDIRECT BYTE-TO-CHANNEL REPRESENTATION")
    print("-" * 78)

    frames = encode_image(original_data)

    complete_frames = sum(
        1 for frame in frames
        if frame["valid_bytes"] == CHANNEL_COUNT
    )

    partial_frames = len(frames) - complete_frames

    print(f"Channel count: {CHANNEL_COUNT}")
    print("Byte mapping range: [-1, +1]")
    print(f"Total frames: {len(frames):,}")
    print(f"Complete frames: {complete_frames:,}")
    print(f"Partial frames: {partial_frames:,}")

    # --------------------------------------------------------
    # RELATIONSHIP EXPANSION
    # --------------------------------------------------------

    print("\nEXPLICIT RELATIONSHIP EXPANSION")
    print("-" * 78)

    print("Relationship orders: 1, 2, 3 and 4")
    print("Classifier: None")
    print("Hidden nonlinear learning: None")

    statistics = analyse_relationships(frames)

    print("\nAGGREGATE RELATIONSHIP STATISTICS")
    print("-" * 78)

    total_features = 0

    for order, stats in statistics.items():

        count = stats["count"]
        total_features += count

        if count:

            mean_absolute = stats["absolute_total"] / count

            print(f"\nOrder {order}")
            print(f"  Feature count: {count:,}")
            print(f"  Minimum: {stats['minimum']:.8f}")
            print(f"  Maximum: {stats['maximum']:.8f}")
            print(f"  Mean absolute value: {mean_absolute:.8f}")

    print(f"\nTotal relationship values: {total_features:,}")

    # --------------------------------------------------------
    # RECONSTRUCTION
    # --------------------------------------------------------

    print("\nIMAGE RECONSTRUCTION")
    print("-" * 78)

    reconstructed_data = decode_frames(frames)

    output_path = get_unique_output_path(input_path)

    output_path.write_bytes(reconstructed_data)

    reconstructed_hash = calculate_sha256(reconstructed_data)

    print(f"Reconstructed file: {output_path}")
    print(f"Reconstructed size: {len(reconstructed_data):,} bytes")

    print(f"Original SHA-256:      {original_hash}")
    print(f"Reconstructed SHA-256: {reconstructed_hash}")

    byte_match = original_data == reconstructed_data
    hash_match = original_hash == reconstructed_hash

    print(f"\nByte-for-byte match: {byte_match}")
    print(f"SHA-256 match: {hash_match}")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 78)

    if byte_match and hash_match:

        print("FRC19 RESULT: PASS")
        print("Image file represented and reconstructed exactly.")

    else:

        print("FRC19 RESULT: FAIL")
        print("Image reconstruction did not match the original.")

    print("=" * 78)


if __name__ == "__main__":
    main()
