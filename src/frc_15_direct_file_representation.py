
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 15: DIRECT FILE REPRESENTATION

Purpose:
    Convert an ordinary file into ordered eight-channel numerical
    frames, reconstruct the original file, and verify exact recovery.

Representation:
    8 consecutive bytes = 1 frame
    Each byte is mapped from [0, 255] to [-1.0, +1.0]

No audio waveform is generated.
No FFT is performed.
No classifier or learning algorithm is used.

Design principle:
    The representation handles the transformation.
    Future FRC classifiers remain strictly linear.
"""

from pathlib import Path
import hashlib
import sys


# ------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ------------------------------------------------------------

CHANNEL_COUNT = 8

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

SAMPLE_INPUT = OUTPUT_DIR / "frc_15_sample_input.txt"


# ------------------------------------------------------------
# FILE UTILITIES
# ------------------------------------------------------------

def create_sample_file_if_needed():
    """
    Create a small test file if the user has not supplied a file.
    Never overwrite an existing sample.
    """

    if not SAMPLE_INPUT.exists():
        sample_text = (
            "FREQUENCY RELATIONSHIP COMPUTING\n"
            "FRC 15: DIRECT FILE REPRESENTATION\n"
            "Testing eight-channel file conversion.\n"
        )

        SAMPLE_INPUT.write_text(
            sample_text,
            encoding="utf-8"
        )

    return SAMPLE_INPUT


def calculate_sha256(data):
    """Calculate a SHA-256 checksum for byte data."""

    return hashlib.sha256(data).hexdigest()


def create_output_path(input_path):
    """
    Create a reconstruction filename without overwriting
    an existing file.
    """

    base_name = input_path.stem + "_reconstructed"
    extension = input_path.suffix

    output_path = OUTPUT_DIR / (base_name + extension)

    counter = 1

    while output_path.exists():
        output_path = OUTPUT_DIR / (
            f"{base_name}_{counter}{extension}"
        )
        counter += 1

    return output_path


# ------------------------------------------------------------
# BYTE-TO-CHANNEL CONVERSION
# ------------------------------------------------------------

def byte_to_channel(byte_value):
    """
    Convert an integer byte from [0, 255] into [-1.0, +1.0].
    """

    return (byte_value * 2.0 / 255.0) - 1.0


def channel_to_byte(channel_value):
    """
    Reverse the channel mapping and recover the original byte.
    """

    byte_value = round(
        (channel_value + 1.0) * 255.0 / 2.0
    )

    return max(0, min(255, byte_value))


# ------------------------------------------------------------
# FILE REPRESENTATION
# ------------------------------------------------------------

def encode_file(data):
    """
    Convert file bytes into sequential eight-channel frames.

    Each frame contains:
        - Eight normalised channel values
        - The number of original bytes in that frame

    The valid-byte count prevents padding from being mistaken
    for original file data.
    """

    frames = []

    for start in range(0, len(data), CHANNEL_COUNT):

        block = data[start:start + CHANNEL_COUNT]

        valid_byte_count = len(block)

        # Convert original bytes into channel values.
        frame = [
            byte_to_channel(value)
            for value in block
        ]

        # Pad incomplete frames to eight channels.
        while len(frame) < CHANNEL_COUNT:
            frame.append(byte_to_channel(0))

        frames.append({
            "channels": frame,
            "valid_bytes": valid_byte_count
        })

    return frames


# ------------------------------------------------------------
# FILE RECONSTRUCTION
# ------------------------------------------------------------

def decode_frames(frames):
    """
    Reconstruct original bytes from the numerical frames.

    Only the valid original channels in each frame are used.
    """

    reconstructed = bytearray()

    for frame in frames:

        channels = frame["channels"]
        valid_byte_count = frame["valid_bytes"]

        for channel_value in channels[:valid_byte_count]:

            reconstructed.append(
                channel_to_byte(channel_value)
            )

    return bytes(reconstructed)


# ------------------------------------------------------------
# DISPLAY UTILITIES
# ------------------------------------------------------------

def display_frames(frames, maximum_frames=3):
    """Display a small preview of the numerical representation."""

    print("\nFRAME REPRESENTATION PREVIEW")
    print("-" * 70)

    for frame_number, frame in enumerate(
        frames[:maximum_frames], start=1
    ):

        print(
            f"\nFrame {frame_number} "
            f"({frame['valid_bytes']} valid bytes)"
        )

        for index, value in enumerate(frame["channels"]):

            channel_name = chr(ord("A") + index)

            print(
                f"  Channel {channel_name}: {value:+.6f}"
            )

    if len(frames) > maximum_frames:
        print(
            f"\n... {len(frames) - maximum_frames} "
            "additional frames not displayed."
        )


# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 15: DIRECT FILE REPRESENTATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Select the input file.
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).expanduser()

        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path

    else:
        input_path = create_sample_file_if_needed()

    if not input_path.is_file():
        print("\nERROR: Input file could not be found.")
        print(f"Path: {input_path}")
        return

    # Read the original file.
    original_data = input_path.read_bytes()

    print("\nINPUT FILE")
    print("-" * 70)
    print(f"File: {input_path.name}")
    print(f"Original size: {len(original_data):,} bytes")
    print(f"SHA-256: {calculate_sha256(original_data)}")

    # Convert bytes into FRC frames.
    frames = encode_file(original_data)

    print("\nREPRESENTATION")
    print("-" * 70)
    print(f"Channel count: {CHANNEL_COUNT}")
    print(f"Frame count: {len(frames):,}")
    print("Mapping: byte [0,255] -> channel [-1,+1]")
    print("Frame order: preserved")

    display_frames(frames)

    # Reconstruct the original bytes.
    reconstructed_data = decode_frames(frames)

    # Write the reconstructed file without overwriting.
    output_path = create_output_path(input_path)
    output_path.write_bytes(reconstructed_data)

    # Verify exact recovery.
    exact_match = original_data == reconstructed_data

    print("\nRECONSTRUCTION VERIFICATION")
    print("-" * 70)
    print(f"Reconstructed file: {output_path.name}")
    print(f"Reconstructed size: {len(reconstructed_data):,} bytes")
    print(f"SHA-256: {calculate_sha256(reconstructed_data)}")
    print(f"Byte-for-byte match: {exact_match}")

    if exact_match:
        print("\nRESULT: PASS")
        print("Original file reconstructed exactly.")

    else:
        print("\nRESULT: FAIL")
        print("Original and reconstructed files differ.")

    print("\n" + "=" * 70)
    print("FRC 15 EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
