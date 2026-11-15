
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 16: REPRESENTATION VERIFICATION

Purpose:
Verify direct byte-to-channel representation across different
file sizes, frame boundaries, and binary content.

The original FRC15 experiment remains unchanged.

No external packages are required.
"""

import hashlib
import struct
import zlib
import binascii


# ============================================================
# CONFIGURATION
# ============================================================

CHANNEL_COUNT = 8


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
# FILE REPRESENTATION
# ============================================================

def encode_file(file_data):
    """Convert bytes into ordered eight-channel frames."""

    frames = []

    for start in range(0, len(file_data), CHANNEL_COUNT):

        block = file_data[start:start + CHANNEL_COUNT]

        valid_bytes = len(block)

        channels = [
            byte_to_channel(value)
            for value in block
        ]

        # Pad incomplete frames using the representation of byte 0.
        while len(channels) < CHANNEL_COUNT:
            channels.append(byte_to_channel(0))

        frames.append({
            "channels": channels,
            "valid_bytes": valid_bytes
        })

    return frames


def decode_frames(frames):
    """Reconstruct the original bytes from channel frames."""

    reconstructed = bytearray()

    for frame in frames:

        channels = frame["channels"]
        valid_bytes = frame["valid_bytes"]

        for channel in channels[:valid_bytes]:
            reconstructed.append(channel_to_byte(channel))

    return bytes(reconstructed)


# ============================================================
# TEST DATA GENERATION
# ============================================================

def create_png_chunk(chunk_type, data):
    """Create a valid PNG chunk with its CRC checksum."""

    chunk = chunk_type + data

    checksum = binascii.crc32(chunk) & 0xFFFFFFFF

    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", checksum)
    )


def create_test_png():
    """
    Generate a small valid 2x2 RGB PNG using standard Python
    libraries. No external image package is required.
    """

    signature = b"\x89PNG\r\n\x1a\n"

    width = 2
    height = 2

    # PNG image header: 8-bit RGB colour.
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        8,
        2,
        0,
        0,
        0
    )

    # Each row begins with filter byte 0.
    image_pixels = (
        b"\x00"
        + bytes([
            255, 0, 0,
            0, 255, 0
        ])
        + b"\x00"
        + bytes([
            0, 0, 255,
            255, 255, 255
        ])
    )

    compressed_pixels = zlib.compress(image_pixels)

    return (
        signature
        + create_png_chunk(b"IHDR", ihdr)
        + create_png_chunk(b"IDAT", compressed_pixels)
        + create_png_chunk(b"IEND", b"")
    )


def is_valid_png(data):
    """Check the PNG signature and basic chunk structure."""

    signature = b"\x89PNG\r\n\x1a\n"

    if not data.startswith(signature):
        return False

    position = len(signature)
    found_ihdr = False
    found_iend = False

    while position + 12 <= len(data):

        length = struct.unpack(
            ">I",
            data[position:position + 4]
        )[0]

        chunk_type = data[position + 4:position + 8]

        chunk_end = position + 12 + length

        if chunk_end > len(data):
            return False

        chunk_data = data[position + 8:position + 8 + length]

        stored_crc = struct.unpack(
            ">I",
            data[position + 8 + length:chunk_end]
        )[0]

        calculated_crc = binascii.crc32(
            chunk_type + chunk_data
        ) & 0xFFFFFFFF

        if stored_crc != calculated_crc:
            return False

        if chunk_type == b"IHDR":
            found_ihdr = True

        if chunk_type == b"IEND":
            found_iend = True

            # IEND must be the final chunk.
            return found_ihdr and chunk_end == len(data)

        position = chunk_end

    return False


# ============================================================
# VERIFICATION
# ============================================================

def verify_case(name, original_data):

    frames = encode_file(original_data)

    reconstructed = decode_frames(frames)

    expected_frames = (
        len(original_data) + CHANNEL_COUNT - 1
    ) // CHANNEL_COUNT

    # Check frame count.
    frame_count_ok = len(frames) == expected_frames

    # Check every frame contains exactly eight channels.
    channel_count_ok = all(
        len(frame["channels"]) == CHANNEL_COUNT
        for frame in frames
    )

    # Check valid byte counts.
    valid_byte_count_ok = (
        sum(frame["valid_bytes"] for frame in frames)
        == len(original_data)
    )

    # Check every channel remains in the permitted range.
    channel_range_ok = all(
        -1.0 <= channel <= 1.0
        for frame in frames
        for channel in frame["channels"]
    )

    # Check exact reconstruction.
    reconstruction_ok = reconstructed == original_data

    passed = all([
        frame_count_ok,
        channel_count_ok,
        valid_byte_count_ok,
        channel_range_ok,
        reconstruction_ok
    ])

    original_hash = hashlib.sha256(original_data).hexdigest()
    reconstructed_hash = hashlib.sha256(reconstructed).hexdigest()

    print(f"\nTEST: {name}")
    print("-" * 65)
    print(f"Original size:          {len(original_data)} bytes")
    print(f"Expected frames:        {expected_frames}")
    print(f"Generated frames:      {len(frames)}")
    print(f"Frame count:            {'PASS' if frame_count_ok else 'FAIL'}")
    print(f"Eight channels/frame:   {'PASS' if channel_count_ok else 'FAIL'}")
    print(f"Valid byte counts:      {'PASS' if valid_byte_count_ok else 'FAIL'}")
    print(f"Channel range:          {'PASS' if channel_range_ok else 'FAIL'}")
    print(f"Exact reconstruction:   {'PASS' if reconstruction_ok else 'FAIL'}")
    print(f"SHA-256 match:          {'PASS' if original_hash == reconstructed_hash else 'FAIL'}")

    print(f"RESULT:                 {'PASS' if passed else 'FAIL'}")

    return passed


# ============================================================
# BYTE MAPPING EXHAUSTIVE TEST
# ============================================================

def verify_all_byte_values():

    print("\nBYTE MAPPING EXHAUSTIVE TEST")
    print("=" * 65)

    failures = []

    for value in range(256):

        channel = byte_to_channel(value)
        reconstructed = channel_to_byte(channel)

        if reconstructed != value:
            failures.append(value)

    if not failures:
        print("Values tested: 256")
        print("Values reconstructed correctly: 256")
        print("RESULT: PASS")
        return True

    print(f"Failed byte values: {failures}")
    print("RESULT: FAIL")

    return False


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 16: REPRESENTATION VERIFICATION")
    print("=" * 65)

    # Construct test inputs.
    test_cases = [
        ("Empty file", b""),

        ("Single-byte file", bytes([127])),

        ("Smaller than eight bytes",
         bytes([10, 20, 30, 40, 50])),

        ("Exactly eight bytes",
         bytes([0, 32, 64, 96, 128, 160, 192, 255])),

        ("Incomplete final frame",
         bytes(range(19))),

        ("Larger text file",
         (
             "Frequency Relationship Computing\n"
             "Direct File Representation Verification\n"
             "FRC 16 tests ordered eight-channel frames.\n"
         ).encode("utf-8") * 500
        ),

        ("Valid PNG image",
         create_test_png())
    ]

    print("\nTEST CONFIGURATION")
    print("-" * 65)
    print(f"Channel count: {CHANNEL_COUNT}")
    print("Mapping: byte [0,255] -> channel [-1,+1]")
    print("Frame order: preserved")
    print("Reconstruction: byte-for-byte comparison")

    results = []

    # Verify every test case.
    for name, data in test_cases:

        if name == "Valid PNG image":
            print(
                "\nPNG structure:",
                "VALID" if is_valid_png(data) else "INVALID"
            )

        results.append(
            verify_case(name, data)
        )

    # Test all possible byte values.
    byte_mapping_result = verify_all_byte_values()

    # Final report.
    total_tests = len(results) + 1
    passed_tests = sum(results) + int(byte_mapping_result)

    print("\n" + "=" * 65)
    print("FRC16 FINAL RESULTS")
    print("=" * 65)

    print(f"Tests completed: {total_tests}")
    print(f"Tests passed:    {passed_tests}")
    print(f"Tests failed:    {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("\nRESULT: ALL TESTS PASSED")
    else:
        print("\nRESULT: VERIFICATION FAILED")

    print("=" * 65)


if __name__ == "__main__":
    main()
