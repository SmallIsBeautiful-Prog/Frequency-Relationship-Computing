
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 07: INCREMENTAL RELATIONSHIPS

Purpose:
Build an ordered, persistent representation of all
pairwise relationships recovered during growth-state
construction and cancellation.

Each relationship records:
1. Its two original carrier frequencies.
2. Its difference frequency.
3. The growth state in which it was introduced.
4. Its position in the overall introduction sequence.
5. Its measured amplitude in the incremental residual.

The script also builds cumulative relationship snapshots.

A snapshot at Gk contains all relationships introduced
up to and including that growth state.

Inputs:
    output/frc_05_growth_states.npz
    output/frc_06_state_cancellation.npz

Outputs:
    output/frc_07_incremental_relationships.npz
    output/frc_07_relationship_sequence.csv
    output/frc_07_growth_snapshots.csv
"""

from pathlib import Path
import csv
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("output")

GROWTH_FILE = OUTPUT_DIR / "frc_05_growth_states.npz"
CANCELLATION_FILE = (
    OUTPUT_DIR / "frc_06_state_cancellation.npz"
)

SAMPLE_RATE = 96000
EXPECTED_AMPLITUDE = 0.08

EXPECTED_RELATIONSHIP_AMPLITUDE = (
    EXPECTED_AMPLITUDE ** 2
)

AMPLITUDE_TOLERANCE = 1e-8


# ============================================================
# LOAD PREVIOUS EXPERIMENTS
# ============================================================

def load_previous_results():

    if not GROWTH_FILE.exists():

        raise FileNotFoundError(
            "FRC 05 output is missing. "
            "Run frc_05_growth_states.py first."
        )

    if not CANCELLATION_FILE.exists():

        raise FileNotFoundError(
            "FRC 06 output is missing. "
            "Run frc_06_state_cancellation.py first."
        )

    growth_data = np.load(GROWTH_FILE)
    cancellation_data = np.load(CANCELLATION_FILE)

    frequencies = growth_data["frequencies"]
    time = growth_data["time"]

    growth_states = [
        growth_data[f"G{index + 1}"].copy()
        for index in range(len(frequencies))
    ]

    residuals = [
        cancellation_data[f"R{index + 1}"].copy()
        for index in range(len(frequencies))
    ]

    return frequencies, time, growth_states, residuals


# ============================================================
# FREQUENCY MEASUREMENT
# ============================================================

def measure_frequency_amplitude(signal, frequency):

    spectrum = np.fft.rfft(signal)

    frequency_bins = np.fft.rfftfreq(
        len(signal),
        d=1 / SAMPLE_RATE
    )

    index = np.argmin(
        np.abs(frequency_bins - frequency)
    )

    measured_frequency = frequency_bins[index]

    if not np.isclose(
        measured_frequency,
        frequency,
        atol=1e-9
    ):

        raise ValueError(
            f"{frequency} Hz does not align with an FFT bin."
        )

    amplitude = (
        2 * np.abs(spectrum[index]) / len(signal)
    )

    return float(amplitude)


# ============================================================
# CREATE ORDERED RELATIONSHIP RECORDS
# ============================================================

def create_relationship_sequence(frequencies, residuals):

    records = []
    sequence_position = 0

    print("\nORDERED RELATIONSHIP INTRODUCTION")
    print("-" * 78)

    for new_index in range(len(frequencies)):

        # G1 has no previous frequency to form a pair with.
        if new_index == 0:
            continue

        new_frequency = int(frequencies[new_index])

        # Every earlier frequency forms one new relationship
        # with the frequency introduced at this growth state.
        for previous_index in range(new_index):

            previous_frequency = int(
                frequencies[previous_index]
            )

            difference = abs(
                previous_frequency - new_frequency
            )

            sequence_position += 1

            residual = residuals[new_index]

            measured_amplitude = measure_frequency_amplitude(
                residual,
                difference
            )

            passed = np.isclose(
                measured_amplitude,
                EXPECTED_RELATIONSHIP_AMPLITUDE,
                atol=AMPLITUDE_TOLERANCE
            )

            record = {
                "relationship_id": f"R{sequence_position:03d}",
                "sequence_position": sequence_position,
                "introduced_at_state": new_index + 1,
                "previous_frequency_hz": previous_frequency,
                "new_frequency_hz": new_frequency,
                "difference_frequency_hz": difference,
                "measured_amplitude": measured_amplitude,
                "passed": passed,
            }

            records.append(record)

            print(
                f"{record['relationship_id']} | "
                f"Position {sequence_position:2d} | "
                f"G{new_index + 1} | "
                f"{previous_frequency} - "
                f"{new_frequency} = "
                f"{difference} Hz | "
                f"Amplitude: {measured_amplitude:.6f} | "
                f"{'PASS' if passed else 'FAIL'}"
            )

    return records


# ============================================================
# BUILD CUMULATIVE RELATIONSHIP SNAPSHOTS
# ============================================================

def build_cumulative_snapshots(
    frequencies,
    growth_states,
    records,
):

    number_of_states = len(frequencies)
    number_of_relationships = len(records)

    # Rows represent growth states.
    # Columns represent relationships in introduction order.
    amplitude_snapshots = np.zeros(
        (number_of_states, number_of_relationships)
    )

    presence_snapshots = np.zeros(
        (number_of_states, number_of_relationships),
        dtype=bool
    )

    print("\nCUMULATIVE RELATIONSHIP SNAPSHOTS")
    print("-" * 78)

    for state_index, growth_state in enumerate(growth_states):

        squared_state = growth_state ** 2

        relationship_count = 0

        for relationship_index, record in enumerate(records):

            introduced_state = record["introduced_at_state"]

            # A relationship is available from its
            # introduction state onward.
            if introduced_state <= state_index + 1:

                difference = record[
                    "difference_frequency_hz"
                ]

                amplitude = measure_frequency_amplitude(
                    squared_state,
                    difference
                )

                amplitude_snapshots[
                    state_index,
                    relationship_index
                ] = amplitude

                presence_snapshots[
                    state_index,
                    relationship_index
                ] = True

                relationship_count += 1

        print(
            f"G{state_index + 1}: "
            f"{relationship_count} relationships present | "
            f"{int(np.sum(presence_snapshots[state_index]))} "
            f"recorded"
        )

    return amplitude_snapshots, presence_snapshots


# ============================================================
# VERIFY ORDER AND CUMULATIVE REPRESENTATION
# ============================================================

def verify_representation(
    records,
    amplitude_snapshots,
    presence_snapshots,
):

    expected_total = 28

    sequence_positions = [
        record["sequence_position"]
        for record in records
    ]

    relationship_frequencies = [
        record["difference_frequency_hz"]
        for record in records
    ]

    sequence_is_complete = (
        sequence_positions
        == list(range(1, expected_total + 1))
    )

    differences_are_unique = (
        len(set(relationship_frequencies))
        == len(relationship_frequencies)
    )

    all_incremental_measurements_passed = all(
        record["passed"]
        for record in records
    )

    expected_counts = [
        0, 1, 3, 6, 10, 15, 21, 28
    ]

    actual_counts = [
        int(np.sum(row))
        for row in presence_snapshots
    ]

    snapshot_counts_passed = (
        actual_counts == expected_counts
    )

    final_snapshot = amplitude_snapshots[-1]

    final_amplitudes_passed = np.allclose(
        final_snapshot,
        EXPECTED_RELATIONSHIP_AMPLITUDE,
        atol=AMPLITUDE_TOLERANCE
    )

    print("\nREPRESENTATION VERIFICATION")
    print("-" * 78)

    print(
        f"Relationship records created: "
        f"{len(records)}"
    )

    print(
        f"Sequence positions complete: "
        f"{'YES' if sequence_is_complete else 'NO'}"
    )

    print(
        f"Difference frequencies unique: "
        f"{'YES' if differences_are_unique else 'NO'}"
    )

    print(
        f"Incremental measurements passed: "
        f"{'YES' if all_incremental_measurements_passed else 'NO'}"
    )

    print(
        f"Cumulative snapshot counts correct: "
        f"{'YES' if snapshot_counts_passed else 'NO'}"
    )

    print(
        f"Final snapshot amplitudes correct: "
        f"{'YES' if final_amplitudes_passed else 'NO'}"
    )

    return {
        "sequence_is_complete": sequence_is_complete,
        "differences_are_unique": differences_are_unique,
        "incremental_measurements_passed": (
            all_incremental_measurements_passed
        ),
        "snapshot_counts_passed": snapshot_counts_passed,
        "final_amplitudes_passed": final_amplitudes_passed,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    frequencies,
    records,
    amplitude_snapshots,
    presence_snapshots,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the numerical relationship representation.
    np.savez_compressed(
        OUTPUT_DIR / "frc_07_incremental_relationships.npz",
        frequencies=frequencies,
        relationship_amplitudes=amplitude_snapshots,
        relationship_presence=presence_snapshots,
        sequence_positions=np.array([
            record["sequence_position"]
            for record in records
        ]),
        introduction_states=np.array([
            record["introduced_at_state"]
            for record in records
        ]),
        difference_frequencies=np.array([
            record["difference_frequency_hz"]
            for record in records
        ]),
    )

    # Save the ordered relationship event records.
    sequence_path = (
        OUTPUT_DIR / "frc_07_relationship_sequence.csv"
    )

    with open(
        sequence_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    # Save the cumulative snapshot summary.
    snapshot_path = (
        OUTPUT_DIR / "frc_07_growth_snapshots.csv"
    )

    with open(
        snapshot_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "growth_state",
            "relationships_present",
            "relationships_remaining_to_introduce",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index, row in enumerate(presence_snapshots):

            count = int(np.sum(row))

            writer.writerow({
                "growth_state": index + 1,
                "relationships_present": count,
                "relationships_remaining_to_introduce": (
                    len(records) - count
                ),
            })

    print("\nSAVED FILES")
    print("-" * 78)
    print("output/frc_07_incremental_relationships.npz")
    print("output/frc_07_relationship_sequence.csv")
    print("output/frc_07_growth_snapshots.csv")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 78)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 07: INCREMENTAL RELATIONSHIPS")
    print("=" * 78)

    frequencies, time, growth_states, residuals = (
        load_previous_results()
    )

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 78)
    print(f"Carrier frequencies: {frequencies.tolist()}")
    print(f"Number of carriers: {len(frequencies)}")
    print(f"Number of growth states: {len(growth_states)}")
    print(f"Samples per state: {len(time)}")

    records = create_relationship_sequence(
        frequencies,
        residuals
    )

    amplitude_snapshots, presence_snapshots = (
        build_cumulative_snapshots(
            frequencies,
            growth_states,
            records
        )
    )

    checks = verify_representation(
        records,
        amplitude_snapshots,
        presence_snapshots
    )

    save_results(
        frequencies,
        records,
        amplitude_snapshots,
        presence_snapshots
    )

    all_checks_passed = all(checks.values())

    print("\nEXPERIMENT SUMMARY")
    print("-" * 78)

    print(
        f"Ordered relationships recorded: {len(records)}"
    )

    print(
        f"Final cumulative relationships: "
        f"{int(np.sum(presence_snapshots[-1]))}"
    )

    print(
        f"Final snapshot shape: "
        f"{amplitude_snapshots[-1].shape}"
    )

    print(
        f"Full snapshot matrix shape: "
        f"{amplitude_snapshots.shape}"
    )

    if all_checks_passed:

        print("\nRESULT: ALL INCREMENTAL REPRESENTATION CHECKS PASSED")

    else:

        print("\nRESULT: ONE OR MORE CHECKS FAILED")

    print("\nFRC 07 COMPLETE")


if __name__ == "__main__":
    main()
