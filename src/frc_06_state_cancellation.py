
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 06: STATE CANCELLATION

Purpose:
Square consecutive growth states and subtract them.

R_k = G_k^2 - G_(k-1)^2

Mathematically:

R_k = 2 * G_(k-1) * s_k + s_k^2

This should expose the new pairwise relationships
created when each new frequency is introduced.

The experiment verifies:
1. Squared-state subtraction.
2. Agreement with the mathematical expansion.
3. Recovery of every newly introduced difference frequency.
4. Cancellation of previously established relationships.
5. Accumulation of all 28 relationships by G8.

Input:
    output/frc_05_growth_states.npz

Outputs:
    output/frc_06_state_cancellation.npz
    output/frc_06_state_cancellation.csv
"""

from pathlib import Path
import csv
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("output")

INPUT_FILE = OUTPUT_DIR / "frc_05_growth_states.npz"

SAMPLE_RATE = 96000
EXPECTED_AMPLITUDE = 0.08

AMPLITUDE_TOLERANCE = 1e-8
WAVEFORM_TOLERANCE = 1e-10


# ============================================================
# LOAD FRC 05 GROWTH STATES
# ============================================================

def load_growth_states():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Run FRC 05 first."
        )

    data = np.load(INPUT_FILE)

    frequencies = data["frequencies"]
    time = data["time"]

    growth_states = [
        data[f"G{index + 1}"].copy()
        for index in range(len(frequencies))
    ]

    phases = data["phases"]

    return frequencies, time, phases, growth_states


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
# BUILD EXPECTED PAIRWISE RELATIONSHIPS
# ============================================================

def get_pairwise_relationships(frequencies):

    relationships = []

    for i in range(len(frequencies)):

        for j in range(i + 1, len(frequencies)):

            difference = abs(
                int(frequencies[i]) - int(frequencies[j])
            )

            relationships.append({
                "frequency_a": int(frequencies[i]),
                "frequency_b": int(frequencies[j]),
                "difference_hz": difference,
                "index_a": i,
                "index_b": j,
            })

    return relationships


# ============================================================
# SQUARED-STATE CANCELLATION
# ============================================================

def calculate_incremental_residuals(growth_states):

    residuals = []
    expected_residuals = []
    individual_components = []

    print("\nSQUARED-STATE CANCELLATION")
    print("-" * 75)

    for index, current_state in enumerate(growth_states):

        if index == 0:

            previous_state = np.zeros_like(current_state)

            new_component = current_state.copy()

        else:

            previous_state = growth_states[index - 1]

            new_component = (
                current_state - previous_state
            )

        # Square both cumulative states.
        current_squared = current_state ** 2

        previous_squared = previous_state ** 2

        # Subtract the previous squared state.
        residual = (
            current_squared - previous_squared
        )

        # Verify against the mathematical expansion.
        expected_residual = (
            2 * previous_state * new_component
            + new_component ** 2
        )

        waveform_error = np.max(
            np.abs(residual - expected_residual)
        )

        residuals.append(residual)

        expected_residuals.append(
            expected_residual
        )

        individual_components.append(
            new_component
        )

        print(
            f"G{index + 1}: "
            f"New frequency: {index + 1} "
            f"| Residual error: {waveform_error:.12f}"
        )

    return (
        residuals,
        expected_residuals,
        individual_components,
    )


# ============================================================
# VERIFY NEW RELATIONSHIPS
# ============================================================

def verify_incremental_relationships(
    frequencies,
    growth_states,
    residuals,
    relationships,
):

    relationship_results = []
    state_results = []

    print("\nNEW PAIRWISE RELATIONSHIPS")
    print("-" * 75)

    for state_index, residual in enumerate(residuals):

        # G1 introduces no pairwise relationships.
        if state_index == 0:
            new_relationships = []
            previous_relationships = []

        else:

            new_relationships = [
                relationship
                for relationship in relationships
                if relationship["index_b"] == state_index
            ]

            previous_relationships = [
                relationship
                for relationship in relationships
                if relationship["index_b"] < state_index
            ]

        new_passes = 0
        measured_amplitudes = []

        for relationship in new_relationships:

            difference = relationship["difference_hz"]

            measured = measure_frequency_amplitude(
                residual,
                difference
            )

            passed = np.isclose(
                measured,
                EXPECTED_AMPLITUDE ** 2,
                atol=AMPLITUDE_TOLERANCE
            )

            if passed:
                new_passes += 1

            measured_amplitudes.append(measured)

            relationship_results.append({
                "growth_state": state_index + 1,
                "frequency_a": relationship["frequency_a"],
                "frequency_b": relationship["frequency_b"],
                "difference_hz": difference,
                "expected_amplitude": EXPECTED_AMPLITUDE ** 2,
                "measured_amplitude": measured,
                "passed": passed,
            })

            print(
                f"G{state_index + 1}: "
                f"{relationship['frequency_a']} - "
                f"{relationship['frequency_b']} = "
                f"{difference} Hz | "
                f"Amplitude: {measured:.6f} | "
                f"{'PASS' if passed else 'FAIL'}"
            )

        # Previously established differences should cancel.
        previous_leakage = []

        for relationship in previous_relationships:

            difference = relationship["difference_hz"]

            measured = measure_frequency_amplitude(
                residual,
                difference
            )

            previous_leakage.append(measured)

        max_previous_leakage = (
            max(previous_leakage)
            if previous_leakage
            else 0.0
        )

        total_new = len(new_relationships)

        cumulative_relationships = sum(
            1 for relationship in relationships
            if relationship["index_b"] <= state_index
        )

        state_results.append({
            "growth_state": state_index + 1,
            "new_relationships": total_new,
            "relationships_created_total": cumulative_relationships,
            "new_relationships_passed": new_passes,
            "max_previous_relationship_leakage": max_previous_leakage,
        })

        print(
            f"\nG{state_index + 1} SUMMARY: "
            f"{new_passes}/{total_new} new relationships passed | "
            f"Total relationships: {cumulative_relationships} | "
            f"Previous leakage: {max_previous_leakage:.10f}"
        )

    return relationship_results, state_results


# ============================================================
# SAVE EXPERIMENT RESULTS
# ============================================================

def save_results(
    frequencies,
    time,
    phases,
    residuals,
    expected_residuals,
    relationship_results,
    state_results,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the actual incremental residual waveforms.
    residual_data = {
        "frequencies": frequencies,
        "time": time,
        "phases": phases,
    }

    for index, residual in enumerate(residuals):

        residual_data[f"R{index + 1}"] = residual

        residual_data[
            f"expected_R{index + 1}"
        ] = expected_residuals[index]

    np.savez_compressed(
        OUTPUT_DIR / "frc_06_state_cancellation.npz",
        **residual_data
    )

    # Save relationship measurements.
    relationship_path = (
        OUTPUT_DIR / "frc_06_relationships.csv"
    )

    with open(
        relationship_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(relationship_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(relationship_results)

    # Save growth-state summaries.
    summary_path = (
        OUTPUT_DIR / "frc_06_state_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(state_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(state_results)

    print("\nSAVED FILES")
    print("-" * 75)
    print("output/frc_06_state_cancellation.npz")
    print("output/frc_06_relationships.csv")
    print("output/frc_06_state_summary.csv")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 75)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 06: STATE CANCELLATION")
    print("=" * 75)

    frequencies, time, phases, growth_states = (
        load_growth_states()
    )

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 75)
    print(f"Carrier frequencies: {frequencies.tolist()}")
    print(f"Number of carriers: {len(frequencies)}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Samples per state: {len(time)}")
    print(f"Amplitude: {EXPECTED_AMPLITUDE:.4f}")

    relationships = get_pairwise_relationships(
        frequencies
    )

    print(
        f"Expected total pairwise relationships: "
        f"{len(relationships)}"
    )

    (
        residuals,
        expected_residuals,
        individual_components,
    ) = calculate_incremental_residuals(
        growth_states
    )

    relationship_results, state_results = (
        verify_incremental_relationships(
            frequencies,
            growth_states,
            residuals,
            relationships,
        )
    )

    # Check the residual waveform against its expansion.
    maximum_waveform_error = max(
        np.max(
            np.abs(residuals[index] - expected_residuals[index])
        )
        for index in range(len(residuals))
    )

    # Check every new relationship.
    all_relationships_passed = all(
        result["passed"]
        for result in relationship_results
    )

    # Check that previously existing relationships cancel.
    maximum_previous_leakage = max(
        result["max_previous_relationship_leakage"]
        for result in state_results
    )

    previous_relationships_cancelled = (
        maximum_previous_leakage < AMPLITUDE_TOLERANCE
    )

    total_new_relationships = sum(
        result["new_relationships"]
        for result in state_results
    )

    total_created_relationships = (
        state_results[-1]["relationships_created_total"]
    )

    save_results(
        frequencies,
        time,
        phases,
        residuals,
        expected_residuals,
        relationship_results,
        state_results,
    )

    print("\nEXPERIMENT SUMMARY")
    print("-" * 75)

    print(
        f"New relationships recovered: "
        f"{total_new_relationships}"
    )

    print(
        f"Expected relationships: "
        f"{len(relationships)}"
    )

    print(
        f"Relationships accumulated by G8: "
        f"{total_created_relationships}"
    )

    print(
        f"Maximum mathematical expansion error: "
        f"{maximum_waveform_error:.12f}"
    )

    print(
        f"All new relationships passed: "
        f"{'YES' if all_relationships_passed else 'NO'}"
    )

    print(
        f"Previous relationships cancelled: "
        f"{'YES' if previous_relationships_cancelled else 'NO'}"
    )

    if (
        maximum_waveform_error < WAVEFORM_TOLERANCE
        and all_relationships_passed
        and previous_relationships_cancelled
        and total_new_relationships == 28
        and total_created_relationships == 28
    ):

        print("\nRESULT: ALL STATE CANCELLATION CHECKS PASSED")

    else:

        print("\nRESULT: ONE OR MORE CHECKS FAILED")

    print("\nFRC 06 COMPLETE")


if __name__ == "__main__":
    main()
