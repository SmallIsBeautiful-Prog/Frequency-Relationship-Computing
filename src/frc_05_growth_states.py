
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 05: GROWTH STATES

Purpose:
Build a sequence of cumulative frequency states.

Each new state retains all previously introduced frequencies
and adds one new frequency.

This stage verifies:
1. Each state contains the expected frequencies.
2. Previous states remain unchanged.
3. Subtracting consecutive states recovers the newly added signal.
4. The final state contains all eight frequencies.

The next stage will use these saved states to investigate
recovery of new components and their relationships.
"""

from pathlib import Path
import csv
import numpy as np


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

FREQUENCIES = [
    12000,
    11900,
    11770,
    11610,
    11420,
    11200,
    10950,
    10670,
]

SAMPLE_RATE = 96000
DURATION = 0.1
AMPLITUDE = 0.08
RANDOM_SEED = 505

OUTPUT_DIR = Path("output")


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_frequency_signal(frequency, phase, time):
    """
    Generate one sine-wave component.

    The same frequency always uses the same phase and time base.
    This is important because later stages will rely on exact
    subtraction between saved growth states.
    """

    return AMPLITUDE * np.sin(
        2 * np.pi * frequency * time + phase
    )


def generate_growth_states():
    """
    Introduce frequencies one at a time.

    G1 = A
    G2 = A + B
    G3 = A + B + C
    ...
    G8 = A + B + C + ... + H

    Every cumulative state is saved separately.
    """

    sample_count = int(SAMPLE_RATE * DURATION)

    time = np.arange(sample_count) / SAMPLE_RATE

    rng = np.random.default_rng(RANDOM_SEED)

    phases = rng.uniform(
        0,
        2 * np.pi,
        size=len(FREQUENCIES)
    )

    individual_signals = []

    for frequency, phase in zip(FREQUENCIES, phases):

        signal = generate_frequency_signal(
            frequency,
            phase,
            time
        )

        individual_signals.append(signal)

    growth_states = []

    current_state = np.zeros(sample_count)

    for signal in individual_signals:

        current_state = current_state + signal

        # Save a separate copy of this growth state.
        growth_states.append(current_state.copy())

    return (
        time,
        phases,
        individual_signals,
        growth_states,
    )


# ============================================================
# FREQUENCY MEASUREMENT
# ============================================================

def measure_frequency_amplitude(signal, frequency):
    """
    Measure the amplitude at an exact FFT frequency bin.

    The experiment uses a 0.1-second signal, giving 10 Hz
    frequency resolution. All selected frequencies align
    with these FFT bins.
    """

    spectrum = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1 / SAMPLE_RATE
    )

    index = np.argmin(
        np.abs(frequencies - frequency)
    )

    measured_frequency = frequencies[index]

    if not np.isclose(
        measured_frequency,
        frequency,
        atol=1e-9
    ):
        raise ValueError(
            f"Frequency {frequency} Hz does not align "
            f"with an FFT bin."
        )

    amplitude = (
        2 * np.abs(spectrum[index]) / len(signal)
    )

    return float(amplitude)


# ============================================================
# GROWTH STATE VERIFICATION
# ============================================================

def verify_growth_states(
    individual_signals,
    growth_states
):
    """
    Check that each state contains exactly the expected
    cumulative signal.

    Also subtract consecutive states to recover the
    newly introduced frequency component.
    """

    results = []

    print("\nGROWTH STATE VERIFICATION")
    print("-" * 65)

    for index, state in enumerate(growth_states):

        expected_state = np.sum(
            individual_signals[:index + 1],
            axis=0
        )

        state_error = np.max(
            np.abs(state - expected_state)
        )

        if index == 0:

            recovered_signal = state
            expected_signal = individual_signals[0]

        else:

            previous_state = growth_states[index - 1]

            recovered_signal = state - previous_state

            expected_signal = individual_signals[index]

        recovery_error = np.max(
            np.abs(recovered_signal - expected_signal)
        )

        new_frequency = FREQUENCIES[index]

        recovered_amplitude = measure_frequency_amplitude(
            recovered_signal,
            new_frequency
        )

        results.append({
            "growth_state": index + 1,
            "frequencies_present": index + 1,
            "new_frequency_hz": new_frequency,
            "state_error": state_error,
            "recovery_error": recovery_error,
            "recovered_amplitude": recovered_amplitude,
        })

        print(
            f"G{index + 1}: "
            f"{index + 1} frequencies | "
            f"Added: {new_frequency:5d} Hz | "
            f"State error: {state_error:.10f} | "
            f"Recovery error: {recovery_error:.10f} | "
            f"Amplitude: {recovered_amplitude:.6f}"
        )

    return results


# ============================================================
# FINAL STATE VERIFICATION
# ============================================================

def verify_final_state(final_state):

    print("\nFINAL GROWTH STATE")
    print("-" * 65)

    all_present = True

    for frequency in FREQUENCIES:

        measured = measure_frequency_amplitude(
            final_state,
            frequency
        )

        passed = np.isclose(
            measured,
            AMPLITUDE,
            atol=1e-8
        )

        if not passed:
            all_present = False

        print(
            f"{frequency:5d} Hz | "
            f"Measured amplitude: {measured:.6f} | "
            f"{'PASS' if passed else 'FAIL'}"
        )

    return all_present


# ============================================================
# SAVE GROWTH STATES
# ============================================================

def save_results(
    time,
    phases,
    growth_states,
    results
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save all cumulative states and their time base.
    state_data = {
        "time": time,
        "frequencies": np.array(FREQUENCIES),
        "phases": phases,
    }

    for index, state in enumerate(growth_states):

        state_data[f"G{index + 1}"] = state

    np.savez_compressed(
        OUTPUT_DIR / "frc_05_growth_states.npz",
        **state_data
    )

    # Save the verification report.
    csv_path = OUTPUT_DIR / "frc_05_growth_states.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSAVED FILES")
    print("-" * 65)
    print("output/frc_05_growth_states.npz")
    print("output/frc_05_growth_states.csv")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 05: GROWTH STATES")
    print("=" * 65)

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 65)
    print(f"Carrier frequencies: {FREQUENCIES}")
    print(f"Number of frequencies: {len(FREQUENCIES)}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Amplitude: {AMPLITUDE}")

    (
        time,
        phases,
        individual_signals,
        growth_states,
    ) = generate_growth_states()

    print("\nCONSTRUCTED GROWTH STATES")
    print("-" * 65)

    for index, state in enumerate(growth_states):

        print(
            f"G{index + 1}: "
            f"{index + 1} frequencies | "
            f"{len(state)} samples | "
            f"Peak amplitude: {np.max(np.abs(state)):.6f}"
        )

    results = verify_growth_states(
        individual_signals,
        growth_states
    )

    final_state_passed = verify_final_state(
        growth_states[-1]
    )

    save_results(
        time,
        phases,
        growth_states,
        results
    )

    print("\nEXPERIMENT SUMMARY")
    print("-" * 65)

    max_state_error = max(
        result["state_error"]
        for result in results
    )

    max_recovery_error = max(
        result["recovery_error"]
        for result in results
    )

    print(
        f"Growth states constructed: {len(growth_states)}"
    )

    print(
        f"Maximum state error: {max_state_error:.12f}"
    )

    print(
        f"Maximum recovery error: {max_recovery_error:.12f}"
    )

    print(
        f"Final state contains all frequencies: "
        f"{'YES' if final_state_passed else 'NO'}"
    )

    if (
        max_state_error < 1e-10
        and max_recovery_error < 1e-10
        and final_state_passed
    ):

        print("\nRESULT: ALL GROWTH STATE CHECKS PASSED")

    else:

        print("\nRESULT: ONE OR MORE CHECKS FAILED")

    print("\nFRC 05 COMPLETE")


if __name__ == "__main__":
    main()
