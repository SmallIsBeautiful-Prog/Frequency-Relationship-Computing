
"""
FREQUENCY RELATIONSHIP COMPUTING
SCRIPT 03: SCALABLE FREQUENCY REPRESENTATION

Purpose:
    Automatically construct a scalable representation of
    carrier frequencies and all their unique pairwise relationships.

Experiments:
    4 carriers  ->  6 relationships
    6 carriers  -> 15 relationships
    8 carriers  -> 28 relationships

Two simultaneous learning tasks:
    Output 1 = XOR(A, B)
    Output 2 = XOR(C, D)

The same linear least-squares learning method is used
for every configuration.

This script investigates whether the representation can
grow without manually defining each pairwise feature.
"""

import numpy as np
from itertools import product


# ------------------------------------------------------------
# 1. EXPERIMENT CONFIGURATION
# ------------------------------------------------------------

SAMPLE_RATE = 96000
DURATION = 0.1

# The full frequency set.
# Each experiment uses the first 4, 6 or 8 frequencies.
FREQUENCIES = [
    12000,
    11900,
    11770,
    11610,
    11420,
    11200,
    10950,
    10670
]

# The configurations we want to test.
CARRIER_COUNTS = [4, 6, 8]

AMPLITUDE = 0.08
AMPLITUDE_VARIATION = 0.15

EXAMPLES_PER_TARGET_CASE = 400
TRAINING_FRACTION = 0.70

RANDOM_SEED = 301

CHANNEL_NAMES = [
    "A", "B", "C", "D",
    "E", "F", "G", "H"
]

# Two XOR tasks use these channel pairs.
TARGET_PAIRS = [
    (0, 1),
    (2, 3)
]


# ------------------------------------------------------------
# 2. PREPARE THE TIME AXIS
# ------------------------------------------------------------

SAMPLE_COUNT = int(SAMPLE_RATE * DURATION)

time = np.arange(SAMPLE_COUNT) / SAMPLE_RATE


# ------------------------------------------------------------
# 3. AUTOMATICALLY GENERATE PAIRWISE RELATIONSHIPS
# ------------------------------------------------------------

def build_relationships(frequencies):
    """
    Generate every unique pair of carrier frequencies.

    The function does not need to know how many carriers
    will be supplied.

    Each relationship contains:
        First carrier index
        Second carrier index
        Difference frequency
    """

    relationships = []

    for i in range(len(frequencies)):

        for j in range(i + 1, len(frequencies)):

            difference = abs(
                frequencies[i] - frequencies[j]
            )

            relationships.append((i, j, difference))

    return relationships


# ------------------------------------------------------------
# 4. VALIDATE THE FREQUENCY CONFIGURATION
# ------------------------------------------------------------

def validate_configuration(frequencies, relationships):
    """
    Check that the frequency configuration is suitable
    for independent FFT-based relationship extraction.

    In particular:
        - Carrier frequencies must be unique.
        - Pairwise difference frequencies must be unique.
        - Frequencies must align with FFT bins.
        - Frequencies must remain below Nyquist.
    """

    if len(set(frequencies)) != len(frequencies):

        raise ValueError(
            "Carrier frequencies must be unique."
        )

    differences = [
        relationship[2]
        for relationship in relationships
    ]

    if len(set(differences)) != len(differences):

        raise ValueError(
            "Pairwise difference-frequency collision detected."
        )

    all_measured_frequencies = list(frequencies) + differences

    for frequency in all_measured_frequencies:

        if frequency >= SAMPLE_RATE / 2:

            raise ValueError(
                f"{frequency} Hz exceeds the Nyquist limit."
            )

        bin_position = frequency * DURATION

        if not np.isclose(
            bin_position,
            round(bin_position),
            atol=1e-8
        ):

            raise ValueError(
                f"{frequency} Hz does not align with an FFT bin."
            )


# ------------------------------------------------------------
# 5. MEASURE A FREQUENCY COMPONENT
# ------------------------------------------------------------

def measure_frequency(signal, frequency):
    """
    Extract the amplitude of one frequency using an FFT.

    Our experiment uses exact FFT-bin alignment.
    """

    spectrum = np.fft.rfft(signal)

    amplitudes = (
        np.abs(spectrum) * (2.0 / len(signal))
    )

    bin_index = int(round(frequency * DURATION))

    return amplitudes[bin_index]


# ------------------------------------------------------------
# 6. GENERATE ONE BINARY INPUT EXAMPLE
# ------------------------------------------------------------

def generate_inputs(targets, carrier_count, rng):
    """
    Generate binary carrier inputs satisfying two XOR targets.

    The remaining carriers, if any, receive random binary inputs.

    This allows us to keep the same two learning tasks while
    increasing the number of available carriers.
    """

    inputs = np.zeros(carrier_count, dtype=int)

    # Generate the first XOR pair.
    if targets[0] == 0:

        pair = [
            (0, 0),
            (1, 1)
        ][rng.integers(0, 2)]

    else:

        pair = [
            (0, 1),
            (1, 0)
        ][rng.integers(0, 2)]

    inputs[0] = pair[0]
    inputs[1] = pair[1]

    # Generate the second XOR pair.
    if targets[1] == 0:

        pair = [
            (0, 0),
            (1, 1)
        ][rng.integers(0, 2)]

    else:

        pair = [
            (0, 1),
            (1, 0)
        ][rng.integers(0, 2)]

    inputs[2] = pair[0]
    inputs[3] = pair[1]

    # Any additional carrier inputs are independent.
    if carrier_count > 4:

        inputs[4:] = rng.integers(
            0, 2,
            size=carrier_count - 4
        )

    return inputs


# ------------------------------------------------------------
# 7. GENERATE THE FREQUENCY REPRESENTATION
# ------------------------------------------------------------

def generate_example(
    inputs,
    frequencies,
    relationships,
    rng
):
    """
    Construct the combined waveform and automatically
    extract all original and pairwise features.

    Original features:
        One measured amplitude per carrier.

    Pairwise features:
        One measured difference-frequency amplitude
        per automatically generated relationship.
    """

    carrier_signals = []

    # Construct all carrier waveforms.
    for i, frequency in enumerate(frequencies):

        if inputs[i] == 1:

            amplitude = AMPLITUDE * (
                1.0 + rng.uniform(
                    -AMPLITUDE_VARIATION,
                    AMPLITUDE_VARIATION
                )
            )

        else:

            amplitude = 0.0

        signal = amplitude * np.sin(
            2.0 * np.pi * frequency * time
        )

        carrier_signals.append(signal)

    # Combine all carriers.
    combined_signal = np.sum(
        carrier_signals,
        axis=0
    )

    # Apply square-law processing.
    squared_signal = combined_signal ** 2

    # Extract original carrier amplitudes.
    carrier_features = []

    for frequency in frequencies:

        measured = measure_frequency(
            combined_signal,
            frequency
        )

        carrier_features.append(measured)

    # Extract all automatically generated relationships.
    interaction_features = []

    for _, _, difference in relationships:

        measured = measure_frequency(
            squared_signal,
            difference
        )

        interaction_features.append(measured)

    # Combine both feature groups.
    features = (
        carrier_features + interaction_features
    )

    return features


# ------------------------------------------------------------
# 8. BUILD A DATASET
# ------------------------------------------------------------

def build_dataset(
    frequencies,
    relationships,
    rng
):
    """
    Generate balanced examples for all four combinations
    of the two XOR targets.
    """

    carrier_count = len(frequencies)

    features = []
    targets = []
    groups = []

    target_cases = list(
        product([0, 1], repeat=2)
    )

    for target_case in target_cases:

        for _ in range(EXAMPLES_PER_TARGET_CASE):

            inputs = generate_inputs(
                target_case,
                carrier_count,
                rng
            )

            example = generate_example(
                inputs,
                frequencies,
                relationships,
                rng
            )

            features.append(example)
            targets.append(target_case)
            groups.append(target_case)

    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(groups, dtype=int)
    )


# ------------------------------------------------------------
# 9. SPLIT TRAINING AND TEST DATA
# ------------------------------------------------------------

def split_dataset(groups, rng):
    """
    Split each target combination separately.

    This maintains balanced training and testing datasets.
    """

    training_indices = []
    testing_indices = []

    target_cases = list(
        product([0, 1], repeat=2)
    )

    training_count = int(
        EXAMPLES_PER_TARGET_CASE * TRAINING_FRACTION
    )

    for case in target_cases:

        matching = np.where(
            np.all(groups == case, axis=1)
        )[0]

        shuffled = rng.permutation(matching)

        training_indices.extend(
            shuffled[:training_count]
        )

        testing_indices.extend(
            shuffled[training_count:]
        )

    return (
        np.asarray(training_indices),
        np.asarray(testing_indices)
    )


# ------------------------------------------------------------
# 10. TRAIN THE LINEAR CLASSIFIER
# ------------------------------------------------------------

def train_classifier(features, targets):
    """
    Train two output classifiers using ordinary least squares.

    Both outputs use the same representation and training
    procedure, but learn their own numerical weights.
    """

    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)

    feature_std[feature_std == 0] = 1.0

    normalised = (
        features - feature_mean
    ) / feature_std

    design_matrix = np.column_stack([
        np.ones(len(normalised)),
        normalised
    ])

    weights, _, _, _ = np.linalg.lstsq(
        design_matrix,
        targets,
        rcond=None
    )

    return weights, feature_mean, feature_std


# ------------------------------------------------------------
# 11. PREDICT OUTPUTS
# ------------------------------------------------------------

def predict(
    features,
    weights,
    feature_mean,
    feature_std
):
    """
    Use the trained linear classifier to predict both outputs.
    """

    normalised = (
        features - feature_mean
    ) / feature_std

    design_matrix = np.column_stack([
        np.ones(len(normalised)),
        normalised
    ])

    scores = design_matrix @ weights

    return (scores >= 0.5).astype(int)


# ------------------------------------------------------------
# 12. EVALUATE THE RESULTS
# ------------------------------------------------------------

def evaluate(
    predictions,
    actual
):
    """
    Calculate individual and joint classification accuracy.
    """

    accuracies = []

    for output_index in range(2):

        accuracy = np.mean(
            predictions[:, output_index]
            == actual[:, output_index]
        ) * 100.0

        accuracies.append(accuracy)

    joint_accuracy = np.mean(
        np.all(predictions == actual, axis=1)
    ) * 100.0

    return accuracies, joint_accuracy


# ------------------------------------------------------------
# 13. DISPLAY THE MOST IMPORTANT RELATIONSHIPS
# ------------------------------------------------------------

def show_important_relationships(
    weights,
    relationships
):
    """
    Display the three strongest pairwise weights
    for each output.

    This helps us see whether the classifier has learned
    to favour the relationships relevant to its tasks.
    """

    carrier_count = 0

    # The first relationship weight follows the bias
    # and all original carrier weights.
    relationship_start = 1 + len(FREQUENCIES[:0])

    # The actual carrier count is inferred from the
    # relationship indices.
    if relationships:

        carrier_count = max(
            max(i, j)
            for i, j, _ in relationships
        ) + 1

    relationship_start = 1 + carrier_count

    for output_index in range(2):

        print(f"\nOutput {output_index + 1}:")

        ranked = []

        for index, relationship in enumerate(relationships):

            weight = weights[
                relationship_start + index,
                output_index
            ]

            ranked.append((
                abs(weight),
                weight,
                relationship
            ))

        ranked.sort(
            key=lambda item: item[0],
            reverse=True
        )

        for _, weight, (i, j, difference) in ranked[:3]:

            print(
                f"  {CHANNEL_NAMES[i]}-{CHANNEL_NAMES[j]} "
                f"({difference} Hz): {weight:+.6f}"
            )


# ------------------------------------------------------------
# 14. RUN ONE CONFIGURATION
# ------------------------------------------------------------

def run_configuration(carrier_count, seed):
    """
    Run the same experiment with a selected number of carriers.
    """

    frequencies = FREQUENCIES[:carrier_count]

    relationships = build_relationships(frequencies)

    validate_configuration(
        frequencies,
        relationships
    )

    rng = np.random.default_rng(seed)

    features, targets, groups = build_dataset(
        frequencies,
        relationships,
        rng
    )

    training_indices, testing_indices = split_dataset(
        groups,
        rng
    )

    training_targets = targets[training_indices]
    testing_targets = targets[testing_indices]

    # Original carrier representation.
    original_training = features[
        training_indices, :carrier_count
    ]

    original_testing = features[
        testing_indices, :carrier_count
    ]

    original_weights, original_mean, original_std = (
        train_classifier(
            original_training,
            training_targets
        )
    )

    original_predictions = predict(
        original_testing,
        original_weights,
        original_mean,
        original_std
    )

    # Full pairwise-enhanced representation.
    enhanced_training = features[training_indices]
    enhanced_testing = features[testing_indices]

    enhanced_weights, enhanced_mean, enhanced_std = (
        train_classifier(
            enhanced_training,
            training_targets
        )
    )

    enhanced_predictions = predict(
        enhanced_testing,
        enhanced_weights,
        enhanced_mean,
        enhanced_std
    )

    original_accuracy, original_joint = evaluate(
        original_predictions,
        testing_targets
    )

    enhanced_accuracy, enhanced_joint = evaluate(
        enhanced_predictions,
        testing_targets
    )

    return {
        "carrier_count": carrier_count,
        "relationship_count": len(relationships),
        "feature_count": features.shape[1],
        "training_count": len(training_indices),
        "testing_count": len(testing_indices),
        "original_accuracy": original_accuracy,
        "original_joint": original_joint,
        "enhanced_accuracy": enhanced_accuracy,
        "enhanced_joint": enhanced_joint,
        "weights": enhanced_weights,
        "relationships": relationships
    }


# ------------------------------------------------------------
# 15. RUN ALL SCALABILITY EXPERIMENTS
# ------------------------------------------------------------

def main():

    print("=" * 65)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("SCRIPT 03: SCALABLE FREQUENCY REPRESENTATION")
    print("=" * 65)

    print("\nEXPERIMENT CONFIGURATION")
    print("------------------------")
    print(f"Carrier configurations: {CARRIER_COUNTS}")
    print(f"Examples per target combination: "
          f"{EXAMPLES_PER_TARGET_CASE}")
    print(f"Training fraction: {TRAINING_FRACTION:.0%}")
    print("Learning method: Ordinary least squares")

    results = []

    for carrier_count in CARRIER_COUNTS:

        print("\n" + "=" * 65)
        print(f"CONFIGURATION: {carrier_count} CARRIERS")
        print("=" * 65)

        result = run_configuration(
            carrier_count,
            RANDOM_SEED + carrier_count
        )

        results.append(result)

        print("\nAUTOMATICALLY GENERATED REPRESENTATION")
        print("--------------------------------------")

        print(
            f"Carriers: {result['carrier_count']}"
        )

        print(
            f"Pairwise relationships: "
            f"{result['relationship_count']}"
        )

        print(
            f"Total enhanced features: "
            f"{result['feature_count']}"
        )

        print(
            f"Training examples: "
            f"{result['training_count']}"
        )

        print(
            f"Testing examples: "
            f"{result['testing_count']}"
        )

        print("\nCLASSIFICATION RESULTS")
        print("----------------------")

        print("Original carrier features:")

        for i, accuracy in enumerate(
            result["original_accuracy"]
        ):

            print(
                f"  Output {i + 1}: {accuracy:.2f}%"
            )

        print(
            f"  Both outputs correct: "
            f"{result['original_joint']:.2f}%"
        )

        print("\nPairwise-enhanced features:")

        for i, accuracy in enumerate(
            result["enhanced_accuracy"]
        ):

            print(
                f"  Output {i + 1}: {accuracy:.2f}%"
            )

        print(
            f"  Both outputs correct: "
            f"{result['enhanced_joint']:.2f}%"
        )

        print("\nSTRONGEST LEARNED PAIRWISE WEIGHTS")
        print("----------------------------------")

        show_important_relationships(
            result["weights"],
            result["relationships"]
        )

    # --------------------------------------------------------
    # FINAL SCALABILITY SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SCALABILITY SUMMARY")
    print("=" * 65)

    print(
        "\nCarriers | Relationships | Features | "
        "Original Joint | Enhanced Joint"
    )

    print("-" * 65)

    for result in results:

        print(
            f"{result['carrier_count']:8d} | "
            f"{result['relationship_count']:13d} | "
            f"{result['feature_count']:8d} | "
            f"{result['original_joint']:13.2f}% | "
            f"{result['enhanced_joint']:13.2f}%"
        )

    print("\nEXPERIMENT COMPLETE")
    print("-------------------")


if __name__ == "__main__":
    main()
