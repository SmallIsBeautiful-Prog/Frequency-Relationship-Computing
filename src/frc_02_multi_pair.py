
"""
FREQUENCY RELATIONSHIP COMPUTING
SCRIPT 02: MULTI-PAIR LEARNING

Purpose:
    Test whether one linear learning architecture can learn
    multiple pairwise XOR relationships simultaneously.

Four carrier frequencies produce six unique pairwise
difference-frequency components.

Two outputs are learned:
    Output 1 = XOR(A, B)
    Output 2 = XOR(C, D)

Two representations are compared:
    1. Original carrier amplitudes: A, B, C, D
    2. Enhanced representation: four carriers + six interactions

The learning method remains ordinary least squares.
"""

import numpy as np
from itertools import product


# ------------------------------------------------------------
# 1. EXPERIMENT CONFIGURATION
# ------------------------------------------------------------

SAMPLE_RATE = 96000
DURATION = 0.1

FREQUENCIES = np.array([
    12000,
    11900,
    11770,
    11610
])

AMPLITUDE = 0.08
AMPLITUDE_VARIATION = 0.15

EXAMPLES_PER_CASE = 200
TRAINING_PER_CASE = 140

RANDOM_SEED = 201

CHANNEL_NAMES = ["A", "B", "C", "D"]


# ------------------------------------------------------------
# 2. CONSTRUCT THE PAIRWISE RELATIONSHIPS
# ------------------------------------------------------------

def build_relationships():
    """
    Generate every unique pair of carrier frequencies.

    Each relationship is identified by:
        channel index 1
        channel index 2
        absolute frequency difference
    """

    relationships = []

    for i in range(len(FREQUENCIES)):

        for j in range(i + 1, len(FREQUENCIES)):

            difference = abs(
                FREQUENCIES[i] - FREQUENCIES[j]
            )

            relationships.append((i, j, difference))

    return relationships


RELATIONSHIPS = build_relationships()


# ------------------------------------------------------------
# 3. PREPARE THE TIME AXIS
# ------------------------------------------------------------

SAMPLE_COUNT = int(SAMPLE_RATE * DURATION)

time = np.arange(SAMPLE_COUNT) / SAMPLE_RATE


# ------------------------------------------------------------
# 4. MEASURE A FREQUENCY COMPONENT
# ------------------------------------------------------------

def measure_frequency(signal, frequency):
    """
    Measure the amplitude of a specified frequency using FFT.

    The selected frequencies align with exact FFT bins.
    """

    spectrum = np.fft.rfft(signal)

    amplitudes = (
        np.abs(spectrum) * (2.0 / len(signal))
    )

    bin_index = int(round(frequency * DURATION))

    return amplitudes[bin_index]


# ------------------------------------------------------------
# 5. GENERATE ONE EXAMPLE
# ------------------------------------------------------------

def generate_example(inputs, rng):
    """
    Generate four carrier waveforms from four binary inputs.

    Combine them, square the combined waveform, and measure:
        - Four original carrier amplitudes
        - Six pairwise difference-frequency amplitudes

    The two learning targets are independent XOR tasks.
    """

    carrier_signals = []

    for i in range(len(FREQUENCIES)):

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
            2.0 * np.pi * FREQUENCIES[i] * time
        )

        carrier_signals.append(signal)

    # Combine all four frequency components.
    combined_signal = np.sum(
        carrier_signals,
        axis=0
    )

    # Apply square-law processing.
    squared_signal = combined_signal ** 2

    # Measure the original carrier amplitudes.
    carrier_features = []

    for frequency in FREQUENCIES:

        measured = measure_frequency(
            combined_signal,
            frequency
        )

        carrier_features.append(measured)

    # Measure all six pairwise difference frequencies.
    interaction_features = []

    for _, _, difference in RELATIONSHIPS:

        measured = measure_frequency(
            squared_signal,
            difference
        )

        interaction_features.append(measured)

    # Combine original and pairwise features.
    features = (
        carrier_features + interaction_features
    )

    # Two simultaneous XOR targets.
    target_1 = inputs[0] ^ inputs[1]
    target_2 = inputs[2] ^ inputs[3]

    targets = [target_1, target_2]

    return features, targets


# ------------------------------------------------------------
# 6. BUILD THE DATASET
# ------------------------------------------------------------

def build_dataset(rng):
    """
    Generate examples for all 16 possible combinations
    of the four binary inputs.
    """

    features = []
    targets = []
    groups = []

    input_cases = list(
        product([0, 1], repeat=4)
    )

    for inputs in input_cases:

        for _ in range(EXAMPLES_PER_CASE):

            example, target = generate_example(
                inputs,
                rng
            )

            features.append(example)
            targets.append(target)
            groups.append(inputs)

    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(groups, dtype=int)
    )


# ------------------------------------------------------------
# 7. SPLIT TRAINING AND TEST DATA
# ------------------------------------------------------------

def split_dataset(groups, rng):
    """
    Split every input combination separately.

    This preserves equal representation of all 16
    combinations in both training and testing data.
    """

    training_indices = []
    testing_indices = []

    input_cases = list(
        product([0, 1], repeat=4)
    )

    for case in input_cases:

        matching = np.where(
            np.all(groups == case, axis=1)
        )[0]

        shuffled = rng.permutation(matching)

        training_indices.extend(
            shuffled[:TRAINING_PER_CASE]
        )

        testing_indices.extend(
            shuffled[TRAINING_PER_CASE:]
        )

    return (
        np.asarray(training_indices),
        np.asarray(testing_indices)
    )


# ------------------------------------------------------------
# 8. TRAIN THE LINEAR CLASSIFIER
# ------------------------------------------------------------

def train_classifier(features, targets):
    """
    Train a multi-output linear classifier using least squares.

    Each output receives its own learned weights, but both
    outputs use the same input representation and training
    procedure.

    Normalisation is calculated from training data only.
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
# 9. PREDICT OUTPUTS
# ------------------------------------------------------------

def predict(features, weights, feature_mean, feature_std):
    """
    Generate predictions for both XOR outputs.
    """

    normalised = (
        features - feature_mean
    ) / feature_std

    design_matrix = np.column_stack([
        np.ones(len(normalised)),
        normalised
    ])

    scores = design_matrix @ weights

    predictions = (scores >= 0.5).astype(int)

    return predictions


# ------------------------------------------------------------
# 10. EVALUATE THE CLASSIFIER
# ------------------------------------------------------------

def evaluate(name, predictions, actual):
    """
    Report individual output accuracy and joint accuracy.

    Joint accuracy requires both XOR outputs to be correct
    on the same example.
    """

    print(f"\n{name}")
    print("-" * len(name))

    for output_index in range(2):

        accuracy = np.mean(
            predictions[:, output_index]
            == actual[:, output_index]
        ) * 100.0

        print(
            f"Output {output_index + 1} accuracy: "
            f"{accuracy:.2f}%"
        )

    joint_accuracy = np.mean(
        np.all(predictions == actual, axis=1)
    ) * 100.0

    print(
        f"Both outputs correct: {joint_accuracy:.2f}%"
    )


# ------------------------------------------------------------
# 11. RUN THE EXPERIMENT
# ------------------------------------------------------------

def main():

    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("SCRIPT 02: MULTI-PAIR LEARNING")
    print("=" * 60)

    print("\nEXPERIMENT CONFIGURATION")
    print("------------------------")

    for name, frequency in zip(
        CHANNEL_NAMES, FREQUENCIES
    ):

        print(f"Carrier {name}: {frequency} Hz")

    print(f"\nCarrier count: {len(FREQUENCIES)}")
    print(f"Pairwise relationships: {len(RELATIONSHIPS)}")

    print("\nPAIRWISE RELATIONSHIPS")
    print("----------------------")

    for i, j, difference in RELATIONSHIPS:

        print(
            f"{CHANNEL_NAMES[i]}-{CHANNEL_NAMES[j]}: "
            f"{difference} Hz"
        )

    print("\nDATASET")
    print("-------")

    features, targets, groups = build_dataset(rng)

    training_indices, testing_indices = split_dataset(
        groups,
        rng
    )

    print(f"Total examples: {len(features)}")
    print(f"Training examples: {len(training_indices)}")
    print(f"Testing examples: {len(testing_indices)}")

    training_targets = targets[training_indices]
    testing_targets = targets[testing_indices]

    # --------------------------------------------------------
    # CLASSIFIER 1: ORIGINAL CARRIER FEATURES
    # --------------------------------------------------------

    original_training = features[
        training_indices, :4
    ]

    original_testing = features[
        testing_indices, :4
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

    # --------------------------------------------------------
    # CLASSIFIER 2: ALL PAIRWISE FEATURES
    # --------------------------------------------------------

    enhanced_training = features[
        training_indices, :
    ]

    enhanced_testing = features[
        testing_indices, :
    ]

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

    # --------------------------------------------------------
    # REPORT RESULTS
    # --------------------------------------------------------

    print("\nFEATURE REPRESENTATIONS")
    print("-----------------------")
    print("Original: A, B, C, D")
    print("Enhanced: A, B, C, D + 6 pairwise relationships")

    evaluate(
        "ORIGINAL LINEAR CLASSIFIER",
        original_predictions,
        testing_targets
    )

    evaluate(
        "PAIRWISE-ENHANCED LINEAR CLASSIFIER",
        enhanced_predictions,
        testing_targets
    )

    print("\nLEARNED PAIRWISE WEIGHTS")
    print("------------------------")

    for output_index in range(2):

        print(f"\nOutput {output_index + 1}:")

        for index, relationship in enumerate(RELATIONSHIPS):

            weight = enhanced_weights[
                1 + len(FREQUENCIES) + index,
                output_index
            ]

            i, j, difference = relationship

            print(
                f"{CHANNEL_NAMES[i]}-{CHANNEL_NAMES[j]} "
                f"({difference} Hz): {weight:+.6f}"
            )

    print("\nEXPERIMENT COMPLETE")
    print("-------------------")


if __name__ == "__main__":
    main()
