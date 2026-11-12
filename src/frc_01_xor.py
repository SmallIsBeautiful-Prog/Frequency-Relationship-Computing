
"""
FREQUENCY RELATIONSHIP COMPUTING
SCRIPT 01: ORIGINAL XOR LEARNING

Purpose:
    Demonstrate how pairwise frequency relationships allow a
    simple linear classifier to learn XOR.

Representation:
    A       = measured amplitude of the first carrier
    B       = measured amplitude of the second carrier
    A * B   = pairwise interaction, measured at the difference frequency

Two classifiers are compared:
    1. Original features: A, B
    2. Enhanced features: A, B, A*B

The classifier uses ordinary least squares.
No machine-learning libraries are required.
"""

import numpy as np


# ------------------------------------------------------------
# 1. EXPERIMENT CONFIGURATION
# ------------------------------------------------------------

SAMPLE_RATE = 96000
DURATION = 0.1

FREQUENCY_A = 12000
FREQUENCY_B = 11900

AMPLITUDE = 0.08
AMPLITUDE_VARIATION = 0.15

EXAMPLES_PER_CASE = 200
TRAINING_PER_CASE = 140

RANDOM_SEED = 101

# The 100 Hz difference frequency is our pairwise relationship.
DIFFERENCE_FREQUENCY = abs(FREQUENCY_A - FREQUENCY_B)


# ------------------------------------------------------------
# 2. GENERATE THE TIME AXIS
# ------------------------------------------------------------

SAMPLE_COUNT = int(SAMPLE_RATE * DURATION)

time = np.arange(SAMPLE_COUNT) / SAMPLE_RATE


# ------------------------------------------------------------
# 3. MEASURE A FREQUENCY COMPONENT
# ------------------------------------------------------------

def measure_frequency(signal, frequency):
    """
    Measure the amplitude of a specific frequency using an FFT.

    Our experiment uses a duration that contains an exact
    number of cycles, so no window function is required.
    """

    spectrum = np.fft.rfft(signal)

    # Convert the FFT result into a single-sided amplitude spectrum.
    amplitudes = np.abs(spectrum) * (2.0 / len(signal))

    # Find the FFT bin corresponding to the requested frequency.
    bin_index = int(round(frequency * DURATION))

    return amplitudes[bin_index]


# ------------------------------------------------------------
# 4. GENERATE ONE XOR EXAMPLE
# ------------------------------------------------------------

def generate_example(input_a, input_b, rng):
    """
    Generate two frequency components.

    A binary input of 1 activates its frequency.
    A binary input of 0 leaves that frequency inactive.

    Small amplitude variations make the task less artificial.
    """

    if input_a == 1:
        amplitude_a = AMPLITUDE * (
            1.0 + rng.uniform(-AMPLITUDE_VARIATION,
                              AMPLITUDE_VARIATION)
        )
    else:
        amplitude_a = 0.0

    if input_b == 1:
        amplitude_b = AMPLITUDE * (
            1.0 + rng.uniform(-AMPLITUDE_VARIATION,
                              AMPLITUDE_VARIATION)
        )
    else:
        amplitude_b = 0.0

    # Construct the two carrier waveforms.
    signal_a = amplitude_a * np.sin(
        2.0 * np.pi * FREQUENCY_A * time
    )

    signal_b = amplitude_b * np.sin(
        2.0 * np.pi * FREQUENCY_B * time
    )

    # Combine both frequencies.
    combined_signal = signal_a + signal_b

    # Apply the square-law transformation.
    squared_signal = combined_signal ** 2

    # Measure the original carrier frequencies from the
    # original combined waveform.
    measured_a = measure_frequency(
        combined_signal, FREQUENCY_A
    )

    measured_b = measure_frequency(
        combined_signal, FREQUENCY_B
    )

    # Measure the pairwise difference frequency from
    # the squared waveform.
    measured_interaction = measure_frequency(
        squared_signal, DIFFERENCE_FREQUENCY
    )

    # XOR is true when exactly one input is active.
    target = input_a ^ input_b

    return [
        measured_a,
        measured_b,
        measured_interaction
    ], target


# ------------------------------------------------------------
# 5. BUILD THE DATASET
# ------------------------------------------------------------

def build_dataset(rng):
    """
    Generate an equal number of examples for all four
    possible combinations of binary inputs.
    """

    features = []
    targets = []
    groups = []

    input_cases = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]

    for input_a, input_b in input_cases:

        for _ in range(EXAMPLES_PER_CASE):

            example, target = generate_example(
                input_a, input_b, rng
            )

            features.append(example)
            targets.append(target)

            # Retain the original input case so that we can
            # split each case evenly into training and testing.
            groups.append((input_a, input_b))

    return (
        np.asarray(features),
        np.asarray(targets, dtype=float),
        np.asarray(groups)
    )


# ------------------------------------------------------------
# 6. SPLIT TRAINING AND TEST DATA
# ------------------------------------------------------------

def split_dataset(targets, groups, rng):
    """
    Split each of the four input combinations separately.

    This keeps the training and testing datasets balanced.
    """

    training_indices = []
    testing_indices = []

    input_cases = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]

    for case in input_cases:

        matching_indices = np.where(
            np.all(groups == case, axis=1)
        )[0]

        shuffled_indices = rng.permutation(matching_indices)

        training_indices.extend(
            shuffled_indices[:TRAINING_PER_CASE]
        )

        testing_indices.extend(
            shuffled_indices[TRAINING_PER_CASE:]
        )

    return (
        np.asarray(training_indices),
        np.asarray(testing_indices)
    )


# ------------------------------------------------------------
# 7. TRAIN THE LINEAR CLASSIFIER
# ------------------------------------------------------------

def train_classifier(training_features, training_targets):
    """
    Learn numerical weights using ordinary least squares.

    Features are standardised using training data only.

    A bias column is added so the classifier can learn
    an adjustable decision boundary.
    """

    feature_mean = np.mean(training_features, axis=0)
    feature_std = np.std(training_features, axis=0)

    # Avoid division by zero if a feature is constant.
    feature_std[feature_std == 0] = 1.0

    normalised_features = (
        training_features - feature_mean
    ) / feature_std

    # Add the bias column.
    design_matrix = np.column_stack([
        np.ones(len(normalised_features)),
        normalised_features
    ])

    # Find the weights that minimise squared prediction error.
    weights, _, _, _ = np.linalg.lstsq(
        design_matrix,
        training_targets,
        rcond=None
    )

    return weights, feature_mean, feature_std


# ------------------------------------------------------------
# 8. MAKE PREDICTIONS
# ------------------------------------------------------------

def predict(features, weights, feature_mean, feature_std):
    """
    Apply the learned linear weights to new examples.

    A score of 0.5 or higher is classified as XOR = 1.
    """

    normalised_features = (
        features - feature_mean
    ) / feature_std

    design_matrix = np.column_stack([
        np.ones(len(normalised_features)),
        normalised_features
    ])

    scores = design_matrix @ weights

    predictions = (scores >= 0.5).astype(int)

    return predictions


# ------------------------------------------------------------
# 9. EVALUATE CLASSIFIER PERFORMANCE
# ------------------------------------------------------------

def evaluate(name, predictions, actual):
    """
    Report accuracy and a simple confusion matrix.
    """

    accuracy = np.mean(predictions == actual) * 100.0

    true_negative = np.sum(
        (predictions == 0) & (actual == 0)
    )

    false_positive = np.sum(
        (predictions == 1) & (actual == 0)
    )

    false_negative = np.sum(
        (predictions == 0) & (actual == 1)
    )

    true_positive = np.sum(
        (predictions == 1) & (actual == 1)
    )

    print(f"\n{name}")
    print("-" * len(name))

    print(f"Accuracy: {accuracy:.2f}%")

    print("Confusion matrix:")
    print("                 Predicted")
    print("                 0       1")
    print(f"Actual 0      {true_negative:4d}    {false_positive:4d}")
    print(f"Actual 1      {false_negative:4d}    {true_positive:4d}")


# ------------------------------------------------------------
# 10. RUN THE EXPERIMENT
# ------------------------------------------------------------

def main():

    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("SCRIPT 01: ORIGINAL XOR LEARNING")
    print("=" * 60)

    print("\nEXPERIMENT CONFIGURATION")
    print("------------------------")
    print(f"Carrier A: {FREQUENCY_A} Hz")
    print(f"Carrier B: {FREQUENCY_B} Hz")
    print(f"Pairwise difference: {DIFFERENCE_FREQUENCY} Hz")
    print(f"Examples: {EXAMPLES_PER_CASE * 4}")
    print(f"Training examples: {TRAINING_PER_CASE * 4}")

    # Generate the examples.
    features, targets, groups = build_dataset(rng)

    # Separate training and testing examples.
    training_indices, testing_indices = split_dataset(
        targets, groups, rng
    )

    training_targets = targets[training_indices]
    testing_targets = targets[testing_indices]

    # --------------------------------------------------------
    # CLASSIFIER 1: ORIGINAL FEATURES
    # --------------------------------------------------------

    # Use only the two original carrier amplitudes.
    original_training = features[training_indices, :2]
    original_testing = features[testing_indices, :2]

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
    # CLASSIFIER 2: PAIRWISE-ENHANCED FEATURES
    # --------------------------------------------------------

    # Include the measured difference-frequency amplitude.
    enhanced_training = features[training_indices, :]
    enhanced_testing = features[testing_indices, :]

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
    print("Original: A, B")
    print("Enhanced: A, B, A*B")

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

    print("\nLEARNED ENHANCED CLASSIFIER WEIGHTS")
    print("-----------------------------------")
    print(f"Bias: {enhanced_weights[0]:.6f}")
    print(f"A:    {enhanced_weights[1]:.6f}")
    print(f"B:    {enhanced_weights[2]:.6f}")
    print(f"A*B:  {enhanced_weights[3]:.6f}")

    print("\nEXPERIMENT COMPLETE")
    print("-------------------")


if __name__ == "__main__":
    main()
