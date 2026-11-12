
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 04: RELATIONSHIP LEARNING

Purpose:
    Build a reusable linear learning model that can:

    1. Learn from original carrier amplitudes and pairwise relationships.
    2. Store and report its learned weights.
    3. Predict multiple outputs.
    4. Test the importance of individual relationships by removing them.

Experiment:
    Eight frequency carriers.
    Twenty-eight pairwise relationships.

    Output 1 = XOR(A, B)
    Output 2 = XOR(C, D)

The classifier remains a linear least-squares model.
The expanded representation supplies the pairwise features.
"""

import numpy as np


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

FREQUENCIES = np.array([
    12000,
    11900,
    11770,
    11610,
    11420,
    11200,
    10950,
    10670
], dtype=float)

SAMPLE_RATE = 96000
DURATION = 0.1

BASE_AMPLITUDE = 0.08
AMPLITUDE_VARIATION = 0.15

EXAMPLES_PER_TARGET = 400
TRAINING_FRACTION = 0.70

RANDOM_SEED = 404

CARRIER_COUNT = len(FREQUENCIES)


# ============================================================
# 2. GENERATE ALL PAIRWISE RELATIONSHIPS
# ============================================================

def create_relationships(frequencies):
    """
    Generate every unique pair of carriers.

    Each relationship contains:
        First carrier index
        Second carrier index
        Absolute frequency difference
    """

    relationships = []

    for i in range(len(frequencies)):
        for j in range(i + 1, len(frequencies)):

            difference = abs(frequencies[i] - frequencies[j])

            relationships.append((i, j, difference))

    return relationships


RELATIONSHIPS = create_relationships(FREQUENCIES)


# ============================================================
# 3. GENERATE THE LEARNING DATA
# ============================================================

def generate_dataset():
    """
    Generate examples for two simultaneous XOR tasks.

    Output 1:
        XOR of carriers A and B.

    Output 2:
        XOR of carriers C and D.

    The remaining four carrier inputs are random.

    Each possible combination of the two target outputs
    receives the same number of examples.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    inputs = []
    targets = []

    for target_1 in (0, 1):
        for target_2 in (0, 1):

            for _ in range(EXAMPLES_PER_TARGET):

                # Randomly choose the first input of each XOR pair.
                a = rng.integers(0, 2)
                c = rng.integers(0, 2)

                # Set the second input to produce the required XOR.
                b = a ^ target_1
                d = c ^ target_2

                # Four additional independent binary inputs.
                extra_inputs = rng.integers(0, 2, size=4)

                example = [
                    a, b, c, d,
                    *extra_inputs
                ]

                inputs.append(example)
                targets.append([target_1, target_2])

    inputs = np.asarray(inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)

    # Shuffle examples while keeping inputs and targets together.
    order = rng.permutation(len(inputs))

    return inputs[order], targets[order]


# ============================================================
# 4. CONVERT INPUTS INTO FREQUENCY FEATURES
# ============================================================

def extract_features(binary_inputs):
    """
    Convert binary inputs into measured frequency features.

    Original carrier amplitudes are measured from the
    combined waveform.

    Pairwise difference amplitudes are measured from
    the squared waveform.

    The squared waveform applies the square-law
    transformation that exposes pairwise interactions.
    """

    sample_count = int(SAMPLE_RATE * DURATION)

    time = np.arange(sample_count) / SAMPLE_RATE

    # Precompute the eight carrier waveforms.
    carrier_basis = np.array([
        np.sin(2 * np.pi * frequency * time)
        for frequency in FREQUENCIES
    ])

    # Convert frequencies into their exact FFT bins.
    carrier_bins = np.rint(
        FREQUENCIES * DURATION
    ).astype(int)

    relationship_bins = np.array([
        int(round(difference * DURATION))
        for _, _, difference in RELATIONSHIPS
    ])

    # Give every feature a readable name.
    feature_names = [
        f"{chr(65 + i)} ({int(frequency)} Hz)"
        for i, frequency in enumerate(FREQUENCIES)
    ]

    for i, j, difference in RELATIONSHIPS:

        name = (
            f"{chr(65 + i)}-{chr(65 + j)} "
            f"({int(difference)} Hz)"
        )

        feature_names.append(name)

    # One reproducible generator for all amplitude variations.
    rng = np.random.default_rng(RANDOM_SEED + 2)

    features = []

    for binary_example in binary_inputs:

        # Start with all carrier amplitudes switched off.
        amplitudes = np.zeros(CARRIER_COUNT)

        # Find which carriers are active.
        active = np.where(binary_example == 1)[0]

        # Give active carriers a small random amplitude variation.
        if len(active):

            amplitudes[active] = (
                BASE_AMPLITUDE
                * rng.uniform(
                    1.0 - AMPLITUDE_VARIATION,
                    1.0 + AMPLITUDE_VARIATION,
                    size=len(active)
                )
            )

        # Combine the original carrier waveforms.
        combined_waveform = amplitudes @ carrier_basis

        # Apply the square-law transformation.
        squared_waveform = combined_waveform ** 2

        # Measure original carrier amplitudes.
        carrier_spectrum = np.abs(
            np.fft.rfft(combined_waveform)
        ) * (2.0 / sample_count)

        measured_carriers = carrier_spectrum[carrier_bins]

        # Measure pairwise difference-frequency amplitudes.
        interaction_spectrum = np.abs(
            np.fft.rfft(squared_waveform)
        ) * (2.0 / sample_count)

        measured_relationships = (
            interaction_spectrum[relationship_bins]
        )

        # Combine original and pairwise features.
        example_features = np.concatenate([
            measured_carriers,
            measured_relationships
        ])

        features.append(example_features)

    return np.asarray(features), feature_names


# ============================================================
# 5. REUSABLE LINEAR RELATIONSHIP MODEL
# ============================================================

class LinearRelationshipModel:
    """
    Reusable multi-output linear least-squares classifier.

    The model:

        1. Standardises input features using training data.
        2. Adds a bias term.
        3. Learns weights using least squares.
        4. Converts scores into binary predictions.

    It can be reused with different frequency features
    and different target outputs.
    """

    def __init__(self):

        self.feature_means = None
        self.feature_scales = None
        self.weights = None

    def fit(self, features, targets):

        # Learn normalisation values from training data only.
        self.feature_means = np.mean(features, axis=0)
        self.feature_scales = np.std(features, axis=0)

        # Avoid division by zero for constant features.
        self.feature_scales[
            self.feature_scales < 1e-12
        ] = 1.0

        normalised = (
            features - self.feature_means
        ) / self.feature_scales

        # Add a constant bias feature.
        design_matrix = np.column_stack([
            np.ones(len(normalised)),
            normalised
        ])

        # Learn all output weights in one least-squares operation.
        self.weights, _, _, _ = np.linalg.lstsq(
            design_matrix,
            targets,
            rcond=None
        )

        return self

    def predict_scores(self, features):

        normalised = (
            features - self.feature_means
        ) / self.feature_scales

        design_matrix = np.column_stack([
            np.ones(len(normalised)),
            normalised
        ])

        return design_matrix @ self.weights

    def predict(self, features):

        scores = self.predict_scores(features)

        return (scores >= 0.5).astype(int)

    def feature_weights(self):
        """
        Return learned feature weights, excluding the bias.
        """

        return self.weights[1:, :]


# ============================================================
# 6. EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(model, features, targets):

    predictions = model.predict(features)

    output_1_accuracy = np.mean(
        predictions[:, 0] == targets[:, 0]
    )

    output_2_accuracy = np.mean(
        predictions[:, 1] == targets[:, 1]
    )

    joint_accuracy = np.mean(
        np.all(predictions == targets, axis=1)
    )

    return (
        output_1_accuracy,
        output_2_accuracy,
        joint_accuracy
    )


def print_accuracy(label, results):

    output_1, output_2, joint = results

    print(f"\n{label}")
    print("-" * len(label))

    print(f"Output 1 accuracy: {output_1 * 100:.2f}%")
    print(f"Output 2 accuracy: {output_2 * 100:.2f}%")
    print(f"Both outputs correct: {joint * 100:.2f}%")


# ============================================================
# 7. REPORT THE LEARNED FEATURE WEIGHTS
# ============================================================

def report_weights(model, feature_names):

    weights = model.feature_weights()

    output_names = [
        "Output 1: XOR(A, B)",
        "Output 2: XOR(C, D)"
    ]

    print("\nLEARNED FEATURE WEIGHTS")
    print("=======================")

    for output_index, output_name in enumerate(output_names):

        print(f"\n{output_name}")
        print("-" * len(output_name))

        ranked_features = sorted(
            range(len(feature_names)),
            key=lambda i: abs(weights[i, output_index]),
            reverse=True
        )

        print(
            f"{'Feature':<26}"
            f"{'Weight':>12}"
        )

        for feature_index in ranked_features[:10]:

            print(
                f"{feature_names[feature_index]:<26}"
                f"{weights[feature_index, output_index]:>12.6f}"
            )


# ============================================================
# 8. TEST THE IMPORTANCE OF EACH RELATIONSHIP
# ============================================================

def run_relationship_ablation(
    training_features,
    training_targets,
    test_features,
    test_targets,
    feature_names
):
    """
    Remove one pairwise relationship at a time.

    A new model is trained without that relationship.

    We then measure how much test performance changes.

    This tests whether the model actually needs a particular
    relationship to solve either task.
    """

    results = []

    for relationship_index, relationship in enumerate(RELATIONSHIPS):

        feature_index = (
            CARRIER_COUNT + relationship_index
        )

        # Keep every feature except the selected relationship.
        keep_indices = [
            i for i in range(training_features.shape[1])
            if i != feature_index
        ]

        reduced_training = training_features[:, keep_indices]
        reduced_test = test_features[:, keep_indices]

        # Train a fresh model without the selected feature.
        model = LinearRelationshipModel()

        model.fit(
            reduced_training,
            training_targets
        )

        accuracy = evaluate_model(
            model,
            reduced_test,
            test_targets
        )

        results.append((
            feature_names[feature_index],
            accuracy
        ))

    print("\nPAIRWISE RELATIONSHIP ABLATION")
    print("==============================")
    print("Each row shows performance after removing one relationship.")
    print()

    print(
        f"{'Removed relationship':<26}"
        f"{'Output 1':>12}"
        f"{'Output 2':>12}"
        f"{'Both correct':>16}"
    )

    for name, accuracy in results:

        output_1, output_2, joint = accuracy

        print(
            f"{name:<26}"
            f"{output_1 * 100:>11.2f}%"
            f"{output_2 * 100:>11.2f}%"
            f"{joint * 100:>15.2f}%"
        )

    return results


# ============================================================
# 9. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 04: RELATIONSHIP LEARNING")
    print("=" * 65)

    print("\nEXPERIMENT CONFIGURATION")
    print("------------------------")

    print(f"Carrier frequencies: {FREQUENCIES.astype(int).tolist()}")
    print(f"Number of carriers: {CARRIER_COUNT}")
    print(f"Pairwise relationships: {len(RELATIONSHIPS)}")
    print(f"Examples per target combination: {EXAMPLES_PER_TARGET}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")

    # Generate the input examples and target outputs.
    binary_inputs, targets = generate_dataset()

    # Convert inputs into carrier and relationship features.
    features, feature_names = extract_features(binary_inputs)

    print(f"\nTotal examples: {len(features)}")
    print(f"Original carrier features: {CARRIER_COUNT}")
    print(f"Pairwise relationship features: {len(RELATIONSHIPS)}")
    print(f"Total expanded features: {features.shape[1]}")

    # Split into training and testing examples.
    rng = np.random.default_rng(RANDOM_SEED + 1)

    order = rng.permutation(len(features))

    training_count = int(
        len(features) * TRAINING_FRACTION
    )

    training_indices = order[:training_count]
    test_indices = order[training_count:]

    training_features = features[training_indices]
    test_features = features[test_indices]

    training_targets = targets[training_indices]
    test_targets = targets[test_indices]

    print(f"\nTraining examples: {len(training_indices)}")
    print(f"Testing examples: {len(test_indices)}")

    # --------------------------------------------------------
    # BASELINE: ORIGINAL CARRIER FEATURES ONLY
    # --------------------------------------------------------

    baseline_model = LinearRelationshipModel()

    baseline_model.fit(
        training_features[:, :CARRIER_COUNT],
        training_targets
    )

    baseline_results = evaluate_model(
        baseline_model,
        test_features[:, :CARRIER_COUNT],
        test_targets
    )

    print_accuracy(
        "BASELINE: ORIGINAL CARRIER FEATURES ONLY",
        baseline_results
    )

    # --------------------------------------------------------
    # FULL MODEL: CARRIERS + ALL PAIRWISE RELATIONSHIPS
    # --------------------------------------------------------

    relationship_model = LinearRelationshipModel()

    relationship_model.fit(
        training_features,
        training_targets
    )

    full_results = evaluate_model(
        relationship_model,
        test_features,
        test_targets
    )

    print_accuracy(
        "EXPANDED MODEL: CARRIERS + PAIRWISE RELATIONSHIPS",
        full_results
    )

    # --------------------------------------------------------
    # REPORT LEARNED WEIGHTS
    # --------------------------------------------------------

    report_weights(
        relationship_model,
        feature_names
    )

    # --------------------------------------------------------
    # REMOVE EACH RELATIONSHIP IN TURN
    # --------------------------------------------------------

    run_relationship_ablation(
        training_features,
        training_targets,
        test_features,
        test_targets,
        feature_names
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FRC 04 EXPERIMENT COMPLETE")
    print("=" * 65)

    print("\nThe model has now demonstrated:")

    print("1. Reusable linear model training.")
    print("2. Learning weights for multiple outputs.")
    print("3. Ranking original and pairwise features.")
    print("4. Testing individual relationship importance.")

    print("\nResearch note:")
    print(
        "Ablation measures how much performance changes when "
        "a feature is removed and the model is retrained."
    )

    print(
        "A large change suggests that the removed feature "
        "was useful for the tested task."
    )


if __name__ == "__main__":
    main()
