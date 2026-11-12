
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 10: BOOLEAN LEARNING

Purpose:
    Test whether the same linear classifier can learn several
    Boolean operations simultaneously using pairwise features.

Boolean operations:
    Output 1: AND(A, B)
    Output 2: OR(C, D)
    Output 3: XOR(E, F)
    Output 4: NAND(G, H)

Representations:
    1. Original carrier amplitudes: 8 features
    2. Pairwise-enhanced representation: 8 + 28 = 36 features

The pairwise features are amplitude products. These are the
direct arithmetic equivalent of the difference-frequency
amplitudes produced by square-law processing.

Classifier:
    Multi-output least-squares linear model.

No separate classifier is created for each Boolean operation.
One model learns all four outputs together.
"""

import itertools
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
])

NUMBER_OF_CARRIERS = len(FREQUENCIES)

AMPLITUDE = 0.08
AMPLITUDE_VARIATION = 0.15

EXAMPLES_PER_COMBINATION = 200

TRAIN_RATIO = 0.70
RANDOM_SEED = 1001

THRESHOLD = 0.5


# ============================================================
# 2. GENERATE ALL BINARY INPUT COMBINATIONS
# ============================================================

def generate_binary_combinations():
    """
    Generate every possible combination of eight binary inputs.

    Eight binary inputs produce:

        2^8 = 256 combinations

    Each combination is repeated to create multiple examples
    with slightly different carrier amplitudes.
    """

    return np.array(
        list(itertools.product([0, 1], repeat=NUMBER_OF_CARRIERS)),
        dtype=int
    )


# ============================================================
# 3. DEFINE THE FOUR BOOLEAN OPERATIONS
# ============================================================

def calculate_targets(binary_inputs):
    """
    Calculate four Boolean outputs.

    Each output uses a different pair of input channels.
    """

    A = binary_inputs[:, 0]
    B = binary_inputs[:, 1]

    C = binary_inputs[:, 2]
    D = binary_inputs[:, 3]

    E = binary_inputs[:, 4]
    F = binary_inputs[:, 5]

    G = binary_inputs[:, 6]
    H = binary_inputs[:, 7]

    output_and = A & B
    output_or = C | D
    output_xor = E ^ F
    output_nand = 1 - (G & H)

    return np.column_stack([
        output_and,
        output_or,
        output_xor,
        output_nand
    ])


# ============================================================
# 4. CREATE CARRIER AMPLITUDES
# ============================================================

def generate_carrier_amplitudes(binary_inputs, rng):
    """
    Convert binary inputs into carrier amplitudes.

    A binary 0 produces no carrier amplitude.

    A binary 1 produces an amplitude of approximately 0.08,
    with independent variation of plus or minus 15%.

    This prevents the classifier from relying on perfectly
    identical amplitude values.
    """

    variation = rng.uniform(
        1.0 - AMPLITUDE_VARIATION,
        1.0 + AMPLITUDE_VARIATION,
        size=binary_inputs.shape
    )

    amplitudes = (
        binary_inputs
        * AMPLITUDE
        * variation
    )

    return amplitudes


# ============================================================
# 5. GENERATE ALL 28 PAIRWISE RELATIONSHIPS
# ============================================================

def generate_pairwise_features(amplitudes):
    """
    Calculate every unique pairwise amplitude product.

    Eight carriers produce:

        8 * 7 / 2 = 28 relationships

    Each feature represents the relationship between two
    carrier amplitudes.

    Under square-law processing, the corresponding
    difference-frequency component has amplitude proportional
    to this product.

    We use direct multiplication here because earlier FRC
    experiments established its numerical equivalence to
    extracting these components from the squared waveform.
    """

    relationships = []
    relationship_descriptions = []

    for i in range(NUMBER_OF_CARRIERS):

        for j in range(i + 1, NUMBER_OF_CARRIERS):

            product = amplitudes[:, i] * amplitudes[:, j]

            relationships.append(product)

            difference_frequency = abs(
                FREQUENCIES[i] - FREQUENCIES[j]
            )

            description = (
                f"{FREQUENCIES[i]}-{FREQUENCIES[j]} "
                f"({difference_frequency} Hz)"
            )

            relationship_descriptions.append(description)

    relationship_features = np.column_stack(relationships)

    return relationship_features, relationship_descriptions


# ============================================================
# 6. BUILD THE TWO FEATURE REPRESENTATIONS
# ============================================================

def build_representations(amplitudes, relationship_features):

    # Original carrier measurements: 8 features
    original_features = amplitudes

    # Original carriers plus all 28 pairwise relationships
    enhanced_features = np.column_stack([
        amplitudes,
        relationship_features
    ])

    return original_features, enhanced_features


# ============================================================
# 7. CREATE A STRATIFIED TRAINING AND TEST SPLIT
# ============================================================

def split_dataset(features, targets, combination_ids, rng):
    """
    Split examples while preserving every binary input
    combination in both training and testing sets.

    70% of each combination goes into training.
    30% goes into testing.
    """

    train_indices = []
    test_indices = []

    for combination in np.unique(combination_ids):

        indices = np.where(
            combination_ids == combination
        )[0]

        indices = rng.permutation(indices)

        split_point = int(len(indices) * TRAIN_RATIO)

        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    return (
        features[train_indices],
        features[test_indices],
        targets[train_indices],
        targets[test_indices]
    )


# ============================================================
# 8. TRAIN THE LINEAR CLASSIFIER
# ============================================================

def train_linear_classifier(train_features, train_targets):
    """
    Train one multi-output least-squares linear classifier.

    Feature standardisation is calculated using training
    data only.

    A bias column is added so the model can learn an offset.

    np.linalg.lstsq finds the coefficients that minimise
    the sum of squared prediction errors.
    """

    means = np.mean(train_features, axis=0)
    standard_deviations = np.std(train_features, axis=0)

    standard_deviations = np.where(
        standard_deviations < 1e-12,
        1.0,
        standard_deviations
    )

    normalised_features = (
        train_features - means
    ) / standard_deviations

    # Add the bias column.
    design_matrix = np.column_stack([
        np.ones(len(normalised_features)),
        normalised_features
    ])

    # One fit learns all four outputs simultaneously.
    weights, residuals, rank, singular_values = np.linalg.lstsq(
        design_matrix,
        train_targets,
        rcond=None
    )

    return {
        "weights": weights,
        "means": means,
        "standard_deviations": standard_deviations,
        "rank": rank
    }


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

def predict(model, features):
    """
    Apply the trained linear model to new examples.
    """

    normalised_features = (
        features - model["means"]
    ) / model["standard_deviations"]

    design_matrix = np.column_stack([
        np.ones(len(normalised_features)),
        normalised_features
    ])

    scores = design_matrix @ model["weights"]

    predictions = (scores >= THRESHOLD).astype(int)

    return predictions


# ============================================================
# 10. EVALUATE EACH OUTPUT
# ============================================================

def evaluate_model(name, model, test_features, test_targets):

    predictions = predict(model, test_features)

    print()
    print("=" * 68)
    print(name)
    print("=" * 68)

    output_names = [
        "AND",
        "OR",
        "XOR",
        "NAND"
    ]

    output_accuracies = []

    for index, output_name in enumerate(output_names):

        actual = test_targets[:, index]
        predicted = predictions[:, index]

        accuracy = np.mean(actual == predicted) * 100

        true_negative = np.sum(
            (actual == 0) & (predicted == 0)
        )

        false_positive = np.sum(
            (actual == 0) & (predicted == 1)
        )

        false_negative = np.sum(
            (actual == 1) & (predicted == 0)
        )

        true_positive = np.sum(
            (actual == 1) & (predicted == 1)
        )

        output_accuracies.append(accuracy)

        print()
        print(f"{output_name} accuracy: {accuracy:.2f}%")

        print(
            "Confusion matrix: "
            f"TN={true_negative} "
            f"FP={false_positive} "
            f"FN={false_negative} "
            f"TP={true_positive}"
        )

    joint_accuracy = np.mean(
        np.all(predictions == test_targets, axis=1)
    ) * 100

    print()
    print(f"Average output accuracy: {np.mean(output_accuracies):.2f}%")
    print(f"All four outputs correct: {joint_accuracy:.2f}%")

    return predictions, output_accuracies, joint_accuracy


# ============================================================
# 11. DISPLAY LEARNED FEATURE WEIGHTS
# ============================================================

def display_relationship_weights(model, relationship_descriptions):
    """
    Display the strongest learned relationship weights
    for each output.

    This helps us see whether the model has learned to use
    the relationship associated with each Boolean operation.
    """

    weights = model["weights"]

    output_names = [
        "AND",
        "OR",
        "XOR",
        "NAND"
    ]

    number_of_carriers = NUMBER_OF_CARRIERS

    print()
    print("=" * 68)
    print("LEARNED PAIRWISE RELATIONSHIP WEIGHTS")
    print("=" * 68)

    for output_index, output_name in enumerate(output_names):

        relationship_weights = weights[
            number_of_carriers + 1:,
            output_index
        ]

        strongest_indices = np.argsort(
            np.abs(relationship_weights)
        )[::-1][:3]

        print()
        print(f"{output_name}:")

        for relationship_index in strongest_indices:

            weight = relationship_weights[relationship_index]

            description = relationship_descriptions[
                relationship_index
            ]

            print(f"  {description}: {weight:+.6f}")


# ============================================================
# 12. MAIN EXPERIMENT
# ============================================================

def main():

    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 68)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 10: BOOLEAN LEARNING")
    print("=" * 68)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print("EXPERIMENT CONFIGURATION")
    print("-" * 68)

    print(f"Carrier frequencies: {FREQUENCIES.tolist()}")
    print(f"Number of carriers: {NUMBER_OF_CARRIERS}")

    number_of_relationships = (
        NUMBER_OF_CARRIERS
        * (NUMBER_OF_CARRIERS - 1)
        // 2
    )

    print(f"Pairwise relationships: {number_of_relationships}")

    print("Boolean operations: AND, OR, XOR, NAND")
    print(f"Examples per input combination: {EXAMPLES_PER_COMBINATION}")

    # --------------------------------------------------------
    # Generate every possible binary input combination
    # --------------------------------------------------------

    binary_combinations = generate_binary_combinations()

    number_of_combinations = len(binary_combinations)

    print()
    print("DATASET GENERATION")
    print("-" * 68)

    print(f"Possible binary combinations: {number_of_combinations}")

    repeated_inputs = np.repeat(
        binary_combinations,
        EXAMPLES_PER_COMBINATION,
        axis=0
    )

    combination_ids = np.repeat(
        np.arange(number_of_combinations),
        EXAMPLES_PER_COMBINATION
    )

    number_of_examples = len(repeated_inputs)

    print(f"Total examples: {number_of_examples}")

    # --------------------------------------------------------
    # Calculate target Boolean outputs
    # --------------------------------------------------------

    targets = calculate_targets(repeated_inputs)

    print()
    print("TARGET OPERATIONS")
    print("-" * 68)

    print("Output 1: AND(A, B)")
    print("Output 2: OR(C, D)")
    print("Output 3: XOR(E, F)")
    print("Output 4: NAND(G, H)")

    # --------------------------------------------------------
    # Generate amplitude representations
    # --------------------------------------------------------

    amplitudes = generate_carrier_amplitudes(
        repeated_inputs,
        rng
    )

    relationship_features, relationship_descriptions = (
        generate_pairwise_features(amplitudes)
    )

    original_features, enhanced_features = build_representations(
        amplitudes,
        relationship_features
    )

    print()
    print("FEATURE REPRESENTATIONS")
    print("-" * 68)

    print(f"Original carrier features: {original_features.shape[1]}")

    print(
        "Pairwise relationship features: "
        f"{relationship_features.shape[1]}"
    )

    print(f"Enhanced total features: {enhanced_features.shape[1]}")

    # --------------------------------------------------------
    # Split the dataset
    # --------------------------------------------------------

    (
        original_train,
        original_test,
        train_targets,
        test_targets
    ) = split_dataset(
        original_features,
        targets,
        combination_ids,
        rng
    )

    (
        enhanced_train,
        enhanced_test,
        _,
        _
    ) = split_dataset(
        enhanced_features,
        targets,
        combination_ids,
        np.random.default_rng(RANDOM_SEED)
    )

    print()
    print("TRAINING AND TEST SPLIT")
    print("-" * 68)

    print(f"Training examples: {len(train_targets)}")
    print(f"Testing examples: {len(test_targets)}")

    # --------------------------------------------------------
    # Train original carrier model
    # --------------------------------------------------------

    original_model = train_linear_classifier(
        original_train,
        train_targets
    )

    # --------------------------------------------------------
    # Train pairwise-enhanced model
    # --------------------------------------------------------

    enhanced_model = train_linear_classifier(
        enhanced_train,
        train_targets
    )

    # --------------------------------------------------------
    # Evaluate both models
    # --------------------------------------------------------

    original_results = evaluate_model(
        "MODEL 1: ORIGINAL CARRIER FEATURES",
        original_model,
        original_test,
        test_targets
    )

    enhanced_results = evaluate_model(
        "MODEL 2: PAIRWISE-ENHANCED FEATURES",
        enhanced_model,
        enhanced_test,
        test_targets
    )

    # --------------------------------------------------------
    # Inspect learned relationship weights
    # --------------------------------------------------------

    display_relationship_weights(
        enhanced_model,
        relationship_descriptions
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 68)
    print("FRC 10: FINAL SUMMARY")
    print("=" * 68)

    print(
        "Original average accuracy: "
        f"{np.mean(original_results[1]):.2f}%"
    )

    print(
        "Enhanced average accuracy: "
        f"{np.mean(enhanced_results[1]):.2f}%"
    )

    print(
        "Original joint accuracy: "
        f"{original_results[2]:.2f}%"
    )

    print(
        "Enhanced joint accuracy: "
        f"{enhanced_results[2]:.2f}%"
    )

    print()
    print("Experiment completed.")


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":
    main()
