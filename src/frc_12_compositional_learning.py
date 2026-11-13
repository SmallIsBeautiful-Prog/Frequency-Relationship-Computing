
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 12: COMPOSITIONAL LEARNING

Purpose:
    Investigate whether higher-order relationships can be
    constructed by combining previously established features.

Core idea:
    Pairwise relationships can become inputs to further
    relationship construction.

    R_AB = A * B

    R_ABC = R_AB * C

    R_ABCD = R_ABC * D

Tasks:
    Output 1: A XOR B XOR C
    Output 2: D XOR E XOR F
    Output 3: A XOR B XOR C XOR D

Representations:
    1. Original carrier features
    2. Pairwise-enhanced features
    3. Direct third-order features
    4. Composed third-order features
    5. Composed third- and fourth-order features

Classifier:
    Multi-output least-squares linear model.

Important:
    Direct and composed products are mathematically equivalent
    when constructed from the same underlying amplitudes.

    This experiment tests compositional construction and
    representation, not a computational efficiency advantage.
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

EXAMPLES_PER_COMBINATION = 100

TRAIN_RATIO = 0.70
RANDOM_SEED = 1201

THRESHOLD = 0.5


# ============================================================
# 2. GENERATE BINARY INPUT COMBINATIONS
# ============================================================

def generate_binary_combinations():

    return np.array(
        list(itertools.product(
            [0, 1],
            repeat=NUMBER_OF_CARRIERS
        )),
        dtype=int
    )


# ============================================================
# 3. DEFINE BOOLEAN TARGETS
# ============================================================

def calculate_targets(binary_inputs):

    A = binary_inputs[:, 0]
    B = binary_inputs[:, 1]
    C = binary_inputs[:, 2]
    D = binary_inputs[:, 3]

    E = binary_inputs[:, 4]
    F = binary_inputs[:, 5]

    output_1 = A ^ B ^ C

    output_2 = D ^ E ^ F

    output_3 = A ^ B ^ C ^ D

    return np.column_stack([
        output_1,
        output_2,
        output_3
    ])


# ============================================================
# 4. GENERATE CARRIER AMPLITUDES
# ============================================================

def generate_amplitudes(binary_inputs, rng):

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
# 5. GENERATE PAIRWISE RELATIONSHIPS
# ============================================================

def generate_pairwise_features(amplitudes):

    features = []
    descriptions = []
    pair_indices = []

    for i in range(NUMBER_OF_CARRIERS):

        for j in range(i + 1, NUMBER_OF_CARRIERS):

            product = (
                amplitudes[:, i]
                * amplitudes[:, j]
            )

            features.append(product)

            descriptions.append(
                f"{FREQUENCIES[i]}-{FREQUENCIES[j]}"
            )

            pair_indices.append((i, j))

    return (
        np.column_stack(features),
        descriptions,
        pair_indices
    )


# ============================================================
# 6. GENERATE DIRECT THIRD-ORDER RELATIONSHIPS
# ============================================================

def generate_direct_third_order(amplitudes):

    features = []
    descriptions = []
    triplet_indices = []

    for i, j, k in itertools.combinations(
        range(NUMBER_OF_CARRIERS), 3
    ):

        product = (
            amplitudes[:, i]
            * amplitudes[:, j]
            * amplitudes[:, k]
        )

        features.append(product)

        descriptions.append(
            f"{FREQUENCIES[i]}-"
            f"{FREQUENCIES[j]}-"
            f"{FREQUENCIES[k]}"
        )

        triplet_indices.append((i, j, k))

    return (
        np.column_stack(features),
        descriptions,
        triplet_indices
    )


# ============================================================
# 7. COMPOSE THIRD-ORDER RELATIONSHIPS
# ============================================================

def generate_composed_third_order(
    amplitudes,
    pairwise_features,
    pair_indices
):
    """
    Construct three-way relationships by combining an existing
    pairwise feature with a third carrier.

    Example:

        R_AB = A * B

        R_ABC = R_AB * C

    Every unique three-carrier combination is constructed once.

    This is a staged construction:
        1. Retrieve an existing pairwise relationship.
        2. Combine it with the remaining carrier.
        3. Store the resulting three-way relationship.
    """

    features = []
    descriptions = []

    pair_lookup = {
        pair: index
        for index, pair in enumerate(pair_indices)
    }

    for i, j, k in itertools.combinations(
        range(NUMBER_OF_CARRIERS), 3
    ):

        pair_index = pair_lookup[(i, j)]

        existing_relationship = pairwise_features[:, pair_index]

        new_relationship = (
            existing_relationship
            * amplitudes[:, k]
        )

        features.append(new_relationship)

        descriptions.append(
            f"({FREQUENCIES[i]}-{FREQUENCIES[j]})"
            f" x {FREQUENCIES[k]}"
        )

    return np.column_stack(features), descriptions


# ============================================================
# 8. COMPOSE FOURTH-ORDER RELATIONSHIPS
# ============================================================

def generate_composed_fourth_order(
    amplitudes,
    direct_third_features,
    triplet_indices
):
    """
    Construct fourth-order relationships by combining an
    existing three-way relationship with a fourth carrier.

    Example:

        R_ABC = A * B * C

        R_ABCD = R_ABC * D

    Each unique four-carrier combination is constructed once.
    """

    features = []
    descriptions = []

    triplet_lookup = {
        triplet: index
        for index, triplet in enumerate(triplet_indices)
    }

    for i, j, k, l in itertools.combinations(
        range(NUMBER_OF_CARRIERS), 4
    ):

        triplet_index = triplet_lookup[(i, j, k)]

        existing_relationship = (
            direct_third_features[:, triplet_index]
        )

        new_relationship = (
            existing_relationship
            * amplitudes[:, l]
        )

        features.append(new_relationship)

        descriptions.append(
            f"({FREQUENCIES[i]}-"
            f"{FREQUENCIES[j]}-"
            f"{FREQUENCIES[k]})"
            f" x {FREQUENCIES[l]}"
        )

    return np.column_stack(features), descriptions


# ============================================================
# 9. VERIFY DIRECT AND COMPOSED THIRD-ORDER FEATURES
# ============================================================

def verify_third_order(direct, composed):

    differences = np.abs(direct - composed)

    maximum_error = np.max(differences)

    mean_error = np.mean(differences)

    print()
    print("=" * 68)
    print("THIRD-ORDER COMPOSITION VERIFICATION")
    print("=" * 68)

    print(f"Direct features: {direct.shape[1]}")
    print(f"Composed features: {composed.shape[1]}")

    print(f"Mean absolute error: {mean_error:.12f}")
    print(f"Maximum absolute error: {maximum_error:.12f}")

    passed = np.allclose(
        direct,
        composed,
        rtol=1e-10,
        atol=1e-12
    )

    print(
        "Composition verification:",
        "PASS" if passed else "FAIL"
    )

    return passed


# ============================================================
# 10. CREATE FEATURE REPRESENTATIONS
# ============================================================

def build_representations(
    amplitudes,
    pairwise,
    direct_third,
    composed_third,
    composed_fourth
):

    original = amplitudes

    pairwise_enhanced = np.column_stack([
        amplitudes,
        pairwise
    ])

    direct_third_enhanced = np.column_stack([
        amplitudes,
        pairwise,
        direct_third
    ])

    composed_third_enhanced = np.column_stack([
        amplitudes,
        pairwise,
        composed_third
    ])

    composed_fourth_enhanced = np.column_stack([
        amplitudes,
        pairwise,
        composed_third,
        composed_fourth
    ])

    return {
        "Original carriers": original,
        "Pairwise-enhanced": pairwise_enhanced,
        "Direct third-order": direct_third_enhanced,
        "Composed third-order": composed_third_enhanced,
        "Composed third and fourth-order": composed_fourth_enhanced
    }


# ============================================================
# 11. STRATIFIED TRAINING AND TEST SPLIT
# ============================================================

def split_dataset(features, targets, combination_ids, rng):

    train_indices = []
    test_indices = []

    for combination in np.unique(combination_ids):

        indices = np.where(
            combination_ids == combination
        )[0]

        indices = rng.permutation(indices)

        split_point = int(
            len(indices) * TRAIN_RATIO
        )

        train_indices.extend(
            indices[:split_point]
        )

        test_indices.extend(
            indices[split_point:]
        )

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    return (
        features[train_indices],
        features[test_indices],
        targets[train_indices],
        targets[test_indices]
    )


# ============================================================
# 12. TRAIN LINEAR CLASSIFIER
# ============================================================

def train_classifier(train_features, train_targets):

    means = np.mean(train_features, axis=0)

    standard_deviations = np.std(
        train_features,
        axis=0
    )

    standard_deviations = np.where(
        standard_deviations < 1e-12,
        1.0,
        standard_deviations
    )

    normalised = (
        train_features - means
    ) / standard_deviations

    design_matrix = np.column_stack([
        np.ones(len(normalised)),
        normalised
    ])

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
# 13. PREDICTION
# ============================================================

def predict(model, features):

    normalised = (
        features - model["means"]
    ) / model["standard_deviations"]

    design_matrix = np.column_stack([
        np.ones(len(normalised)),
        normalised
    ])

    scores = design_matrix @ model["weights"]

    return (
        scores >= THRESHOLD
    ).astype(int)


# ============================================================
# 14. EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    model,
    test_features,
    test_targets
):

    predictions = predict(
        model,
        test_features
    )

    output_names = [
        "PARITY(A, B, C)",
        "PARITY(D, E, F)",
        "PARITY(A, B, C, D)"
    ]

    accuracies = []

    print()
    print("=" * 68)
    print(name.upper())
    print("=" * 68)

    for index, output_name in enumerate(output_names):

        actual = test_targets[:, index]

        predicted = predictions[:, index]

        accuracy = np.mean(
            actual == predicted
        ) * 100

        tn = np.sum(
            (actual == 0) & (predicted == 0)
        )

        fp = np.sum(
            (actual == 0) & (predicted == 1)
        )

        fn = np.sum(
            (actual == 1) & (predicted == 0)
        )

        tp = np.sum(
            (actual == 1) & (predicted == 1)
        )

        accuracies.append(accuracy)

        print()
        print(f"{output_name}: {accuracy:.2f}%")

        print(
            f"Confusion matrix: "
            f"TN={tn} FP={fp} FN={fn} TP={tp}"
        )

    joint_accuracy = np.mean(
        np.all(
            predictions == test_targets,
            axis=1
        )
    ) * 100

    print()
    print(
        f"Average output accuracy: "
        f"{np.mean(accuracies):.2f}%"
    )

    print(
        f"All three outputs correct: "
        f"{joint_accuracy:.2f}%"
    )

    return {
        "accuracies": accuracies,
        "joint_accuracy": joint_accuracy
    }


# ============================================================
# 15. MAIN EXPERIMENT
# ============================================================

def main():

    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 68)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 12: COMPOSITIONAL LEARNING")
    print("=" * 68)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print("EXPERIMENT CONFIGURATION")
    print("-" * 68)

    print(f"Carrier frequencies: {FREQUENCIES.tolist()}")
    print(f"Number of carriers: {NUMBER_OF_CARRIERS}")

    number_of_pairs = (
        NUMBER_OF_CARRIERS
        * (NUMBER_OF_CARRIERS - 1)
        // 2
    )

    number_of_triplets = (
        NUMBER_OF_CARRIERS
        * (NUMBER_OF_CARRIERS - 1)
        * (NUMBER_OF_CARRIERS - 2)
        // 6
    )

    number_of_quadruplets = (
        NUMBER_OF_CARRIERS
        * (NUMBER_OF_CARRIERS - 1)
        * (NUMBER_OF_CARRIERS - 2)
        * (NUMBER_OF_CARRIERS - 3)
        // 24
    )

    print(f"Pairwise relationships: {number_of_pairs}")
    print(f"Three-way relationships: {number_of_triplets}")
    print(f"Four-way relationships: {number_of_quadruplets}")

    # --------------------------------------------------------
    # Generate dataset
    # --------------------------------------------------------

    binary_combinations = generate_binary_combinations()

    repeated_inputs = np.repeat(
        binary_combinations,
        EXAMPLES_PER_COMBINATION,
        axis=0
    )

    combination_ids = np.repeat(
        np.arange(len(binary_combinations)),
        EXAMPLES_PER_COMBINATION
    )

    targets = calculate_targets(
        repeated_inputs
    )

    amplitudes = generate_amplitudes(
        repeated_inputs,
        rng
    )

    print()
    print("TARGET TASKS")
    print("-" * 68)

    print("Output 1: A XOR B XOR C")
    print("Output 2: D XOR E XOR F")
    print("Output 3: A XOR B XOR C XOR D")

    print()
    print("DATASET")
    print("-" * 68)

    print(
        f"Possible binary combinations: "
        f"{len(binary_combinations)}"
    )

    print(
        f"Examples per combination: "
        f"{EXAMPLES_PER_COMBINATION}"
    )

    print(
        f"Total examples: "
        f"{len(repeated_inputs)}"
    )

    # --------------------------------------------------------
    # Build relationships
    # --------------------------------------------------------

    pairwise, pair_descriptions, pair_indices = (
        generate_pairwise_features(amplitudes)
    )

    direct_third, triplet_descriptions, triplet_indices = (
        generate_direct_third_order(amplitudes)
    )

    composed_third, composed_descriptions = (
        generate_composed_third_order(
            amplitudes,
            pairwise,
            pair_indices
        )
    )

    composed_fourth, fourth_descriptions = (
        generate_composed_fourth_order(
            amplitudes,
            direct_third,
            triplet_indices
        )
    )

    # --------------------------------------------------------
    # Verify composition
    # --------------------------------------------------------

    composition_passed = verify_third_order(
        direct_third,
        composed_third
    )

    if not composition_passed:
        raise RuntimeError(
            "Third-order composition verification failed."
        )

    # --------------------------------------------------------
    # Feature representations
    # --------------------------------------------------------

    representations = build_representations(
        amplitudes,
        pairwise,
        direct_third,
        composed_third,
        composed_fourth
    )

    print()
    print("FEATURE REPRESENTATIONS")
    print("-" * 68)

    for name, features in representations.items():

        print(
            f"{name}: {features.shape[1]} features"
        )

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    print()
    print("TRAINING AND TEST SPLIT")
    print("-" * 68)

    for name, features in representations.items():

        (
            train_features,
            test_features,
            train_targets,
            test_targets
        ) = split_dataset(
            features,
            targets,
            combination_ids,
            np.random.default_rng(RANDOM_SEED)
        )

        print(
            f"{name}: "
            f"{len(train_features)} training, "
            f"{len(test_features)} testing"
        )

    # --------------------------------------------------------
    # Train and evaluate each representation
    # --------------------------------------------------------

    results = {}

    for name, features in representations.items():

        (
            train_features,
            test_features,
            train_targets,
            test_targets
        ) = split_dataset(
            features,
            targets,
            combination_ids,
            np.random.default_rng(RANDOM_SEED)
        )

        model = train_classifier(
            train_features,
            train_targets
        )

        results[name] = evaluate_model(
            f"MODEL: {name}",
            model,
            test_features,
            test_targets
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 68)
    print("FRC 12: FINAL COMPARISON")
    print("=" * 68)

    for name, result in results.items():

        print()
        print(name)

        print(
            f"Average accuracy: "
            f"{np.mean(result['accuracies']):.2f}%"
        )

        print(
            f"Joint accuracy: "
            f"{result['joint_accuracy']:.2f}%"
        )

    print()
    print("Experiment completed.")


# ============================================================
# 16. RUN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    main()
