
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 11: HIGHER-ORDER RELATIONSHIPS

Purpose:
    Test whether three-way frequency relationships extend
    the capabilities of a linear classifier beyond pairwise
    feature expansion.

Tasks:
    Output 1: three-input parity of A, B, C
    Output 2: three-input parity of D, E, F

Representations:
    1. Original carrier amplitudes: 8 features
    2. Pairwise expansion: 8 + 28 = 36 features
    3. Third-order expansion: 8 + 28 + 56 = 92 features

One multi-output least-squares linear classifier is trained
for each representation.

All representations use exactly the same training and
testing examples.
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
RANDOM_SEED = 1101

THRESHOLD = 0.5

OUTPUT_NAMES = [
    "PARITY(A, B, C)",
    "PARITY(D, E, F)"
]


# ============================================================
# 2. GENERATE ALL BINARY INPUT COMBINATIONS
# ============================================================

def generate_binary_combinations():
    """
    Eight binary inputs produce 256 possible combinations.
    """

    return np.array(
        list(itertools.product(
            [0, 1],
            repeat=NUMBER_OF_CARRIERS
        )),
        dtype=int
    )


# ============================================================
# 3. DEFINE THE THREE-INPUT PARITY TASKS
# ============================================================

def calculate_targets(binary_inputs):
    """
    Three-input parity returns 1 when an odd number of
    the three inputs are active.

    This is equivalent to applying XOR twice:

        (A XOR B) XOR C
    """

    A = binary_inputs[:, 0]
    B = binary_inputs[:, 1]
    C = binary_inputs[:, 2]

    D = binary_inputs[:, 3]
    E = binary_inputs[:, 4]
    F = binary_inputs[:, 5]

    output_1 = A ^ B ^ C
    output_2 = D ^ E ^ F

    return np.column_stack([
        output_1,
        output_2
    ])


# ============================================================
# 4. GENERATE CARRIER AMPLITUDES
# ============================================================

def generate_carrier_amplitudes(binary_inputs, rng):
    """
    Convert binary inputs into amplitude measurements.

    Active inputs receive an amplitude of approximately 0.08,
    with independent variation of plus or minus 15%.

    Inactive inputs have zero amplitude.
    """

    variation = rng.uniform(
        1.0 - AMPLITUDE_VARIATION,
        1.0 + AMPLITUDE_VARIATION,
        size=binary_inputs.shape
    )

    return (
        binary_inputs
        * AMPLITUDE
        * variation
    )


# ============================================================
# 5. GENERATE PAIRWISE RELATIONSHIPS
# ============================================================

def generate_pairwise_features(amplitudes):
    """
    Generate all unique two-carrier products.

    For eight carriers:

        8 choose 2 = 28 pairwise relationships.
    """

    features = []
    descriptions = []

    for i, j in itertools.combinations(
        range(NUMBER_OF_CARRIERS), 2
    ):

        features.append(
            amplitudes[:, i] * amplitudes[:, j]
        )

        difference = abs(
            FREQUENCIES[i] - FREQUENCIES[j]
        )

        descriptions.append(
            f"{FREQUENCIES[i]}-{FREQUENCIES[j]} "
            f"({difference} Hz)"
        )

    return np.column_stack(features), descriptions


# ============================================================
# 6. GENERATE THREE-WAY RELATIONSHIPS
# ============================================================

def generate_triple_features(amplitudes):
    """
    Generate all unique three-carrier products.

    For eight carriers:

        8 choose 3 = 56 three-way relationships.

    Each feature represents the simultaneous product of
    three carrier amplitudes:

        A * B * C

    These are third-order polynomial interaction features.
    """

    features = []
    descriptions = []

    for i, j, k in itertools.combinations(
        range(NUMBER_OF_CARRIERS), 3
    ):

        triple_product = (
            amplitudes[:, i]
            * amplitudes[:, j]
            * amplitudes[:, k]
        )

        features.append(triple_product)

        descriptions.append(
            f"{FREQUENCIES[i]}-"
            f"{FREQUENCIES[j]}-"
            f"{FREQUENCIES[k]}"
        )

    return np.column_stack(features), descriptions


# ============================================================
# 7. BUILD ALL THREE REPRESENTATIONS
# ============================================================

def build_representations(
    amplitudes,
    pairwise_features,
    triple_features
):
    """
    Build nested representations.

    Original:
        Individual carrier amplitudes.

    Pairwise:
        Individual amplitudes plus all pairwise products.

    Third-order:
        Individual amplitudes, pairwise products and
        all three-way products.
    """

    original = amplitudes

    pairwise = np.column_stack([
        amplitudes,
        pairwise_features
    ])

    third_order = np.column_stack([
        amplitudes,
        pairwise_features,
        triple_features
    ])

    return original, pairwise, third_order


# ============================================================
# 8. CREATE ONE SHARED STRATIFIED DATASET SPLIT
# ============================================================

def create_split(combination_ids, rng):
    """
    Every binary input combination contributes:

        70 training examples
        30 testing examples

    This ensures that every possible input combination
    appears in both datasets.

    The same indices are used for all representations.
    """

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

        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    return (
        np.array(train_indices),
        np.array(test_indices)
    )


# ============================================================
# 9. TRAIN THE MULTI-OUTPUT LINEAR CLASSIFIER
# ============================================================

def train_classifier(train_features, train_targets):
    """
    Train a single multi-output least-squares model.

    Standardisation is calculated from training data only.

    A bias column is included.

    The model learns both output operations simultaneously.
    """

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

    weights, residuals, rank, singular_values = (
        np.linalg.lstsq(
            design_matrix,
            train_targets,
            rcond=None
        )
    )

    return {
        "weights": weights,
        "means": means,
        "standard_deviations": standard_deviations,
        "rank": rank
    }


# ============================================================
# 10. MAKE PREDICTIONS
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
# 11. EVALUATE A REPRESENTATION
# ============================================================

def evaluate_model(
    representation_name,
    model,
    test_features,
    test_targets
):

    predictions = predict(
        model,
        test_features
    )

    print()
    print("=" * 68)
    print(representation_name)
    print("=" * 68)

    accuracies = []

    for output_index, output_name in enumerate(OUTPUT_NAMES):

        actual = test_targets[:, output_index]
        predicted = predictions[:, output_index]

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

    average_accuracy = np.mean(accuracies)

    joint_accuracy = np.mean(
        np.all(
            predictions == test_targets,
            axis=1
        )
    ) * 100

    print()
    print(f"Average output accuracy: {average_accuracy:.2f}%")
    print(f"Both outputs correct: {joint_accuracy:.2f}%")

    return {
        "predictions": predictions,
        "accuracies": accuracies,
        "average_accuracy": average_accuracy,
        "joint_accuracy": joint_accuracy
    }


# ============================================================
# 12. DISPLAY THE LEARNED THREE-WAY WEIGHTS
# ============================================================

def display_triple_weights(
    model,
    triple_descriptions
):
    """
    Display the strongest three-way feature weights
    for each output.

    Weight indexing:
        0                  = bias
        1 to 8             = carrier features
        9 to 36            = pairwise features
        37 onwards         = triple features
    """

    weights = model["weights"]

    triple_start = (
        1
        + NUMBER_OF_CARRIERS
        + NUMBER_OF_CARRIERS * (NUMBER_OF_CARRIERS - 1) // 2
    )

    triple_weights = weights[triple_start:, :]

    print()
    print("=" * 68)
    print("STRONGEST THREE-WAY RELATIONSHIP WEIGHTS")
    print("=" * 68)

    for output_index, output_name in enumerate(OUTPUT_NAMES):

        column = triple_weights[:, output_index]

        strongest = np.argsort(
            np.abs(column)
        )[::-1][:5]

        print()
        print(output_name)

        for index in strongest:

            print(
                f"  {triple_descriptions[index]}: "
                f"{column[index]:+.6f}"
            )


# ============================================================
# 13. MAIN EXPERIMENT
# ============================================================

def main():

    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 68)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 11: HIGHER-ORDER RELATIONSHIPS")
    print("=" * 68)

    # --------------------------------------------------------
    # Experiment configuration
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

    number_of_triples = (
        NUMBER_OF_CARRIERS
        * (NUMBER_OF_CARRIERS - 1)
        * (NUMBER_OF_CARRIERS - 2)
        // 6
    )

    print(f"Pairwise relationships: {number_of_pairs}")
    print(f"Three-way relationships: {number_of_triples}")

    print()
    print("TARGET TASKS")
    print("-" * 68)

    print("Output 1: A XOR B XOR C")
    print("Output 2: D XOR E XOR F")

    # --------------------------------------------------------
    # Generate the dataset
    # --------------------------------------------------------

    binary_combinations = generate_binary_combinations()

    number_of_combinations = len(binary_combinations)

    repeated_inputs = np.repeat(
        binary_combinations,
        EXAMPLES_PER_COMBINATION,
        axis=0
    )

    combination_ids = np.repeat(
        np.arange(number_of_combinations),
        EXAMPLES_PER_COMBINATION
    )

    targets = calculate_targets(repeated_inputs)

    number_of_examples = len(repeated_inputs)

    print()
    print("DATASET")
    print("-" * 68)

    print(f"Possible binary combinations: {number_of_combinations}")
    print(f"Examples per combination: {EXAMPLES_PER_COMBINATION}")
    print(f"Total examples: {number_of_examples}")

    # --------------------------------------------------------
    # Generate amplitudes and relationship features
    # --------------------------------------------------------

    amplitudes = generate_carrier_amplitudes(
        repeated_inputs,
        rng
    )

    pairwise_features, pair_descriptions = (
        generate_pairwise_features(amplitudes)
    )

    triple_features, triple_descriptions = (
        generate_triple_features(amplitudes)
    )

    original, pairwise, third_order = build_representations(
        amplitudes,
        pairwise_features,
        triple_features
    )

    print()
    print("FEATURE REPRESENTATIONS")
    print("-" * 68)

    print(f"Original carrier features: {original.shape[1]}")

    print(f"Pairwise-enhanced features: {pairwise.shape[1]}")

    print(f"Third-order features: {third_order.shape[1]}")

    # --------------------------------------------------------
    # Create one shared train/test split
    # --------------------------------------------------------

    train_indices, test_indices = create_split(
        combination_ids,
        np.random.default_rng(RANDOM_SEED)
    )

    train_targets = targets[train_indices]
    test_targets = targets[test_indices]

    print()
    print("TRAINING AND TEST SPLIT")
    print("-" * 68)

    print(f"Training examples: {len(train_indices)}")
    print(f"Testing examples: {len(test_indices)}")

    # --------------------------------------------------------
    # Prepare representations using the shared indices
    # --------------------------------------------------------

    representations = [
        (
            "MODEL 1: ORIGINAL CARRIER FEATURES",
            original
        ),
        (
            "MODEL 2: PAIRWISE-ENHANCED FEATURES",
            pairwise
        ),
        (
            "MODEL 3: THIRD-ORDER FEATURES",
            third_order
        )
    ]

    results = {}

    # --------------------------------------------------------
    # Train and evaluate each representation
    # --------------------------------------------------------

    for name, features in representations:

        train_features = features[train_indices]
        test_features = features[test_indices]

        model = train_classifier(
            train_features,
            train_targets
        )

        result = evaluate_model(
            name,
            model,
            test_features,
            test_targets
        )

        results[name] = {
            "model": model,
            "result": result
        }

        if name.startswith("MODEL 3"):

            display_triple_weights(
                model,
                triple_descriptions
            )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 68)
    print("FRC 11: FINAL COMPARISON")
    print("=" * 68)

    for name, data in results.items():

        result = data["result"]

        print()
        print(name)

        print(
            f"Average accuracy: "
            f"{result['average_accuracy']:.2f}%"
        )

        print(
            f"Joint accuracy: "
            f"{result['joint_accuracy']:.2f}%"
        )

    print()
    print("Experiment completed.")


# ============================================================
# 14. RUN
# ============================================================

if __name__ == "__main__":
    main()
