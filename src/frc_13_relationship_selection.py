
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 13: RELATIONSHIP SELECTION

Purpose:
Investigate whether FRC can automatically identify the
relationships most useful for individual classification tasks.

The experiment compares:
    1. Original carrier features.
    2. Pairwise-enhanced features.
    3. Full first-to-fourth-order representation.
    4. Automatically selected task-specific features.

The same least-squares linear classifier is retained.

Feature selection uses training and validation data.
The final test set remains untouched until evaluation.
"""

import itertools
import numpy as np


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

FREQUENCIES = [
    12000, 11900, 11770, 11610,
    11420, 11200, 10950, 10670
]

CHANNELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

BASE_AMPLITUDE = 0.08
AMPLITUDE_VARIATION = 0.15

EXAMPLES_PER_COMBINATION = 100

TRAIN_PER_COMBINATION = 60
VALIDATION_PER_COMBINATION = 20
TEST_PER_COMBINATION = 20

RANDOM_SEED = 1301

# Minimum validation accuracy accepted during selection.
SELECTION_TARGET_ACCURACY = 99.9


# ============================================================
# DATA GENERATION
# ============================================================

def generate_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    combinations = list(
        itertools.product([0, 1], repeat=len(FREQUENCIES))
    )

    all_amplitudes = []
    all_targets = []

    train_indices = []
    validation_indices = []
    test_indices = []

    current_index = 0

    for bits in combinations:

        bits = np.array(bits, dtype=int)

        # Generate independent amplitude variations.
        variation = rng.uniform(
            1.0 - AMPLITUDE_VARIATION,
            1.0 + AMPLITUDE_VARIATION,
            size=(EXAMPLES_PER_COMBINATION, len(FREQUENCIES))
        )

        amplitudes = (
            BASE_AMPLITUDE
            * bits[np.newaxis, :]
            * variation
        )

        # Three simultaneous classification tasks.
        target_1 = bits[0] ^ bits[1] ^ bits[2]
        target_2 = bits[3] ^ bits[4] ^ bits[5]
        target_3 = bits[0] ^ bits[1] ^ bits[2] ^ bits[3]

        targets = np.tile(
            [target_1, target_2, target_3],
            (EXAMPLES_PER_COMBINATION, 1)
        )

        all_amplitudes.append(amplitudes)
        all_targets.append(targets)

        # Split repetitions of each binary combination.
        # This preserves all 256 combinations in every split.
        indices = np.arange(
            current_index,
            current_index + EXAMPLES_PER_COMBINATION
        )

        rng.shuffle(indices)

        train_indices.extend(
            indices[:TRAIN_PER_COMBINATION]
        )

        validation_indices.extend(
            indices[
                TRAIN_PER_COMBINATION:
                TRAIN_PER_COMBINATION + VALIDATION_PER_COMBINATION
            ]
        )

        test_indices.extend(
            indices[
                TRAIN_PER_COMBINATION + VALIDATION_PER_COMBINATION:
            ]
        )

        current_index += EXAMPLES_PER_COMBINATION

    amplitudes = np.vstack(all_amplitudes)
    targets = np.vstack(all_targets)

    return (
        amplitudes,
        targets,
        np.array(train_indices),
        np.array(validation_indices),
        np.array(test_indices)
    )


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def build_features(amplitudes):

    feature_columns = []
    feature_names = []

    # --------------------------------------------------------
    # FIRST ORDER: ORIGINAL CARRIERS
    # --------------------------------------------------------

    for i, name in enumerate(CHANNELS):

        feature_columns.append(amplitudes[:, i])
        feature_names.append(name)

    original_count = len(feature_columns)

    # --------------------------------------------------------
    # SECOND ORDER: PAIRWISE RELATIONSHIPS
    # --------------------------------------------------------

    for i, j in itertools.combinations(
        range(len(CHANNELS)), 2
    ):

        feature_columns.append(
            amplitudes[:, i] * amplitudes[:, j]
        )

        feature_names.append(
            f"{CHANNELS[i]}*{CHANNELS[j]}"
        )

    pairwise_count = len(feature_columns)

    # --------------------------------------------------------
    # THIRD ORDER: THREE-WAY RELATIONSHIPS
    # --------------------------------------------------------

    for i, j, k in itertools.combinations(
        range(len(CHANNELS)), 3
    ):

        feature_columns.append(
            amplitudes[:, i]
            * amplitudes[:, j]
            * amplitudes[:, k]
        )

        feature_names.append(
            f"{CHANNELS[i]}*{CHANNELS[j]}*{CHANNELS[k]}"
        )

    third_order_count = len(feature_columns)

    # --------------------------------------------------------
    # FOURTH ORDER: FOUR-WAY RELATIONSHIPS
    # --------------------------------------------------------

    for i, j, k, l in itertools.combinations(
        range(len(CHANNELS)), 4
    ):

        feature_columns.append(
            amplitudes[:, i]
            * amplitudes[:, j]
            * amplitudes[:, k]
            * amplitudes[:, l]
        )

        feature_names.append(
            f"{CHANNELS[i]}*{CHANNELS[j]}*"
            f"{CHANNELS[k]}*{CHANNELS[l]}"
        )

    features = np.column_stack(feature_columns)

    return (
        features,
        feature_names,
        original_count,
        pairwise_count,
        third_order_count
    )


# ============================================================
# STANDARDISATION
# ============================================================

def standardise_training_data(X_train):

    means = np.mean(X_train, axis=0)
    scales = np.std(X_train, axis=0)

    scales[scales < 1e-12] = 1.0

    return means, scales


def apply_standardisation(X, means, scales):

    return (X - means) / scales


# ============================================================
# LINEAR CLASSIFIER
# ============================================================

def train_classifier(X, targets):

    means, scales = standardise_training_data(X)

    X_scaled = apply_standardisation(
        X, means, scales
    )

    design = np.column_stack([
        np.ones(len(X_scaled)),
        X_scaled
    ])

    weights, _, _, _ = np.linalg.lstsq(
        design,
        targets,
        rcond=None
    )

    return weights, means, scales


def predict_classifier(X, weights, means, scales):

    X_scaled = apply_standardisation(
        X, means, scales
    )

    design = np.column_stack([
        np.ones(len(X_scaled)),
        X_scaled
    ])

    scores = design @ weights

    return (scores >= 0.5).astype(int)


# ============================================================
# EVALUATION
# ============================================================

def confusion_counts(actual, predicted):

    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    tp = np.sum((actual == 1) & (predicted == 1))

    return tn, fp, fn, tp


def evaluate_model(name, predictions, targets):

    accuracies = []

    print("\n" + "=" * 68)
    print(f"MODEL: {name}")
    print("=" * 68)

    output_names = [
        "PARITY(A, B, C)",
        "PARITY(D, E, F)",
        "PARITY(A, B, C, D)"
    ]

    for output_index, output_name in enumerate(output_names):

        actual = targets[:, output_index]
        predicted = predictions[:, output_index]

        accuracy = np.mean(actual == predicted) * 100

        tn, fp, fn, tp = confusion_counts(
            actual, predicted
        )

        accuracies.append(accuracy)

        print(f"\n{output_name}: {accuracy:.2f}%")

        print(
            f"Confusion matrix: "
            f"TN={tn} FP={fp} FN={fn} TP={tp}"
        )

    average_accuracy = np.mean(accuracies)

    joint_accuracy = np.mean(
        np.all(predictions == targets, axis=1)
    ) * 100

    print(f"\nAverage output accuracy: {average_accuracy:.2f}%")
    print(f"All three outputs correct: {joint_accuracy:.2f}%")

    return average_accuracy, joint_accuracy


# ============================================================
# AUTOMATIC RELATIONSHIP SELECTION
# ============================================================

def select_relationships(
    X_train,
    y_train,
    X_validation,
    y_validation,
    feature_names
):

    # First train using every available feature.
    full_weights, means, scales = train_classifier(
        X_train,
        y_train
    )

    X_validation_scaled = apply_standardisation(
        X_validation,
        means,
        scales
    )

    selected_features = []

    print("\n" + "=" * 68)
    print("AUTOMATIC RELATIONSHIP SELECTION")
    print("=" * 68)

    for output_index in range(y_train.shape[1]):

        # Rank features using absolute learned weights.
        feature_weights = full_weights[1:, output_index]

        ranking = np.argsort(
            np.abs(feature_weights)
        )[::-1]

        intercept = full_weights[0, output_index]

        selected = None

        # ----------------------------------------------------
        # Find the smallest ranked feature set that reaches
        # the validation accuracy target using full-model
        # coefficients.
        # ----------------------------------------------------

        for count in range(1, len(ranking) + 1):

            candidate = ranking[:count]

            scores = (
                intercept
                + X_validation_scaled[:, candidate]
                @ feature_weights[candidate]
            )

            predictions = (scores >= 0.5).astype(int)

            accuracy = np.mean(
                predictions == y_validation[:, output_index]
            ) * 100

            if accuracy >= SELECTION_TARGET_ACCURACY:

                selected = list(candidate)
                break

        if selected is None:
            selected = list(ranking)

        # ----------------------------------------------------
        # Refit using only selected features.
        #
        # If the refitted model does not meet the validation
        # target, add the next ranked feature and retry.
        # ----------------------------------------------------

        remaining = [
            index for index in ranking
            if index not in selected
        ]

        while True:

            selected_array = np.array(
                selected, dtype=int
            )

            weights, model_means, model_scales = train_classifier(
                X_train[:, selected_array],
                y_train[:, output_index:output_index + 1]
            )

            validation_predictions = predict_classifier(
                X_validation[:, selected_array],
                weights,
                model_means,
                model_scales
            )

            validation_accuracy = np.mean(
                validation_predictions[:, 0]
                == y_validation[:, output_index]
            ) * 100

            if (
                validation_accuracy >= SELECTION_TARGET_ACCURACY
                or not remaining
            ):
                break

            selected.append(remaining.pop(0))

        selected_features.append(selected)

        print(f"\nOUTPUT {output_index + 1}")

        print(f"Selected features: {len(selected)}")

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.2f}%"
        )

        print("Selected relationships:")

        for index in selected:
            print(f"  {feature_names[index]}")

    return selected_features


# ============================================================
# FINAL TASK-SPECIFIC MODELS
# ============================================================

def train_selected_models(
    X_train,
    y_train,
    X_test,
    selected_features
):

    predictions = np.zeros(
        (len(X_test), y_train.shape[1]),
        dtype=int
    )

    for output_index, selected in enumerate(selected_features):

        selected_array = np.array(
            selected, dtype=int
        )

        weights, means, scales = train_classifier(
            X_train[:, selected_array],
            y_train[:, output_index:output_index + 1]
        )

        predictions[:, output_index] = predict_classifier(
            X_test[:, selected_array],
            weights,
            means,
            scales
        )[:, 0]

    return predictions


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 68)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 13: RELATIONSHIP SELECTION")
    print("=" * 68)

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 68)

    print(f"Carrier frequencies: {FREQUENCIES}")
    print(f"Number of carriers: {len(FREQUENCIES)}")

    print("\nTARGET TASKS")
    print("-" * 68)

    print("Output 1: A XOR B XOR C")
    print("Output 2: D XOR E XOR F")
    print("Output 3: A XOR B XOR C XOR D")

    # --------------------------------------------------------
    # Generate data
    # --------------------------------------------------------

    (
        amplitudes,
        targets,
        train_indices,
        validation_indices,
        test_indices
    ) = generate_dataset()

    print("\nDATASET")
    print("-" * 68)

    print(f"Total examples: {len(amplitudes)}")
    print(f"Training examples: {len(train_indices)}")
    print(f"Validation examples: {len(validation_indices)}")
    print(f"Testing examples: {len(test_indices)}")

    # --------------------------------------------------------
    # Construct all features
    # --------------------------------------------------------

    (
        features,
        feature_names,
        original_count,
        pairwise_count,
        third_order_count
    ) = build_features(amplitudes)

    print("\nFEATURE REPRESENTATIONS")
    print("-" * 68)

    print(f"Original carriers: {original_count}")
    print(f"Pairwise-enhanced: {pairwise_count}")
    print(f"Third-order: {third_order_count}")
    print(f"Full fourth-order: {features.shape[1]}")

    X_train = features[train_indices]
    X_validation = features[validation_indices]
    X_test = features[test_indices]

    y_train = targets[train_indices]
    y_validation = targets[validation_indices]
    y_test = targets[test_indices]

    # --------------------------------------------------------
    # Baseline: original carriers
    # --------------------------------------------------------

    original_weights, original_means, original_scales = (
        train_classifier(
            X_train[:, :original_count],
            y_train
        )
    )

    original_predictions = predict_classifier(
        X_test[:, :original_count],
        original_weights,
        original_means,
        original_scales
    )

    original_results = evaluate_model(
        "ORIGINAL CARRIERS",
        original_predictions,
        y_test
    )

    # --------------------------------------------------------
    # Pairwise representation
    # --------------------------------------------------------

    pairwise_weights, pairwise_means, pairwise_scales = (
        train_classifier(
            X_train[:, :pairwise_count],
            y_train
        )
    )

    pairwise_predictions = predict_classifier(
        X_test[:, :pairwise_count],
        pairwise_weights,
        pairwise_means,
        pairwise_scales
    )

    pairwise_results = evaluate_model(
        "PAIRWISE-ENHANCED",
        pairwise_predictions,
        y_test
    )

    # --------------------------------------------------------
    # Full first-to-fourth-order representation
    # --------------------------------------------------------

    full_weights, full_means, full_scales = train_classifier(
        X_train,
        y_train
    )

    full_predictions = predict_classifier(
        X_test,
        full_weights,
        full_means,
        full_scales
    )

    full_results = evaluate_model(
        "FULL FOURTH-ORDER REPRESENTATION",
        full_predictions,
        y_test
    )

    # --------------------------------------------------------
    # Select task-specific relationships
    # --------------------------------------------------------

    selected_features = select_relationships(
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_names
    )

    # --------------------------------------------------------
    # Calculate overall feature reduction
    # --------------------------------------------------------

    union_features = sorted(
        set(
            index
            for selection in selected_features
            for index in selection
        )
    )

    print("\n" + "=" * 68)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 68)

    for index, selection in enumerate(selected_features):

        print(
            f"Output {index + 1}: "
            f"{len(selection)} of {features.shape[1]} features"
        )

    print(f"\nUnique features required across all tasks: {len(union_features)}")

    reduction = (
        1.0 - len(union_features) / features.shape[1]
    ) * 100

    print(f"Overall feature reduction: {reduction:.2f}%")

    # --------------------------------------------------------
    # Final training using training + validation data.
    # The test data has not been used for feature selection.
    # --------------------------------------------------------

    final_train_indices = np.concatenate([
        train_indices,
        validation_indices
    ])

    final_X_train = features[final_train_indices]
    final_y_train = targets[final_train_indices]

    selected_predictions = train_selected_models(
        final_X_train,
        final_y_train,
        X_test,
        selected_features
    )

    selected_results = evaluate_model(
        "AUTOMATICALLY SELECTED RELATIONSHIPS",
        selected_predictions,
        y_test
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("FRC 13: FINAL COMPARISON")
    print("=" * 68)

    results = [
        ("Original carriers", original_results),
        ("Pairwise-enhanced", pairwise_results),
        ("Full fourth-order", full_results),
        ("Selected relationships", selected_results)
    ]

    for name, result in results:

        print(f"\n{name}")
        print(f"Average accuracy: {result[0]:.2f}%")
        print(f"Joint accuracy: {result[1]:.2f}%")

    print("\n" + "=" * 68)
    print("FRC 13: EXPERIMENT COMPLETED")
    print("=" * 68)


if __name__ == "__main__":
    main()
