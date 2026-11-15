
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 14: ADAPTIVE REPRESENTATION

Purpose:
Investigate whether FRC can progressively increase the
complexity of its feature representation when necessary.

The system tests:
    1. Original carrier features.
    2. Pairwise relationships.
    3. Third-order relationships.
    4. Fourth-order relationships.

Each task independently selects the lowest representation
complexity that reaches the validation accuracy target.

The same least-squares linear classifier is retained.

The final test dataset is not used for adaptive selection.
"""

import itertools
import csv
import os
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

RANDOM_SEED = 1401

# The system stops expanding a task when validation
# accuracy reaches this threshold.
SELECTION_TARGET_ACCURACY = 99.9

OUTPUT_DIRECTORY = "output"


# ============================================================
# TASK DEFINITIONS
# ============================================================

TASK_NAMES = [
    "PARITY(A, B)",
    "PARITY(C, D, E)",
    "PARITY(E, F, G, H)"
]


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

        # Three tasks with different parity complexity.
        target_1 = bits[0] ^ bits[1]

        target_2 = bits[2] ^ bits[3] ^ bits[4]

        target_3 = (
            bits[4] ^ bits[5] ^ bits[6] ^ bits[7]
        )

        targets = np.tile(
            [target_1, target_2, target_3],
            (EXAMPLES_PER_COMBINATION, 1)
        )

        all_amplitudes.append(amplitudes)
        all_targets.append(targets)

        # Split repetitions of each binary combination.
        # Every combination appears in all three datasets,
        # but with independent amplitude variations.

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
# PROGRESSIVE FEATURE CONSTRUCTION
# ============================================================

def build_features(amplitudes):

    feature_columns = []
    feature_names = []
    order_endpoints = {}

    # --------------------------------------------------------
    # FIRST ORDER: ORIGINAL CARRIERS
    # --------------------------------------------------------

    for i, name in enumerate(CHANNELS):

        feature_columns.append(amplitudes[:, i])
        feature_names.append(name)

    order_endpoints[1] = len(feature_columns)

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

    order_endpoints[2] = len(feature_columns)

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

    order_endpoints[3] = len(feature_columns)

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

    order_endpoints[4] = len(feature_columns)

    features = np.column_stack(feature_columns)

    return features, feature_names, order_endpoints


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
        X,
        means,
        scales
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
        X,
        means,
        scales
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


def evaluate_predictions(name, predictions, targets):

    accuracies = []

    print("\n" + "=" * 68)
    print(f"MODEL: {name}")
    print("=" * 68)

    for output_index, task_name in enumerate(TASK_NAMES):

        actual = targets[:, output_index]
        predicted = predictions[:, output_index]

        accuracy = np.mean(
            actual == predicted
        ) * 100

        tn, fp, fn, tp = confusion_counts(
            actual,
            predicted
        )

        accuracies.append(accuracy)

        print(f"\n{task_name}: {accuracy:.2f}%")

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
# ADAPTIVE REPRESENTATION SELECTION
# ============================================================

def select_representation(
    X_train,
    y_train,
    X_validation,
    y_validation,
    order_endpoints
):

    selected_orders = []
    validation_history = []

    print("\n" + "=" * 68)
    print("ADAPTIVE REPRESENTATION SELECTION")
    print("=" * 68)

    for output_index, task_name in enumerate(TASK_NAMES):

        print(f"\n{task_name}")
        print("-" * 68)

        selected_order = 4
        selected_accuracy = None

        for order in range(1, 5):

            feature_count = order_endpoints[order]

            # Train using all features up to this order.
            X_train_level = X_train[:, :feature_count]
            X_validation_level = X_validation[:, :feature_count]

            weights, means, scales = train_classifier(
                X_train_level,
                y_train[:, output_index:output_index + 1]
            )

            predictions = predict_classifier(
                X_validation_level,
                weights,
                means,
                scales
            )[:, 0]

            accuracy = np.mean(
                predictions == y_validation[:, output_index]
            ) * 100

            validation_history.append({
                "task": task_name,
                "order": order,
                "features": feature_count,
                "validation_accuracy": accuracy
            })

            print(
                f"Order {order}: "
                f"{feature_count} features, "
                f"validation accuracy {accuracy:.2f}%"
            )

            # Stop at the first representation that
            # reaches the required validation accuracy.
            if accuracy >= SELECTION_TARGET_ACCURACY:

                selected_order = order
                selected_accuracy = accuracy
                break

        if selected_accuracy is None:
            selected_accuracy = accuracy

        selected_orders.append(selected_order)

        print(
            f"\nSelected order: {selected_order}"
        )

        print(
            f"Selected feature count: "
            f"{order_endpoints[selected_order]}"
        )

        print(
            f"Selection validation accuracy: "
            f"{selected_accuracy:.2f}%"
        )

    return selected_orders, validation_history


# ============================================================
# FINAL MODEL TRAINING
# ============================================================

def train_fixed_order_models(
    X_train,
    y_train,
    X_test,
    order_endpoints,
    order
):

    feature_count = order_endpoints[order]

    predictions = np.zeros(
        (len(X_test), y_train.shape[1]),
        dtype=int
    )

    for output_index in range(y_train.shape[1]):

        weights, means, scales = train_classifier(
            X_train[:, :feature_count],
            y_train[:, output_index:output_index + 1]
        )

        predictions[:, output_index] = predict_classifier(
            X_test[:, :feature_count],
            weights,
            means,
            scales
        )[:, 0]

    return predictions


def train_adaptive_models(
    X_train,
    y_train,
    X_test,
    selected_orders,
    order_endpoints
):

    predictions = np.zeros(
        (len(X_test), y_train.shape[1]),
        dtype=int
    )

    for output_index, order in enumerate(selected_orders):

        feature_count = order_endpoints[order]

        weights, means, scales = train_classifier(
            X_train[:, :feature_count],
            y_train[:, output_index:output_index + 1]
        )

        predictions[:, output_index] = predict_classifier(
            X_test[:, :feature_count],
            weights,
            means,
            scales
        )[:, 0]

    return predictions


# ============================================================
# SAVE EXPERIMENT RESULTS
# ============================================================

def save_results(
    validation_history,
    selected_orders,
    feature_names,
    order_endpoints,
    test_accuracies
):

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    history_path = os.path.join(
        OUTPUT_DIRECTORY,
        "frc_14_adaptive_validation.csv"
    )

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "task",
            "order",
            "features",
            "validation_accuracy"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(validation_history)

    selection_path = os.path.join(
        OUTPUT_DIRECTORY,
        "frc_14_selected_features.csv"
    )

    with open(
        selection_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "task",
            "selected_order",
            "feature_count",
            "test_accuracy",
            "features"
        ])

        for i, order in enumerate(selected_orders):

            count = order_endpoints[order]

            writer.writerow([
                TASK_NAMES[i],
                order,
                count,
                test_accuracies[i],
                "; ".join(feature_names[:count])
            ])

    print("\nSaved experiment files:")
    print(f"  {history_path}")
    print(f"  {selection_path}")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 68)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 14: ADAPTIVE REPRESENTATION")
    print("=" * 68)

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 68)

    print(f"Carrier frequencies: {FREQUENCIES}")
    print(f"Number of carriers: {len(FREQUENCIES)}")

    print("\nTARGET TASKS")
    print("-" * 68)

    for task in TASK_NAMES:
        print(task)

    # --------------------------------------------------------
    # Generate dataset
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
    # Construct progressive feature representation
    # --------------------------------------------------------

    (
        features,
        feature_names,
        order_endpoints
    ) = build_features(amplitudes)

    print("\nFEATURE REPRESENTATIONS")
    print("-" * 68)

    for order, count in order_endpoints.items():

        print(
            f"Order {order}: {count} cumulative features"
        )

    # --------------------------------------------------------
    # Split datasets
    # --------------------------------------------------------

    X_train = features[train_indices]
    X_validation = features[validation_indices]
    X_test = features[test_indices]

    y_train = targets[train_indices]
    y_validation = targets[validation_indices]
    y_test = targets[test_indices]

    # --------------------------------------------------------
    # Select the required representation independently
    # for each task using validation data.
    # --------------------------------------------------------

    (
        selected_orders,
        validation_history
    ) = select_representation(
        X_train,
        y_train,
        X_validation,
        y_validation,
        order_endpoints
    )

    # --------------------------------------------------------
    # Final training uses training + validation data.
    # The test data remains untouched until evaluation.
    # --------------------------------------------------------

    final_train_indices = np.concatenate([
        train_indices,
        validation_indices
    ])

    final_X_train = features[final_train_indices]
    final_y_train = targets[final_train_indices]

    # --------------------------------------------------------
    # Evaluate fixed representations
    # --------------------------------------------------------

    fixed_results = {}

    for order in range(1, 5):

        predictions = train_fixed_order_models(
            final_X_train,
            final_y_train,
            X_test,
            order_endpoints,
            order
        )

        fixed_results[order] = evaluate_predictions(
            f"FIXED ORDER {order}",
            predictions,
            y_test
        )

    # --------------------------------------------------------
    # Evaluate adaptive representation
    # --------------------------------------------------------

    adaptive_predictions = train_adaptive_models(
        final_X_train,
        final_y_train,
        X_test,
        selected_orders,
        order_endpoints
    )

    adaptive_results = evaluate_predictions(
        "ADAPTIVE REPRESENTATION",
        adaptive_predictions,
        y_test
    )

    # --------------------------------------------------------
    # Calculate feature usage
    # --------------------------------------------------------

    adaptive_feature_connections = sum(
        order_endpoints[order]
        for order in selected_orders
    )

    full_feature_connections = (
        len(TASK_NAMES) * order_endpoints[4]
    )

    feature_reduction = (
        1.0
        - adaptive_feature_connections / full_feature_connections
    ) * 100

    print("\n" + "=" * 68)
    print("ADAPTIVE REPRESENTATION SUMMARY")
    print("=" * 68)

    for i, order in enumerate(selected_orders):

        print(
            f"\n{TASK_NAMES[i]}"
        )

        print(f"Selected order: {order}")

        print(
            f"Active features: {order_endpoints[order]}"
        )

        print(
            f"Test accuracy: "
            f"{np.mean(adaptive_predictions[:, i] == y_test[:, i]) * 100:.2f}%"
        )

    print("\nTOTAL FEATURE USAGE")

    print(
        f"Fixed fourth-order connections: "
        f"{full_feature_connections}"
    )

    print(
        f"Adaptive feature connections: "
        f"{adaptive_feature_connections}"
    )

    print(
        f"Feature reduction: {feature_reduction:.2f}%"
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("FRC 14: FINAL COMPARISON")
    print("=" * 68)

    for order in range(1, 5):

        result = fixed_results[order]

        print(f"\nFixed order {order}")
        print(f"Average accuracy: {result[0]:.2f}%")
        print(f"Joint accuracy: {result[1]:.2f}%")

    print("\nAdaptive representation")
    print(f"Average accuracy: {adaptive_results[0]:.2f}%")
    print(f"Joint accuracy: {adaptive_results[1]:.2f}%")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    test_accuracies = [
        np.mean(
            adaptive_predictions[:, i] == y_test[:, i]
        ) * 100
        for i in range(len(TASK_NAMES))
    ]

    save_results(
        validation_history,
        selected_orders,
        feature_names,
        order_endpoints,
        test_accuracies
    )

    print("\n" + "=" * 68)
    print("FRC 14: EXPERIMENT COMPLETED")
    print("=" * 68)


if __name__ == "__main__":
    main()
