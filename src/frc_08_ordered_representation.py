
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 08: ORDERED REPRESENTATION

PURPOSE
-------
Test whether preserving the introduction order of pairwise
relationships gives a linear classifier additional capability.

TASK
----
Determine whether relationship A-B was introduced before
relationship C-D.

REPRESENTATIONS
---------------
1. Original carrier amplitudes.
2. Unordered pairwise relationship amplitudes.
3. Ordered representation:
   - Original carrier amplitudes.
   - Pairwise relationship amplitudes.
   - Relationship introduction positions.

DATASET
-------
All 40,320 possible introduction orders of eight frequencies.

CLASSIFIER
----------
The same least-squares linear classifier is used for all
three representations.

SCIENTIFIC NOTE
---------------
Relationship amplitudes are calculated using amplitude products,
which are mathematically equivalent to the pairwise cross-term
amplitudes established in earlier FRC experiments.

Introduction positions are explicit growth-history information.
They are not inferred from the final unordered relationship set.
"""

import csv
import itertools
from pathlib import Path

import numpy as np


# ============================================================
# 1. EXPERIMENT CONFIGURATION
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

NUMBER_OF_CARRIERS = len(FREQUENCIES)

AMPLITUDE = 0.08
AMPLITUDE_VARIATION = 0.15

TRAINING_FRACTION = 0.70

RANDOM_SEED = 808

OUTPUT_DIR = Path("output")

# Relationships used by the classification task.

TARGET_PAIR_1 = (12000, 11900)
TARGET_PAIR_2 = (11770, 11610)


# ============================================================
# 2. GENERATE ALL PAIRWISE RELATIONSHIPS
# ============================================================

RELATIONSHIP_PAIRS = [
    (FREQUENCIES[i], FREQUENCIES[j])
    for i in range(NUMBER_OF_CARRIERS)
    for j in range(i + 1, NUMBER_OF_CARRIERS)
]

NUMBER_OF_RELATIONSHIPS = len(RELATIONSHIP_PAIRS)

# Always store relationship keys in ascending frequency order.
#
# This ensures that (12000, 11900) and (11900, 12000)
# are treated as the same relationship.

RELATIONSHIP_INDEX = {
    tuple(sorted(pair)): index
    for index, pair in enumerate(RELATIONSHIP_PAIRS)
}

# Find the two target relationship indices.

TARGET_INDEX_1 = RELATIONSHIP_INDEX[
    tuple(sorted(TARGET_PAIR_1))
]

TARGET_INDEX_2 = RELATIONSHIP_INDEX[
    tuple(sorted(TARGET_PAIR_2))
]


# ============================================================
# 3. CALCULATE RELATIONSHIP INTRODUCTION ORDER
# ============================================================

def calculate_relationship_order(frequency_order):
    """
    Calculate the order in which relationships are introduced.

    Each new frequency forms a relationship with every frequency
    that was introduced before it.

    The returned vector follows the canonical relationship order
    in RELATIONSHIP_PAIRS.

    Each value records the event position, from 1 to 28.
    """

    introduced = []

    event_positions = {}

    event_number = 0

    for new_frequency in frequency_order:

        # Form relationships with all previously introduced
        # frequencies.

        for previous_frequency in introduced:

            pair = tuple(
                sorted((new_frequency, previous_frequency))
            )

            event_number += 1

            event_positions[pair] = event_number

        introduced.append(new_frequency)

    # Convert the relationship dictionary into a consistently
    # ordered numerical feature vector.

    positions = np.zeros(
        NUMBER_OF_RELATIONSHIPS,
        dtype=float
    )

    for index, pair in enumerate(RELATIONSHIP_PAIRS):

        # IMPORTANT: sort the pair when retrieving its position.
        positions[index] = event_positions[tuple(sorted(pair))]

    return positions


# ============================================================
# 4. GENERATE ONE EXPERIMENTAL EXAMPLE
# ============================================================

def generate_example(frequency_order, rng):
    """
    Generate the three representations for one introduction order.
    """

    # --------------------------------------------------------
    # Carrier amplitudes
    # --------------------------------------------------------

    # Generate independent amplitude variation for each carrier.

    carrier_amplitudes = rng.uniform(
        AMPLITUDE * (1 - AMPLITUDE_VARIATION),
        AMPLITUDE * (1 + AMPLITUDE_VARIATION),
        NUMBER_OF_CARRIERS
    )

    # Associate each frequency with its amplitude.

    amplitude_lookup = {
        frequency: carrier_amplitudes[index]
        for index, frequency in enumerate(FREQUENCIES)
    }

    # --------------------------------------------------------
    # Pairwise relationship amplitudes
    # --------------------------------------------------------

    # The square-law cross-term amplitude is proportional
    # to the product of the two carrier amplitudes.

    relationship_amplitudes = np.zeros(
        NUMBER_OF_RELATIONSHIPS,
        dtype=float
    )

    for index, pair in enumerate(RELATIONSHIP_PAIRS):

        frequency_a, frequency_b = pair

        amplitude_a = amplitude_lookup[frequency_a]
        amplitude_b = amplitude_lookup[frequency_b]

        relationship_amplitudes[index] = (
            amplitude_a * amplitude_b
        )

    # --------------------------------------------------------
    # Relationship introduction positions
    # --------------------------------------------------------

    relationship_positions = calculate_relationship_order(
        frequency_order
    )

    # --------------------------------------------------------
    # Build the three representations
    # --------------------------------------------------------

    # Representation 1: original carrier amplitudes.

    original_features = carrier_amplitudes.copy()

    # Representation 2: unordered pairwise amplitudes.

    unordered_features = relationship_amplitudes.copy()

    # Representation 3: carrier amplitudes, relationship
    # amplitudes, and their introduction positions.

    ordered_features = np.concatenate([
        carrier_amplitudes,
        relationship_amplitudes,
        relationship_positions
    ])

    # --------------------------------------------------------
    # Classification target
    # --------------------------------------------------------

    position_1 = relationship_positions[TARGET_INDEX_1]
    position_2 = relationship_positions[TARGET_INDEX_2]

    # Class 1: A-B was introduced before C-D.
    # Class 0: C-D was introduced before A-B.

    label = int(position_1 < position_2)

    return (
        original_features,
        unordered_features,
        ordered_features,
        label
    )


# ============================================================
# 5. BUILD THE COMPLETE DATASET
# ============================================================

def build_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    original_data = []
    unordered_data = []
    ordered_data = []
    labels = []

    # Generate all possible introduction orders.

    all_orders = itertools.permutations(FREQUENCIES)

    for frequency_order in all_orders:

        (
            original_features,
            unordered_features,
            ordered_features,
            label
        ) = generate_example(frequency_order, rng)

        original_data.append(original_features)
        unordered_data.append(unordered_features)
        ordered_data.append(ordered_features)

        labels.append(label)

    return (
        np.asarray(original_data),
        np.asarray(unordered_data),
        np.asarray(ordered_data),
        np.asarray(labels, dtype=int)
    )


# ============================================================
# 6. CREATE A BALANCED TRAINING AND TESTING SPLIT
# ============================================================

def split_dataset(labels):

    rng = np.random.default_rng(RANDOM_SEED + 1)

    class_zero = np.where(labels == 0)[0]
    class_one = np.where(labels == 1)[0]

    rng.shuffle(class_zero)
    rng.shuffle(class_one)

    split_zero = int(
        len(class_zero) * TRAINING_FRACTION
    )

    split_one = int(
        len(class_one) * TRAINING_FRACTION
    )

    train_indices = np.concatenate([
        class_zero[:split_zero],
        class_one[:split_one]
    ])

    test_indices = np.concatenate([
        class_zero[split_zero:],
        class_one[split_one:]
    ])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


# ============================================================
# 7. LINEAR LEAST-SQUARES CLASSIFIER
# ============================================================

class LinearRelationshipModel:

    def __init__(self):

        self.mean = None
        self.scale = None
        self.weights = None

    def fit(self, features, labels):

        # Calculate feature normalisation using training data only.

        self.mean = np.mean(features, axis=0)

        self.scale = np.std(features, axis=0)

        # Prevent division by zero for constant features.

        self.scale[self.scale < 1e-12] = 1.0

        normalised = (
            features - self.mean
        ) / self.scale

        # Add a bias column.

        design_matrix = np.column_stack([
            np.ones(len(normalised)),
            normalised
        ])

        # Learn the linear weights.

        self.weights, _, _, _ = np.linalg.lstsq(
            design_matrix,
            labels.astype(float),
            rcond=None
        )

    def predict(self, features):

        normalised = (
            features - self.mean
        ) / self.scale

        design_matrix = np.column_stack([
            np.ones(len(normalised)),
            normalised
        ])

        scores = design_matrix @ self.weights

        return (scores >= 0.5).astype(int)


# ============================================================
# 8. EVALUATE ONE REPRESENTATION
# ============================================================

def evaluate_model(
    name,
    features,
    labels,
    train_indices,
    test_indices
):

    model = LinearRelationshipModel()

    model.fit(
        features[train_indices],
        labels[train_indices]
    )

    predictions = model.predict(
        features[test_indices]
    )

    actual = labels[test_indices]

    accuracy = np.mean(predictions == actual) * 100

    true_negative = np.sum(
        (actual == 0) & (predictions == 0)
    )

    false_positive = np.sum(
        (actual == 0) & (predictions == 1)
    )

    false_negative = np.sum(
        (actual == 1) & (predictions == 0)
    )

    true_positive = np.sum(
        (actual == 1) & (predictions == 1)
    )

    print(f"\n{name}")
    print("-" * 65)

    print(f"Feature count: {features.shape[1]}")
    print(f"Accuracy:      {accuracy:.2f}%")

    print("\nConfusion matrix:")

    print(f"True negatives:  {true_negative}")
    print(f"False positives: {false_positive}")
    print(f"False negatives: {false_negative}")
    print(f"True positives:  {true_positive}")

    return {
        "representation": name,
        "feature_count": features.shape[1],
        "accuracy": accuracy,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive
    }


# ============================================================
# 9. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 78)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 08: ORDERED REPRESENTATION")
    print("=" * 78)

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 78)

    print(f"Carrier frequencies: {FREQUENCIES}")

    print(f"Number of carriers: {NUMBER_OF_CARRIERS}")

    print(
        f"Number of pairwise relationships: "
        f"{NUMBER_OF_RELATIONSHIPS}"
    )

    print(
        f"Possible introduction orders: "
        f"{NUMBER_OF_CARRIERS}! = "
        f"{len(list(itertools.permutations(FREQUENCIES))):,}"
    )

    print("\nORDER-DEPENDENT TASK")
    print("-" * 78)

    print(f"Relationship A-B: {TARGET_PAIR_1}")
    print(f"Relationship C-D: {TARGET_PAIR_2}")

    print("Predict whether A-B was introduced before C-D.")

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    print("\nGENERATING DATASET")
    print("-" * 78)

    (
        original_data,
        unordered_data,
        ordered_data,
        labels
    ) = build_dataset()

    print(f"Total examples: {len(labels):,}")

    print(
        f"Class 0 (C-D introduced first): "
        f"{np.sum(labels == 0):,}"
    )

    print(
        f"Class 1 (A-B introduced first): "
        f"{np.sum(labels == 1):,}"
    )

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    train_indices, test_indices = split_dataset(labels)

    print("\nDATASET SPLIT")
    print("-" * 78)

    print(f"Training examples: {len(train_indices):,}")
    print(f"Testing examples:  {len(test_indices):,}")

    # --------------------------------------------------------
    # Evaluate representations
    # --------------------------------------------------------

    print("\nCLASSIFICATION RESULTS")
    print("=" * 78)

    results = []

    results.append(
        evaluate_model(
            "1. ORIGINAL CARRIER AMPLITUDES",
            original_data,
            labels,
            train_indices,
            test_indices
        )
    )

    results.append(
        evaluate_model(
            "2. UNORDERED PAIRWISE RELATIONSHIPS",
            unordered_data,
            labels,
            train_indices,
            test_indices
        )
    )

    results.append(
        evaluate_model(
            "3. ORDERED PAIRWISE REPRESENTATION",
            ordered_data,
            labels,
            train_indices,
            test_indices
        )
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("EXPERIMENT SUMMARY")
    print("=" * 78)

    print(
        f"{'Representation':<43}"
        f"{'Features':>12}"
        f"{'Accuracy':>15}"
    )

    print("-" * 78)

    for result in results:

        print(
            f"{result['representation']:<43}"
            f"{result['feature_count']:>12}"
            f"{result['accuracy']:>14.2f}%"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(exist_ok=True)

    results_path = OUTPUT_DIR / "frc_08_results.csv"

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nOUTPUT")
    print("-" * 78)

    print(f"Results saved to: {results_path}")

    print("\nFRC 08 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
