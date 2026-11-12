
"""
FREQUENCY RELATIONSHIP COMPUTING
FRC 09: GROWTH LEARNING

PURPOSE
-------
Build ordered relationship representations from successive
cumulative growth snapshots, then train a linear classifier
using the recovered growth information.

EXPERIMENT
----------
1. Generate all 40,320 possible frequency introduction orders.
2. Construct eight cumulative growth snapshots for each order.
3. Record which relationships exist in each snapshot.
4. Recover the introduction state of every relationship.
5. Verify the recovered states against the known construction.
6. Train and compare three linear classifiers.

CLASSIFICATION TASK
-------------------
Determine whether relationship A-B was introduced before
relationship C-D.

REPRESENTATIONS
---------------
1. Original carrier amplitudes.
2. Unordered pairwise relationship amplitudes.
3. Growth-derived representation:
   - Original carrier amplitudes.
   - Pairwise relationship amplitudes.
   - Recovered relationship introduction states.

CLASSIFIER
----------
The same least-squares linear classifier is used throughout.

SCIENTIFIC NOTE
---------------
Growth snapshots are represented as cumulative relationship
presence matrices. Introduction states are recovered by
detecting the first snapshot in which each relationship appears.

This is a digital representation of the growth process.
It does not yet perform waveform subtraction for every
individual training example.
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

NUMBER_OF_RELATIONSHIPS = (
    NUMBER_OF_CARRIERS * (NUMBER_OF_CARRIERS - 1) // 2
)

AMPLITUDE = 0.08
AMPLITUDE_VARIATION = 0.15

TRAINING_FRACTION = 0.70

RANDOM_SEED = 909

OUTPUT_DIR = Path("output")

# Target relationships.

TARGET_PAIR_1 = (12000, 11900)
TARGET_PAIR_2 = (11770, 11610)


# ============================================================
# 2. GENERATE CANONICAL RELATIONSHIP INDEX
# ============================================================

RELATIONSHIP_PAIRS = [
    (FREQUENCIES[i], FREQUENCIES[j])
    for i in range(NUMBER_OF_CARRIERS)
    for j in range(i + 1, NUMBER_OF_CARRIERS)
]

# Sort every pair so that lookup is independent of direction.

RELATIONSHIP_INDEX = {
    tuple(sorted(pair)): index
    for index, pair in enumerate(RELATIONSHIP_PAIRS)
}

TARGET_INDEX_1 = RELATIONSHIP_INDEX[
    tuple(sorted(TARGET_PAIR_1))
]

TARGET_INDEX_2 = RELATIONSHIP_INDEX[
    tuple(sorted(TARGET_PAIR_2))
]


# ============================================================
# 3. CONSTRUCT CUMULATIVE GROWTH SNAPSHOTS
# ============================================================

def construct_growth_snapshots(frequency_order):
    """
    Construct cumulative relationship-presence snapshots.

    Each snapshot represents the relationships that exist
    after a particular frequency has been introduced.

    Snapshot 1:
        First frequency introduced.
        No pairwise relationships yet.

    Snapshot 2:
        Second frequency introduced.
        One relationship exists.

    Snapshot 3:
        Third frequency introduced.
        Three relationships exist.

    This continues until all 28 relationships exist.

    Returns:
        A matrix with shape:

            (number of growth states, number of relationships)

        A value of 1 means the relationship exists.
        A value of 0 means it has not yet appeared.
    """

    snapshots = np.zeros(
        (NUMBER_OF_CARRIERS, NUMBER_OF_RELATIONSHIPS),
        dtype=np.int8
    )

    introduced = []

    current_relationships = np.zeros(
        NUMBER_OF_RELATIONSHIPS,
        dtype=np.int8
    )

    for state_index, new_frequency in enumerate(frequency_order):

        # The new frequency forms relationships with every
        # frequency that has already been introduced.

        for previous_frequency in introduced:

            pair = tuple(
                sorted((new_frequency, previous_frequency))
            )

            relationship_index = RELATIONSHIP_INDEX[pair]

            current_relationships[relationship_index] = 1

        introduced.append(new_frequency)

        # Save a cumulative snapshot of the current state.

        snapshots[state_index] = current_relationships.copy()

    return snapshots


# ============================================================
# 4. RECOVER RELATIONSHIP INTRODUCTION STATES
# ============================================================

def recover_introduction_states(snapshots):
    """
    Recover when each relationship first appeared.

    The first snapshot containing a relationship determines
    its introduction state.

    States are numbered G1 to G8.

    Relationships that do not appear in any snapshot
    are treated as an error.
    """

    introduction_states = np.zeros(
        NUMBER_OF_RELATIONSHIPS,
        dtype=float
    )

    for relationship_index in range(NUMBER_OF_RELATIONSHIPS):

        presence = snapshots[:, relationship_index]

        appearance = np.where(presence == 1)[0]

        if len(appearance) == 0:
            raise ValueError(
                f"Relationship {relationship_index} "
                "was never introduced."
            )

        # Convert zero-based array index to growth-state number.

        introduction_states[relationship_index] = (
            appearance[0] + 1
        )

    return introduction_states


# ============================================================
# 5. VERIFY RECOVERED STATES
# ============================================================

def calculate_expected_states(frequency_order):
    """
    Calculate the expected introduction state directly
    from the known frequency introduction order.

    This is used only as a verification reference.
    """

    frequency_positions = {
        frequency: index + 1
        for index, frequency in enumerate(frequency_order)
    }

    expected_states = np.zeros(
        NUMBER_OF_RELATIONSHIPS,
        dtype=float
    )

    for index, pair in enumerate(RELATIONSHIP_PAIRS):

        position_a = frequency_positions[pair[0]]
        position_b = frequency_positions[pair[1]]

        # A relationship appears when its second frequency
        # is introduced.

        expected_states[index] = max(
            position_a,
            position_b
        )

    return expected_states


# ============================================================
# 6. GENERATE ONE EXPERIMENTAL EXAMPLE
# ============================================================

def generate_example(frequency_order, rng):
    """
    Generate carrier, relationship, and growth-derived features.
    """

    # --------------------------------------------------------
    # Generate carrier amplitudes
    # --------------------------------------------------------

    carrier_amplitudes = rng.uniform(
        AMPLITUDE * (1 - AMPLITUDE_VARIATION),
        AMPLITUDE * (1 + AMPLITUDE_VARIATION),
        NUMBER_OF_CARRIERS
    )

    amplitude_lookup = {
        frequency: carrier_amplitudes[index]
        for index, frequency in enumerate(FREQUENCIES)
    }

    # --------------------------------------------------------
    # Calculate pairwise relationship amplitudes
    # --------------------------------------------------------

    relationship_amplitudes = np.zeros(
        NUMBER_OF_RELATIONSHIPS,
        dtype=float
    )

    for index, pair in enumerate(RELATIONSHIP_PAIRS):

        amplitude_a = amplitude_lookup[pair[0]]
        amplitude_b = amplitude_lookup[pair[1]]

        relationship_amplitudes[index] = (
            amplitude_a * amplitude_b
        )

    # --------------------------------------------------------
    # Construct cumulative growth snapshots
    # --------------------------------------------------------

    snapshots = construct_growth_snapshots(
        frequency_order
    )

    # --------------------------------------------------------
    # Recover introduction states from snapshots
    # --------------------------------------------------------

    recovered_states = recover_introduction_states(
        snapshots
    )

    # --------------------------------------------------------
    # Verify against known construction order
    # --------------------------------------------------------

    expected_states = calculate_expected_states(
        frequency_order
    )

    if not np.array_equal(recovered_states, expected_states):

        raise ValueError(
            "Growth-state reconstruction failed verification."
        )

    # --------------------------------------------------------
    # Build the three feature representations
    # --------------------------------------------------------

    original_features = carrier_amplitudes.copy()

    unordered_features = relationship_amplitudes.copy()

    growth_features = np.concatenate([
        carrier_amplitudes,
        relationship_amplitudes,
        recovered_states
    ])

    # --------------------------------------------------------
    # Classification target
    # --------------------------------------------------------

    state_1 = recovered_states[TARGET_INDEX_1]
    state_2 = recovered_states[TARGET_INDEX_2]

    # Class 1: A-B appeared before C-D.
    # Class 0: C-D appeared before A-B.

    label = int(state_1 < state_2)

    return (
        original_features,
        unordered_features,
        growth_features,
        label,
        snapshots,
        recovered_states
    )


# ============================================================
# 7. BUILD COMPLETE DATASET
# ============================================================

def build_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    original_data = []
    unordered_data = []
    growth_data = []
    labels = []

    total_verified = 0

    all_orders = itertools.permutations(FREQUENCIES)

    for frequency_order in all_orders:

        (
            original_features,
            unordered_features,
            growth_features,
            label,
            snapshots,
            recovered_states
        ) = generate_example(frequency_order, rng)

        original_data.append(original_features)
        unordered_data.append(unordered_features)
        growth_data.append(growth_features)

        labels.append(label)

        total_verified += 1

    return (
        np.asarray(original_data),
        np.asarray(unordered_data),
        np.asarray(growth_data),
        np.asarray(labels, dtype=int),
        total_verified
    )


# ============================================================
# 8. CREATE BALANCED TRAINING AND TESTING SPLIT
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
# 9. LINEAR LEAST-SQUARES CLASSIFIER
# ============================================================

class LinearRelationshipModel:

    def __init__(self):

        self.mean = None
        self.scale = None
        self.weights = None

    def fit(self, features, labels):

        # Normalisation is calculated using training data only.

        self.mean = np.mean(features, axis=0)

        self.scale = np.std(features, axis=0)

        self.scale[self.scale < 1e-12] = 1.0

        normalised = (
            features - self.mean
        ) / self.scale

        # Add bias column.

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
# 10. EVALUATE ONE REPRESENTATION
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
# 11. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 78)
    print("FREQUENCY RELATIONSHIP COMPUTING")
    print("FRC 09: GROWTH LEARNING")
    print("=" * 78)

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 78)

    print(f"Carrier frequencies: {FREQUENCIES}")

    print(f"Number of carriers: {NUMBER_OF_CARRIERS}")

    print(
        f"Number of relationships: "
        f"{NUMBER_OF_RELATIONSHIPS}"
    )

    print(
        f"Possible introduction orders: "
        f"{NUMBER_OF_CARRIERS}! = "
        f"{NUMBER_OF_CARRIERS and 40320:,}"
    )

    print("\nGROWTH SNAPSHOT CONFIGURATION")
    print("-" * 78)

    print(f"Number of growth states: {NUMBER_OF_CARRIERS}")

    for state in range(1, NUMBER_OF_CARRIERS + 1):

        expected_relationships = (
            state * (state - 1) // 2
        )

        print(
            f"G{state}: "
            f"{expected_relationships} cumulative relationships"
        )

    print("\nORDER-DEPENDENT TASK")
    print("-" * 78)

    print(f"Relationship A-B: {TARGET_PAIR_1}")
    print(f"Relationship C-D: {TARGET_PAIR_2}")

    print("Predict whether A-B was introduced before C-D.")

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    print("\nGENERATING GROWTH DATASET")
    print("-" * 78)

    (
        original_data,
        unordered_data,
        growth_data,
        labels,
        total_verified
    ) = build_dataset()

    print(f"Total examples: {len(labels):,}")

    print(f"Verified growth sequences: {total_verified:,}")

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
            "3. GROWTH-DERIVED REPRESENTATION",
            growth_data,
            labels,
            train_indices,
            test_indices
        )
    )

    # --------------------------------------------------------
    # Summary
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

    results_path = OUTPUT_DIR / "frc_09_results.csv"

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

    print("\nFRC 09 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
