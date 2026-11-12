# FRC 07 — Incremental Relationships

## Purpose

Create an ordered representation of pairwise relationships, preserving the growth state in which each relationship was introduced.

## Implementation

The script uses the saved growth states from FRC 05 and incremental residuals from FRC 06.

Each relationship receives:
- A unique identifier and sequence position.
- Its two original carrier frequencies.
- Its difference frequency.
- Its introduction growth state.
- Its measured amplitude.

Eight cumulative snapshots are constructed to represent the relationship structure at each growth stage.

## Results

- 28 relationship records created.
- All 28 measurements passed.
- All difference frequencies unique.
- Introduction positions complete from 1 to 28.
- Cumulative relationship counts verified: 0, 1, 3, 6, 10, 15, 21, 28.
- Final snapshot contains all 28 relationships.

## What this demonstrates

FRC can maintain an ordered record of relationship formation and preserve cumulative representations of the developing frequency structure.

The representation contains both relationship identity and its recorded introduction position.

## Research status

Successful digital construction and verification. Introduction order is currently supplied by the known growth sequence rather than independently inferred by a learning system.

## Next step

Develop an explicit ordered feature representation that allows a linear classifier to distinguish relationship structures with different introduction histories.