# FRC 06 — State Cancellation

## Purpose

Investigate whether subtracting consecutive squared growth states recovers the new pairwise frequency relationships introduced at each stage.

## Implementation

The experiment loads the eight saved growth states from FRC 05 and calculates:

\[
R_k=G_k^2-G_{k-1}^2
\]

Using the expansion:

\[
R_k=2G_{k-1}s_k+s_k^2
\]

The resulting residual is analysed for the new pairwise difference frequencies.

## Results

- All 28 pairwise relationships recovered.
- Each new relationship measured at amplitude 0.0064.
- All 28 relationships accumulated by G8.
- Mathematical expansion error: zero at displayed precision.
- Previous difference-frequency leakage: zero at displayed precision.

## What this demonstrates

Squared-state subtraction can isolate the incremental relationship contributions associated with each newly introduced frequency.

Each new frequency forms a pairwise relationship with every previously introduced frequency.

This preserves the incremental structure of relationship formation rather than relying exclusively on the final combined representation.

## Research status

Successful digital verification under controlled conditions. Robustness to noise, imperfect cancellation and physical implementation remains untested.

## Next step

Develop an ordered relationship representation that records when each relationship was introduced, preparing the representation for learning and order-dependent tasks.