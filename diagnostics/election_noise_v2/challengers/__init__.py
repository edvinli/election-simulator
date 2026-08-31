"""Preregistered ElectionNoise v2 challenger implementations (Part 4).

Challenger A — variance-corrected smoothed empirical bootstrap (§C).
Challenger B — Ledoit-Wolf-regularized joint Gaussian residual model (§C).

Nothing in this package is imported by the frozen evaluator, and nothing here
modifies it. Both challengers feed the unchanged production
``apply_batch_simplex_transfer``.
"""
