"""Two-dimensional voting map without centre-pull (rule 3).

Encoding: Ja = +1, Avstår = 0, Nej = -1, Frånvarande = missing (NaN). Absence
is never imputed -- imputing 0 (or the vote mean) would drag rarely present
members towards the centre of the map.

1. Map votes: analysis-set votes whose minority side is at least 2.5 % of the
   Ja+Nej ballots cast (``NEAR_UNANIMOUS_MINORITY_SHARE``), and on which every
   party has a line (a tie or an all-absent party would otherwise need
   imputation).
2. Geometry: PCA on the party x vote matrix X of party-line codes, centred per
   vote (mu_i = mean over parties of X[:, i]). With Xc = U S V^T, party scores
   are (U S)[:, :2] and the vote loadings are W = V[:, :2].
3. Members: for member m with observed (cast) map-vote set O,
       s_m = argmin_s sum_{i in O} (v_mi - mu_i - W_i . s)^2
           = (W_O^T W_O)^{-1} W_O^T (v_O - mu_O),
   using only votes the member cast. A member who always votes the party line
   lands on (near) the party's point however many votes they missed. Members
   with fewer than ``MIN_MP_MAP_VOTES`` cast map votes are left unplaced.

Signs: an SVD axis is defined only up to sign; axis k is flipped so that
``MAP_AXIS_ANCHORS[k]`` scores positive (fallback: first party in
``PARTY_ORDER`` with a non-zero score). Axes are not named left/right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import (
    MAP_AXIS_ANCHORS,
    MAP_DIMENSIONS,
    MIN_MP_MAP_VOTES,
    NEAR_UNANIMOUS_MINORITY_SHARE,
    PARTY_ORDER,
)

_ZERO = 1e-9


def is_near_unanimous(ja: int, nej: int, threshold: float = NEAR_UNANIMOUS_MINORITY_SHARE) -> bool:
    """True when the minority of Ja+Nej ballots is below ``threshold`` (or nobody voted Ja/Nej)."""
    total = ja + nej
    if total == 0:
        return True
    return min(ja, nej) / total < threshold


@dataclass(frozen=True)
class PartyMap:
    parties: tuple[str, ...]
    mu: np.ndarray  # (V,)
    loadings: np.ndarray  # W, (V, k)
    party_scores: np.ndarray  # (P, k)
    singular_values: np.ndarray  # all, descending
    explained_variance_ratio: np.ndarray  # all components

    def project(self, codes: np.ndarray, min_votes: int = MIN_MP_MAP_VOTES) -> tuple[np.ndarray | None, int, str | None]:
        return project_member(codes, self.mu, self.loadings, min_votes=min_votes)


def _orient(scores: np.ndarray, loadings: np.ndarray, parties: Sequence[str], anchors: Sequence[str]) -> None:
    for axis in range(scores.shape[1]):
        order = []
        if axis < len(anchors) and anchors[axis] in parties:
            order.append(anchors[axis])
        order += [party for party in PARTY_ORDER if party in parties and party not in order]
        order += sorted(party for party in parties if party not in order)
        for party in order:
            value = scores[list(parties).index(party), axis]
            if abs(value) > _ZERO:
                if value < 0:
                    scores[:, axis] *= -1
                    loadings[:, axis] *= -1
                break


def fit_party_map(
    party_codes: np.ndarray,
    parties: Sequence[str],
    *,
    dimensions: int = MAP_DIMENSIONS,
    anchors: Sequence[str] = MAP_AXIS_ANCHORS,
) -> PartyMap:
    """PCA of a complete party x vote code matrix (no NaN allowed)."""
    matrix = np.asarray(party_codes, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(parties):
        raise ValueError("party_codes must be parties x votes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("party matrix must be complete; drop votes with missing party lines first")
    if matrix.shape[0] <= dimensions or matrix.shape[1] < dimensions:
        raise ValueError("too few parties or votes for the requested dimensions")
    mu = matrix.mean(axis=0)
    centred = matrix - mu
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    scores = u[:, :dimensions] * s[:dimensions]
    loadings = vt[:dimensions].T.copy()
    _orient(scores, loadings, parties, anchors)
    total = float(np.sum(s**2))
    ratio = (s**2) / total if total > 0 else np.zeros_like(s)
    return PartyMap(tuple(parties), mu, loadings, scores, s, ratio)


def project_member(
    codes: np.ndarray,
    mu: np.ndarray,
    loadings: np.ndarray,
    *,
    min_votes: int = MIN_MP_MAP_VOTES,
) -> tuple[np.ndarray | None, int, str | None]:
    """Least-squares position from the member's cast votes only.

    Returns ``(position or None, n_cast, reason_if_unplaced)``.
    """
    codes = np.asarray(codes, dtype=float)
    observed = np.isfinite(codes)
    n_cast = int(observed.sum())
    if n_cast < min_votes:
        return None, n_cast, f"fewer than {min_votes} cast map votes (n={n_cast})"
    w = loadings[observed]
    target = codes[observed] - mu[observed]
    solution, _residuals, rank, _sv = np.linalg.lstsq(w, target, rcond=None)
    if rank < loadings.shape[1]:
        return None, n_cast, "cast map votes do not span the map dimensions"
    return solution, n_cast, None
