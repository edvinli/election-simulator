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

Signs: an SVD axis is defined only up to sign. A term map flips axis k so that
``MAP_AXIS_ANCHORS[k]`` (M, V) scores positive, provided its |score| is at
least ``MAP_ANCHOR_MIN_SHARE`` of the axis's largest; otherwise the first
party in ``PARTY_ORDER`` that clears that bar. A session map is aligned with
its term's map axis by axis (``align_to_reference``). Axes are not named
left/right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import (
    MAP_ALIGNMENT_MIN_COSINE,
    MAP_ANCHOR_MIN_SHARE,
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


def _orient_axis_by_anchor(
    scores: np.ndarray, loadings: np.ndarray, parties: Sequence[str], axis: int, anchor: str | None
) -> str | None:
    """Make the anchor (or the first qualifying party) positive on ``axis``;
    returns the party used. A party qualifies when its |score| is at least
    ``MAP_ANCHOR_MIN_SHARE`` of the largest |score| on the axis."""
    column = scores[:, axis]
    bar = max(_ZERO, MAP_ANCHOR_MIN_SHARE * float(np.max(np.abs(column)))) if column.size else _ZERO
    parties = list(parties)
    order = [anchor] if anchor in parties else []
    order += [party for party in PARTY_ORDER if party in parties and party not in order]
    order += sorted(party for party in parties if party not in order)
    for party in order:
        value = column[parties.index(party)]
        if abs(value) >= bar:
            if value < 0:
                scores[:, axis] *= -1
                loadings[:, axis] *= -1
            return party
    return None


def _orient(scores: np.ndarray, loadings: np.ndarray, parties: Sequence[str], anchors: Sequence[str]) -> list[str | None]:
    return [
        _orient_axis_by_anchor(scores, loadings, parties, axis, anchors[axis] if axis < len(anchors) else None)
        for axis in range(scores.shape[1])
    ]


def orient_by_anchors(fitted: "PartyMap", anchors: Sequence[str] = MAP_AXIS_ANCHORS) -> list[dict]:
    """The term rule, applied in place (idempotent); one record per axis."""
    used = _orient(fitted.party_scores, fitted.loadings, fitted.parties, anchors)
    return [{"method": "anchor", "anchor": party} for party in used]


def align_to_reference(
    fitted: "PartyMap",
    reference: dict[str, Sequence[float]],
    *,
    anchors: Sequence[str] = MAP_AXIS_ANCHORS,
    min_cosine: float = MAP_ALIGNMENT_MIN_COSINE,
) -> list[dict]:
    """Flip the axes of ``fitted`` in place to agree with ``reference`` (party
    -> coordinates on the term map). Per axis: the sign of the dot product over
    the parties in both; if |cosine| < ``min_cosine`` (or no shared party), the
    anchor rule. Returns one record per axis."""
    shared = [party for party in fitted.parties if party in reference]
    records = []
    for axis in range(fitted.party_scores.shape[1]):
        own = np.array([fitted.party_scores[fitted.parties.index(p), axis] for p in shared], dtype=float)
        ref = np.array([float(reference[p][axis]) for p in shared], dtype=float)
        norm = float(np.linalg.norm(own) * np.linalg.norm(ref)) if shared else 0.0
        cosine = float(own @ ref) / norm if norm > _ZERO else 0.0
        if abs(cosine) >= min_cosine:
            if cosine < 0:
                fitted.party_scores[:, axis] *= -1
                fitted.loadings[:, axis] *= -1
            records.append({"method": "aligned", "cosine": abs(cosine)})
        else:
            anchor = anchors[axis] if axis < len(anchors) else None
            party = _orient_axis_by_anchor(fitted.party_scores, fitted.loadings, fitted.parties, axis, anchor)
            records.append({"method": "anchor", "cosine": cosine, "anchor": party})
    return records


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
