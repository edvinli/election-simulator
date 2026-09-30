"""Causal Kalman filter for latent party support in percentage points.

State: the eight named-party shares ``s`` (other parties are the remainder)
and one additive house effect vector ``b_h`` per pollster. Support follows a
random walk; a poll reads ``s + b_h`` at its fieldwork midpoint and an
election reads ``s`` on election day, each with multinomial-shaped noise.
Because elections read ``s`` without a house effect, they are what anchors
house effects; a pollster's effect drifts slowly between them.

An estimate "as of" a date runs the filter over exactly the observations
available by then, in fieldwork order, and reports the filtered state on that
date. Nothing observed later can change it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from .config import HYPERPARAMETER_NAMES, INITIAL_STATE_SD_PP, PARTIES
from .data import Observation

K = len(PARTIES)
LOG_2PI = math.log(2.0 * math.pi)


def multinomial_shape(shares_pp: np.ndarray) -> np.ndarray:
    """1e4 * (diag(pi) - pi pi^T) for the named parties, in pp^2 per respondent.

    The shape is taken at the full composition, other parties included and
    renormalized, so it stays positive definite even if the named shares of
    an estimate sum to 100 or more.
    """

    named = np.clip(np.asarray(shares_pp, dtype=float), 0.05, 99.0)
    composition = np.append(named, max(100.0 - named.sum(), 0.05))
    pi = (composition / composition.sum())[: len(named)]
    return 1e4 * (np.diag(pi) - np.outer(pi, pi))


@dataclass(frozen=True)
class Hyperparameters:
    process: float
    house_prior: float
    house_drift: float
    design_effect: float
    election_n: float

    @classmethod
    def from_log(cls, values: Sequence[float]) -> "Hyperparameters":
        return cls(*(float(math.exp(v)) for v in values))

    def to_log(self) -> list[float]:
        return [math.log(getattr(self, name)) for name in HYPERPARAMETER_NAMES]

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in HYPERPARAMETER_NAMES}


@dataclass
class FilterState:
    day: date
    mean: np.ndarray
    cov: np.ndarray
    loglik: float
    n_obs: int
    house_last_day: dict[int, date]

    def copy(self) -> "FilterState":
        return FilterState(
            self.day, self.mean.copy(), self.cov.copy(), self.loglik, self.n_obs, dict(self.house_last_day)
        )


class SupportFilter:
    """The filter for one fixed house list, reference composition and parameters."""

    def __init__(
        self,
        houses: Sequence[str],
        reference_shares: Sequence[float],
        params: Hyperparameters,
    ) -> None:
        self.house_index = {house: i for i, house in enumerate(sorted(houses))}
        self.params = params
        self.dim = K * (1 + len(self.house_index))
        # House effects are measured in the multinomial metric of a fixed
        # reference composition, so their prior and drift do not move with s.
        self.reference_shape = multinomial_shape(np.asarray(reference_shares, dtype=float))

    def initial_state(self, start: date, start_shares: Sequence[float]) -> FilterState:
        mean = np.zeros(self.dim)
        mean[:K] = start_shares
        cov = np.zeros((self.dim, self.dim))
        cov[:K, :K] = np.eye(K) * INITIAL_STATE_SD_PP**2
        house_block = self.params.house_prior * self.reference_shape
        for i in range(len(self.house_index)):
            lo = K * (1 + i)
            cov[lo : lo + K, lo : lo + K] = house_block
        return FilterState(start, mean, cov, 0.0, 0, {})

    def predict(self, state: FilterState, day: date) -> None:
        gap = (day - state.day).days
        if gap < 0:
            raise ValueError(f"Filter cannot move backwards from {state.day} to {day}")
        if gap:
            state.cov[:K, :K] += self.params.process * gap * multinomial_shape(state.mean[:K])
            state.day = day

    def _drift_house(self, state: FilterState, house: int) -> None:
        # Drift increments are independent of everything else, so applying
        # them only when the house is next read is exact.
        last = state.house_last_day.get(house)
        if last is not None:
            gap = (state.day - last).days
            if gap > 0:
                lo = K * (1 + house)
                state.cov[lo : lo + K, lo : lo + K] += self.params.house_drift * gap * self.reference_shape
        state.house_last_day[house] = state.day

    def update(self, state: FilterState, obs: Observation) -> None:
        self.predict(state, obs.obs_date)
        s = state.mean[:K]
        idx = np.arange(K)
        if obs.kind == "poll":
            house = self.house_index[obs.house]
            self._drift_house(state, house)
            idx = np.concatenate([idx, K * (1 + house) + np.arange(K)])
            noise = self.params.design_effect * multinomial_shape(s) / obs.sample_size
            noise += np.eye(K) * obs.rounding_pp**2 / 12.0
            predicted = s + state.mean[K * (1 + house) : K * (2 + house)]
        else:
            noise = multinomial_shape(s) / self.params.election_n
            predicted = s

        # H selects s (and b_h for a poll); P H^T is the sum of those columns.
        pht = state.cov[:, idx[:K]]
        if obs.kind == "poll":
            pht = pht + state.cov[:, idx[K:]]
        innovation_cov = pht[idx[:K]] + (pht[idx[K:]] if obs.kind == "poll" else 0.0) + noise
        innovation = np.asarray(obs.shares) - predicted

        chol = np.linalg.cholesky(innovation_cov)
        whitened = np.linalg.solve(chol, innovation)
        gain_t = np.linalg.solve(chol.T, np.linalg.solve(chol, pht.T))
        state.mean += gain_t.T @ innovation
        state.cov -= pht @ gain_t
        state.cov = 0.5 * (state.cov + state.cov.T)
        state.loglik -= 0.5 * (K * LOG_2PI + 2.0 * np.log(np.diag(chol)).sum() + whitened @ whitened)
        state.n_obs += 1

    def run(
        self,
        state: FilterState,
        observations: Sequence[Observation],
        checkpoints: list[tuple[str, FilterState]] | None = None,
    ) -> FilterState:
        for obs in observations:
            self.update(state, obs)
            if checkpoints is not None:
                checkpoints.append((obs.key, state.copy()))
        return state

    def consensus_reading_at(
        self, state: FilterState, day: date, weights: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of what a weighted mix of pollsters would read on ``day``.

        The reading is ``s + sum_h w_h b_h``: latent support plus the
        weighted mean house effect. It does not alter the filter; it only
        reports the level on the poll-consensus scale. Each house effect's
        drift since it was last read is included, as the filter would add it.
        """

        gap = (day - state.day).days
        if gap < 0:
            raise ValueError(f"Cannot report {day} from a state on {state.day}")
        total = sum(weights.values())
        selector = np.zeros((K, self.dim))
        selector[:, :K] = np.eye(K)
        cov = state.cov.copy()
        cov[:K, :K] += self.params.process * gap * multinomial_shape(state.mean[:K])
        for house, weight in weights.items():
            index = self.house_index[house]
            lo = K * (1 + index)
            selector[:, lo : lo + K] = (weight / total) * np.eye(K)
            last = state.house_last_day.get(index)
            if last is not None:
                cov[lo : lo + K, lo : lo + K] += self.params.house_drift * (day - last).days * self.reference_shape
        return selector @ state.mean, selector @ cov @ selector.T

    def support_at(self, state: FilterState, day: date) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of the named shares on ``day`` (not before the state)."""

        gap = (day - state.day).days
        if gap < 0:
            raise ValueError(f"Cannot report {day} from a state on {state.day}")
        cov = state.cov[:K, :K] + self.params.process * gap * multinomial_shape(state.mean[:K])
        return state.mean[:K].copy(), cov
