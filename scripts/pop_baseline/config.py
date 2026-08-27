"""Configuration for the separately versioned Poll of Polls baseline.

The values here are evidence-backed defaults, not parameters fitted to the
election outcomes in this repository.  See ``docs/pop_baseline.md`` for the
source record and the limitations of the reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.pollofpolls.state_config import ALL_CATEGORIES


BASELINE_VERSION = "PoPBaseline-v1.0"
MODEL_ID = "pop_baseline_v1"
PARTY_ORDER: tuple[str, ...] = tuple(ALL_CATEGORIES)
PARLIAMENTARY_PARTIES: tuple[str, ...] = tuple(ALL_CATEGORIES[:8])
REFERENCE_CATEGORY = "REST"

# The 2018 first-party method explicitly used three historical horizons and
# combined equal-sized simulation batches.  The 2022/2026 posts later mention
# an 88-day step, but this baseline intentionally reconstructs the historical
# method the project set out to improve rather than silently substituting that
# later variant.
DEFAULT_STEP_WINDOWS: tuple[int, ...] = (21, 28, 35)
MIN_TRANSITIONS = 1
MIN_SHARE_PCT = 0.01

# First-party 2018 support-vote formula.  FI is part of REST in our canonical
# nine-category data, so it cannot be represented as an independent recipient;
# the supported recipients below are the parliamentary parties for which the
# source formula can be evaluated without inventing a FI category.
SUPPORT_VOTE_TARGETS: tuple[str, ...] = ("L", "C", "KD", "MP", "V")
RIGHT_BLOCK: tuple[str, ...] = ("M", "L", "C", "KD")
LEFT_BLOCK: tuple[str, ...] = ("S", "V", "MP")


@dataclass(frozen=True)
class PoPBaselineConfig:
    """Explicit baseline choices; no option changes Candidate A."""

    step_windows: tuple[int, ...] = DEFAULT_STEP_WINDOWS
    random_sign: bool = True
    compositional_space: str = "clr_aitchison_perturbation"
    apply_support_voting: bool = True
    support_voting_targets: tuple[str, ...] = SUPPORT_VOTE_TARGETS
    partial_step_policy: str = "linear_clr_fraction"

    def __post_init__(self) -> None:
        if not self.step_windows or any(int(w) <= 0 for w in self.step_windows):
            raise ValueError("step_windows must contain positive integers")
        if len(set(self.step_windows)) != len(self.step_windows):
            raise ValueError("step_windows must not contain duplicates")
        if self.compositional_space != "clr_aitchison_perturbation":
            raise ValueError("Only the documented CLR/Aitchison perturbation is supported")
        if self.partial_step_policy != "linear_clr_fraction":
            raise ValueError("Only the documented fractional final step is supported")
        unknown = set(self.support_voting_targets) - set(PARTY_ORDER)
        if unknown:
            raise ValueError(f"Unknown support-voting target(s): {sorted(unknown)}")


DEFAULT_CONFIG = PoPBaselineConfig()
