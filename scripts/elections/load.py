"""Load and align official election returns into 9-category forecast space."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping
import pandas as pd

from .config import DEFAULT_PROCESSED_DIR, PARLIAMENTARY_PARTIES
from .parse import _clean_html_text


def load_election_targets_for_forecasting(
    processed_file: Path | str | None = None,
) -> dict[date, dict[str, float]]:
    """Load official election results and aggregate FI + OTHER -> REST using exact integer votes.

    Returns:
        dict mapping election_date -> {party: vote_share_pct} for the 9 forecast categories:
        M, L, C, KD, S, V, MP, SD, REST.
    """
    csv_path = Path(processed_file) if processed_file else DEFAULT_PROCESSED_DIR / "riksdag_election_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing canonical election results file at {csv_path}")

    df = pd.read_csv(csv_path)

    election_targets: dict[date, dict[str, float]] = {}

    for election_date_str, group in df.groupby("election_date"):
        d_val = date.fromisoformat(str(election_date_str))
        valid_total = int(group["valid_votes_total"].iloc[0])

        votes_by_party: dict[str, int] = {p: 0 for p in PARLIAMENTARY_PARTIES}
        rest_votes = 0

        for _, row in group.iterrows():
            party = str(row["party"]).strip().upper()
            votes = int(row["votes"])

            if party in PARLIAMENTARY_PARTIES:
                votes_by_party[party] = votes
            elif party in ("FI", "OTHER"):
                rest_votes += votes
            else:
                raise ValueError(f"Unknown party in canonical dataset: {party}")

        votes_by_party["REST"] = rest_votes

        # Verify sum of integer votes matches valid_total exactly
        if sum(votes_by_party.values()) != valid_total:
            raise ValueError(
                f"Election {d_val}: Sum of votes ({sum(votes_by_party.values())}) "
                f"does not match valid_votes_total ({valid_total})"
            )

        # Calculate exact shares in percentage space
        target_shares = {
            party: round((votes / valid_total) * 100.0, 6)
            for party, votes in votes_by_party.items()
        }

        # Normalize sum to exactly 100.0% if slight floating rounding
        sum_sh = sum(target_shares.values())
        if abs(sum_sh - 100.0) > 0.001:
            raise ValueError(f"Election {d_val}: Target shares sum ({sum_sh:.4f}%) deviates from 100%")

        election_targets[d_val] = target_shares

    return election_targets


def get_election_target(
    election_date: date | str,
    processed_file: Path | str | None = None,
) -> dict[str, float]:
    """Get the 9-party target composition for a specific election date."""
    d_val = date.fromisoformat(election_date) if isinstance(election_date, str) else election_date
    targets = load_election_targets_for_forecasting(processed_file)
    if d_val not in targets:
        raise KeyError(f"No election targets found for date {d_val}. Available: {list(targets.keys())}")
    return targets[d_val]
