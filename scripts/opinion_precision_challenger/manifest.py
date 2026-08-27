"""Canonical rolling backtest case manifest generator (Experiment 2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple
import pandas as pd

from scripts.pollofpolls.normalize import parse_date
from .config import (
    DEFAULT_HORIZONS,
    DEFAULT_ORIGIN_STEP_DAYS,
    POP_TIMESERIES_FILE,
    START_ORIGIN_DATE,
)


@dataclass(frozen=True)
class RollingCaseManifestRecord:
    """A single canonical backtest evaluation case."""

    origin_date: date
    horizon_days: int
    target_date: date
    calendar_year: int
    calendar_block_6m: str  # e.g. "2014_H1", "2014_H2"


def generate_6m_block_id(d: date) -> str:
    """Generate 6-month calendar block identifier: YYYY_H1 (Jan-Jun) or YYYY_H2 (Jul-Dec)."""
    half = "H1" if d.month <= 6 else "H2"
    return f"{d.year}_{half}"


def build_canonical_rolling_manifest(
    pop_file: Path | str = POP_TIMESERIES_FILE,
    start_date: date = START_ORIGIN_DATE,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    step_days: int = DEFAULT_ORIGIN_STEP_DAYS,
) -> List[RollingCaseManifestRecord]:
    """Generate the exact canonical rolling backtest case manifest.

    A case (origin_date, horizon) is eligible if:
        1. origin_date is on weekly stepping >= start_date
        2. origin_date exists in PoP series
        3. (origin_date + horizon) exists in PoP series
    """
    df = pd.read_csv(pop_file)
    available_dates = set(parse_date(d) for d in df["date"])
    latest_date = max(available_dates)

    manifest: List[RollingCaseManifestRecord] = []
    current_origin = start_date

    while current_origin <= latest_date:
        if current_origin in available_dates:
            for h in horizons:
                target_d = current_origin + timedelta(days=h)
                if target_d in available_dates:
                    rec = RollingCaseManifestRecord(
                        origin_date=current_origin,
                        horizon_days=h,
                        target_date=target_d,
                        calendar_year=current_origin.year,
                        calendar_block_6m=generate_6m_block_id(current_origin),
                    )
                    manifest.append(rec)
        current_origin += timedelta(days=step_days)

    return manifest
